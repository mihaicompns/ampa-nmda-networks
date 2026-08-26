"""
Run and save complete balanced conical Crank-Nicolson simulations.

This module owns the production save workflow that tests should exercise:
receive simulation parameters, prepare or receive spike trains, run the real
balanced simulation, save inputs/outputs/metadata/statistics, and return the
generated results.
"""

from dataclasses import dataclass, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import multiprocessing
import re
from typing import Optional, Any

import numpy as np
from brian2 import Hz, second, um, meter, have_same_dimensions
from scipy.stats import expon, uniform

from TaperredDendritesBalancedCrankNicolson import (
    plot_crank_nicolson_unitless_closed_tapered_cone_balance,
    simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param,
)
from TaperredDendritesPDE import ConicalNumericalCableParameters, create_delta_pulses
from src.iteration_19_tapered_dendrites.data import to_SI
from CylindricalDendritesEventSimulation import (
    simulate_crank_nicolson_unitless_closed_cylinder_balance_with_param,
)


@dataclass(frozen=True)
class SavedBalancedConicalSimulation:
    save_dir: Path
    inputs_file: Path
    outputs_file: Path
    metadata_file: Path
    statistics_file: Path
    graphs_dir: Path
    graph_file: Optional[Path]
    times: np.ndarray
    V_s: np.ndarray
    excitatory_events: np.ndarray
    inhibitory_events: np.ndarray
    metadata: dict
    statistics: dict


@dataclass(frozen=True)
class BalancedSimulationMetadata:
    """
    Parameters that identify a saved balanced simulation on disk.
    """
    simulation_label: str
    e_limits: Optional[tuple[float, float]]
    i_limits: Optional[tuple[float, float]]
    r_e_density: Any
    r_i_density: Any
    t_max: float
    dt: float = 1e-8
    spatial_points: int = 101
    g: Optional[float] = None
    seed: Optional[int] = None
    gamma: Optional[float] = None
    base_rate: Any = None
    base_I_e_strength: Any = None

    @classmethod
    def from_brunel_params(
            cls,
            simulation_label: str,
            e_limits: tuple[float, float],
            i_limits: tuple[float, float],
            t_max,
            gamma: float,
            g: float,
            dt=1e-8,
            spatial_points: int = 101,
            base_rate=0.4 * Hz / um,
            base_I_e_strength=None,
            seed: Optional[int] = None):
        """
        Build simulation metadata from Brunel-style balance parameters.

        gamma scales the inhibitory event density relative to the excitatory
        density. g records the inhibitory-to-excitatory strength ratio.
        """
        return cls(
            simulation_label=simulation_label,
            e_limits=e_limits,
            i_limits=i_limits,
            r_e_density=base_rate,
            r_i_density=gamma * base_rate,
            t_max=to_SI(t_max),
            dt=to_SI(dt),
            spatial_points=spatial_points,
            g=g,
            seed=seed,
            gamma=gamma,
            base_rate=base_rate,
            base_I_e_strength=base_I_e_strength,
        )

    @classmethod
    def from_brunnel_params(cls, *args, **kwargs):
        return cls.from_brunel_params(*args, **kwargs)

    # Folder design:
    # <label>__e_<left>_<right>um__i_<left>_<right>um__
    # re_<hz_per_um>hz_per_um__ri_<hz_per_um>hz_per_um__
    # t_<duration_ms>ms__dt_<time_step_ps>ps__mol_n_<spatial_points>__g_<ratio>
    def save_label(self) -> str:
        pieces = [
            self.simulation_label,
            _limits_label("e", self.e_limits),
            _limits_label("i", self.i_limits),
            f"re_{_rate_density_label(self.r_e_density)}",
            f"ri_{_rate_density_label(self.r_i_density)}",
            f"t_{_duration_label(self.t_max)}",
            f"dt_{_picosecond_label(self.dt)}ps",
            f"mol_n_{int(self.spatial_points)}",
        ]
        if self.g is not None:
            pieces.append(f"g_{_number_label(self.g)}")
        return _safe_label("__".join(piece for piece in pieces if piece))


@dataclass
class BalancedComparisonFutures:
    save_dir: Path
    executor: ProcessPoolExecutor
    futures: dict

    def done(self):
        return all(future.done() for future in self.futures)

    def completed(self):
        results = {}
        for future, geometry in self.futures.items():
            if future.done():
                results[geometry] = future.result()
        return results

    def result(self):
        results = {}
        for future in as_completed(self.futures):
            geometry = self.futures[future]
            results[geometry] = future.result()
        self.executor.shutdown(wait=False)
        return {
            "save_dir": self.save_dir,
            "conical": results["conical"],
            "cylindrical": results["cylindrical"],
        }


def _safe_label(label: str) -> str:
    label = label.replace("/", "_")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", label.strip().lower()).strip("_")


def _as_float(value) -> float:
    try:
        return float(to_SI(value))
    except Exception:
        return float(value)


def _number_label(value, precision: int = 6) -> str:
    text = f"{_as_float(value):.{precision}g}"
    return text.replace("-", "m").replace(".", "p")


def _um_value(value) -> float:
    return _as_float(value) / 1e-6


def _um_label(value) -> str:
    value_um = _um_value(value)
    if abs(value_um - round(value_um)) < 1e-9:
        return str(int(round(value_um)))
    return _number_label(value_um)


def _limits_label(prefix: str, limits: Optional[tuple[float, float]]) -> str:
    if limits is None:
        return ""
    left, right = limits
    return f"{prefix}_{_um_label(left)}_{_um_label(right)}um"


def _rate_density_label(value) -> str:
    return f"{_number_label(_rate_density_hz_per_um(value))}hz_per_um"


def _rate_density_hz_per_um(value) -> float:
    try:
        if have_same_dimensions(value, Hz / um):
            return float(value / (Hz / um))
    except Exception:
        pass
    return _as_float(value)


def _duration_label(value) -> str:
    seconds = _as_float(value)
    milliseconds = seconds * 1000.0
    if abs(milliseconds - round(milliseconds)) < 1e-9:
        return f"{int(round(milliseconds))}ms"
    return f"{_number_label(milliseconds)}ms"


def _picosecond_label(value) -> str:
    picoseconds = _as_float(value) / 1e-12
    if abs(picoseconds - round(picoseconds)) < 1e-6:
        return str(int(round(picoseconds)))
    return _number_label(picoseconds)


def _simulation_metadata(
        simulation_label: str,
        t_max: float,
        e_limits: Optional[tuple[float, float]],
        i_limits: Optional[tuple[float, float]],
        r_e_density,
        r_i_density,
        g: Optional[float],
        seed: Optional[int],
        dt: float,
        spatial_points: int,
        experiment_metadata: Optional[BalancedSimulationMetadata]) -> BalancedSimulationMetadata:
    if experiment_metadata is not None:
        return replace(
            experiment_metadata,
            e_limits=e_limits,
            i_limits=i_limits,
            t_max=t_max,
            dt=dt,
            spatial_points=spatial_points,
        )
    return BalancedSimulationMetadata(
        simulation_label=simulation_label,
        e_limits=e_limits,
        i_limits=i_limits,
        r_e_density=r_e_density,
        r_i_density=r_i_density,
        t_max=t_max,
        dt=dt,
        spatial_points=spatial_points,
        g=g,
        seed=seed,
    )


def _event_count(events: np.ndarray) -> int:
    if events.size == 0:
        return 0
    return int(events.shape[1])


def _full_cable_limits(p) -> tuple[float, float]:
    return float(p.x[0]), float(p.x[-1])


def _limits_from_static_events_or_cable(events: np.ndarray, p) -> tuple[float, float]:
    if _event_count(events) > 2:
        positions = events[1]
        return float(np.min(positions)), float(np.max(positions))
    return _full_cable_limits(p)


def _complete_limits_for_label(
        p,
        excitatory_events: np.ndarray,
        inhibitory_events: np.ndarray,
        e_limits: Optional[tuple[float, float]],
        i_limits: Optional[tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    if e_limits is None:
        e_limits = _limits_from_static_events_or_cable(excitatory_events, p)
    if i_limits is None:
        i_limits = _limits_from_static_events_or_cable(inhibitory_events, p)
    return e_limits, i_limits


def _events_as_list(events: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in events.tolist()]

def _generate_uniform_events(
        t_max: float,
        e_limits: tuple[float, float],
        i_limits: tuple[float, float],
        seed: Optional[int],
        r_e_density=0.4 * Hz / um,
        r_i_density=0.1 * Hz / um):
    e_left, e_right = e_limits
    i_left, i_right = i_limits

    r_e = r_e_density * (e_right - e_left) * meter
    r_i = r_i_density * (i_right - i_left) * meter

    excitatory_events = create_delta_pulses(
        t_max=t_max,
        x_distribution=uniform(loc=e_left, scale=e_right - e_left),
        t_distribution=expon(scale=1.0 / r_e),
        seed=seed,
    )
    inhibitory_events = create_delta_pulses(
        t_max=t_max,
        x_distribution=uniform(loc=i_left, scale=i_right - i_left),
        t_distribution=expon(scale=1.0 / r_i),
        seed=None if seed is None else seed + 1,
    )

    return excitatory_events, inhibitory_events


def _default_desired_positions(p) -> list[float]:
    return [
        float(0.2 * p.L / 1e-6),
        float(0.5 * p.L / 1e-6),
        float(0.8 * p.L / 1e-6),
    ]


def _geometry_metadata(p, geometry_type: str) -> dict:
    geometry = {
        "cable_length_um": float(p.L / 1e-6),
        "spatial_points": int(len(p.x)),
        "spatial_range_um": [float(p.x[0] / 1e-6), float(p.x[-1] / 1e-6)],
        "dx_um": float(p.dx / 1e-6),
    }
    if geometry_type == "tapered_cone":
        geometry.update({
            "r_at_0_um": float(p.r_at_0 / 1e-6),
            "r_at_L_um": float(p.r_at_L / 1e-6),
        })
    else:
        geometry["r0_um"] = float(p.r0 / 1e-6)
    return geometry


def _output_statistics(excitatory_events: np.ndarray, inhibitory_events: np.ndarray, V_s: np.ndarray) -> dict:
    return {
        "total_events": int(excitatory_events.shape[1] + inhibitory_events.shape[1]),
        "excitatory_events": int(excitatory_events.shape[1]),
        "inhibitory_events": int(inhibitory_events.shape[1]),
        "voltage_min_mV": float(np.min(V_s) * 1000.0),
        "voltage_max_mV": float(np.max(V_s) * 1000.0),
        "voltage_mean_mV": float(np.mean(V_s) * 1000.0),
        "voltage_std_mV": float(np.std(V_s) * 1000.0),
        "voltage_range_mV": float((np.max(V_s) - np.min(V_s)) * 1000.0),
    }


def _prepare_balanced_save_folders(save_dir: Path):
    inputs_dir = save_dir / "inputs"
    outputs_dir = save_dir / "outputs"
    graphs_dir = save_dir / "graphs"
    metadata_dir = save_dir / "metadata"
    statistics_dir = save_dir / "statistics"
    for directory in (inputs_dir, outputs_dir, graphs_dir, metadata_dir, statistics_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return inputs_dir, outputs_dir, graphs_dir, metadata_dir, statistics_dir


def _save_balanced_simulation_result(
        p,
        t_max: float,
        output_root: Path,
        simulation_label: str,
        times: np.ndarray,
        V_s: np.ndarray,
        excitatory_events: np.ndarray,
        inhibitory_events: np.ndarray,
        e_limits: tuple[float, float],
        i_limits: tuple[float, float],
        r_e_density,
        r_i_density,
        g: Optional[float],
        experiment_metadata: Optional[BalancedSimulationMetadata],
        include_metadata_in_save_dir: bool,
        seed: Optional[int],
        plot: bool,
        show_plot: bool,
        verbose: bool,
        desired_positions: Optional[list[float]],
        simulator_name: str,
        geometry_type: str) -> SavedBalancedConicalSimulation:
    simulation_metadata = _simulation_metadata(
        simulation_label=simulation_label,
        t_max=t_max,
        e_limits=e_limits,
        i_limits=i_limits,
        r_e_density=r_e_density,
        r_i_density=r_i_density,
        g=g,
        seed=seed,
        dt=p.dt,
        spatial_points=len(p.x),
        experiment_metadata=experiment_metadata,
    )
    save_label = simulation_metadata.save_label()
    save_dir_name = save_label if include_metadata_in_save_dir else _safe_label(simulation_label)
    save_dir = Path(output_root) / save_dir_name

    inputs_dir, outputs_dir, graphs_dir, metadata_dir, statistics_dir = _prepare_balanced_save_folders(save_dir)
    inputs_file = inputs_dir / "inputs.npz"
    outputs_file = outputs_dir / "outputs.npz"
    metadata_file = metadata_dir / "metadata.json"
    statistics_file = statistics_dir / "simulation_stats.json"
    graph_file = graphs_dir / "simulation_graph.png" if plot else None

    if desired_positions is None:
        desired_positions = _default_desired_positions(p)

    if plot:
        plot_crank_nicolson_unitless_closed_tapered_cone_balance(
            times=times,
            V_s=V_s,
            p=p,
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            simulation_label=simulation_label,
            desired_positions=desired_positions,
            save=True,
            graph_save_name="simulation_graph",
            graph_out_dir=graphs_dir,
            show_plot=show_plot,
            verbose=verbose,
        )

    np.savez_compressed(
        inputs_file,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
    )
    np.savez_compressed(
        outputs_file,
        times=times,
        voltage=V_s,
        x=p.x,
    )

    statistics = _output_statistics(excitatory_events, inhibitory_events, V_s)
    metadata = {
        "simulation_info": {
            "label": simulation_label,
            "save_label": save_label,
            "deterministic": True,
            "simulator": simulator_name,
            "geometry_type": geometry_type,
        },
        "geometry": _geometry_metadata(p, geometry_type),
        "electrical_properties": {
            "tau_ms": float(p.tau * 1000.0),
            "dt_ms": float(p.dt * 1000.0),
            "t_max_ms": float(t_max * 1000.0),
            "I_e_pA": float(p.I_e / 1e-12),
            "I_i_pA": float(p.I_i / 1e-12),
            "g": None if g is None else float(g),
        },
        "input_info": {
            "seed": seed,
            "e_limits_m": [float(e_limits[0]), float(e_limits[1])],
            "i_limits_m": [float(i_limits[0]), float(i_limits[1])],
            "r_e_density_hz_per_um": _rate_density_hz_per_um(simulation_metadata.r_e_density),
            "r_i_density_hz_per_um": _rate_density_hz_per_um(simulation_metadata.r_i_density),
            "total_events": statistics["total_events"],
            "excitatory_events": statistics["excitatory_events"],
            "inhibitory_events": statistics["inhibitory_events"],
            "excitatory_spike_train": _events_as_list(excitatory_events),
            "inhibitory_spike_train": _events_as_list(inhibitory_events),
        },
        "brunel_parameters": {
            "gamma": simulation_metadata.gamma,
            "g": simulation_metadata.g,
            "base_rate_hz_per_um": None
            if simulation_metadata.base_rate is None
            else _rate_density_hz_per_um(simulation_metadata.base_rate),
            "base_I_e_strength": None
            if simulation_metadata.base_I_e_strength is None
            else _as_float(simulation_metadata.base_I_e_strength),
        },
        "files": {
            "inputs": str(inputs_file.relative_to(save_dir)),
            "outputs": str(outputs_file.relative_to(save_dir)),
            "metadata": str(metadata_file.relative_to(save_dir)),
            "statistics": str(statistics_file.relative_to(save_dir)),
            "graphs": str(graphs_dir.relative_to(save_dir)),
            "graph": None if graph_file is None else str(graph_file.relative_to(save_dir)),
        },
        "output_statistics": statistics,
    }

    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    statistics_file.write_text(json.dumps(statistics, indent=2), encoding="utf-8")

    return SavedBalancedConicalSimulation(
        save_dir=save_dir,
        inputs_file=inputs_file,
        outputs_file=outputs_file,
        metadata_file=metadata_file,
        statistics_file=statistics_file,
        graphs_dir=graphs_dir,
        graph_file=graph_file,
        times=times,
        V_s=V_s,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        metadata=metadata,
        statistics=statistics,
    )


def run_and_save_balanced_conical_simulation(
        p: ConicalNumericalCableParameters,
        t_max,
        output_root: Path,
        simulation_label: str,
        excitatory_events: Optional[np.ndarray] = None,
        inhibitory_events: Optional[np.ndarray] = None,
        e_limits: Optional[tuple[float, float]] = None,
        i_limits: Optional[tuple[float, float]] = None,
        r_e_density=0.4 * Hz / um,
        r_i_density=0.1 * Hz / um,
        g: Optional[float] = None,
        experiment_metadata: Optional[BalancedSimulationMetadata] = None,
        include_metadata_in_save_dir: bool = True,
        seed: Optional[int] = None,
        saved_frames: int = 1200,
        verbose: bool = False,
        plot: bool = True,
        show_plot: bool = True,
        desired_positions: Optional[list[float]] = None) -> SavedBalancedConicalSimulation:
    """
    Run a real balanced conical Crank-Nicolson simulation and save all artifacts.

    If excitatory_events/inhibitory_events are provided, the simulation is fully
    deterministic from those static inputs. If they are omitted, e_limits and
    i_limits are used to generate deterministic uniform spike trains when seed
    is provided.
    """
    t_max = to_SI(t_max)

    if excitatory_events is None or inhibitory_events is None:
        if e_limits is None or i_limits is None:
            raise ValueError(
                "Provide either static excitatory/inhibitory events or both e_limits and i_limits."
            )
        excitatory_events, inhibitory_events = _generate_uniform_events(
            t_max=t_max,
            e_limits=e_limits,
            i_limits=i_limits,
            r_e_density=r_e_density,
            r_i_density=r_i_density,
            seed=seed,
        )

    excitatory_events = np.asarray(excitatory_events, dtype=float)
    inhibitory_events = np.asarray(inhibitory_events, dtype=float)
    e_limits, i_limits = _complete_limits_for_label(
        p=p,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        e_limits=e_limits,
        i_limits=i_limits,
    )

    times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param(
        p=p,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        t_max=t_max,
        saved_frames=saved_frames,
        verbose=verbose,
    )

    return _save_balanced_simulation_result(
        p=p,
        t_max=t_max,
        output_root=output_root,
        simulation_label=simulation_label,
        times=times,
        V_s=V_s,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        e_limits=e_limits,
        i_limits=i_limits,
        r_e_density=r_e_density,
        r_i_density=r_i_density,
        g=g,
        experiment_metadata=experiment_metadata,
        include_metadata_in_save_dir=include_metadata_in_save_dir,
        seed=seed,
        plot=plot,
        show_plot=show_plot,
        verbose=verbose,
        desired_positions=desired_positions,
        simulator_name="simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param",
        geometry_type="tapered_cone",
    )


def run_and_save_balanced_cylindrical_simulation(
        p,
        t_max,
        output_root: Path,
        simulation_label: str,
        excitatory_events: np.ndarray,
        inhibitory_events: np.ndarray,
        e_limits: Optional[tuple[float, float]] = None,
        i_limits: Optional[tuple[float, float]] = None,
        r_e_density=0.4 * Hz / um,
        r_i_density=0.1 * Hz / um,
        g: Optional[float] = None,
        experiment_metadata: Optional[BalancedSimulationMetadata] = None,
        include_metadata_in_save_dir: bool = True,
        seed: Optional[int] = None,
        saved_frames: int = 1200,
        verbose: bool = False,
        plot: bool = True,
        show_plot: bool = True,
        desired_positions: Optional[list[float]] = None) -> SavedBalancedConicalSimulation:
    """
    Run and save the balanced cylindrical comparison simulation with static E/I events.
    """
    t_max = to_SI(t_max)
    excitatory_events = np.asarray(excitatory_events, dtype=float)
    inhibitory_events = np.asarray(inhibitory_events, dtype=float)
    e_limits, i_limits = _complete_limits_for_label(
        p=p,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        e_limits=e_limits,
        i_limits=i_limits,
    )

    times, V_s = simulate_crank_nicolson_unitless_closed_cylinder_balance_with_param(
        p=p,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        t_max=t_max,
        saved_frames=saved_frames,
        verbose=verbose,
    )

    return _save_balanced_simulation_result(
        p=p,
        t_max=t_max,
        output_root=output_root,
        simulation_label=simulation_label,
        times=times,
        V_s=V_s,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        e_limits=e_limits,
        i_limits=i_limits,
        r_e_density=r_e_density,
        r_i_density=r_i_density,
        g=g,
        experiment_metadata=experiment_metadata,
        include_metadata_in_save_dir=include_metadata_in_save_dir,
        seed=seed,
        plot=plot,
        show_plot=show_plot,
        verbose=verbose,
        desired_positions=desired_positions,
        simulator_name="simulate_crank_nicolson_unitless_closed_cylinder_balance_with_param",
        geometry_type="uniform_cylinder",
    )


def sim_conical(**kwargs):
    """
    Command-style wrapper: one conical simulation plus saving.
    """
    return run_and_save_balanced_conical_simulation(**kwargs)


def sim_cylindrical(**kwargs):
    """
    Command-style wrapper: one cylindrical simulation plus saving.
    """
    return run_and_save_balanced_cylindrical_simulation(**kwargs)


def _balanced_comparison_commands(
        conical_p: ConicalNumericalCableParameters,
        t_max,
        output_root: Path,
        simulation_label: str,
        e_limits: Optional[tuple[float, float]] = None,
        i_limits: Optional[tuple[float, float]] = None,
        r_e_density=0.4 * Hz / um,
        r_i_density=0.1 * Hz / um,
        g: Optional[float] = None,
        experiment_metadata: Optional[BalancedSimulationMetadata] = None,
        excitatory_events: Optional[np.ndarray] = None,
        inhibitory_events: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        saved_frames: int = 1200,
        verbose: bool = False,
        plot: bool = True,
        show_plot: bool = True):
    t_max = to_SI(t_max)
    if excitatory_events is None or inhibitory_events is None:
        if e_limits is None or i_limits is None:
            raise ValueError(
                "Provide either static excitatory/inhibitory events or both e_limits and i_limits."
            )
        excitatory_events, inhibitory_events = _generate_uniform_events(
            t_max=t_max,
            e_limits=e_limits,
            i_limits=i_limits,
            r_e_density=r_e_density,
            r_i_density=r_i_density,
            seed=seed,
        )
    excitatory_events = np.asarray(excitatory_events, dtype=float)
    inhibitory_events = np.asarray(inhibitory_events, dtype=float)
    e_limits, i_limits = _complete_limits_for_label(
        p=conical_p,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        e_limits=e_limits,
        i_limits=i_limits,
    )

    simulation_metadata = _simulation_metadata(
        simulation_label=simulation_label,
        t_max=t_max,
        e_limits=e_limits,
        i_limits=i_limits,
        r_e_density=r_e_density,
        r_i_density=r_i_density,
        g=g,
        seed=seed,
        dt=conical_p.dt,
        spatial_points=len(conical_p.x),
        experiment_metadata=experiment_metadata,
    )
    save_label = simulation_metadata.save_label()
    base_dir = Path(output_root) / save_label
    cylindrical_p = conical_p.to_numerical_cylindrical_params()

    common_kwargs = {
        "t_max": t_max,
        "output_root": base_dir,
        "excitatory_events": excitatory_events,
        "inhibitory_events": inhibitory_events,
        "e_limits": e_limits,
        "i_limits": i_limits,
        "r_e_density": r_e_density,
        "r_i_density": r_i_density,
        "g": g,
        "experiment_metadata": simulation_metadata,
        "include_metadata_in_save_dir": False,
        "seed": seed,
        "saved_frames": saved_frames,
        "verbose": verbose,
        "plot": plot,
        "show_plot": show_plot,
    }
    return base_dir, [
        ("conical", sim_conical, {
            **common_kwargs,
            "p": conical_p,
            "simulation_label": "conical",
        }),
        ("cylindrical", sim_cylindrical, {
            **common_kwargs,
            "p": cylindrical_p,
            "simulation_label": "cylindrical",
        }),
    ]


def submit_balanced_conical_with_cylindrical_comparison(
        conical_p: ConicalNumericalCableParameters,
        t_max,
        output_root: Path,
        simulation_label: str,
        e_limits: Optional[tuple[float, float]] = None,
        i_limits: Optional[tuple[float, float]] = None,
        r_e_density=0.4 * Hz / um,
        r_i_density=0.1 * Hz / um,
        g: Optional[float] = None,
        experiment_metadata: Optional[BalancedSimulationMetadata] = None,
        excitatory_events: Optional[np.ndarray] = None,
        inhibitory_events: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        saved_frames: int = 1200,
        verbose: bool = False,
        plot: bool = True,
        show_plot: bool = True,
        max_workers: int = 2) -> BalancedComparisonFutures:
    """
    Submit cone and cylinder simulations and return immediately with futures.

    Results are saved as siblings:
      output_root/simulation_label/conical
      output_root/simulation_label/cylindrical
    """
    base_dir, commands = _balanced_comparison_commands(
        conical_p=conical_p,
        t_max=t_max,
        output_root=output_root,
        simulation_label=simulation_label,
        e_limits=e_limits,
        i_limits=i_limits,
        r_e_density=r_e_density,
        r_i_density=r_i_density,
        g=g,
        experiment_metadata=experiment_metadata,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        seed=seed,
        saved_frames=saved_frames,
        verbose=verbose,
        plot=plot,
        show_plot=show_plot,
    )

    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
    )
    futures = {
        executor.submit(function, **kwargs): geometry
        for geometry, function, kwargs in commands
    }

    return BalancedComparisonFutures(
        save_dir=base_dir,
        executor=executor,
        futures=futures,
    )


def run_and_save_balanced_conical_with_cylindrical_comparison(
        *args,
        **kwargs):
    """
    Blocking convenience wrapper around submit_balanced_conical_with_cylindrical_comparison.

    Use submit_balanced_conical_with_cylindrical_comparison for long IDE runs
    where the caller should keep working while simulations are running.
    """
    submitted = submit_balanced_conical_with_cylindrical_comparison(*args, **kwargs)
    return submitted.result()


def run_balanced_comparison_sweep(
        comparison_kwargs,
        max_workers: int = 2):
    """
    Run many conical/cylindrical comparisons through one bounded process pool.

    Only `max_workers` simulation commands are submitted at a time. Once one
    command finishes, the next command is submitted, so the executor queue does
    not accumulate every heavy simulation payload up front.
    """
    comparison_commands = []
    comparison_results = []
    for comparison_index, kwargs in enumerate(comparison_kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("max_workers", None)
        save_dir, commands = _balanced_comparison_commands(**kwargs)
        comparison_results.append({"save_dir": save_dir})
        for geometry, function, command_kwargs in commands:
            comparison_commands.append((comparison_index, geometry, function, command_kwargs))

    if not comparison_commands:
        return []

    max_workers = max(1, int(max_workers))
    command_iter = iter(comparison_commands)
    active_futures = {}

    def submit_next(executor):
        try:
            comparison_index, geometry, function, command_kwargs = next(command_iter)
        except StopIteration:
            return False
        future = executor.submit(function, **command_kwargs)
        active_futures[future] = (comparison_index, geometry)
        return True

    with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn")) as executor:
        for _ in range(min(max_workers, len(comparison_commands))):
            submit_next(executor)

        while active_futures:
            for future in as_completed(tuple(active_futures)):
                comparison_index, geometry = active_futures.pop(future)
                comparison_results[comparison_index][geometry] = future.result()
                submit_next(executor)
                break

    return comparison_results

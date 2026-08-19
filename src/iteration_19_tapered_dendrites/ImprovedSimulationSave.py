"""
Run and save complete balanced conical Crank-Nicolson simulations.

This module owns the production save workflow that tests should exercise:
receive simulation parameters, prepare or receive spike trains, run the real
balanced simulation, save inputs/outputs/metadata/statistics, and return the
generated results.
"""

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import multiprocessing
import re
from typing import Optional

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


def _events_as_list(events: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in events.tolist()]

def _generate_uniform_events(
        t_max: float,
        e_limits: tuple[float, float],
        i_limits: tuple[float, float],
        seed: Optional[int]):
    r_e_density = 0.02 * Hz / um
    r_i_density = 0.01 * Hz / um

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


def run_and_save_balanced_conical_simulation(
        p: ConicalNumericalCableParameters,
        t_max,
        output_root: Path,
        simulation_label: str,
        excitatory_events: Optional[np.ndarray] = None,
        inhibitory_events: Optional[np.ndarray] = None,
        e_limits: Optional[tuple[float, float]] = None,
        i_limits: Optional[tuple[float, float]] = None,
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
            seed=seed,
        )

    excitatory_events = np.asarray(excitatory_events, dtype=float)
    inhibitory_events = np.asarray(inhibitory_events, dtype=float)

    save_dir = Path(output_root) / _safe_label(simulation_label)
    inputs_dir = save_dir / "inputs"
    outputs_dir = save_dir / "outputs"
    graphs_dir = save_dir / "graphs"
    metadata_dir = save_dir / "metadata"
    statistics_dir = save_dir / "statistics"
    for directory in (inputs_dir, outputs_dir, graphs_dir, metadata_dir, statistics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    inputs_file = inputs_dir / "inputs.npz"
    outputs_file = outputs_dir / "outputs.npz"
    metadata_file = metadata_dir / "metadata.json"
    statistics_file = statistics_dir / "simulation_stats.json"
    graph_file = graphs_dir / "simulation_graph.png" if plot else None
    if desired_positions is None:
        desired_positions = [
            float(0.2 * p.L / 1e-6),
            float(0.5 * p.L / 1e-6),
            float(0.8 * p.L / 1e-6),
        ]

    times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param(
        p=p,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        t_max=t_max,
        saved_frames=saved_frames,
        verbose=verbose,
    )

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

    statistics = {
        "total_events": int(excitatory_events.shape[1] + inhibitory_events.shape[1]),
        "excitatory_events": int(excitatory_events.shape[1]),
        "inhibitory_events": int(inhibitory_events.shape[1]),
        "voltage_min_mV": float(np.min(V_s) * 1000.0),
        "voltage_max_mV": float(np.max(V_s) * 1000.0),
        "voltage_mean_mV": float(np.mean(V_s) * 1000.0),
        "voltage_std_mV": float(np.std(V_s) * 1000.0),
        "voltage_range_mV": float((np.max(V_s) - np.min(V_s)) * 1000.0),
    }

    metadata = {
        "simulation_info": {
            "label": simulation_label,
            "deterministic": True,
            "simulator": "simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param",
            "geometry_type": "tapered_cone",
        },
        "geometry": {
            "cable_length_um": float(p.L / 1e-6),
            "spatial_points": int(len(p.x)),
            "spatial_range_um": [float(p.x[0] / 1e-6), float(p.x[-1] / 1e-6)],
            "dx_um": float(p.dx / 1e-6),
            "r_at_0_um": float(p.r_at_0 / 1e-6),
            "r_at_L_um": float(p.r_at_L / 1e-6),
        },
        "electrical_properties": {
            "tau_ms": float(p.tau * 1000.0),
            "dt_ms": float(p.dt * 1000.0),
            "t_max_ms": float(t_max * 1000.0),
            "I_e_pA": float(p.I_e / 1e-12),
            "I_i_pA": float(p.I_i / 1e-12),
        },
        "input_info": {
            "seed": seed,
            "e_limits_m": None if e_limits is None else [float(e_limits[0]), float(e_limits[1])],
            "i_limits_m": None if i_limits is None else [float(i_limits[0]), float(i_limits[1])],
            "total_events": statistics["total_events"],
            "excitatory_events": statistics["excitatory_events"],
            "inhibitory_events": statistics["inhibitory_events"],
            "excitatory_spike_train": _events_as_list(excitatory_events),
            "inhibitory_spike_train": _events_as_list(inhibitory_events),
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


def run_and_save_balanced_cylindrical_simulation(
        p,
        t_max,
        output_root: Path,
        simulation_label: str,
        excitatory_events: np.ndarray,
        inhibitory_events: np.ndarray,
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

    save_dir = Path(output_root) / _safe_label(simulation_label)
    inputs_dir = save_dir / "inputs"
    outputs_dir = save_dir / "outputs"
    graphs_dir = save_dir / "graphs"
    metadata_dir = save_dir / "metadata"
    statistics_dir = save_dir / "statistics"
    for directory in (inputs_dir, outputs_dir, graphs_dir, metadata_dir, statistics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    inputs_file = inputs_dir / "inputs.npz"
    outputs_file = outputs_dir / "outputs.npz"
    metadata_file = metadata_dir / "metadata.json"
    statistics_file = statistics_dir / "simulation_stats.json"
    graph_file = graphs_dir / "simulation_graph.png" if plot else None

    if desired_positions is None:
        desired_positions = [
            float(0.2 * p.L / 1e-6),
            float(0.5 * p.L / 1e-6),
            float(0.8 * p.L / 1e-6),
        ]

    times, V_s = simulate_crank_nicolson_unitless_closed_cylinder_balance_with_param(
        p=p,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        t_max=t_max,
        saved_frames=saved_frames,
        verbose=verbose,
    )

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

    statistics = {
        "total_events": int(excitatory_events.shape[1] + inhibitory_events.shape[1]),
        "excitatory_events": int(excitatory_events.shape[1]),
        "inhibitory_events": int(inhibitory_events.shape[1]),
        "voltage_min_mV": float(np.min(V_s) * 1000.0),
        "voltage_max_mV": float(np.max(V_s) * 1000.0),
        "voltage_mean_mV": float(np.mean(V_s) * 1000.0),
        "voltage_std_mV": float(np.std(V_s) * 1000.0),
        "voltage_range_mV": float((np.max(V_s) - np.min(V_s)) * 1000.0),
    }

    metadata = {
        "simulation_info": {
            "label": simulation_label,
            "deterministic": True,
            "simulator": "simulate_crank_nicolson_unitless_closed_cylinder_balance_with_param",
            "geometry_type": "uniform_cylinder",
        },
        "geometry": {
            "cable_length_um": float(p.L / 1e-6),
            "spatial_points": int(len(p.x)),
            "spatial_range_um": [float(p.x[0] / 1e-6), float(p.x[-1] / 1e-6)],
            "dx_um": float(p.dx / 1e-6),
            "r0_um": float(p.r0 / 1e-6),
        },
        "electrical_properties": {
            "tau_ms": float(p.tau * 1000.0),
            "dt_ms": float(p.dt * 1000.0),
            "t_max_ms": float(t_max * 1000.0),
            "I_e_pA": float(p.I_e / 1e-12),
            "I_i_pA": float(p.I_i / 1e-12),
        },
        "input_info": {
            "seed": seed,
            "total_events": statistics["total_events"],
            "excitatory_events": statistics["excitatory_events"],
            "inhibitory_events": statistics["inhibitory_events"],
            "excitatory_spike_train": _events_as_list(excitatory_events),
            "inhibitory_spike_train": _events_as_list(inhibitory_events),
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


def submit_balanced_conical_with_cylindrical_comparison(
        conical_p: ConicalNumericalCableParameters,
        t_max,
        output_root: Path,
        simulation_label: str,
        e_limits: Optional[tuple[float, float]] = None,
        i_limits: Optional[tuple[float, float]] = None,
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
            seed=seed,
        )
    excitatory_events = np.asarray(excitatory_events, dtype=float)
    inhibitory_events = np.asarray(inhibitory_events, dtype=float)

    base_dir = Path(output_root) / _safe_label(simulation_label)
    cylindrical_p = conical_p.to_numerical_cylindrical_params()

    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
    )
    futures = {
        executor.submit(
            sim_conical,
            p=conical_p,
            t_max=t_max,
            output_root=base_dir,
            simulation_label="conical",
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            e_limits=e_limits,
            i_limits=i_limits,
            seed=seed,
            saved_frames=saved_frames,
            verbose=verbose,
            plot=plot,
            show_plot=show_plot,
        ): "conical",
        executor.submit(
            sim_cylindrical,
            p=cylindrical_p,
            t_max=t_max,
            output_root=base_dir,
            simulation_label="cylindrical",
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            seed=seed,
            saved_frames=saved_frames,
            verbose=verbose,
            plot=plot,
            show_plot=show_plot,
        ): "cylindrical",
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

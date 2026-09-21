import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, mV, nF, nS
from scipy.optimize import least_squares

from src.iteration_20_nonlin_dynamics_NMDA.nmda_spike_model import (
    NMDASpikeParameters,
    simulate_nmda_spike_plain,
)


REFERENCE_DATA_DIR = (
    Path(__file__).resolve().parent
    / "reference_data"
    / "apic20_standard_e_l_moving_nullcline_animation"
)
STANDARD_APIC20_TRACE_CSV = (
    REFERENCE_DATA_DIR / "csv" / "apic20_e_l_minus_70_above_threshold_3_spike.csv"
)
STANDARD_APIC20_METADATA_CSV = (
    REFERENCE_DATA_DIR / "csv" / "apic20_e_l_minus_70_metadata.csv"
)


@dataclass(frozen=True)
class Apic20ReferenceTrace:
    trace_path: Path
    metadata_path: Path
    condition: str
    n_spikes: int
    spike_times_ms: np.ndarray
    t_ms: np.ndarray
    v_local_mV: np.ndarray
    g_nmda_nS: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class SingleCompartmentFitParameters:
    g_nmda_max_nS: float = 13.9
    capacitance_nF: float = 0.5
    g_leak_nS: float = 0.0
    tau_rise_ms: float = 2.0
    tau_decay_ms: float = 100.0
    alpha_per_ms: float = 0.5
    presynaptic_spike_weight: float = 1.0

    def to_nmda_parameters(self, target_t_ms, initial_v_mV):
        target_t_ms = np.asarray(target_t_ms, dtype=float)
        dt_ms = _uniform_dt_ms(target_t_ms)
        return NMDASpikeParameters(
            dt=dt_ms * ms,
            t_stop=float(target_t_ms[-1]) * ms,
            capacitance=self.capacitance_nF * nF,
            g_leak=self.g_leak_nS * nS,
            e_leak=initial_v_mV * mV,
            g_nmda_max=self.g_nmda_max_nS * nS,
            tau_rise=self.tau_rise_ms * ms,
            tau_decay=self.tau_decay_ms * ms,
            alpha=self.alpha_per_ms / ms,
            presynaptic_spike_weight=self.presynaptic_spike_weight,
            initial_v=initial_v_mV * mV,
        )


@dataclass(frozen=True)
class SingleCompartmentFitResult:
    fit_params: SingleCompartmentFitParameters
    simulation: dict
    metrics: dict
    optimizer_result: object
    history: tuple = ()


@dataclass(frozen=True)
class SingleCompartmentFitHistoryFrame:
    iteration_index: int
    fit_params: SingleCompartmentFitParameters
    simulation: dict
    metrics: dict


def read_standard_apic20_reference(
    trace_path=STANDARD_APIC20_TRACE_CSV,
    metadata_path=STANDARD_APIC20_METADATA_CSV,
    condition="above threshold",
    n_spikes=3,
):
    trace_path = Path(trace_path)
    metadata_path = Path(metadata_path)
    trace = np.genfromtxt(trace_path, delimiter=",", names=True, dtype=None, encoding=None)
    metadata = _read_reference_metadata_row(
        metadata_path=metadata_path,
        condition=condition,
        n_spikes=n_spikes,
    )
    spike_times_ms = spike_times_from_reference_metadata(metadata)
    return Apic20ReferenceTrace(
        trace_path=trace_path,
        metadata_path=metadata_path,
        condition=metadata["condition"],
        n_spikes=int(metadata["n_spikes"]),
        spike_times_ms=spike_times_ms,
        t_ms=np.asarray(trace["t_ms"], dtype=float),
        v_local_mV=np.asarray(trace["v_local_mV"], dtype=float),
        g_nmda_nS=np.asarray(trace["g_nmda_nS"], dtype=float),
        metadata=metadata,
    )


def spike_times_from_reference_metadata(metadata):
    first_spike_time_ms = float(metadata["first_spike_time_ms"])
    spike_dt_ms = float(metadata["spike_dt_ms"])
    n_spikes = int(metadata["n_spikes"])
    return first_spike_time_ms + spike_dt_ms * np.arange(n_spikes, dtype=float)


def simulate_single_compartment_trace(
    fit_params,
    target_t_ms,
    spike_times_ms,
    initial_v_mV,
):
    params = fit_params.to_nmda_parameters(
        target_t_ms=target_t_ms,
        initial_v_mV=initial_v_mV,
    )
    spike_times = [float(spike_time_ms) * ms for spike_time_ms in spike_times_ms]
    simulation = simulate_nmda_spike_plain(spike_times=spike_times, params=params)
    expected_n = len(np.asarray(target_t_ms))
    return {
        "t_ms": np.asarray(simulation["t_ms"][:expected_n], dtype=float),
        "V_mV": np.asarray(simulation["V_mV"][:expected_n], dtype=float),
        "x": np.asarray(simulation["x"][:expected_n], dtype=float),
        "s": np.asarray(simulation["s"][:expected_n], dtype=float),
        "I_NMDA_nA": np.asarray(simulation["I_NMDA_nA"][:expected_n], dtype=float),
    }


def fit_single_compartment_to_reference(
    reference=None,
    initial=SingleCompartmentFitParameters(),
    max_nfev=250,
    fit_start_ms=45.0,
    free_parameter_names=None,
    collect_history=False,
):
    if reference is None:
        reference = read_standard_apic20_reference()
    initial_v_mV = float(reference.v_local_mV[0])
    return fit_single_compartment_to_trace(
        target_t_ms=reference.t_ms,
        target_v_mV=reference.v_local_mV,
        spike_times_ms=reference.spike_times_ms,
        initial_v_mV=initial_v_mV,
        initial=initial,
        max_nfev=max_nfev,
        fit_start_ms=fit_start_ms,
        free_parameter_names=free_parameter_names,
        collect_history=collect_history,
    )


def fit_single_compartment_to_reference_with_fixed_passive_parameters(
    reference=None,
    initial=SingleCompartmentFitParameters(),
    fixed_capacitance_nF=None,
    fixed_g_leak_nS=None,
    fixed_tau_rise_ms=None,
    fixed_tau_decay_ms=None,
    fixed_alpha_per_ms=None,
    max_nfev=250,
    fit_start_ms=45.0,
    free_parameter_names=None,
    collect_history=False,
):
    if reference is None:
        reference = read_standard_apic20_reference()
    initial_v_mV = float(reference.v_local_mV[0])
    return fit_single_compartment_to_trace_with_fixed_passive_parameters(
        target_t_ms=reference.t_ms,
        target_v_mV=reference.v_local_mV,
        spike_times_ms=reference.spike_times_ms,
        initial_v_mV=initial_v_mV,
        initial=initial,
        fixed_capacitance_nF=fixed_capacitance_nF,
        fixed_g_leak_nS=fixed_g_leak_nS,
        fixed_tau_rise_ms=fixed_tau_rise_ms,
        fixed_tau_decay_ms=fixed_tau_decay_ms,
        fixed_alpha_per_ms=fixed_alpha_per_ms,
        max_nfev=max_nfev,
        fit_start_ms=fit_start_ms,
        free_parameter_names=free_parameter_names,
        collect_history=collect_history,
    )


def fit_single_compartment_to_trace_with_fixed_passive_parameters(
    target_t_ms,
    target_v_mV,
    spike_times_ms,
    initial_v_mV,
    initial=SingleCompartmentFitParameters(),
    fixed_capacitance_nF=None,
    fixed_g_leak_nS=None,
    fixed_tau_rise_ms=None,
    fixed_tau_decay_ms=None,
    fixed_alpha_per_ms=None,
    max_nfev=250,
    fit_start_ms=45.0,
    free_parameter_names=None,
    collect_history=False,
):
    if fixed_capacitance_nF is None:
        raise ValueError("fixed_capacitance_nF must be provided")
    if fixed_g_leak_nS is None:
        raise ValueError("fixed_g_leak_nS must be provided")
    if fixed_tau_rise_ms is None:
        raise ValueError("fixed_tau_rise_ms must be provided")
    if fixed_tau_decay_ms is None:
        raise ValueError("fixed_tau_decay_ms must be provided")
    if fixed_alpha_per_ms is None:
        raise ValueError("fixed_alpha_per_ms must be provided")
    if free_parameter_names is None:
        free_parameter_names = (
            "g_nmda_max_nS",
            "presynaptic_spike_weight",
        )
    fixed_parameter_names = {
        "capacitance_nF",
        "g_leak_nS",
        "tau_rise_ms",
        "tau_decay_ms",
        "alpha_per_ms",
    }
    unexpected_fixed_parameter_names = fixed_parameter_names.intersection(
        free_parameter_names
    )
    if unexpected_fixed_parameter_names:
        raise ValueError(
            "free_parameter_names cannot include fixed passive parameters: "
            f"{sorted(unexpected_fixed_parameter_names)}"
        )

    fixed_initial = replace(
        initial,
        capacitance_nF=float(fixed_capacitance_nF),
        g_leak_nS=float(fixed_g_leak_nS),
        tau_rise_ms=float(fixed_tau_rise_ms),
        tau_decay_ms=float(fixed_tau_decay_ms),
        alpha_per_ms=float(fixed_alpha_per_ms),
    )
    return fit_single_compartment_to_trace(
        target_t_ms=target_t_ms,
        target_v_mV=target_v_mV,
        spike_times_ms=spike_times_ms,
        initial_v_mV=initial_v_mV,
        initial=fixed_initial,
        max_nfev=max_nfev,
        fit_start_ms=fit_start_ms,
        free_parameter_names=free_parameter_names,
        collect_history=collect_history,
    )


def fit_single_compartment_to_trace(
    target_t_ms,
    target_v_mV,
    spike_times_ms,
    initial_v_mV,
    initial=SingleCompartmentFitParameters(),
    max_nfev=250,
    fit_start_ms=45.0,
    free_parameter_names=None,
    collect_history=False,
):
    target_t_ms = np.asarray(target_t_ms, dtype=float)
    target_v_mV = np.asarray(target_v_mV, dtype=float)
    if free_parameter_names is None:
        free_parameter_names = (
            "g_nmda_max_nS",
            "capacitance_nF",
            "g_leak_nS",
            "tau_decay_ms",
            "alpha_per_ms",
            "presynaptic_spike_weight",
        )

    fit_mask = target_t_ms >= fit_start_ms
    if not np.any(fit_mask):
        raise ValueError("fit_start_ms excludes the whole target trace")

    history = []

    def build_simulation_and_metrics(log_values):
        fit_params = _replace_log_parameters(
            initial=initial,
            parameter_names=free_parameter_names,
            log_values=log_values,
        )
        simulation = simulate_single_compartment_trace(
            fit_params=fit_params,
            target_t_ms=target_t_ms,
            spike_times_ms=spike_times_ms,
            initial_v_mV=initial_v_mV,
        )
        metrics = compute_trace_fit_metrics(
            target_t_ms=target_t_ms,
            target_v_mV=target_v_mV,
            fitted_t_ms=simulation["t_ms"],
            fitted_v_mV=simulation["V_mV"],
        )
        metrics["fit_start_ms"] = float(fit_start_ms)
        return fit_params, simulation, metrics

    def append_history_frame(log_values):
        fit_params, simulation, metrics = build_simulation_and_metrics(log_values)
        history.append(
            SingleCompartmentFitHistoryFrame(
                iteration_index=len(history),
                fit_params=fit_params,
                simulation=simulation,
                metrics=metrics,
            )
        )

    def residuals(log_values):
        fit_params = _replace_log_parameters(
            initial=initial,
            parameter_names=free_parameter_names,
            log_values=log_values,
        )
        simulation = simulate_single_compartment_trace(
            fit_params=fit_params,
            target_t_ms=target_t_ms,
            spike_times_ms=spike_times_ms,
            initial_v_mV=initial_v_mV,
        )
        return simulation["V_mV"][fit_mask] - target_v_mV[fit_mask]

    lower_bounds, upper_bounds = _log_bounds_for_parameters(free_parameter_names)
    x0 = _pack_log_parameters(initial, free_parameter_names)
    if collect_history:
        append_history_frame(x0)
    optimizer_result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=max_nfev,
        callback=append_history_frame if collect_history else None,
    )
    fit_params, simulation, metrics = build_simulation_and_metrics(optimizer_result.x)
    return SingleCompartmentFitResult(
        fit_params=fit_params,
        simulation=simulation,
        metrics=metrics,
        optimizer_result=optimizer_result,
        history=tuple(history),
    )


def compute_trace_fit_metrics(target_t_ms, target_v_mV, fitted_t_ms, fitted_v_mV):
    target_t_ms = np.asarray(target_t_ms, dtype=float)
    target_v_mV = np.asarray(target_v_mV, dtype=float)
    fitted_v_on_target = np.interp(
        target_t_ms,
        np.asarray(fitted_t_ms, dtype=float),
        np.asarray(fitted_v_mV, dtype=float),
    )
    error = fitted_v_on_target - target_v_mV
    return {
        "rmse_mV": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_mV": float(np.max(np.abs(error))),
        "peak_error_mV": float(np.max(fitted_v_on_target) - np.max(target_v_mV)),
        "target_peak_time_ms": float(target_t_ms[np.argmax(target_v_mV)]),
        "fitted_peak_time_ms": float(target_t_ms[np.argmax(fitted_v_on_target)]),
    }


def plot_reference_and_fit_two_panels(reference, fit_result):
    return plot_reference_and_simulation_two_panels(
        reference=reference,
        simulation=fit_result.simulation,
        simulation_label="single compartment fit",
        title="Apical 20 multicompartment trace vs single-compartment NMDA equivalent",
    )


def plot_reference_and_simulation_two_panels(
    reference,
    simulation,
    simulation_label="single compartment",
    title=None,
):
    fitted_v_on_reference_t = np.interp(
        reference.t_ms,
        np.asarray(simulation["t_ms"], dtype=float),
        np.asarray(simulation["V_mV"], dtype=float),
    )
    voltage_difference_mV = fitted_v_on_reference_t - reference.v_local_mV

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(8, 10),
        sharex=True,
    )
    axes[0].plot(
        reference.t_ms,
        reference.v_local_mV,
        color="black",
        linewidth=2,
        label="multicompartment CSV",
    )
    axes[1].plot(
        simulation["t_ms"],
        simulation["V_mV"],
        color="#1f77b4",
        linewidth=2,
        label=simulation_label,
    )
    axes[2].plot(
        reference.t_ms,
        reference.v_local_mV,
        color="black",
        linewidth=2,
        label="multicompartment CSV",
    )
    axes[2].plot(
        simulation["t_ms"],
        simulation["V_mV"],
        color="#1f77b4",
        linewidth=2,
        label=simulation_label,
    )
    axes[3].axhline(0.0, color="0.35", linewidth=1, linestyle="--")
    axes[3].plot(
        reference.t_ms,
        voltage_difference_mV,
        color="#d62728",
        linewidth=2,
        label=f"{simulation_label} - multicompartment CSV",
    )
    for ax in axes:
        ax.legend(loc="best")
    for ax in axes[:3]:
        ax.set_ylabel(r"$V_{\mathrm{local}}$ [mV]")
    axes[3].set_ylabel(r"$\Delta V$ [mV]")
    axes[3].set_xlabel(r"$t$ [ms]")
    if title is None:
        title = "Apical 20 multicompartment trace vs single-compartment simulation"
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_standard_apic20_reference_trace(reference=None):
    if reference is None:
        reference = read_standard_apic20_reference()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(
        reference.t_ms,
        reference.v_local_mV,
        color="black",
        linewidth=2,
        label="multicompartment CSV",
    )
    ax.set_title("Apical 20 multicompartment reference trace")
    ax.set_xlabel(r"$t$ [ms]")
    ax.set_ylabel(r"$V_{\mathrm{local}}$ [mV]")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax, reference


def _read_reference_metadata_row(metadata_path, condition, n_spikes):
    with Path(metadata_path).open(newline="") as metadata_file:
        for row in csv.DictReader(metadata_file):
            if row["condition"] == condition and int(row["n_spikes"]) == int(n_spikes):
                return row
    raise ValueError(f"No metadata row for condition={condition}, n_spikes={n_spikes}")


def _uniform_dt_ms(t_ms):
    t_ms = np.asarray(t_ms, dtype=float)
    dts = np.diff(t_ms)
    if len(dts) == 0:
        raise ValueError("target_t_ms must contain at least two samples")
    dt_ms = float(dts[0])
    if not np.allclose(dts, dt_ms):
        raise ValueError("target_t_ms must be uniformly sampled")
    return dt_ms


def _pack_log_parameters(params, parameter_names):
    lower_bounds, _ = _log_bounds_for_parameters(parameter_names)
    raw_values = np.asarray([getattr(params, name) for name in parameter_names], dtype=float)
    clipped_values = np.maximum(raw_values, np.exp(lower_bounds))
    return np.log(clipped_values)


def _replace_log_parameters(initial, parameter_names, log_values):
    values = {name: float(np.exp(value)) for name, value in zip(parameter_names, log_values)}
    return SingleCompartmentFitParameters(
        g_nmda_max_nS=values.get("g_nmda_max_nS", initial.g_nmda_max_nS),
        capacitance_nF=values.get("capacitance_nF", initial.capacitance_nF),
        g_leak_nS=values.get("g_leak_nS", initial.g_leak_nS),
        tau_rise_ms=values.get("tau_rise_ms", initial.tau_rise_ms),
        tau_decay_ms=values.get("tau_decay_ms", initial.tau_decay_ms),
        alpha_per_ms=values.get("alpha_per_ms", initial.alpha_per_ms),
        presynaptic_spike_weight=values.get(
            "presynaptic_spike_weight",
            initial.presynaptic_spike_weight,
        ),
    )


def _log_bounds_for_parameters(parameter_names):
    bounds = {
        "g_nmda_max_nS": (0.01, 500.0),
        "capacitance_nF": (0.01, 50.0),
        "g_leak_nS": (1e-6, 100.0),
        "tau_rise_ms": (0.1, 20.0),
        "tau_decay_ms": (5.0, 500.0),
        "alpha_per_ms": (0.001, 10.0),
        "presynaptic_spike_weight": (0.001, 20.0),
    }
    lower_bounds = []
    upper_bounds = []
    for parameter_name in parameter_names:
        lower, upper = bounds[parameter_name]
        lower_bounds.append(np.log(lower))
        upper_bounds.append(np.log(upper))
    return np.asarray(lower_bounds), np.asarray(upper_bounds)

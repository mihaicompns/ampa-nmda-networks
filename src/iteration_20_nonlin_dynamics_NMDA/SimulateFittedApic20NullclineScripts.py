import csv
import json
import unittest
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, Slider
import numpy as np

from src.Plotting import show_plots_non_blocking
from src.iteration_20_nonlin_dynamics_NMDA.FitApic20NMDAStateScripts import (
    sigma_nmda_block,
    compute_voltage_error_summary,
    fit_capacitance_and_g_leak_for_voltage_replay,
    fit_nmda_s_state_to_apic20_reference,
    simulate_nmda_x_s_state,
    simulate_voltage_from_nmda_state,
)


def amplification_label(amplification):
    label = f"{float(amplification):.2f}".rstrip("0").rstrip(".")
    return f"{label}x".replace("-", "m").replace(".", "p")


def g_nmda_max_label(g_nmda_max_nS):
    label = f"{float(g_nmda_max_nS):.12g}"
    if "." in label:
        label = label.rstrip("0").rstrip(".")
    return label.replace("-", "m").replace(".", "p")


def protocol_path_label(label):
    cleaned = []
    previous_was_separator = False
    for character in str(label).strip().lower():
        if character.isalnum():
            cleaned.append(character)
            previous_was_separator = False
        elif not previous_was_separator:
            cleaned.append("_")
            previous_was_separator = True
    return "".join(cleaned).strip("_") or "manual_run"


FITTED_APIC20_SCAN_SPIKES_MS = np.asarray([50.0, 70.0, 90.0], dtype=float)


def fitted_apic20_g_nmda_amplification_protocol_name(amplification, spikes=None):
    if spikes is None:
        spikes = FITTED_APIC20_SCAN_SPIKES_MS
    spike_count = len(np.asarray(spikes, dtype=float))
    return (
        f"nmda_scan_{spike_count}_spike_s_model/"
        f"amp_{amplification_label(amplification)}"
    )


def fitted_apic20_three_spike_g_nmda_max_sweep_protocol_name(g_nmda_max_nS):
    return (
        "fitted_apic20/three_protocol_sweep/"
        f"g_max_{g_nmda_max_label(g_nmda_max_nS)}"
    )


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"
FITTED_NULLCLINE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "fitted_apic20_nullcline"
FITTED_APIC20_THREE_SPIKE_G_NMDA_MAX_SWEEP_OUTPUT_DIR = (
        DEFAULT_OUTPUT_DIR / "fitted_apic20" / "three_protocol_sweep"
)
FITTED_NULLCLINE_SPIKE_SNAPSHOT_OUTPUT_DIR = (
        DEFAULT_OUTPUT_DIR / "fitted_apic20_nullcline_spike_snapshots"
)
FITTED_NULLCLINE_SATURATING_OUTPUT_DIR = (
        DEFAULT_OUTPUT_DIR / "fitted_apic20_nullcline_with_s_saturation"
)
FITTED_NULLCLINE_SATURATING_TWO_SPIKE_OUTPUT_DIR = (
        DEFAULT_OUTPUT_DIR / "fitted_apic20_nullcline_with_s_saturation_two_spike"
)
FITTED_NULLCLINE_SATURATING_G_NMDA_AMPLIFICATION_OUTPUT_DIR = (
        DEFAULT_OUTPUT_DIR / "fitted_apic20_nullcline_with_s_saturation_g_nmda_amplification"
)
FITTED_NULLCLINE_INTERACTIVE_CSV_DIR = (
        DEFAULT_OUTPUT_DIR / "fitted_apic20_interactive_csv"
)
FITTED_APIC20_PARAMETER_FIT_DIR = DEFAULT_OUTPUT_DIR / "fitted_apic20_parameter_fit"
FITTED_APIC20_PARAMETER_CSV_PATH = (
        FITTED_APIC20_PARAMETER_FIT_DIR / "fitted_apic20_parameters.csv"
)
INTERACTIVE_CSV_PROTOCOL_NAME = "current_balance_three_spike"
INTERACTIVE_T_MIN_MS = 40.0
INTERACTIVE_T_MAX_MS = 200.0
TWO_SPIKE_SELECTION_MAX_MS = 89.0
TWO_SPIKE_T_MAX_MS = INTERACTIVE_T_MAX_MS
PARALLEL_SCRIPT_N_JOBS = 8
G_NMDA_MAX_AMPLIFICATIONS = np.arange(1, 10)
FITTED_APIC20_G_NMDA_AMPLIFICATION_PROTOCOL_NAMES = tuple(
    fitted_apic20_g_nmda_amplification_protocol_name(amplification)
    for amplification in G_NMDA_MAX_AMPLIFICATIONS
)
FITTED_APIC20_INTERACTIVE_PROTOCOL_NAMES = (
    "standard_three_spike",
    "current_balance_three_spike",
    "standard_two_spike",
    "current_balance_two_spike",
    "standard_three_spike_with_s_saturation",
    "current_balance_three_spike_with_s_saturation",
    "standard_two_spike_with_s_saturation",
    "current_balance_two_spike_with_s_saturation",
    "current_balance_three_spike_e_l_minus_50",
    "current_balance_three_spike_e_l_minus_50_g_nmda_max_1_nS",
    *(
        f"current_balance_three_spike_e_l_minus_50_g_nmda_max_{gmax}_nS"
        for gmax in range(2, 11)
    ),
    "current_balance_three_spike_e_l_minus_60",
    "current_balance_three_spike_e_l_minus_60_g_nmda_max_1_nS",
    *(
        f"current_balance_three_spike_e_l_minus_60_g_nmda_max_{gmax}_nS"
        for gmax in range(2, 11)
    ),
    "current_balance_three_spike_e_l_minus_70",
    *(
        f"current_balance_three_spike_e_l_minus_70_g_nmda_max_{gmax}_nS"
        for gmax in range(1, 11)
    ),
    *FITTED_APIC20_G_NMDA_AMPLIFICATION_PROTOCOL_NAMES,
)

FIRST_SPIKE_TIME_MS = 50.0
SPIKE_DT_MS = 20.0
N_SPIKES = 3
T_MIN_MS = 40.0
T_MAX_MS = 170.0
FRAME_COUNT = 140
FPS = 14
E_LEAK_MV = -70.0
E_EXCITATORY_MV = 0.0
STABLE_FIXED_POINT_MARKERSIZE = 10
UNSTABLE_FIXED_POINT_MARKERSIZE = 12


@dataclass(frozen=True)
class FittedApic20NullclineParameters:
    spike_weight: float
    tau_rise_ms: float
    tau_decay_ms: float
    alpha_per_ms: float
    capacitance_nF: float
    g_leak_nS: float
    g_nmda_max_nS: float
    e_leak_mV: float
    e_excitatory_mV: float


def fitted_apic20_spike_times_ms():
    return FITTED_APIC20_SCAN_SPIKES_MS.copy()


def fitted_apic20_spike_times_up_to_ms(max_time_ms):
    return np.asarray(
        [
            spike_time_ms
            for spike_time_ms in fitted_apic20_spike_times_ms()
            if spike_time_ms <= float(max_time_ms)
        ],
        dtype=float,
    )


def fitted_apic20_spike_snapshot_times_up_to_ms(max_time_ms=89.0, spike_times_ms=None):
    if spike_times_ms is None:
        spike_times_ms = fitted_apic20_spike_times_up_to_ms(max_time_ms)
    else:
        spike_times_ms = np.asarray(spike_times_ms, dtype=float)
    return {
        f"spike_{spike_index}_t{spike_time_ms:g}ms": float(spike_time_ms)
        for spike_index, spike_time_ms in enumerate(
            spike_times_ms,
            start=1,
        )
        if spike_time_ms <= float(max_time_ms)
    }


def fitted_apic20_snapshot_times_up_to_ms(max_time_ms=89.0, spike_times_ms=None):
    snapshot_times = {
        label: time_ms
        for label, time_ms in {
            "t67ms": 67.0,
            "t89ms": 89.0,
            "t97ms": 97.0,
            "t98ms": 98.0,
            "t140ms": 140.0,
        }.items()
        if time_ms <= float(max_time_ms)
    }
    snapshot_times.update(
        fitted_apic20_spike_snapshot_times_up_to_ms(
            max_time_ms,
            spike_times_ms=spike_times_ms,
        )
    )
    return dict(sorted(snapshot_times.items(), key=lambda item: item[1]))


def peak_voltage_snapshot_times_between_spikes_ms(trace, t_max_ms):
    t_ms = np.asarray(trace["t_ms"], dtype=float)
    v_mV = np.asarray(trace["v_local_mV"], dtype=float)
    spike_times_ms = np.asarray(trace["spike_times_ms"], dtype=float)
    peak_times = {}
    for spike_index, spike_time_ms in enumerate(spike_times_ms, start=1):
        interval_end_ms = (
            spike_times_ms[spike_index]
            if spike_index < len(spike_times_ms)
            else float(t_max_ms)
        )
        if spike_index < len(spike_times_ms):
            interval_mask = (t_ms >= spike_time_ms) & (t_ms < interval_end_ms)
        else:
            interval_mask = (t_ms >= spike_time_ms) & (t_ms <= interval_end_ms)
        interval_indexes = np.flatnonzero(interval_mask)
        if len(interval_indexes) == 0:
            continue
        peak_index = interval_indexes[int(np.argmax(v_mV[interval_indexes]))]
        peak_times[f"vmax_after_spike_{spike_index}"] = float(t_ms[peak_index])
    return peak_times


def estimate_zero_crossings(x_values, y_values, atol=1e-12):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    crossings = []
    exact_indexes = np.flatnonzero(np.isclose(y_values, 0.0, atol=atol))
    crossings.extend(x_values[exact_indexes].tolist())
    sign_change_indexes = np.flatnonzero(y_values[:-1] * y_values[1:] < 0.0)
    for index in sign_change_indexes:
        x0 = x_values[index]
        x1 = x_values[index + 1]
        y0 = y_values[index]
        y1 = y_values[index + 1]
        crossings.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    if not crossings:
        return np.asarray([], dtype=float)
    return np.unique(np.asarray(crossings, dtype=float))


def classify_intersection_stability(intersections_mV):
    intersections_mV = np.asarray(intersections_mV, dtype=float)
    if len(intersections_mV) == 3:
        return intersections_mV[[0, 2]], intersections_mV[[1]]
    return intersections_mV, np.asarray([], dtype=float)


def detect_nullcline_zero_count_transition_times_ms(
        trace,
        parameters,
        v_values_mV=None,
        t_min_ms=T_MIN_MS,
        t_max_ms=T_MAX_MS,
        max_snapshots=None,
):
    if v_values_mV is None:
        v_values_mV = np.linspace(-90.0, 20.0, 2500)
    t_ms = np.asarray(trace["t_ms"], dtype=float)
    candidate_indexes = np.flatnonzero((t_ms >= float(t_min_ms)) & (t_ms <= float(t_max_ms)))
    counts = []
    for index in candidate_indexes:
        nullcline_dvdt = compute_local_nullcline_dvdt_mV_per_ms(
            parameters,
            trace["s"][index],
            v_values_mV,
        )
        counts.append(len(estimate_zero_crossings(v_values_mV, nullcline_dvdt)))
    return intersection_count_transition_times_ms(
        t_ms,
        candidate_indexes,
        counts,
        label_prefix="nullcline_bifurcation",
        max_snapshots=max_snapshots,
    )


def detect_current_balance_intersection_count_transition_times_ms(
        trace,
        parameters,
        v_values_mV=None,
        t_min_ms=T_MIN_MS,
        t_max_ms=T_MAX_MS,
        max_snapshots=None,
):
    from src.iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20CurrentNullclineScripts import (
        compute_current_balance_nullcline_pA,
        estimate_current_balance_intersections_mV,
    )

    if v_values_mV is None:
        v_values_mV = np.linspace(-90.0, 20.0, 2500)
    t_ms = np.asarray(trace["t_ms"], dtype=float)
    candidate_indexes = np.flatnonzero((t_ms >= float(t_min_ms)) & (t_ms <= float(t_max_ms)))
    counts = []
    for index in candidate_indexes:
        currents = compute_current_balance_nullcline_pA(
            parameters,
            selected_s=trace["s"][index],
            v_values_mV=v_values_mV,
        )
        counts.append(len(estimate_current_balance_intersections_mV(currents)))
    return intersection_count_transition_times_ms(
        t_ms,
        candidate_indexes,
        counts,
        label_prefix="current_bifurcation",
        max_snapshots=max_snapshots,
    )


def intersection_count_transition_times_ms(
        t_ms,
        candidate_indexes,
        counts,
        label_prefix,
        max_snapshots=None,
):
    transition_times = {}
    if len(candidate_indexes) < 2:
        return transition_times
    transition_number = 1
    previous_count = int(counts[0])
    for candidate_position, current_count in enumerate(counts[1:], start=1):
        current_count = int(current_count)
        if current_count == previous_count:
            continue
        previous_index = int(candidate_indexes[candidate_position - 1])
        index = int(candidate_indexes[candidate_position])
        label = f"{label_prefix}_{transition_number}_{previous_count}_to_{current_count}"
        transition_times[label] = float((t_ms[previous_index] + t_ms[index]) / 2.0)
        transition_number += 1
        previous_count = current_count
        if max_snapshots is not None and len(transition_times) >= int(max_snapshots):
            break
    return transition_times


def fit_and_save_fitted_apic20_parameter_csv(
        output_csv_path=FITTED_APIC20_PARAMETER_CSV_PATH,
):
    (
        reference,
        _target_s,
        simulation,
        _fitted_g_nmda_nS,
        fitted_v_mV,
        fit_result,
    ) = fit_nmda_s_state_to_apic20_reference()
    (
        fitted_capacitance_nF,
        fitted_g_leak_nS,
        fitted_v_with_fitted_capacitance_and_g_leak_mV,
        _voltage_rmse_mV,
        _optimizer_result,
    ) = fit_capacitance_and_g_leak_for_voltage_replay(
        reference=reference,
        s=simulation["s"],
        g_nmda_max_nS=fit_result.fitted_g_nmda_max_nS,
        e_leak_mV=fit_result.e_leak_mV,
        e_excitatory_mV=fit_result.e_excitatory_mV,
        initial_capacitance_nF=fit_result.capacitance_nF,
        initial_g_leak_nS=fit_result.g_leak_nS,
    )
    parameters = FittedApic20NullclineParameters(
        spike_weight=fit_result.spike_weight,
        tau_rise_ms=fit_result.tau_rise_ms,
        tau_decay_ms=fit_result.tau_decay_ms,
        alpha_per_ms=fit_result.alpha_per_ms,
        capacitance_nF=fitted_capacitance_nF,
        g_leak_nS=fitted_g_leak_nS,
        g_nmda_max_nS=fit_result.fitted_g_nmda_max_nS,
        e_leak_mV=fit_result.e_leak_mV,
        e_excitatory_mV=fit_result.e_excitatory_mV,
    )
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_fitted_apic20_parameter_csv(parameters, output_csv_path)
    fit_trace = simulate_fitted_apic20_trace(reference.t_ms, parameters)
    return output_csv_path, reference, parameters, fit_trace


def write_fitted_apic20_parameter_csv(parameters, output_csv_path):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["parameter", "value"])
        for key, value in asdict(parameters).items():
            writer.writerow([key, repr(float(value))])


def read_fitted_apic20_parameters_from_csv(
        csv_path=FITTED_APIC20_PARAMETER_CSV_PATH,
):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing fitted APIC20 parameter CSV: {csv_path}. "
            "Run SimulateFittedApic20ParameterFitScriptTestCases."
            "test_fit_and_save_fitted_apic20_parameters first."
        )
    values = {}
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            values[row["parameter"]] = float(row["value"])
    return FittedApic20NullclineParameters(**values)


def set_fitted_apic20_g_nmda_max(parameters, g_nmda_max_nS):
    if g_nmda_max_nS is None:
        return parameters
    return replace(parameters, g_nmda_max_nS=float(g_nmda_max_nS))


def read_fit_and_simulate_fitted_apic20_reference(
        spike_times_ms=None,
        with_s_saturation=False,
        g_nmda_max_nS=None,
        parameters=None,
):
    from src.iteration_20_nonlin_dynamics_NMDA.single_compartment_equivalence import (
        read_standard_apic20_reference,
    )

    reference = read_standard_apic20_reference()
    if parameters is None:
        parameters = read_fitted_apic20_parameters_from_csv()
    parameters = set_fitted_apic20_g_nmda_max(
        parameters,
        g_nmda_max_nS=g_nmda_max_nS,
    )
    local_trace = simulate_fitted_apic20_trace(
        reference.t_ms,
        parameters,
        spike_times_ms=spike_times_ms,
        with_s_saturation=with_s_saturation,
    )
    local_trace["reference_v_local_mV"] = reference.v_local_mV
    return reference, parameters, local_trace


def simulate_fitted_apic20_trace(t_ms, parameters, spike_times_ms=None, with_s_saturation=False):
    if spike_times_ms is None:
        spike_times_ms = fitted_apic20_spike_times_ms()
    state_function = (
        simulate_saturating_nmda_x_s_state if with_s_saturation else simulate_nmda_x_s_state
    )
    state = state_function(
        t_ms=t_ms,
        spike_times_ms=spike_times_ms,
        spike_weight=parameters.spike_weight,
        tau_rise_ms=parameters.tau_rise_ms,
        tau_decay_ms=parameters.tau_decay_ms,
        alpha_per_ms=parameters.alpha_per_ms,
    )
    v_mV = simulate_voltage_from_nmda_state(
        t_ms=t_ms,
        s=state["s"],
        initial_v_mV=parameters.e_leak_mV,
        capacitance_nF=parameters.capacitance_nF,
        g_leak_nS=parameters.g_leak_nS,
        e_leak_mV=parameters.e_leak_mV,
        e_excitatory_mV=parameters.e_excitatory_mV,
        g_nmda_max_nS=parameters.g_nmda_max_nS,
    )
    sigma = sigma_nmda_block(v_mV)
    g_nmda_nS = parameters.g_nmda_max_nS * sigma * state["s"]
    empirical_dvdt = np.gradient(v_mV, t_ms)
    empirical_d2vdt2 = np.gradient(empirical_dvdt, t_ms)
    individual_s = simulate_individual_s_kernels(
        t_ms,
        parameters,
        spike_times_ms,
        with_s_saturation=with_s_saturation,
    )
    return {
        "condition": (
            "fitted single compartment saturating s"
            if with_s_saturation
            else "fitted single compartment"
        ),
        "n_spikes": len(spike_times_ms),
        "spike_times_ms": np.asarray(spike_times_ms, dtype=float),
        "with_s_saturation": bool(with_s_saturation),
        "t_ms": np.asarray(t_ms, dtype=float),
        "v_local_mV": v_mV,
        "x": state["x"],
        "s": state["s"],
        "individual_s": individual_s,
        "g_nmda_nS": g_nmda_nS,
        "sigma": sigma,
        "empirical_dvdt_mV_per_ms": empirical_dvdt,
        "empirical_d2vdt2_mV_per_ms2": empirical_d2vdt2,
    }


def simulate_saturating_nmda_x_s_state(
        t_ms,
        spike_times_ms,
        spike_weight=1.0,
        tau_rise_ms=2.0,
        tau_decay_ms=100.0,
        alpha_per_ms=0.5,
):
    t_ms = np.asarray(t_ms, dtype=float)
    dt_ms = float(np.median(np.diff(t_ms)))
    spike_steps = _spike_steps_for_fitted_trace(t_ms, spike_times_ms)

    x = np.zeros_like(t_ms, dtype=float)
    s = np.zeros_like(t_ms, dtype=float)
    x_current = 0.0
    s_current = 0.0

    for step in range(len(t_ms)):
        if step in spike_steps:
            x_current += spike_steps[step] * float(spike_weight)

        x[step] = x_current
        s[step] = s_current

        if step == len(t_ms) - 1:
            break

        dx = dt_ms * (-x_current / float(tau_rise_ms))
        ds = dt_ms * (
                -s_current / float(tau_decay_ms)
                + float(alpha_per_ms) * x_current * (1.0 - s_current)
        )
        x_current += dx
        s_current += ds

    return {"x": x, "s": s}


def _spike_steps_for_fitted_trace(t_ms, spike_times_ms):
    spike_steps = {}
    for spike_time_ms in spike_times_ms:
        step = int(np.searchsorted(t_ms, float(spike_time_ms), side="left"))
        if 0 <= step < len(t_ms):
            spike_steps[step] = spike_steps.get(step, 0) + 1
    return spike_steps


def simulate_individual_s_kernels(
        t_ms,
        parameters,
        spike_times_ms=None,
        with_s_saturation=False,
):
    if spike_times_ms is None:
        spike_times_ms = fitted_apic20_spike_times_ms()
    state_function = (
        simulate_saturating_nmda_x_s_state if with_s_saturation else simulate_nmda_x_s_state
    )
    kernels = []
    for spike_time_ms in spike_times_ms:
        state = state_function(
            t_ms=t_ms,
            spike_times_ms=[spike_time_ms],
            spike_weight=parameters.spike_weight,
            tau_rise_ms=parameters.tau_rise_ms,
            tau_decay_ms=parameters.tau_decay_ms,
            alpha_per_ms=parameters.alpha_per_ms,
        )
        kernels.append(state["s"])
    return np.asarray(kernels)


def compute_local_nullcline_dvdt_mV_per_ms(parameters, selected_s, v_values_mV):
    v_values_mV = np.asarray(v_values_mV, dtype=float)
    sigma = sigma_nmda_block(v_values_mV)
    leak_pA = parameters.g_leak_nS * (v_values_mV - parameters.e_leak_mV)
    nmda_pA = (
            parameters.g_nmda_max_nS
            * sigma
            * float(selected_s)
            * (v_values_mV - parameters.e_excitatory_mV)
    )
    return -1e-3 * (leak_pA + nmda_pA) / parameters.capacitance_nF


def derivative_ylim(values, pad_fraction=0.08):
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return -1.0, 1.0
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if np.isclose(minimum, maximum):
        margin = max(abs(minimum) * pad_fraction, 1.0)
    else:
        margin = (maximum - minimum) * pad_fraction
    return minimum - margin, maximum + margin


def max_reference_empirical_dvdt_time_ms(reference, t_min_ms=T_MIN_MS, t_max_ms=T_MAX_MS):
    empirical_dvdt = np.gradient(reference.v_local_mV, reference.t_ms)
    mask = (reference.t_ms >= t_min_ms) & (reference.t_ms <= t_max_ms)
    indexes = np.flatnonzero(mask)
    if len(indexes) == 0:
        indexes = np.arange(len(reference.t_ms))
    max_index = indexes[int(np.argmax(empirical_dvdt[indexes]))]
    return float(reference.t_ms[max_index])


def create_fitted_nullcline_figure():
    fig, axes = plt.subplot_mosaic(
        [
            ["model_ax", "model_ax", "model_ax"],
            ["voltage_ax", "g_nmda_ax", "s_ax"],
            ["nullcline_ax", "nullcline_ax", "nullcline_ax"],
        ],
        figsize=(14, 9),
        height_ratios=[0.8, 1.7, 2.2],
    )
    axes["fig"] = fig
    return axes


def add_fitted_nullcline_model_text(model_ax):
    model_ax.axis("off")
    equations = (
        r"$C\,\frac{dV}{dt} = -g_L(V-E_L) - g_{\mathrm{NMDA,max}}\,\sigma(V)\,s(t)\,(V-E_E)$"
        "\n"
        r"$g_{\mathrm{NMDA}}(t) = g_{\mathrm{NMDA,max}}\,\sigma(V)\,s(t)$"
        "\n"
        r"$s(t) = \sum_{t_i \in \{50,70,90\}}\bar{s}"
        r"\left[\exp\left(-\frac{t-t_i}{\tau_D}\right)"
        r"-\exp\left(-\frac{t-t_i}{\tau_r}\right)\right]H(t-t_i)$"
    )
    model_ax.text(
        0.5,
        0.5,
        equations,
        ha="center",
        va="center",
        fontsize=12,
    )


def configure_fitted_nullcline_axes(axes, trace, parameters, t_min_ms, t_max_ms):
    t_ms = trace["t_ms"]
    time_mask = (t_ms >= t_min_ms) & (t_ms <= t_max_ms)
    axes["voltage_ax"].set_xlim(t_min_ms, t_max_ms)
    axes["g_nmda_ax"].set_xlim(t_min_ms, t_max_ms)
    axes["s_ax"].set_xlim(t_min_ms, t_max_ms)

    axes["voltage_ax"].set_ylim(
        float(np.min(trace["v_local_mV"][time_mask])) - 3.0,
        float(np.max(trace["v_local_mV"][time_mask])) + 3.0,
    )
    axes["g_nmda_ax"].set_ylim(0.0, float(np.max(trace["g_nmda_nS"][time_mask])) * 1.1)
    s_max = float(
        np.max(
            [
                np.max(trace["s"][time_mask]),
                np.max(trace["individual_s"][:, time_mask]),
            ]
        )
    )
    axes["s_ax"].set_ylim(0.0, s_max * 1.1)
    axes["nullcline_ax"].set_xlim(-90.0, 20.0)

    add_fitted_nullcline_model_text(axes["model_ax"])
    axes["voltage_ax"].set_title(r"$V(t)$")
    axes["g_nmda_ax"].set_title(r"$g_{\mathrm{NMDA}}(t)$")
    axes["s_ax"].set_title(r"fitted $s(t)$")
    axes["nullcline_ax"].set_title(r"$dV/dt(V; s(t))$")

    axes["voltage_ax"].set_ylabel("V [mV]")
    axes["g_nmda_ax"].set_ylabel(r"$g_{\mathrm{NMDA}}$ [nS]")
    axes["s_ax"].set_ylabel("s")
    axes["nullcline_ax"].set_ylabel(r"$dV/dt$ [mV/ms]")
    for ax in axes.values():
        if ax is axes["fig"] or ax is axes["model_ax"]:
            continue
        ax.set_xlabel("t [ms]" if ax is not axes["nullcline_ax"] else "V [mV]")
    axes["nullcline_ax"].axhline(0.0, color="black", linewidth=1.0)


def plot_residual_nullcline_fixed_points(ax, v_values_mV, nullcline_dvdt, **kwargs):
    intersections_mV = estimate_zero_crossings(v_values_mV, nullcline_dvdt)
    stable_intersections_mV, unstable_intersections_mV = classify_intersection_stability(
        intersections_mV
    )
    stable_artist = ax.plot(
        stable_intersections_mV,
        np.zeros_like(stable_intersections_mV),
        "o",
        color="black",
        markerfacecolor="black",
        markersize=STABLE_FIXED_POINT_MARKERSIZE,
        linestyle="None",
        **kwargs,
    )
    unstable_artist = ax.plot(
        unstable_intersections_mV,
        np.zeros_like(unstable_intersections_mV),
        "o",
        color="black",
        markerfacecolor="none",
        markersize=UNSTABLE_FIXED_POINT_MARKERSIZE,
        linestyle="None",
        **kwargs,
    )
    return stable_artist + unstable_artist


def generate_fitted_apic20_moving_nullcline_animation(
        output_gif_path=FITTED_NULLCLINE_OUTPUT_DIR
                        / "fitted_apic20_moving_nullcline.gif",
        frame_count=FRAME_COUNT,
        t_min_ms=T_MIN_MS,
        t_max_ms=T_MAX_MS,
        fps=FPS,
        show_plot=True,
        spike_times_ms=None,
        with_s_saturation=False,
        g_nmda_max_nS=None,
        parameters=None,
):
    reference, parameters, trace = read_fit_and_simulate_fitted_apic20_reference(
        spike_times_ms=spike_times_ms,
        with_s_saturation=with_s_saturation,
        g_nmda_max_nS=g_nmda_max_nS,
        parameters=parameters,
    )
    output_gif_path = Path(output_gif_path)
    output_gif_path.parent.mkdir(parents=True, exist_ok=True)

    t_ms = trace["t_ms"]
    frame_mask = (t_ms >= t_min_ms) & (t_ms <= t_max_ms)
    candidate_indexes = np.flatnonzero(frame_mask)
    if len(candidate_indexes) == 0:
        candidate_indexes = np.arange(len(t_ms))
    frame_indexes = np.unique(
        np.linspace(
            candidate_indexes[0],
            candidate_indexes[-1],
            min(frame_count, len(candidate_indexes)),
            dtype=int,
        )
    )
    v_values_mV = np.linspace(-90.0, 20.0, 500)
    nullcline_dvdt = [
        compute_local_nullcline_dvdt_mV_per_ms(
            parameters,
            selected_s=trace["s"][index],
            v_values_mV=v_values_mV,
        )
        for index in frame_indexes
    ]

    axes = create_fitted_nullcline_figure()
    fig = axes["fig"]
    configure_fitted_nullcline_axes(axes, trace, parameters, t_min_ms, t_max_ms)
    axes["nullcline_ax"].set_ylim(derivative_ylim(np.concatenate(nullcline_dvdt)))

    axes["voltage_ax"].plot(t_ms, trace["v_local_mV"], color="0.75", linewidth=1.0)
    axes["g_nmda_ax"].plot(t_ms, trace["g_nmda_nS"], color="0.75", linewidth=1.0)
    axes["s_ax"].plot(t_ms, trace["s"], color="0.75", linewidth=1.0)
    for spike_index, individual_s in enumerate(trace["individual_s"], start=1):
        axes["s_ax"].plot(
            t_ms,
            individual_s,
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
            label=f"spike {spike_index}",
        )
    axes["s_ax"].legend(fontsize=7, loc="best")

    active_voltage_line, = axes["voltage_ax"].plot([], [], color="tab:blue", linewidth=2.0)
    active_g_line, = axes["g_nmda_ax"].plot([], [], color="tab:green", linewidth=2.0)
    active_s_line, = axes["s_ax"].plot([], [], color="tab:purple", linewidth=2.0)
    nullcline_line, = axes["nullcline_ax"].plot([], [], color="tab:orange", linewidth=2.0)
    voltage_point, = axes["voltage_ax"].plot([], [], "o", color="tab:blue")
    g_point, = axes["g_nmda_ax"].plot([], [], "o", color="tab:green")
    s_point, = axes["s_ax"].plot([], [], "o", color="tab:purple")
    nullcline_point, = axes["nullcline_ax"].plot([], [], "o", color="tab:orange")
    stable_fixed_points, = axes["nullcline_ax"].plot(
        [],
        [],
        "o",
        color="black",
        markerfacecolor="black",
        markersize=STABLE_FIXED_POINT_MARKERSIZE,
        linestyle="None",
    )
    unstable_fixed_points, = axes["nullcline_ax"].plot(
        [],
        [],
        "o",
        color="black",
        markerfacecolor="none",
        markersize=UNSTABLE_FIXED_POINT_MARKERSIZE,
        linestyle="None",
    )
    nullcline_voltage_line = axes["nullcline_ax"].axvline(
        0.0,
        color="tab:blue",
        linewidth=1.0,
    )
    title = fig.suptitle("")

    def update(frame_number):
        index = int(frame_indexes[frame_number])
        current_t_ms = float(t_ms[index])
        current_v_mV = float(trace["v_local_mV"][index])
        current_g_nmda_nS = float(trace["g_nmda_nS"][index])
        current_s = float(trace["s"][index])
        current_sigma = float(trace["sigma"][index])
        active_voltage_line.set_data(t_ms[: index + 1], trace["v_local_mV"][: index + 1])
        active_g_line.set_data(t_ms[: index + 1], trace["g_nmda_nS"][: index + 1])
        active_s_line.set_data(t_ms[: index + 1], trace["s"][: index + 1])
        nullcline_line.set_data(v_values_mV, nullcline_dvdt[frame_number])
        nullcline_value = float(np.interp(current_v_mV, v_values_mV, nullcline_dvdt[frame_number]))
        intersections_mV = estimate_zero_crossings(v_values_mV, nullcline_dvdt[frame_number])
        stable_intersections_mV, unstable_intersections_mV = classify_intersection_stability(
            intersections_mV
        )
        stable_fixed_points.set_data(
            stable_intersections_mV,
            np.zeros_like(stable_intersections_mV),
        )
        unstable_fixed_points.set_data(
            unstable_intersections_mV,
            np.zeros_like(unstable_intersections_mV),
        )
        voltage_point.set_data([current_t_ms], [current_v_mV])
        g_point.set_data([current_t_ms], [current_g_nmda_nS])
        s_point.set_data([current_t_ms], [current_s])
        nullcline_point.set_data([current_v_mV], [nullcline_value])
        nullcline_voltage_line.set_xdata([current_v_mV, current_v_mV])
        title.set_text(
            "Fitted APIC20 single-compartment moving nullcline, "
            f"{trace['n_spikes']} spikes, "
            f"g_NMDA,max={parameters.g_nmda_max_nS:.6g} nS, "
            f"C_m={parameters.capacitance_nF:.6g} nF, "
            f"g_L={parameters.g_leak_nS:.6g} nS\n"
            f"t={current_t_ms:.1f} ms, V={current_v_mV:.2f} mV, "
            f"g_NMDA(t)={current_g_nmda_nS:.3g} nS, "
            f"s(t)={current_s:.3g}, sigma(V)={current_sigma:.3g}, "
            f"nullcline dV/dt={nullcline_value:.3g} mV/ms"
        )
        return (
            active_voltage_line,
            active_g_line,
            active_s_line,
            nullcline_line,
            voltage_point,
            g_point,
            s_point,
            nullcline_point,
            stable_fixed_points,
            unstable_fixed_points,
            nullcline_voltage_line,
            title,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indexes),
        interval=1000 / fps,
        blit=False,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.86)
    animation.save(output_gif_path, writer=PillowWriter(fps=fps))
    if show_plot:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output_gif_path


def generate_fitted_apic20_moving_nullcline_snapshot(
        snapshot_time_ms,
        output_png_path,
        t_min_ms=T_MIN_MS,
        t_max_ms=T_MAX_MS,
        show_plot=True,
        spike_times_ms=None,
        with_s_saturation=False,
        g_nmda_max_nS=None,
        parameters=None,
):
    reference, parameters, trace = read_fit_and_simulate_fitted_apic20_reference(
        spike_times_ms=spike_times_ms,
        with_s_saturation=with_s_saturation,
        g_nmda_max_nS=g_nmda_max_nS,
        parameters=parameters,
    )
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    t_ms = trace["t_ms"]
    index = nearest_trace_index(t_ms, snapshot_time_ms, min_index=1, max_index=len(t_ms) - 2)
    current_t_ms = float(t_ms[index])
    v_values_mV = np.linspace(-90.0, 20.0, 5000)
    nullcline_dvdt = compute_local_nullcline_dvdt_mV_per_ms(
        parameters,
        selected_s=trace["s"][index],
        v_values_mV=v_values_mV,
    )

    axes = create_fitted_nullcline_figure()
    fig = axes["fig"]
    configure_fitted_nullcline_axes(axes, trace, parameters, t_min_ms, t_max_ms)
    axes["nullcline_ax"].set_ylim(derivative_ylim(nullcline_dvdt))
    plot_snapshot_axes(axes, trace, parameters, index, v_values_mV, nullcline_dvdt)
    fig.suptitle(
        "Fitted APIC20 single-compartment snapshot, "
        f"t={current_t_ms:.1f} ms, "
        f"V={trace['v_local_mV'][index]:.2f} mV, "
        f"g_NMDA={trace['g_nmda_nS'][index]:.3g} nS, "
        f"s={trace['s'][index]:.3g}, "
        f"C_m={parameters.capacitance_nF:.6g} nF, "
        f"g_L={parameters.g_leak_nS:.6g} nS",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.86)
    fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
    if show_plot:
        fig.show()
        plt.show(block=True)
    else:
        plt.close(fig)
    snapshot = snapshot_metrics(reference, trace, parameters, index, v_values_mV, nullcline_dvdt)
    snapshot["plot_path"] = str(output_png_path)
    return snapshot


def plot_snapshot_axes(axes, trace, parameters, index, v_values_mV, nullcline_dvdt):
    t_ms = trace["t_ms"]
    current_t_ms = float(t_ms[index])
    current_v_mV = float(trace["v_local_mV"][index])
    axes["voltage_ax"].plot(t_ms, trace["v_local_mV"], color="0.75", linewidth=1.0)
    axes["voltage_ax"].plot(t_ms[: index + 1], trace["v_local_mV"][: index + 1], color="tab:blue")
    axes["voltage_ax"].plot(current_t_ms, current_v_mV, "o", color="tab:blue")
    axes["voltage_ax"].axvline(current_t_ms, color="0.55", linewidth=1.0)

    axes["g_nmda_ax"].plot(t_ms, trace["g_nmda_nS"], color="0.75", linewidth=1.0)
    axes["g_nmda_ax"].plot(t_ms[: index + 1], trace["g_nmda_nS"][: index + 1], color="tab:green")
    axes["g_nmda_ax"].plot(current_t_ms, trace["g_nmda_nS"][index], "o", color="tab:green")
    axes["g_nmda_ax"].axvline(current_t_ms, color="0.55", linewidth=1.0)

    axes["s_ax"].plot(t_ms, trace["s"], color="0.75", linewidth=1.0)
    for spike_index, individual_s in enumerate(trace["individual_s"], start=1):
        axes["s_ax"].plot(
            t_ms,
            individual_s,
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
            label=f"spike {spike_index}",
        )
    axes["s_ax"].plot(t_ms[: index + 1], trace["s"][: index + 1], color="tab:purple")
    axes["s_ax"].plot(current_t_ms, trace["s"][index], "o", color="tab:purple")
    axes["s_ax"].axvline(current_t_ms, color="0.55", linewidth=1.0)
    axes["s_ax"].legend(fontsize=7, loc="best")

    nullcline_value = float(np.interp(current_v_mV, v_values_mV, nullcline_dvdt))
    axes["nullcline_ax"].plot(v_values_mV, nullcline_dvdt, color="tab:orange")
    axes["nullcline_ax"].plot(current_v_mV, nullcline_value, "o", color="tab:orange")
    plot_residual_nullcline_fixed_points(axes["nullcline_ax"], v_values_mV, nullcline_dvdt)
    axes["nullcline_ax"].axvline(current_v_mV, color="tab:blue", linewidth=1.0)


def snapshot_metrics(reference, trace, parameters, index, v_values_mV, nullcline_dvdt):
    current_v_mV = float(trace["v_local_mV"][index])
    nullcline_value = float(np.interp(current_v_mV, v_values_mV, nullcline_dvdt))
    return {
        "snapshot_time_ms": float(trace["t_ms"][index]),
        "snapshot_v_mV": current_v_mV,
        "snapshot_g_nmda_nS": float(trace["g_nmda_nS"][index]),
        "snapshot_s": float(trace["s"][index]),
        "snapshot_x": float(trace["x"][index]),
        "snapshot_sigma": float(trace["sigma"][index]),
        "snapshot_empirical_dvdt_mV_per_ms": float(
            trace["empirical_dvdt_mV_per_ms"][index]
        ),
        "snapshot_empirical_d2vdt2_mV_per_ms2": float(
            trace["empirical_d2vdt2_mV_per_ms2"][index]
        ),
        "snapshot_nullcline_dvdt_at_current_v_mV_per_ms": nullcline_value,
        "parameters": asdict(parameters),
        "voltage_error_summary": compute_voltage_error_summary(
            reference=reference,
            fitted_v_mV=trace["v_local_mV"],
        ),
    }


def snapshot_metrics_from_csv_trace(trace, parameters, index, v_values_mV, nullcline_dvdt):
    current_v_mV = float(trace["v_local_mV"][index])
    return {
        "snapshot_time_ms": float(trace["t_ms"][index]),
        "snapshot_v_mV": current_v_mV,
        "snapshot_g_nmda_nS": float(trace["g_nmda_nS"][index]),
        "snapshot_s": float(trace["s"][index]),
        "snapshot_x": float(trace["x"][index]),
        "snapshot_sigma": float(trace["sigma"][index]),
        "snapshot_empirical_dvdt_mV_per_ms": float(
            trace["empirical_dvdt_mV_per_ms"][index]
        ),
        "snapshot_empirical_d2vdt2_mV_per_ms2": float(
            trace["empirical_d2vdt2_mV_per_ms2"][index]
        ),
        "snapshot_nullcline_dvdt_at_current_v_mV_per_ms": float(
            np.interp(current_v_mV, v_values_mV, nullcline_dvdt)
        ),
        "snapshot_nullcline_intersections_mV": estimate_zero_crossings(
            v_values_mV,
            nullcline_dvdt,
        ).tolist(),
        "parameters": asdict(parameters),
    }


def generate_fitted_apic20_moving_nullcline_animation_from_trace(
        metadata,
        trace,
        parameters,
        output_gif_path,
        frame_count=FRAME_COUNT,
        fps=FPS,
        show_plot=False,
):
    output_gif_path = Path(output_gif_path)
    output_gif_path.parent.mkdir(parents=True, exist_ok=True)
    t_min_ms = float(metadata["t_min_ms"])
    t_max_ms = float(metadata["t_max_ms"])
    t_ms = trace["t_ms"]
    frame_mask = (t_ms >= t_min_ms) & (t_ms <= t_max_ms)
    candidate_indexes = np.flatnonzero(frame_mask)
    if len(candidate_indexes) == 0:
        candidate_indexes = np.arange(len(t_ms))
    frame_indexes = np.unique(
        np.linspace(
            candidate_indexes[0],
            candidate_indexes[-1],
            min(frame_count, len(candidate_indexes)),
            dtype=int,
        )
    )
    v_values_mV = np.linspace(-90.0, 20.0, 500)
    nullcline_dvdt = [
        compute_local_nullcline_dvdt_mV_per_ms(
            parameters,
            selected_s=trace["s"][index],
            v_values_mV=v_values_mV,
        )
        for index in frame_indexes
    ]

    axes = create_fitted_nullcline_figure()
    fig = axes["fig"]
    configure_fitted_nullcline_axes(axes, trace, parameters, t_min_ms, t_max_ms)
    axes["nullcline_ax"].set_ylim(derivative_ylim(np.concatenate(nullcline_dvdt)))

    axes["voltage_ax"].plot(t_ms, trace["v_local_mV"], color="0.75", linewidth=1.0)
    axes["g_nmda_ax"].plot(t_ms, trace["g_nmda_nS"], color="0.75", linewidth=1.0)
    axes["s_ax"].plot(t_ms, trace["s"], color="0.75", linewidth=1.0)
    for spike_index, individual_s in enumerate(trace["individual_s"], start=1):
        axes["s_ax"].plot(
            t_ms,
            individual_s,
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
            label=f"spike {spike_index}",
        )
    axes["s_ax"].legend(fontsize=7, loc="best")

    active_voltage_line, = axes["voltage_ax"].plot([], [], color="tab:blue", linewidth=2.0)
    active_g_line, = axes["g_nmda_ax"].plot([], [], color="tab:green", linewidth=2.0)
    active_s_line, = axes["s_ax"].plot([], [], color="tab:purple", linewidth=2.0)
    nullcline_line, = axes["nullcline_ax"].plot([], [], color="tab:orange", linewidth=2.0)
    voltage_point, = axes["voltage_ax"].plot([], [], "o", color="tab:blue")
    g_point, = axes["g_nmda_ax"].plot([], [], "o", color="tab:green")
    s_point, = axes["s_ax"].plot([], [], "o", color="tab:purple")
    nullcline_point, = axes["nullcline_ax"].plot([], [], "o", color="tab:orange")
    stable_fixed_points, = axes["nullcline_ax"].plot(
        [],
        [],
        "o",
        color="black",
        markerfacecolor="black",
        markersize=STABLE_FIXED_POINT_MARKERSIZE,
        linestyle="None",
    )
    unstable_fixed_points, = axes["nullcline_ax"].plot(
        [],
        [],
        "o",
        color="black",
        markerfacecolor="none",
        markersize=UNSTABLE_FIXED_POINT_MARKERSIZE,
        linestyle="None",
    )
    nullcline_voltage_line = axes["nullcline_ax"].axvline(
        0.0,
        color="tab:blue",
        linewidth=1.0,
    )
    title = fig.suptitle("")

    def update(frame_number):
        index = int(frame_indexes[frame_number])
        current_t_ms = float(t_ms[index])
        current_v_mV = float(trace["v_local_mV"][index])
        curve = nullcline_dvdt[frame_number]
        nullcline_value = float(np.interp(current_v_mV, v_values_mV, curve))
        intersections_mV = estimate_zero_crossings(v_values_mV, curve)
        stable_intersections_mV, unstable_intersections_mV = classify_intersection_stability(
            intersections_mV
        )
        active_voltage_line.set_data(t_ms[: index + 1], trace["v_local_mV"][: index + 1])
        active_g_line.set_data(t_ms[: index + 1], trace["g_nmda_nS"][: index + 1])
        active_s_line.set_data(t_ms[: index + 1], trace["s"][: index + 1])
        nullcline_line.set_data(v_values_mV, curve)
        voltage_point.set_data([current_t_ms], [current_v_mV])
        g_point.set_data([current_t_ms], [trace["g_nmda_nS"][index]])
        s_point.set_data([current_t_ms], [trace["s"][index]])
        nullcline_point.set_data([current_v_mV], [nullcline_value])
        stable_fixed_points.set_data(
            stable_intersections_mV,
            np.zeros_like(stable_intersections_mV),
        )
        unstable_fixed_points.set_data(
            unstable_intersections_mV,
            np.zeros_like(unstable_intersections_mV),
        )
        nullcline_voltage_line.set_xdata([current_v_mV, current_v_mV])
        title.set_text(
            f"{metadata['label']}, t={current_t_ms:.1f} ms, "
            f"V={current_v_mV:.2f} mV, "
            f"g_NMDA(t)={trace['g_nmda_nS'][index]:.3g} nS, "
            f"s(t)={trace['s'][index]:.3g}"
        )
        return (
            active_voltage_line,
            active_g_line,
            active_s_line,
            nullcline_line,
            voltage_point,
            g_point,
            s_point,
            nullcline_point,
            stable_fixed_points,
            unstable_fixed_points,
            nullcline_voltage_line,
            title,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indexes),
        interval=1000 / fps,
        blit=False,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.9, hspace=0.45)
    animation.save(output_gif_path, writer=PillowWriter(fps=fps))
    if show_plot:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output_gif_path


def generate_fitted_apic20_moving_nullcline_snapshot_from_trace(
        metadata,
        trace,
        parameters,
        snapshot_time_ms,
        output_png_path,
        show_plot=False,
):
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    t_ms = trace["t_ms"]
    index = nearest_trace_index(t_ms, snapshot_time_ms, min_index=1, max_index=len(t_ms) - 2)
    current_t_ms = float(t_ms[index])
    v_values_mV = np.linspace(-90.0, 20.0, 5000)
    nullcline_dvdt = compute_local_nullcline_dvdt_mV_per_ms(
        parameters,
        selected_s=trace["s"][index],
        v_values_mV=v_values_mV,
    )

    axes = create_fitted_nullcline_figure()
    fig = axes["fig"]
    configure_fitted_nullcline_axes(
        axes,
        trace,
        parameters,
        float(metadata["t_min_ms"]),
        float(metadata["t_max_ms"]),
    )
    axes["nullcline_ax"].set_ylim(derivative_ylim(nullcline_dvdt))
    plot_snapshot_axes(axes, trace, parameters, index, v_values_mV, nullcline_dvdt)
    fig.suptitle(
        f"{metadata['label']}, "
        f"t={current_t_ms:.1f} ms, "
        f"V={trace['v_local_mV'][index]:.2f} mV, "
        f"g_NMDA={trace['g_nmda_nS'][index]:.3g} nS, "
        f"s={trace['s'][index]:.3g}",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.9, hspace=0.45)
    fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
    if show_plot:
        fig.show()
        plt.show(block=True)
    else:
        plt.close(fig)
    snapshot = snapshot_metrics_from_csv_trace(
        trace,
        parameters,
        index,
        v_values_mV,
        nullcline_dvdt,
    )
    snapshot["plot_path"] = str(output_png_path)
    return snapshot


def generate_fitted_apic20_standard_protocol_outputs(
        show_plot=True,
        with_s_saturation=False,
        output_dir=None,
):
    reference, _parameters, trace = read_fit_and_simulate_fitted_apic20_reference(
        with_s_saturation=with_s_saturation,
    )
    save_fitted_apic20_protocol_csv_bundle_from_trace(
        (
            "standard_three_spike_with_s_saturation"
            if with_s_saturation
            else "standard_three_spike"
        ),
        _parameters,
        trace,
    )
    if output_dir is None:
        output_dir = (
            FITTED_NULLCLINE_SATURATING_OUTPUT_DIR
            if with_s_saturation
            else FITTED_NULLCLINE_OUTPUT_DIR
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = generate_fitted_apic20_moving_nullcline_animation(
        output_gif_path=output_dir / "fitted_apic20_moving_nullcline.gif",
        show_plot=show_plot,
        with_s_saturation=with_s_saturation,
    )
    snapshot_times = {
        "t67ms": 67.0,
        "t89ms": 89.0,
        "t97ms": 97.0,
        "t98ms": 98.0,
        "t140ms": 140.0,
        "max_reference_empirical_dvdt": max_reference_empirical_dvdt_time_ms(reference),
    }
    snapshot_times.update(peak_voltage_snapshot_times_between_spikes_ms(trace, T_MAX_MS))
    snapshot_times.update(
        detect_current_balance_intersection_count_transition_times_ms(
            trace,
            _parameters,
            t_max_ms=T_MAX_MS,
        )
    )
    snapshots = {}
    for label, snapshot_time_ms in snapshot_times.items():
        snapshots[label] = generate_fitted_apic20_moving_nullcline_snapshot(
            snapshot_time_ms=snapshot_time_ms,
            output_png_path=output_dir / f"fitted_apic20_moving_nullcline_{label}.png",
            show_plot=show_plot,
            with_s_saturation=with_s_saturation,
        )
    metrics_path = output_dir / "fitted_apic20_moving_nullcline_protocol.json"
    metrics_path.write_text(
        json.dumps(
            {
                "gif_path": str(gif_path),
                "snapshots": snapshots,
                "snapshot_times_ms": snapshot_times,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return gif_path, snapshots, metrics_path


def generate_fitted_apic20_three_spike_g_nmda_max_protocol_outputs(
        g_nmda_max_nS,
        show_plot=False,
        output_dir=None,
):
    if not show_plot:
        plt.switch_backend("Agg")
    label = g_nmda_max_label(g_nmda_max_nS)
    protocol_name = fitted_apic20_three_spike_g_nmda_max_sweep_protocol_name(
        g_nmda_max_nS
    )
    spike_times_ms = fitted_apic20_spike_times_ms()
    reference, parameters, trace = read_fit_and_simulate_fitted_apic20_reference(
        spike_times_ms=spike_times_ms,
        with_s_saturation=False,
        g_nmda_max_nS=g_nmda_max_nS,
    )
    trace_path, metadata_path = save_g_nmda_max_sweep_protocol_csv_bundle_from_trace(
        g_nmda_max_nS=g_nmda_max_nS,
        parameters=parameters,
        trace=trace,
        spike_times_ms=spike_times_ms,
        protocol_name=protocol_name,
    )
    if output_dir is None:
        output_dir = (
                FITTED_APIC20_THREE_SPIKE_G_NMDA_MAX_SWEEP_OUTPUT_DIR
                / f"g_max_{label}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gif_path = generate_fitted_apic20_moving_nullcline_animation(
        output_gif_path=(
                output_dir
                / f"fitted_apic20_three_spike_g_nmda_max_{label}_moving_nullcline.gif"
        ),
        show_plot=show_plot,
        spike_times_ms=spike_times_ms,
        with_s_saturation=False,
        g_nmda_max_nS=g_nmda_max_nS,
    )
    snapshot_times = {
        "t67ms": 67.0,
        "t89ms": 89.0,
        "t97ms": 97.0,
        "t98ms": 98.0,
        "t140ms": 140.0,
        "max_reference_empirical_dvdt": max_reference_empirical_dvdt_time_ms(reference),
    }
    snapshot_times.update(peak_voltage_snapshot_times_between_spikes_ms(trace, T_MAX_MS))
    snapshot_times.update(
        detect_current_balance_intersection_count_transition_times_ms(
            trace,
            parameters,
            t_max_ms=T_MAX_MS,
        )
    )
    snapshots = {}
    for snapshot_label, snapshot_time_ms in snapshot_times.items():
        snapshots[snapshot_label] = generate_fitted_apic20_moving_nullcline_snapshot(
            snapshot_time_ms=snapshot_time_ms,
            output_png_path=(
                    output_dir
                    / f"fitted_apic20_three_spike_g_nmda_max_{label}_{snapshot_label}.png"
            ),
            show_plot=show_plot,
            spike_times_ms=spike_times_ms,
            with_s_saturation=False,
            g_nmda_max_nS=g_nmda_max_nS,
        )
    metrics_path = output_dir / f"fitted_apic20_three_spike_g_nmda_max_{label}_protocol.json"
    metrics_path.write_text(
        json.dumps(
            {
                "g_nmda_max_nS": float(parameters.g_nmda_max_nS),
                "protocol_name": protocol_name,
                "gif_path": str(gif_path),
                "trace_path": str(trace_path),
                "metadata_path": str(metadata_path),
                "snapshots": snapshots,
                "snapshot_times_ms": snapshot_times,
                "spike_times_ms": spike_times_ms.tolist(),
                "with_s_saturation": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return gif_path, snapshots, metrics_path


def generate_fitted_apic20_three_spike_g_nmda_max_sweep_outputs(
        g_nmda_max_values_nS,
        show_plot=False,
        n_jobs=PARALLEL_SCRIPT_N_JOBS,
):
    jobs = [
        (g_nmda_max_label(g_nmda_max_nS), float(g_nmda_max_nS))
        for g_nmda_max_nS in g_nmda_max_values_nS
    ]
    if len({label for label, _g_nmda_max_nS in jobs}) != len(jobs):
        raise ValueError(
            "g_nmda_max_values_nS contains values that collapse to the same "
            "folder label; adjust the sweep values or g_nmda_max_label."
        )
    results = Parallel(n_jobs=n_jobs)(
        delayed(generate_fitted_apic20_three_spike_g_nmda_max_protocol_outputs)(
            g_nmda_max_nS=g_nmda_max_nS,
            show_plot=show_plot,
        )
        for _label, g_nmda_max_nS in jobs
    )
    return {
        label: result
        for (label, _g_nmda_max_nS), result in zip(jobs, results)
    }


def generate_saturated_fitted_apic20_g_nmda_max_protocol_outputs(
        g_nmda_max_nS,
        amplification,
        spike_times_ms=None,
        t_max_ms=T_MAX_MS,
        show_plot=False,
        output_dir=None,
):
    if not show_plot:
        plt.switch_backend("Agg")
    amp_label = amplification_label(amplification)
    if spike_times_ms is None:
        spike_times_ms = fitted_apic20_spike_times_ms()
    spike_times_ms = np.asarray(spike_times_ms, dtype=float)
    spike_label = "two_spike" if len(spike_times_ms) == 2 else "three_spike"
    reference, parameters, trace = read_fit_and_simulate_fitted_apic20_reference(
        spike_times_ms=spike_times_ms,
        with_s_saturation=True,
        g_nmda_max_nS=g_nmda_max_nS,
    )
    save_saturated_g_nmda_amplification_protocol_csv_bundle_from_trace(
        amplification=amplification,
        parameters=parameters,
        trace=trace,
        spike_times_ms=spike_times_ms,
        t_max_ms=t_max_ms,
    )
    if output_dir is None:
        output_dir = (
                FITTED_NULLCLINE_SATURATING_G_NMDA_AMPLIFICATION_OUTPUT_DIR
                / f"nmda_scan_{len(spike_times_ms)}_spike_s_model"
                / f"amp_{amp_label}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gif_path = generate_fitted_apic20_moving_nullcline_animation(
        output_gif_path=(
                output_dir
                / f"fitted_apic20_saturated_{spike_label}_g_nmda_max_{amp_label}_moving_nullcline.gif"
        ),
        t_max_ms=t_max_ms,
        show_plot=show_plot,
        spike_times_ms=spike_times_ms,
        with_s_saturation=True,
        g_nmda_max_nS=g_nmda_max_nS,
    )
    if len(spike_times_ms) == 2:
        snapshot_times = fitted_apic20_snapshot_times_up_to_ms(
            TWO_SPIKE_T_MAX_MS,
            spike_times_ms=spike_times_ms,
        )
        snapshot_times.update(
            peak_voltage_snapshot_times_between_spikes_ms(trace, TWO_SPIKE_T_MAX_MS)
        )
        snapshot_times.update(
            detect_current_balance_intersection_count_transition_times_ms(
                trace,
                parameters,
                t_max_ms=TWO_SPIKE_T_MAX_MS,
            )
        )
    else:
        snapshot_times = {
            "t67ms": 67.0,
            "t89ms": 89.0,
            "t97ms": 97.0,
            "t98ms": 98.0,
            "t140ms": 140.0,
            "max_reference_empirical_dvdt": max_reference_empirical_dvdt_time_ms(reference),
        }
        snapshot_times.update(peak_voltage_snapshot_times_between_spikes_ms(trace, T_MAX_MS))
        snapshot_times.update(
            detect_current_balance_intersection_count_transition_times_ms(
                trace,
                parameters,
                t_max_ms=T_MAX_MS,
            )
        )
    snapshots = {}
    for label, snapshot_time_ms in snapshot_times.items():
        snapshots[label] = generate_fitted_apic20_moving_nullcline_snapshot(
            snapshot_time_ms=snapshot_time_ms,
            output_png_path=(
                    output_dir
                    / f"fitted_apic20_saturated_{spike_label}_g_nmda_max_{amp_label}_{label}.png"
            ),
            t_max_ms=t_max_ms,
            show_plot=show_plot,
            spike_times_ms=spike_times_ms,
            with_s_saturation=True,
            g_nmda_max_nS=g_nmda_max_nS,
        )
    metrics_path = (
            output_dir
            / f"fitted_apic20_saturated_{spike_label}_g_nmda_max_{amp_label}_protocol.json"
    )
    metrics_path.write_text(
        json.dumps(
            {
                "amplification": float(amplification),
                "g_nmda_max_nS": float(parameters.g_nmda_max_nS),
                "gif_path": str(gif_path),
                "snapshots": snapshots,
                "snapshot_times_ms": snapshot_times,
                "spike_times_ms": spike_times_ms.tolist(),
                "with_s_saturation": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return gif_path, snapshots, metrics_path


def generate_saturated_fitted_apic20_g_nmda_amplification_sweep_outputs(
        amplifications=G_NMDA_MAX_AMPLIFICATIONS,
        show_plot=False,
        n_jobs=PARALLEL_SCRIPT_N_JOBS,
        two_spike=False,
):
    base_parameters = read_fitted_apic20_parameters_from_csv()
    base_g_nmda_max_nS = float(base_parameters.g_nmda_max_nS)
    spike_times_ms = (
        fitted_apic20_spike_times_up_to_ms(TWO_SPIKE_SELECTION_MAX_MS)
        if two_spike
        else fitted_apic20_spike_times_ms()
    )
    t_max_ms = 300
    jobs = [
        (
            amplification_label(amplification),
            base_g_nmda_max_nS * float(amplification),
            amplification,
        )
        for amplification in amplifications
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(generate_saturated_fitted_apic20_g_nmda_max_protocol_outputs)(
            g_nmda_max_nS=g_nmda_max_nS,
            amplification=amplification,
            spike_times_ms=spike_times_ms,
            t_max_ms=t_max_ms,
            show_plot=show_plot,
        )
        for _amp_label, g_nmda_max_nS, amplification in jobs
    )
    return {
        amp_label: result
        for (amp_label, _g_nmda_max_nS, _amplification), result in zip(jobs, results)
    }


def generate_fitted_apic20_two_spike_protocol_outputs(
        show_plot=True,
        with_s_saturation=False,
        output_dir=None,
):
    if output_dir is None:
        output_dir = (
            FITTED_NULLCLINE_SATURATING_TWO_SPIKE_OUTPUT_DIR
            if with_s_saturation
            else FITTED_NULLCLINE_SPIKE_SNAPSHOT_OUTPUT_DIR
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    spike_times_ms = fitted_apic20_spike_times_up_to_ms(
        max_time_ms=TWO_SPIKE_SELECTION_MAX_MS
    )
    _reference, _parameters, trace = read_fit_and_simulate_fitted_apic20_reference(
        spike_times_ms=spike_times_ms,
        with_s_saturation=with_s_saturation,
    )
    save_fitted_apic20_protocol_csv_bundle_from_trace(
        (
            "standard_two_spike_with_s_saturation"
            if with_s_saturation
            else "standard_two_spike"
        ),
        _parameters,
        trace,
    )
    gif_path = generate_fitted_apic20_moving_nullcline_animation(
        output_gif_path=output_dir / "fitted_apic20_two_spike_moving_nullcline.gif",
        t_max_ms=TWO_SPIKE_T_MAX_MS,
        show_plot=show_plot,
        spike_times_ms=spike_times_ms,
        with_s_saturation=with_s_saturation,
    )
    snapshot_times = fitted_apic20_snapshot_times_up_to_ms(
        max_time_ms=TWO_SPIKE_T_MAX_MS,
        spike_times_ms=spike_times_ms,
    )
    snapshot_times.update(
        peak_voltage_snapshot_times_between_spikes_ms(trace, TWO_SPIKE_T_MAX_MS)
    )
    snapshot_times.update(
        detect_current_balance_intersection_count_transition_times_ms(
            trace,
            _parameters,
            t_max_ms=TWO_SPIKE_T_MAX_MS,
        )
    )
    snapshots = {}
    for label, snapshot_time_ms in snapshot_times.items():
        snapshots[label] = generate_fitted_apic20_moving_nullcline_snapshot(
            snapshot_time_ms=snapshot_time_ms,
            output_png_path=output_dir / f"fitted_apic20_two_spike_moving_nullcline_{label}.png",
            t_max_ms=TWO_SPIKE_T_MAX_MS,
            show_plot=show_plot,
            spike_times_ms=spike_times_ms,
            with_s_saturation=with_s_saturation,
        )
    metrics_path = output_dir / "fitted_apic20_spike_time_snapshot_protocol.json"
    metrics_path.write_text(
        json.dumps(
            {
                "gif_path": str(gif_path),
                "snapshots": snapshots,
                "snapshot_times_ms": snapshot_times,
                "spike_times_ms": spike_times_ms.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return gif_path, snapshots, metrics_path


def plot_fitted_apic20_parameter_replay_from_csv(
        output_png_path=FITTED_APIC20_PARAMETER_FIT_DIR / "fitted_apic20_voltage_replay.png",
        show_plot=True,
):
    from src.iteration_20_nonlin_dynamics_NMDA.single_compartment_equivalence import (
        read_standard_apic20_reference,
    )

    reference = read_standard_apic20_reference()
    parameters = read_fitted_apic20_parameters_from_csv()
    trace = simulate_fitted_apic20_trace(reference.t_ms, parameters)
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(reference.t_ms, reference.v_local_mV, color="black", linewidth=1.5, label="reference")
    ax.plot(trace["t_ms"], trace["v_local_mV"], color="tab:blue", linewidth=1.2, label="fitted replay")
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("V [mV]")
    ax.set_title("Fitted APIC20 voltage replay from saved parameter CSV")
    ax.legend(loc="best")
    fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output_png_path


def fitted_apic20_interactive_protocol_specs():
    specs = {
        "standard_three_spike": {
            "label": "standard nullcline, 3 spikes",
            "spike_times_ms": fitted_apic20_spike_times_ms(),
            "t_min_ms": INTERACTIVE_T_MIN_MS,
            "t_max_ms": INTERACTIVE_T_MAX_MS,
            "with_s_saturation": False,
        },
        "current_balance_three_spike": {
            "label": "current-balance nullcline, 3 spikes",
            "spike_times_ms": fitted_apic20_spike_times_ms(),
            "t_min_ms": INTERACTIVE_T_MIN_MS,
            "t_max_ms": INTERACTIVE_T_MAX_MS,
            "with_s_saturation": False,
        },
        "standard_two_spike": {
            "label": "standard nullcline, 2 spikes",
            "spike_times_ms": fitted_apic20_spike_times_up_to_ms(TWO_SPIKE_SELECTION_MAX_MS),
            "t_min_ms": T_MIN_MS,
            "t_max_ms": TWO_SPIKE_T_MAX_MS,
            "with_s_saturation": False,
        },
        "current_balance_two_spike": {
            "label": "current-balance nullcline, 2 spikes",
            "spike_times_ms": fitted_apic20_spike_times_up_to_ms(TWO_SPIKE_SELECTION_MAX_MS),
            "t_min_ms": T_MIN_MS,
            "t_max_ms": TWO_SPIKE_T_MAX_MS,
            "with_s_saturation": False,
        },
        "standard_three_spike_with_s_saturation": {
            "label": "standard nullcline with s saturation, 3 spikes",
            "spike_times_ms": fitted_apic20_spike_times_ms(),
            "t_min_ms": INTERACTIVE_T_MIN_MS,
            "t_max_ms": INTERACTIVE_T_MAX_MS,
            "with_s_saturation": True,
        },
        "current_balance_three_spike_with_s_saturation": {
            "label": "current-balance nullcline with s saturation, 3 spikes",
            "spike_times_ms": fitted_apic20_spike_times_ms(),
            "t_min_ms": INTERACTIVE_T_MIN_MS,
            "t_max_ms": INTERACTIVE_T_MAX_MS,
            "with_s_saturation": True,
        },
        "standard_two_spike_with_s_saturation": {
            "label": "standard nullcline with s saturation, 2 spikes",
            "spike_times_ms": fitted_apic20_spike_times_up_to_ms(TWO_SPIKE_SELECTION_MAX_MS),
            "t_min_ms": T_MIN_MS,
            "t_max_ms": TWO_SPIKE_T_MAX_MS,
            "with_s_saturation": True,
        },
        "current_balance_two_spike_with_s_saturation": {
            "label": "current-balance nullcline with s saturation, 2 spikes",
            "spike_times_ms": fitted_apic20_spike_times_up_to_ms(TWO_SPIKE_SELECTION_MAX_MS),
            "t_min_ms": T_MIN_MS,
            "t_max_ms": TWO_SPIKE_T_MAX_MS,
            "with_s_saturation": True,
        },
        "current_balance_three_spike_e_l_minus_50": {
            "label": "current-balance nullcline, 3 spikes, E_L=-50 mV",
            "spike_times_ms": fitted_apic20_spike_times_ms(),
            "t_min_ms": INTERACTIVE_T_MIN_MS,
            "t_max_ms": INTERACTIVE_T_MAX_MS,
            "with_s_saturation": False,
        },
        "current_balance_three_spike_e_l_minus_50_g_nmda_max_1_nS": {
            "label": (
                "current-balance nullcline, 3 spikes, E_L=-50 mV, "
                "g_NMDA,max=1 nS"
            ),
            "spike_times_ms": fitted_apic20_spike_times_ms(),
            "t_min_ms": INTERACTIVE_T_MIN_MS,
            "t_max_ms": INTERACTIVE_T_MAX_MS,
            "with_s_saturation": False,
        },
        "current_balance_three_spike_e_l_minus_60": {
            "label": "current-balance nullcline, 3 spikes, E_L=-60 mV",
            "spike_times_ms": fitted_apic20_spike_times_ms(),
            "t_min_ms": INTERACTIVE_T_MIN_MS,
            "t_max_ms": INTERACTIVE_T_MAX_MS,
            "with_s_saturation": False,
        },
        "current_balance_three_spike_e_l_minus_60_g_nmda_max_1_nS": {
            "label": (
                "current-balance nullcline, 3 spikes, E_L=-60 mV, "
                "g_NMDA,max=1 nS"
            ),
            "spike_times_ms": fitted_apic20_spike_times_ms(),
            "t_min_ms": INTERACTIVE_T_MIN_MS,
            "t_max_ms": INTERACTIVE_T_MAX_MS,
            "with_s_saturation": False,
        },
        "current_balance_three_spike_e_l_minus_70": {
            "label": "current-balance nullcline, 3 spikes, E_L=-70 mV",
            "spike_times_ms": fitted_apic20_spike_times_ms(),
            "t_min_ms": INTERACTIVE_T_MIN_MS,
            "t_max_ms": INTERACTIVE_T_MAX_MS,
            "with_s_saturation": False,
        },
    }
    for e_l_mV, gmax_values in ((-50, range(2, 11)),
                                 (-60, range(2, 11)),
                                 (-70, range(1, 11))):
        for gmax in gmax_values:
            specs[
                f"current_balance_three_spike_e_l_minus_{abs(e_l_mV)}_"
                f"g_nmda_max_{gmax}_nS"
            ] = {
                "label": (
                    "current-balance nullcline, 3 spikes, "
                    f"E_L={e_l_mV} mV, g_NMDA,max={gmax} nS"
                ),
                "spike_times_ms": fitted_apic20_spike_times_ms(),
                "t_min_ms": INTERACTIVE_T_MIN_MS,
                "t_max_ms": INTERACTIVE_T_MAX_MS,
                "with_s_saturation": False,
            }
    for amplification in G_NMDA_MAX_AMPLIFICATIONS:
        amp_label = amplification_label(amplification)
        specs[fitted_apic20_g_nmda_amplification_protocol_name(amplification)] = {
            "label": f"standard nullcline with s saturation, g_NMDA,max {amp_label}",
            "spike_times_ms": fitted_apic20_spike_times_ms(),
            "t_min_ms": INTERACTIVE_T_MIN_MS,
            "t_max_ms": INTERACTIVE_T_MAX_MS,
            "with_s_saturation": True,
            "g_nmda_max_amplification": float(amplification),
        }
    return {name: specs[name] for name in FITTED_APIC20_INTERACTIVE_PROTOCOL_NAMES}


def save_fitted_apic20_protocol_csv_bundle(
        protocol_name=INTERACTIVE_CSV_PROTOCOL_NAME,
        output_dir=FITTED_NULLCLINE_INTERACTIVE_CSV_DIR,
):
    specs = fitted_apic20_interactive_protocol_specs()
    if protocol_name not in specs:
        raise ValueError(f"Unknown protocol {protocol_name!r}; choose one of {sorted(specs)}")
    spec = specs[protocol_name]
    output_dir = Path(output_dir) / protocol_name
    output_dir.mkdir(parents=True, exist_ok=True)

    g_nmda_max_nS = None
    if "g_nmda_max_amplification" in spec:
        base_parameters = read_fitted_apic20_parameters_from_csv()
        g_nmda_max_nS = (
                float(base_parameters.g_nmda_max_nS)
                * float(spec["g_nmda_max_amplification"])
        )
    _reference, parameters, trace = read_fit_and_simulate_fitted_apic20_reference(
        spike_times_ms=spec["spike_times_ms"],
        with_s_saturation=spec["with_s_saturation"],
        g_nmda_max_nS=g_nmda_max_nS,
    )
    trace_path = output_dir / "trace.csv"
    metadata_path = output_dir / "metadata.json"
    _save_interactive_trace_csv(trace, trace_path)
    metadata = {
        "protocol_name": protocol_name,
        "label": spec["label"],
        "t_min_ms": spec["t_min_ms"],
        "t_max_ms": spec["t_max_ms"],
        "spike_times_ms": np.asarray(spec["spike_times_ms"], dtype=float).tolist(),
        "with_s_saturation": bool(spec["with_s_saturation"]),
        "parameters": asdict(parameters),
        "trace_csv": str(trace_path),
    }
    if "g_nmda_max_amplification" in spec:
        metadata["g_nmda_max_amplification"] = float(spec["g_nmda_max_amplification"])
        metadata["g_nmda_max_nS"] = float(parameters.g_nmda_max_nS)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return trace_path, metadata_path


def _save_interactive_trace_csv(trace, path):
    columns = [
        ("t_ms", trace["t_ms"]),
        ("v_local_mV", trace["v_local_mV"]),
        ("x", trace["x"]),
        ("s", trace["s"]),
        ("g_nmda_nS", trace["g_nmda_nS"]),
        ("sigma", trace["sigma"]),
        ("empirical_dvdt_mV_per_ms", trace["empirical_dvdt_mV_per_ms"]),
        ("empirical_d2vdt2_mV_per_ms2", trace["empirical_d2vdt2_mV_per_ms2"]),
    ]
    for spike_index, individual_s in enumerate(trace["individual_s"], start=1):
        columns.append((f"individual_s_{spike_index}", individual_s))
    header = ",".join(name for name, _values in columns)
    data = np.column_stack([np.asarray(values, dtype=float) for _name, values in columns])
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def read_fitted_apic20_protocol_csv_bundle(
        protocol_name=INTERACTIVE_CSV_PROTOCOL_NAME,
        output_dir=FITTED_NULLCLINE_INTERACTIVE_CSV_DIR,
):
    bundle_dir = Path(output_dir) / protocol_name
    metadata_path = bundle_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    trace_path = resolve_protocol_trace_csv_path(metadata_path, metadata)
    data = np.genfromtxt(trace_path, delimiter=",", names=True)
    trace = {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}
    individual_names = sorted(
        name for name in trace if name.startswith("individual_s_")
    )
    trace["individual_s"] = np.asarray([trace[name] for name in individual_names])
    trace["spike_times_ms"] = np.asarray(metadata["spike_times_ms"], dtype=float)
    return metadata, trace


def resolve_protocol_trace_csv_path(metadata_path, metadata):
    metadata_path = Path(metadata_path)
    trace_csv = metadata.get("trace_csv", "trace.csv")
    trace_path = Path(trace_csv)
    if trace_path.is_absolute() and trace_path.exists():
        return trace_path
    if trace_path.exists():
        return trace_path
    relative_trace_path = metadata_path.parent / trace_path
    if relative_trace_path.exists():
        return relative_trace_path
    fallback_trace_path = metadata_path.parent / "trace.csv"
    if fallback_trace_path.exists():
        return fallback_trace_path
    return relative_trace_path


def save_fitted_apic20_protocol_csv_bundle_from_trace(
        protocol_name,
        parameters,
        trace,
        output_dir=FITTED_NULLCLINE_INTERACTIVE_CSV_DIR,
):
    specs = fitted_apic20_interactive_protocol_specs()
    if protocol_name not in specs:
        raise ValueError(f"Unknown protocol {protocol_name!r}; choose one of {sorted(specs)}")
    spec = specs[protocol_name]
    bundle_dir = Path(output_dir) / protocol_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    trace_path = bundle_dir / "trace.csv"
    metadata_path = bundle_dir / "metadata.json"
    _save_interactive_trace_csv(trace, trace_path)
    metadata = {
        "protocol_name": protocol_name,
        "label": spec["label"],
        "t_min_ms": spec["t_min_ms"],
        "t_max_ms": spec["t_max_ms"],
        "spike_times_ms": np.asarray(spec["spike_times_ms"], dtype=float).tolist(),
        "with_s_saturation": bool(spec["with_s_saturation"]),
        "parameters": asdict(parameters),
        "trace_csv": str(trace_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return trace_path, metadata_path


def save_saturated_g_nmda_amplification_protocol_csv_bundle_from_trace(
        amplification,
        parameters,
        trace,
        spike_times_ms=None,
        t_max_ms=INTERACTIVE_T_MAX_MS,
        output_dir=FITTED_NULLCLINE_INTERACTIVE_CSV_DIR,
):
    if spike_times_ms is None:
        spike_times_ms = fitted_apic20_spike_times_ms()
    spike_times_ms = np.asarray(spike_times_ms, dtype=float)
    protocol_name = fitted_apic20_g_nmda_amplification_protocol_name(
        amplification,
        spikes=spike_times_ms,
    )
    bundle_dir = Path(output_dir) / protocol_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    trace_path = bundle_dir / "trace.csv"
    metadata_path = bundle_dir / "metadata.json"
    _save_interactive_trace_csv(trace, trace_path)
    metadata = {
        "protocol_name": protocol_name,
        "label": (
            "standard nullcline with s saturation, "
            f"g_NMDA,max {amplification_label(amplification)}, "
            f"{len(spike_times_ms)} spikes"
        ),
        "t_min_ms": INTERACTIVE_T_MIN_MS,
        "t_max_ms": float(t_max_ms),
        "spike_times_ms": spike_times_ms.tolist(),
        "with_s_saturation": True,
        "g_nmda_max_amplification": float(amplification),
        "g_nmda_max_nS": float(parameters.g_nmda_max_nS),
        "parameters": asdict(parameters),
        "trace_csv": str(trace_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return trace_path, metadata_path


def save_g_nmda_max_sweep_protocol_csv_bundle_from_trace(
        g_nmda_max_nS,
        parameters,
        trace,
        spike_times_ms=None,
        protocol_name=None,
        t_max_ms=INTERACTIVE_T_MAX_MS,
        output_dir=FITTED_NULLCLINE_INTERACTIVE_CSV_DIR,
):
    if spike_times_ms is None:
        spike_times_ms = fitted_apic20_spike_times_ms()
    spike_times_ms = np.asarray(spike_times_ms, dtype=float)
    if protocol_name is None:
        protocol_name = fitted_apic20_three_spike_g_nmda_max_sweep_protocol_name(
            g_nmda_max_nS
        )
    bundle_dir = Path(output_dir) / protocol_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    trace_path = bundle_dir / "trace.csv"
    metadata_path = bundle_dir / "metadata.json"
    _save_interactive_trace_csv(trace, trace_path)
    metadata = {
        "protocol_name": protocol_name,
        "label": (
            "standard nullcline, "
            f"g_NMDA,max {float(g_nmda_max_nS):.12g} nS, "
            f"{len(spike_times_ms)} spikes"
        ),
        "t_min_ms": INTERACTIVE_T_MIN_MS,
        "t_max_ms": float(t_max_ms),
        "spike_times_ms": spike_times_ms.tolist(),
        "with_s_saturation": False,
        "g_nmda_max_nS": float(parameters.g_nmda_max_nS),
        "parameters": asdict(parameters),
        "trace_csv": str(trace_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return trace_path, metadata_path


def save_manual_apic20_protocol_csv_bundle_from_trace(
        protocol_name,
        label,
        parameters,
        trace,
        spike_times_ms,
        with_s_saturation,
        t_min_ms=INTERACTIVE_T_MIN_MS,
        t_max_ms=INTERACTIVE_T_MAX_MS,
        output_dir=FITTED_NULLCLINE_INTERACTIVE_CSV_DIR,
):
    spike_times_ms = np.asarray(spike_times_ms, dtype=float)
    bundle_dir = Path(output_dir) / protocol_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    trace_path = bundle_dir / "trace.csv"
    metadata_path = bundle_dir / "metadata.json"
    _save_interactive_trace_csv(trace, trace_path)
    metadata = {
        "protocol_name": protocol_name,
        "label": label,
        "t_min_ms": float(t_min_ms),
        "t_max_ms": float(t_max_ms),
        "spike_times_ms": spike_times_ms.tolist(),
        "with_s_saturation": bool(with_s_saturation),
        "parameters": asdict(parameters),
        "trace_csv": str(trace_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return trace_path, metadata_path


def simulate(
        cm,
        gl,
        gnmdamax,
        presynaptic_spike_train,
        experiment_label,
        with_saturation=False,
        show_plot=False,
        t_min_ms=INTERACTIVE_T_MIN_MS,
        t_max_ms=INTERACTIVE_T_MAX_MS,
):
    spike_times_ms = np.asarray(presynaptic_spike_train, dtype=float)
    base_parameters = read_fitted_apic20_parameters_from_csv()
    parameters = replace(
        base_parameters,
        capacitance_nF=float(cm),
        g_leak_nS=float(gl),
        g_nmda_max_nS=float(gnmdamax),
    )
    reference, parameters, trace = read_fit_and_simulate_fitted_apic20_reference(
        spike_times_ms=spike_times_ms,
        with_s_saturation=with_saturation,
        parameters=parameters,
    )
    protocol_name = f"manual_search/{protocol_path_label(experiment_label)}"
    output_dir = DEFAULT_OUTPUT_DIR / "fitted_apic20_manual_search" / protocol_path_label(
        experiment_label
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path, metadata_path = save_manual_apic20_protocol_csv_bundle_from_trace(
        protocol_name=protocol_name,
        label=experiment_label,
        parameters=parameters,
        trace=trace,
        spike_times_ms=spike_times_ms,
        with_s_saturation=with_saturation,
        t_min_ms=t_min_ms,
        t_max_ms=t_max_ms,
    )
    gif_path = generate_fitted_apic20_moving_nullcline_animation(
        output_gif_path=output_dir / "fitted_apic20_moving_nullcline.gif",
        t_min_ms=t_min_ms,
        t_max_ms=t_max_ms,
        show_plot=show_plot,
        spike_times_ms=spike_times_ms,
        with_s_saturation=with_saturation,
        parameters=parameters,
    )
    snapshot_times = fitted_apic20_snapshot_times_up_to_ms(
        max_time_ms=t_max_ms,
        spike_times_ms=spike_times_ms,
    )
    snapshot_times["max_reference_empirical_dvdt"] = max_reference_empirical_dvdt_time_ms(
        reference,
        t_min_ms=t_min_ms,
        t_max_ms=t_max_ms,
    )
    snapshot_times.update(peak_voltage_snapshot_times_between_spikes_ms(trace, t_max_ms))
    snapshot_times.update(
        detect_current_balance_intersection_count_transition_times_ms(
            trace,
            parameters,
            t_max_ms=t_max_ms,
        )
    )
    snapshots = {}
    for label, snapshot_time_ms in snapshot_times.items():
        snapshots[label] = generate_fitted_apic20_moving_nullcline_snapshot(
            snapshot_time_ms=snapshot_time_ms,
            output_png_path=output_dir / f"fitted_apic20_moving_nullcline_{label}.png",
            t_min_ms=t_min_ms,
            t_max_ms=t_max_ms,
            show_plot=show_plot,
            spike_times_ms=spike_times_ms,
            with_s_saturation=with_saturation,
            parameters=parameters,
        )
    metrics_path = output_dir / "fitted_apic20_manual_search_protocol.json"
    metrics = {
        "experiment_label": experiment_label,
        "protocol_name": protocol_name,
        "gif_path": str(gif_path),
        "trace_path": str(trace_path),
        "metadata_path": str(metadata_path),
        "snapshots": snapshots,
        "snapshot_times_ms": snapshot_times,
        "spike_times_ms": spike_times_ms.tolist(),
        "with_s_saturation": bool(with_saturation),
        "parameters": asdict(parameters),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return {
        "protocol_name": protocol_name,
        "output_dir": output_dir,
        "gif_path": gif_path,
        "trace_path": trace_path,
        "metadata_path": metadata_path,
        "snapshots": snapshots,
        "metrics_path": metrics_path,
        "parameters": parameters,
    }


COMPARISON_RUN_COLORS = ("tab:blue", "tab:orange")


def compare_runs(
        runs_files,
        output_dir=None,
        comparison_label=None,
        show_plot=False,
        frame_count=FRAME_COUNT,
        fps=FPS,
):
    if comparison_label is None:
        raise ValueError("comparison_label is required for compare_runs")
    runs = [resolve_comparison_run(run_file) for run_file in runs_files]

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / "fitted_apic20_run_comparisons" / protocol_path_label(
            comparison_label
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_min_ms, t_max_ms = comparison_time_window(runs)
    snapshot_times = fitted_apic20_snapshot_times_up_to_ms(
        max_time_ms=t_max_ms,
        spike_times_ms=runs[0]["trace"]["spike_times_ms"],
    )
    for run_index, run in enumerate(runs, start=1):
        for label, time_ms in peak_voltage_snapshot_times_between_spikes_ms(
                run["trace"],
                t_max_ms,
        ).items():
            snapshot_times[f"run{run_index}_{label}"] = time_ms
    snapshot_times = dict(sorted(snapshot_times.items(), key=lambda item: item[1]))

    gif_path = output_dir / "apic20_run_comparison.gif"
    generate_run_comparison_animation(
        runs=runs,
        output_gif_path=gif_path,
        t_min_ms=t_min_ms,
        t_max_ms=t_max_ms,
        frame_count=frame_count,
        fps=fps,
        show_plot=show_plot,
    )
    snapshots = {}
    for label, snapshot_time_ms in snapshot_times.items():
        snapshots[label] = generate_run_comparison_snapshot(
            runs=runs,
            snapshot_time_ms=snapshot_time_ms,
            output_png_path=output_dir / f"apic20_run_comparison_{label}.png",
            t_min_ms=t_min_ms,
            t_max_ms=t_max_ms,
            show_plot=show_plot,
        )
    metrics_path = output_dir / "apic20_run_comparison_protocol.json"
    metrics = {
        "comparison_label": comparison_label,
        "gif_path": str(gif_path),
        "snapshot_times_ms": snapshot_times,
        "snapshots": snapshots,
        "runs": [
            {
                "label": run["label"],
                "protocol_name": run["metadata"]["protocol_name"],
                "run_dir": str(run["run_dir"]),
                "metadata_path": str(run["metadata_path"]),
                "trace_path": str(run["trace_path"]),
            }
            for run in runs
        ],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return {
        "gif_path": gif_path,
        "snapshots": snapshots,
        "metrics_path": metrics_path,
        "runs": runs,
    }


def resolve_comparison_run(run_dir):
    run_dir = Path(run_dir)
    if run_dir.name == "fitted_apic20_manual_search":
        manual_run_dirs = sorted(
            path.parent
            for path in run_dir.glob("*/fitted_apic20_manual_search_protocol.json")
        )
        if not manual_run_dirs:
            raise ValueError(f"No manual APIC20 runs found under {run_dir}")
        run_dir = manual_run_dirs[-1]
    protocol_name = None
    manual_metrics_path = run_dir / "fitted_apic20_manual_search_protocol.json"
    if manual_metrics_path.exists():
        manual_metrics = json.loads(manual_metrics_path.read_text())
        protocol_name = manual_metrics["protocol_name"]
    elif (run_dir / "metadata.json").exists():
        metadata = json.loads((run_dir / "metadata.json").read_text())
        protocol_name = metadata["protocol_name"]
    elif run_dir.name == "fitted_apic20_nullcline":
        protocol_name = "standard_three_spike"
    elif run_dir.name == "fitted_apic20_current_nullcline":
        protocol_name = "current_balance_three_spike"
    else:
        raise ValueError(
            f"Could not resolve a CSV-backed APIC20 protocol from {run_dir}"
        )
    metadata, trace = read_fitted_apic20_protocol_csv_bundle(protocol_name)
    metadata_path = FITTED_NULLCLINE_INTERACTIVE_CSV_DIR / protocol_name / "metadata.json"
    trace_path = resolve_protocol_trace_csv_path(metadata_path, metadata)
    return {
        "run_dir": run_dir,
        "metadata_path": metadata_path,
        "trace_path": trace_path,
        "metadata": metadata,
        "trace": trace,
        "parameters": FittedApic20NullclineParameters(**metadata["parameters"]),
        "label": metadata.get("label", protocol_name),
        "slug": protocol_path_label(metadata.get("label", protocol_name)),
    }


def comparison_time_window(runs):
    t_min_ms = max(float(run["metadata"]["t_min_ms"]) for run in runs)
    t_max_ms = min(float(run["metadata"]["t_max_ms"]) for run in runs)
    for run in runs:
        t_ms = np.asarray(run["trace"]["t_ms"], dtype=float)
        t_min_ms = max(t_min_ms, float(np.min(t_ms)))
        t_max_ms = min(t_max_ms, float(np.max(t_ms)))
    if t_max_ms <= t_min_ms:
        raise ValueError(f"Runs do not overlap in time: {t_min_ms} to {t_max_ms}")
    return t_min_ms, t_max_ms


def generate_run_comparison_animation(
        runs,
        output_gif_path,
        t_min_ms,
        t_max_ms,
        frame_count=FRAME_COUNT,
        fps=FPS,
        show_plot=False,
):
    output_gif_path = Path(output_gif_path)
    output_gif_path.parent.mkdir(parents=True, exist_ok=True)
    frame_times_ms = np.linspace(t_min_ms, t_max_ms, frame_count)
    fig, axes = create_run_comparison_figure()

    def update(frame_index):
        plot_run_comparison_axes(
            axes,
            runs,
            frame_times_ms[frame_index],
            t_min_ms,
            t_max_ms,
        )
        return tuple(
            artist
            for ax in axes.values()
            if ax is not fig
            for artist in ax.get_children()
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_times_ms),
        interval=1000 / fps,
        blit=False,
    )
    animation.save(output_gif_path, writer=PillowWriter(fps=fps))
    if show_plot:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output_gif_path


def generate_run_comparison_snapshot(
        runs,
        snapshot_time_ms,
        output_png_path,
        t_min_ms,
        t_max_ms,
        show_plot=False,
):
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = create_run_comparison_figure()
    plot_run_comparison_axes(
        axes,
        runs,
        snapshot_time_ms,
        t_min_ms,
        t_max_ms,
    )
    fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
    if show_plot:
        fig.show()
        plt.show(block=True)
    else:
        plt.close(fig)
    return {"plot_path": str(output_png_path), "snapshot_time_ms": float(snapshot_time_ms)}


def create_run_comparison_figure():
    fig, axes = plt.subplot_mosaic(
        [
            ["voltage_ax", "g_nmda_ax", "s_ax"],
            ["nullcline_ax", "dvdt_ax", "d2vdt2_ax"],
        ],
        figsize=(14, 8),
    )
    axes["fig"] = fig
    return fig, axes


def plot_run_comparison_axes(axes, runs, selected_t_ms, t_min_ms, t_max_ms):
    fig = axes["fig"]
    for name, ax in axes.items():
        if name != "fig":
            ax.cla()
    v_values_mV = np.linspace(-90.0, 20.0, 500)
    curve_specs = [
        ("voltage_ax", "v_local_mV", "V [mV]", r"$V(t)$"),
        ("g_nmda_ax", "g_nmda_nS", r"$g_{\mathrm{NMDA}}$ [nS]", r"$g_{\mathrm{NMDA}}(t)$"),
        ("s_ax", "s", "s", r"fitted $s(t)$"),
        ("dvdt_ax", "empirical_dvdt_mV_per_ms", r"$dV/dt$ [mV/ms]", r"$dV/dt$"),
        ("d2vdt2_ax", "empirical_d2vdt2_mV_per_ms2", r"$d^2V/dt^2$ [mV/ms$^2$]", r"$d^2V/dt^2$"),
    ]
    nullcline_values = []
    for run, color in zip(runs, COMPARISON_RUN_COLORS):
        trace = run["trace"]
        parameters = run["parameters"]
        t_ms = np.asarray(trace["t_ms"], dtype=float)
        selected_index = nearest_trace_index(t_ms, selected_t_ms)
        for ax_name, trace_key, _ylabel, _title in curve_specs:
            ax = axes[ax_name]
            values = np.asarray(trace[trace_key], dtype=float)
            ax.plot(t_ms, values, color=color, linewidth=1.3, alpha=0.85, label=run["label"])
            ax.plot(
                [t_ms[selected_index]],
                [values[selected_index]],
                "o",
                color=color,
                markersize=4,
            )
            ax.axvline(selected_t_ms, color="0.75", linewidth=0.8)
        nullcline_dvdt = compute_local_nullcline_dvdt_mV_per_ms(
            parameters,
            selected_s=trace["s"][selected_index],
            v_values_mV=v_values_mV,
        )
        nullcline_values.append(nullcline_dvdt)
        axes["nullcline_ax"].plot(
            v_values_mV,
            nullcline_dvdt,
            color=color,
            linewidth=1.5,
            alpha=0.9,
            label=run["label"],
        )
        current_v_mV = float(trace["v_local_mV"][selected_index])
        current_nullcline = float(np.interp(current_v_mV, v_values_mV, nullcline_dvdt))
        axes["nullcline_ax"].plot(
            [current_v_mV],
            [current_nullcline],
            "o",
            color=color,
            markersize=4,
        )
    for ax_name, _trace_key, ylabel, title in curve_specs:
        ax = axes[ax_name]
        ax.set_xlim(t_min_ms, t_max_ms)
        ax.set_xlabel("t [ms]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
    axes["nullcline_ax"].set_xlim(-90.0, 20.0)
    axes["nullcline_ax"].set_ylim(derivative_ylim(np.asarray(nullcline_values)))
    axes["nullcline_ax"].set_xlabel("V [mV]")
    axes["nullcline_ax"].set_ylabel(r"$dV/dt$ [mV/ms]")
    axes["nullcline_ax"].set_title(r"$dV/dt(V; s(t))$")
    axes["nullcline_ax"].axhline(0.0, color="black", linewidth=1.0)
    axes["nullcline_ax"].legend(fontsize=7, loc="best")
    fig.suptitle(f"APIC20 run comparison, t={float(selected_t_ms):.1f} ms", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.88, hspace=0.45)


def nearest_trace_index(t_ms, selected_t_ms, min_index=0, max_index=None):
    t_ms = np.asarray(t_ms, dtype=float)
    if max_index is None:
        max_index = len(t_ms) - 1
    insertion_index = int(np.searchsorted(t_ms, float(selected_t_ms), side="left"))
    candidate_indexes = [
        int(np.clip(insertion_index - 1, min_index, max_index)),
        int(np.clip(insertion_index, min_index, max_index)),
    ]
    return min(
        candidate_indexes,
        key=lambda index: abs(float(t_ms[index]) - float(selected_t_ms)),
    )


def ensure_fitted_apic20_protocol_csv_bundle(
        protocol_name=INTERACTIVE_CSV_PROTOCOL_NAME,
        output_dir=FITTED_NULLCLINE_INTERACTIVE_CSV_DIR,
):
    bundle_dir = Path(output_dir) / protocol_name
    trace_path = bundle_dir / "trace.csv"
    metadata_path = bundle_dir / "metadata.json"
    if not trace_path.exists() or not metadata_path.exists():
        return save_fitted_apic20_protocol_csv_bundle(protocol_name, output_dir)
    specs = fitted_apic20_interactive_protocol_specs()
    metadata = json.loads(metadata_path.read_text())
    spec = specs[protocol_name]
    metadata_is_stale = (
            not np.isclose(float(metadata.get("t_min_ms", np.nan)), float(spec["t_min_ms"]))
            or not np.isclose(float(metadata.get("t_max_ms", np.nan)), float(spec["t_max_ms"]))
            or metadata.get("spike_times_ms") != np.asarray(spec["spike_times_ms"], dtype=float).tolist()
            or bool(metadata.get("with_s_saturation", False)) != bool(spec["with_s_saturation"])
            or not np.isclose(
        float(metadata.get("g_nmda_max_amplification", 1.0)),
        float(spec.get("g_nmda_max_amplification", 1.0)),
    )
    )
    if metadata_is_stale:
        return save_fitted_apic20_protocol_csv_bundle(protocol_name, output_dir)
    return trace_path, metadata_path


def available_fitted_apic20_csv_protocol_names(
        output_dir=FITTED_NULLCLINE_INTERACTIVE_CSV_DIR,
):
    output_dir = Path(output_dir)
    configured_names = list(FITTED_APIC20_INTERACTIVE_PROTOCOL_NAMES)
    discovered_names = []
    for metadata_path in output_dir.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
            trace_path = resolve_protocol_trace_csv_path(metadata_path, metadata)
        except Exception:
            continue
        if trace_path.exists():
            discovered_names.append(metadata_path.parent.relative_to(output_dir).as_posix())
    return tuple(dict.fromkeys([*configured_names, *sorted(discovered_names)]))


class MatplotlibDropdown:
    def __init__(
            self,
            fig,
            selector_ax,
            menu_ax,
            options,
            active=0,
            max_visible_options=12,
            on_select=None,
    ):
        self.fig = fig
        self.selector_ax = selector_ax
        self.menu_ax = menu_ax
        self.options = tuple(options)
        self.active = int(active)
        self.max_visible_options = int(max_visible_options)
        self.on_select = on_select
        self.offset = 0
        self.is_open = False
        self._texts = []
        self._connections = [
            fig.canvas.mpl_connect("button_press_event", self._handle_click),
            fig.canvas.mpl_connect("scroll_event", self._handle_scroll),
        ]
        self._configure_axes()
        self._render_selector()
        self._render_menu()

    def _configure_axes(self):
        for ax in [self.selector_ax, self.menu_ax]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_navigate(False)
        self.selector_ax.set_xlim(0.0, 1.0)
        self.selector_ax.set_ylim(0.0, 1.0)
        self.menu_ax.set_xlim(0.0, 1.0)
        self.menu_ax.set_ylim(0.0, 1.0)
        self.menu_ax.set_zorder(20)
        self.menu_ax.set_visible(False)

    def _render_selector(self):
        self.selector_ax.cla()
        self.selector_ax.set_xticks([])
        self.selector_ax.set_yticks([])
        self.selector_ax.set_xlim(0.0, 1.0)
        self.selector_ax.set_ylim(0.0, 1.0)
        self.selector_ax.text(0.02, 0.5, "CSV bundle", va="center", ha="left", fontsize=9)
        selected_label = self.options[self.active] if self.options else ""
        self.selector_ax.text(
            0.18,
            0.5,
            selected_label,
            va="center",
            ha="left",
            fontsize=9,
            clip_on=True,
        )
        self.selector_ax.text(
            0.985,
            0.5,
            "v" if not self.is_open else "^",
            va="center",
            ha="right",
            fontsize=10,
        )

    def _render_menu(self):
        self.menu_ax.cla()
        self.menu_ax.set_xticks([])
        self.menu_ax.set_yticks([])
        self.menu_ax.set_xlim(0.0, 1.0)
        self.menu_ax.set_ylim(0.0, 1.0)
        visible_count = min(self.max_visible_options, len(self.options))
        if visible_count <= 0:
            self.menu_ax.text(0.02, 0.5, "No CSV bundles found", va="center", ha="left", fontsize=9)
            return
        max_offset = max(0, len(self.options) - visible_count)
        self.offset = min(max(self.offset, 0), max_offset)
        self._texts = []
        for row_index in range(visible_count):
            option_index = self.offset + row_index
            y_top = 1.0 - row_index / visible_count
            y_bottom = 1.0 - (row_index + 1) / visible_count
            facecolor = "0.88" if option_index == self.active else "white"
            self.menu_ax.axhspan(y_bottom, y_top, facecolor=facecolor, edgecolor="0.85")
            text = self.menu_ax.text(
                0.02,
                (y_top + y_bottom) / 2.0,
                self.options[option_index],
                va="center",
                ha="left",
                fontsize=8,
                clip_on=True,
            )
            self._texts.append(text)
        if len(self.options) > visible_count:
            self.menu_ax.text(
                0.985,
                0.02,
                f"{self.offset + 1}-{self.offset + visible_count} of {len(self.options)}",
                va="bottom",
                ha="right",
                fontsize=7,
                color="0.35",
            )

    def _toggle_menu(self):
        self.is_open = not self.is_open
        self.menu_ax.set_visible(self.is_open)
        self._render_selector()
        if self.is_open:
            self._render_menu()
        self.fig.canvas.draw_idle()

    def _close_menu(self):
        if not self.is_open:
            return
        self.is_open = False
        self.menu_ax.set_visible(False)
        self._render_selector()
        self.fig.canvas.draw_idle()

    def _select_index(self, option_index):
        if option_index < 0 or option_index >= len(self.options):
            return
        if option_index == self.active:
            self._close_menu()
            return
        self.active = int(option_index)
        self._close_menu()
        if self.on_select is not None:
            self.on_select(self.options[self.active])

    def _handle_click(self, event):
        if event.inaxes == self.selector_ax:
            self._toggle_menu()
            return
        if event.inaxes == self.menu_ax and self.is_open:
            visible_count = min(self.max_visible_options, len(self.options))
            if visible_count <= 0 or event.ydata is None:
                return
            row_index = int((1.0 - float(event.ydata)) * visible_count)
            row_index = min(max(row_index, 0), visible_count - 1)
            self._select_index(self.offset + row_index)
            return
        self._close_menu()

    def _handle_scroll(self, event):
        if event.inaxes != self.menu_ax or not self.is_open:
            return
        visible_count = min(self.max_visible_options, len(self.options))
        max_offset = max(0, len(self.options) - visible_count)
        if max_offset <= 0:
            return
        self.offset = min(max(self.offset - int(event.step), 0), max_offset)
        self._render_menu()
        self.fig.canvas.draw_idle()


def ensure_interactive_matplotlib_backend():
    current_backend = plt.get_backend().lower()
    if "agg" not in current_backend and "inline" not in current_backend:
        return plt.get_backend()
    errors = []
    for backend in ["QtAgg", "Qt5Agg", "TkAgg"]:
        try:
            plt.switch_backend(backend)
            return plt.get_backend()
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError(
        "Could not switch Matplotlib to an interactive GUI backend. "
        "Tried QtAgg, Qt5Agg, and TkAgg. Errors: " + " | ".join(errors)
    )


def launch_fitted_apic20_csv_snapshot_slider(
        protocol_name=INTERACTIVE_CSV_PROTOCOL_NAME,
        output_dir=FITTED_NULLCLINE_INTERACTIVE_CSV_DIR,
        show_plot=True,
):
    if show_plot:
        ensure_interactive_matplotlib_backend()
    if protocol_name in fitted_apic20_interactive_protocol_specs():
        ensure_fitted_apic20_protocol_csv_bundle(protocol_name, output_dir)
    metadata, trace = read_fitted_apic20_protocol_csv_bundle(protocol_name, output_dir)
    parameters = FittedApic20NullclineParameters(**metadata["parameters"])
    return create_fitted_apic20_csv_snapshot_slider_figure(
        metadata=metadata,
        trace=trace,
        parameters=parameters,
        output_dir=output_dir,
        available_protocol_names=available_fitted_apic20_csv_protocol_names(output_dir),
        show_plot=show_plot,
    )


def create_fitted_apic20_csv_snapshot_slider_figure(
        metadata,
        trace,
        parameters,
        output_dir=FITTED_NULLCLINE_INTERACTIVE_CSV_DIR,
        available_protocol_names=FITTED_APIC20_INTERACTIVE_PROTOCOL_NAMES,
        show_plot=True,
):
    v_values_mV = np.linspace(-90.0, 20.0, 500)
    axes = create_current_like_interactive_figure()
    fig = axes["fig"]
    protocols = tuple(available_protocol_names)
    if metadata["protocol_name"] not in protocols:
        protocols = (metadata["protocol_name"], *protocols)
    slider_ax = fig.add_axes([0.12, 0.025, 0.76, 0.025])
    slider = Slider(
        slider_ax,
        "t [ms]",
        float(metadata["t_min_ms"]),
        float(metadata["t_max_ms"]),
        valinit=float(metadata["t_min_ms"]),
        valfmt="%.3f ms",
    )
    dropdown_ax = fig.add_axes([0.12, 0.92, 0.76, 0.035])
    dropdown_menu_ax = fig.add_axes([0.12, 0.53, 0.76, 0.36])
    timestamp_ax = fig.add_axes([0.895, 0.025, 0.085, 0.025])
    timestamp_button = Button(timestamp_ax, "Print time")

    state = {}

    def update_from_slider(selected_t_ms):
        if not state:
            return
        t_ms = state["t_ms"]
        insertion_index = int(np.searchsorted(t_ms, float(selected_t_ms), side="left"))
        if insertion_index <= 0:
            index = 0
        elif insertion_index >= len(t_ms):
            index = len(t_ms) - 1
        else:
            left_index = insertion_index - 1
            right_index = insertion_index
            if abs(t_ms[left_index] - selected_t_ms) <= abs(t_ms[right_index] - selected_t_ms):
                index = left_index
            else:
                index = right_index
        update_interactive_csv_snapshot(
            index=index,
            metadata=state["metadata"],
            trace=state["trace"],
            parameters=state["parameters"],
            v_values_mV=v_values_mV,
            residual_cache=state["residual_cache"],
            artists=state["artists"],
        )
        state["current_index"] = index
        fig.canvas.draw_idle()

    def load_protocol(protocol_name):
        specs = fitted_apic20_interactive_protocol_specs()
        if protocol_name in specs:
            ensure_fitted_apic20_protocol_csv_bundle(protocol_name, output_dir)
        next_metadata, next_trace = read_fitted_apic20_protocol_csv_bundle(
            protocol_name,
            output_dir,
        )
        next_parameters = FittedApic20NullclineParameters(**next_metadata["parameters"])
        t_ms = np.asarray(next_trace["t_ms"], dtype=float)
        sample_indexes = np.unique(
            np.linspace(0, len(t_ms) - 1, min(250, len(t_ms)), dtype=int)
        )
        sampled_residual_curves = np.asarray(
            [
                compute_local_nullcline_dvdt_mV_per_ms(
                    next_parameters,
                    next_trace["s"][index],
                    v_values_mV,
                )
                for index in sample_indexes
            ]
        )
        for name, ax in axes.items():
            if name != "fig":
                ax.cla()
        configure_interactive_csv_axes(
            axes,
            next_trace,
            next_metadata,
            sampled_residual_curves,
        )
        plot_interactive_static_axes(axes, next_trace)
        artists = create_interactive_csv_artists(axes, fig, t_ms[0])
        state.update(
            {
                "metadata": next_metadata,
                "trace": next_trace,
                "parameters": next_parameters,
                "t_ms": t_ms,
                "residual_cache": {},
                "artists": artists,
            }
        )
        visible_indexes = np.flatnonzero(
            (t_ms >= float(next_metadata["t_min_ms"]))
            & (t_ms <= float(next_metadata["t_max_ms"]))
        )
        initial_index = int(visible_indexes[0]) if len(visible_indexes) else 0
        dt_ms = float(np.median(np.diff(t_ms))) if len(t_ms) > 1 else 1.0
        slider.eventson = False
        slider.valmin = float(next_metadata["t_min_ms"])
        slider.valmax = float(next_metadata["t_max_ms"])
        slider.valstep = dt_ms
        slider.ax.set_xlim(slider.valmin, slider.valmax)
        slider.set_val(float(t_ms[initial_index]))
        slider.eventson = True
        update_from_slider(float(t_ms[initial_index]))

    def select_protocol(protocol_name):
        load_protocol(protocol_name)

    def print_current_timestamp(_event):
        if not state:
            return
        current_index = state.get("current_index", 0)
        current_t_ms = float(state["t_ms"][current_index])
        label = f"t{current_t_ms:.3f}ms".replace(".", "p")
        print(f'            "{label}": {current_t_ms:.12g},')

    slider.on_changed(update_from_slider)
    timestamp_button.on_clicked(print_current_timestamp)
    dropdown = MatplotlibDropdown(
        fig=fig,
        selector_ax=dropdown_ax,
        menu_ax=dropdown_menu_ax,
        options=protocols,
        active=protocols.index(metadata["protocol_name"]),
        on_select=select_protocol,
    )
    load_protocol(metadata["protocol_name"])
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.09, top=0.88, hspace=0.45)
    fig._fitted_apic20_csv_slider_widgets = {
        "slider": slider,
        "dropdown": dropdown,
        "timestamp_button": timestamp_button,
    }
    if show_plot:
        plt.show(block=True)
    return fig, axes, slider


def create_interactive_csv_artists(axes, fig, initial_t_ms):
    active_voltage_line, = axes["voltage_ax"].plot([], [], color="tab:blue", linewidth=2.0)
    active_g_line, = axes["g_nmda_ax"].plot([], [], color="tab:green", linewidth=2.0)
    active_s_line, = axes["s_ax"].plot([], [], color="tab:purple", linewidth=2.0)
    residual_line, = axes["nullcline_ax"].plot([], [], color="tab:orange", linewidth=2.0)
    time_lines = [
        axes[name].axvline(initial_t_ms, color="0.45", linewidth=1.0)
        for name in ["voltage_ax", "g_nmda_ax", "s_ax"]
    ]
    residual_v_line = axes["nullcline_ax"].axvline(initial_t_ms, color="tab:blue", linewidth=1.0)
    voltage_point, = axes["voltage_ax"].plot([], [], "o", color="tab:blue")
    g_point, = axes["g_nmda_ax"].plot([], [], "o", color="tab:green")
    s_point, = axes["s_ax"].plot([], [], "o", color="tab:purple")
    residual_point, = axes["nullcline_ax"].plot([], [], "o", color="tab:orange")
    stable_intersection_dots, = axes["nullcline_ax"].plot(
        [],
        [],
        "o",
        color="black",
        markerfacecolor="black",
        markersize=STABLE_FIXED_POINT_MARKERSIZE,
        linestyle="None",
    )
    unstable_intersection_dots, = axes["nullcline_ax"].plot(
        [],
        [],
        "o",
        color="black",
        markerfacecolor="none",
        markersize=UNSTABLE_FIXED_POINT_MARKERSIZE,
        linestyle="None",
    )
    return {
        "active_voltage_line": active_voltage_line,
        "active_g_line": active_g_line,
        "active_s_line": active_s_line,
        "residual_line": residual_line,
        "time_lines": time_lines,
        "residual_v_line": residual_v_line,
        "voltage_point": voltage_point,
        "g_point": g_point,
        "s_point": s_point,
        "residual_point": residual_point,
        "stable_intersection_dots": stable_intersection_dots,
        "unstable_intersection_dots": unstable_intersection_dots,
        "title": fig.suptitle(""),
    }


def create_current_like_interactive_figure():
    fig, axes = plt.subplot_mosaic(
        [
            ["model_ax", "model_ax", "model_ax"],
            ["voltage_ax", "g_nmda_ax", "s_ax"],
            ["nullcline_ax", "nullcline_ax", "nullcline_ax"],
        ],
        figsize=(14, 9),
        height_ratios=[0.8, 1.7, 2.2],
    )
    axes["fig"] = fig
    return axes


def configure_interactive_csv_axes(axes, trace, metadata, residual_curves):
    t_ms = trace["t_ms"]
    time_mask = (t_ms >= metadata["t_min_ms"]) & (t_ms <= metadata["t_max_ms"])
    for name in ["voltage_ax", "g_nmda_ax", "s_ax"]:
        axes[name].set_xlim(metadata["t_min_ms"], metadata["t_max_ms"])
    axes["nullcline_ax"].set_xlim(-90.0, 20.0)
    axes["voltage_ax"].set_ylim(
        float(np.min(trace["v_local_mV"][time_mask])) - 3.0,
        float(np.max(trace["v_local_mV"][time_mask])) + 3.0,
    )
    axes["g_nmda_ax"].set_ylim(0.0, float(np.max(trace["g_nmda_nS"][time_mask])) * 1.1)
    axes["s_ax"].set_ylim(
        0.0,
        float(np.max([np.max(trace["s"][time_mask]), np.max(trace["individual_s"][:, time_mask])])) * 1.1,
    )
    axes["nullcline_ax"].set_ylim(derivative_ylim(residual_curves))
    add_fitted_nullcline_model_text(axes["model_ax"])
    axes["voltage_ax"].set_title(r"$V(t)$")
    axes["g_nmda_ax"].set_title(r"$g_{\mathrm{NMDA}}(t)$")
    axes["s_ax"].set_title(r"fitted $s(t)$")
    axes["nullcline_ax"].set_title(r"$dV/dt(V; s(t))$")
    axes["voltage_ax"].set_ylabel("V [mV]")
    axes["g_nmda_ax"].set_ylabel(r"$g_{\mathrm{NMDA}}$ [nS]")
    axes["s_ax"].set_ylabel("s")
    axes["nullcline_ax"].set_ylabel(r"$dV/dt$ [mV/ms]")
    for name, ax in axes.items():
        if name in {"fig", "model_ax"}:
            continue
        ax.set_xlabel("V [mV]" if "nullcline" in name else "t [ms]")
    axes["nullcline_ax"].axhline(0.0, color="black", linewidth=1.0)


def plot_interactive_static_axes(axes, trace):
    t_ms = trace["t_ms"]
    axes["voltage_ax"].plot(t_ms, trace["v_local_mV"], color="0.75", linewidth=1.0)
    axes["g_nmda_ax"].plot(t_ms, trace["g_nmda_nS"], color="0.75", linewidth=1.0)
    axes["s_ax"].plot(t_ms, trace["s"], color="0.75", linewidth=1.0)
    for spike_index, individual_s in enumerate(trace["individual_s"], start=1):
        axes["s_ax"].plot(t_ms, individual_s, linestyle=":", linewidth=1.0, alpha=0.7)


def update_interactive_csv_snapshot(
        index,
        metadata,
        trace,
        parameters,
        v_values_mV,
        residual_cache,
        artists,
):
    t_ms = trace["t_ms"]
    current_t_ms = float(t_ms[index])
    current_v_mV = float(trace["v_local_mV"][index])
    if index not in residual_cache:
        residual_cache[index] = compute_local_nullcline_dvdt_mV_per_ms(
            parameters,
            trace["s"][index],
            v_values_mV,
        )
    residual_curve = residual_cache[index]
    artists["active_voltage_line"].set_data(t_ms[: index + 1], trace["v_local_mV"][: index + 1])
    artists["active_g_line"].set_data(t_ms[: index + 1], trace["g_nmda_nS"][: index + 1])
    artists["active_s_line"].set_data(t_ms[: index + 1], trace["s"][: index + 1])
    artists["residual_line"].set_data(v_values_mV, residual_curve)
    for line in artists["time_lines"]:
        line.set_xdata([current_t_ms, current_t_ms])
    artists["residual_v_line"].set_xdata([current_v_mV, current_v_mV])
    artists["voltage_point"].set_data([current_t_ms], [current_v_mV])
    artists["g_point"].set_data([current_t_ms], [trace["g_nmda_nS"][index]])
    artists["s_point"].set_data([current_t_ms], [trace["s"][index]])
    artists["residual_point"].set_data(
        [current_v_mV],
        [np.interp(current_v_mV, v_values_mV, residual_curve)],
    )
    intersections_mV = estimate_zero_crossings(v_values_mV, residual_curve)
    stable_intersections_mV, unstable_intersections_mV = classify_intersection_stability(intersections_mV)
    artists["stable_intersection_dots"].set_data(
        stable_intersections_mV,
        np.zeros_like(stable_intersections_mV),
    )
    artists["unstable_intersection_dots"].set_data(
        unstable_intersections_mV,
        np.zeros_like(unstable_intersections_mV),
    )
    artists["title"].set_text(
        f"{metadata['label']} | t={current_t_ms:.3f} ms, "
        f"V={current_v_mV:.3f} mV, "
        f"g_NMDA={trace['g_nmda_nS'][index]:.4g} nS, "
        f"s={trace['s'][index]:.4g}"
    )


class SimulateFittedApic20ParameterFitScriptTestCases(unittest.TestCase):
    """Manual runnables for the one-time fitted APIC20 parameter CSV."""

    def test_fit_and_save_fitted_apic20_parameters(self):
        csv_path, _reference, parameters, trace = fit_and_save_fitted_apic20_parameter_csv()

        print(csv_path)
        self.assertTrue(csv_path.exists())
        self.assertGreater(csv_path.stat().st_size, 0)
        self.assertTrue(np.isfinite(parameters.capacitance_nF))
        self.assertTrue(np.isfinite(parameters.g_leak_nS))
        self.assertTrue(np.isfinite(trace["v_local_mV"]).all())

    def test_plot_fitted_apic20_parameter_replay_from_saved_csv(self):
        output_png_path = plot_fitted_apic20_parameter_replay_from_csv(show_plot=True)

        print(output_png_path)
        self.assertTrue(output_png_path.exists())
        self.assertGreater(output_png_path.stat().st_size, 0)


def _assert_standard_three_spike_outputs(test_case, gif_path, snapshots, metrics_path):
    test_case.assertTrue(gif_path.exists())
    test_case.assertGreater(gif_path.stat().st_size, 0)
    test_case.assertTrue(metrics_path.exists())
    test_case.assertGreater(metrics_path.stat().st_size, 0)
    required_snapshots = {
        "t67ms",
        "t89ms",
        "t97ms",
        "t98ms",
        "t140ms",
        "max_reference_empirical_dvdt",
        "vmax_after_spike_1",
        "vmax_after_spike_2",
        "vmax_after_spike_3",
    }
    test_case.assertTrue(required_snapshots.issubset(snapshots))
    for snapshot in snapshots.values():
        plot_path = Path(snapshot["plot_path"])
        test_case.assertTrue(plot_path.exists())
        test_case.assertGreater(plot_path.stat().st_size, 0)
        test_case.assertTrue(np.isfinite(snapshot["snapshot_v_mV"]))


def _assert_current_three_spike_outputs(test_case, gif_path, snapshots, metrics_path):
    test_case.assertTrue(gif_path.exists())
    test_case.assertGreater(gif_path.stat().st_size, 0)
    test_case.assertTrue(metrics_path.exists())
    test_case.assertGreater(metrics_path.stat().st_size, 0)
    required_snapshots = {
        "t67ms",
        "t89ms",
        "t97ms",
        "t98ms",
        "t140ms",
        "max_reference_empirical_dvdt",
        "vmax_after_spike_1",
        "vmax_after_spike_2",
        "vmax_after_spike_3",
    }
    test_case.assertTrue(required_snapshots.issubset(snapshots))
    for snapshot in snapshots.values():
        plot_path = Path(snapshot["plot_path"])
        test_case.assertTrue(plot_path.exists())
        test_case.assertGreater(plot_path.stat().st_size, 0)
        test_case.assertTrue(np.isfinite(snapshot["snapshot_v_mV"]))
        test_case.assertTrue(np.isfinite(snapshot["snapshot_current_nmda_pA_at_current_v"]))


def _assert_two_spike_outputs(test_case, gif_path, snapshots, metrics_path):
    test_case.assertTrue(gif_path.exists())
    test_case.assertGreater(gif_path.stat().st_size, 0)
    test_case.assertTrue(metrics_path.exists())
    test_case.assertGreater(metrics_path.stat().st_size, 0)
    required_snapshots = {
        "spike_1_t50ms",
        "t67ms",
        "spike_2_t70ms",
        "t89ms",
        "t97ms",
        "t98ms",
        "t140ms",
        "vmax_after_spike_1",
        "vmax_after_spike_2",
    }
    test_case.assertTrue(required_snapshots.issubset(snapshots))
    for snapshot in snapshots.values():
        plot_path = Path(snapshot["plot_path"])
        test_case.assertTrue(plot_path.exists())
        test_case.assertGreater(plot_path.stat().st_size, 0)
        test_case.assertTrue(np.isfinite(snapshot["snapshot_v_mV"]))


def _assert_protocol_csv_bundle(test_case, protocol_name):
    bundle_dir = FITTED_NULLCLINE_INTERACTIVE_CSV_DIR / protocol_name
    trace_path = bundle_dir / "trace.csv"
    metadata_path = bundle_dir / "metadata.json"
    test_case.assertTrue(trace_path.exists(), str(trace_path))
    test_case.assertGreater(trace_path.stat().st_size, 0, str(trace_path))
    test_case.assertTrue(metadata_path.exists(), str(metadata_path))
    test_case.assertGreater(metadata_path.stat().st_size, 0, str(metadata_path))
    metadata = json.loads(metadata_path.read_text())
    for key in [
        "protocol_name",
        "label",
        "t_min_ms",
        "t_max_ms",
        "spike_times_ms",
        "with_s_saturation",
        "parameters",
        "trace_csv",
    ]:
        test_case.assertIn(key, metadata)
    test_case.assertEqual(protocol_name, metadata["protocol_name"])
    test_case.assertEqual(str(trace_path), metadata["trace_csv"])


def _run_standard_three_spike_protocol(test_case, with_s_saturation):
    gif_path, snapshots, metrics_path = generate_fitted_apic20_standard_protocol_outputs(
        show_plot=True,
        with_s_saturation=with_s_saturation,
    )
    print(gif_path)
    _assert_standard_three_spike_outputs(test_case, gif_path, snapshots, metrics_path)
    _assert_protocol_csv_bundle(
        test_case,
        (
            "standard_three_spike_with_s_saturation"
            if with_s_saturation
            else "standard_three_spike"
        ),
    )
    show_plots_non_blocking()


def _run_current_three_spike_protocol(test_case, with_s_saturation):
    from src.iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20CurrentNullclineScripts import (
        generate_fitted_apic20_current_balance_protocol_outputs,
    )

    gif_path, snapshots, metrics_path = generate_fitted_apic20_current_balance_protocol_outputs(
        show_plot=True,
        with_s_saturation=with_s_saturation,
    )
    print(gif_path)
    _assert_current_three_spike_outputs(test_case, gif_path, snapshots, metrics_path)
    _assert_protocol_csv_bundle(
        test_case,
        (
            "current_balance_three_spike_with_s_saturation"
            if with_s_saturation
            else "current_balance_three_spike"
        ),
    )
    show_plots_non_blocking()


def _run_csv_slider_protocol(test_case, protocol_name):
    trace_path, metadata_path = ensure_fitted_apic20_protocol_csv_bundle(
        protocol_name=protocol_name,
    )
    backend = ensure_interactive_matplotlib_backend()
    print(f"Launching slider with Matplotlib backend: {backend}")
    fig, _axes, _slider = launch_fitted_apic20_csv_snapshot_slider(
        protocol_name=protocol_name,
        show_plot=True,
    )
    test_case.assertTrue(trace_path.exists())
    test_case.assertGreater(trace_path.stat().st_size, 0)
    test_case.assertTrue(metadata_path.exists())
    test_case.assertGreater(metadata_path.stat().st_size, 0)
    test_case.assertIsNotNone(fig)


def _run_current_two_spike_protocol(test_case, with_s_saturation):
    from src.iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20CurrentNullclineScripts import (
        generate_fitted_apic20_current_balance_two_spike_protocol_outputs,
    )

    gif_path, snapshots, metrics_path = (
        generate_fitted_apic20_current_balance_two_spike_protocol_outputs(
            show_plot=True,
            with_s_saturation=with_s_saturation,
        )
    )
    print(gif_path)
    _assert_two_spike_outputs(test_case, gif_path, snapshots, metrics_path)
    for snapshot in snapshots.values():
        test_case.assertTrue(np.isfinite(snapshot["snapshot_current_nmda_pA_at_current_v"]))
    _assert_protocol_csv_bundle(
        test_case,
        (
            "current_balance_two_spike_with_s_saturation"
            if with_s_saturation
            else "current_balance_two_spike"
        ),
    )
    show_plots_non_blocking()


def _run_standard_two_spike_protocol(test_case, with_s_saturation):
    gif_path, snapshots, metrics_path = generate_fitted_apic20_two_spike_protocol_outputs(
        show_plot=True,
        with_s_saturation=with_s_saturation,
    )
    print(gif_path)
    _assert_two_spike_outputs(test_case, gif_path, snapshots, metrics_path)
    _assert_protocol_csv_bundle(
        test_case,
        (
            "standard_two_spike_with_s_saturation"
            if with_s_saturation
            else "standard_two_spike"
        ),
    )
    show_plots_non_blocking()


class SimulateFittedApic20WithSaturation(unittest.TestCase):
    """Manual runnables for fitted APIC20 protocols with saturating s dynamics."""

    def test_generate_fitted_apic20_moving_nullcline_animation_and_snapshots(self):
        _run_standard_three_spike_protocol(
            self,
            with_s_saturation=True,
        )

    def test_generate_fitted_apic20_current_balance_animation_and_snapshots(self):
        _run_current_three_spike_protocol(
            self,
            with_s_saturation=True,
        )

    def test_generate_fitted_apic20_current_balance_two_spike_protocol(self):
        _run_current_two_spike_protocol(
            self,
            with_s_saturation=True,
        )

    def test_generate_fitted_apic20_two_spike_nullcline_protocol(self):
        _run_standard_two_spike_protocol(
            self,
            with_s_saturation=True,
        )


class SimulateFittedApic20WithoutSaturation(unittest.TestCase):
    """Manual runnables for fitted APIC20 protocols without saturating s dynamics."""

    def test_generate_fitted_apic20_moving_nullcline_animation_and_snapshots(self):
        _run_standard_three_spike_protocol(
            self,
            with_s_saturation=False,
        )

    def test_generate_fitted_apic20_current_balance_animation_and_snapshots(self):
        _run_current_three_spike_protocol(
            self,
            with_s_saturation=False,
        )

    def test_generate_fitted_apic20_current_balance_two_spike_protocol(self):
        _run_current_two_spike_protocol(
            self,
            with_s_saturation=False,
        )

    def test_generate_fitted_apic20_two_spike_nullcline_protocol(self):
        _run_standard_two_spike_protocol(
            self,
            with_s_saturation=False,
        )


class SimulateManuallySearchApic20(unittest.TestCase):

    def test_scale_cm_gl_g_max_nmda_to_0_3_(self, scale=0.3):
        fitted_parameters = read_fitted_apic20_parameters_from_csv()
        scale = 0.3
        result = simulate(
            fitted_parameters.capacitance_nF * scale,
            fitted_parameters.g_leak_nS * scale,
            fitted_parameters.g_nmda_max_nS * scale,
            fitted_apic20_spike_times_ms(),
            "scale_cm_gl_g_max_nmda_to_0_3",
            with_saturation=False,
        )

        print(result["gif_path"])

        self.assertTrue(result["gif_path"].exists(), str(result["gif_path"]))
        self.assertTrue(result["metadata_path"].exists(), str(result["metadata_path"]))
        self.assertTrue(result["trace_path"].exists(), str(result["trace_path"]))

    def test_scale_cm_gl_g_max_nmda_to_0_1_(self, scale=0.3):
        fitted_parameters = read_fitted_apic20_parameters_from_csv()
        scale = 0.1
        result = simulate(
            fitted_parameters.capacitance_nF * scale,
            fitted_parameters.g_leak_nS * scale,
            fitted_parameters.g_nmda_max_nS * scale,
            fitted_apic20_spike_times_ms(),
            "scale_cm_gl_g_max_nmda_to_0_1",
            with_saturation=False,
        )

        print(result["gif_path"])

        self.assertTrue(result["gif_path"].exists(), str(result["gif_path"]))
        self.assertTrue(result["metadata_path"].exists(), str(result["metadata_path"]))
        self.assertTrue(result["trace_path"].exists(), str(result["trace_path"]))


class SimulateCompareApic20Runs(unittest.TestCase):

    def test_compare_manual_search_against_current_nullcline(self):
        result = compare_runs(
            [
                DEFAULT_OUTPUT_DIR / "fitted_apic20_manual_search" / "scale_cm_gl_g_max_nmda_to_0_3",
                DEFAULT_OUTPUT_DIR / "fitted_apic20_manual_search" / "scale_cm_gl_g_max_nmda_to_0_1",
                DEFAULT_OUTPUT_DIR / "fitted_apic20_nullcline"
            ],
            comparison_label="manual_search_vs_current_nullcline",
        )

        self.assertTrue(result["gif_path"].exists(), str(result["gif_path"]))
        self.assertTrue(result["metrics_path"].exists(), str(result["metrics_path"]))
        self.assertTrue(result["snapshots"])
        for snapshot in result["snapshots"].values():
            plot_path = Path(snapshot["plot_path"])
            self.assertTrue(plot_path.exists(), str(plot_path))


class SimulateFittedApic20GNMDAMaxSweepScriptTestCases(unittest.TestCase):
    """Manual runnable for non-saturated 3-spike g_NMDA,max sweeps."""

    def test_generate_fitted_apic20_three_spike_g_nmda_max_sweep(self):
        fitted_parameters = read_fitted_apic20_parameters_from_csv()
        g_nmda_max_values_nS = (
                fitted_parameters.g_nmda_max_nS
                * np.linspace(0.2, 2.0, 10)
        )
        g_nmda_max_values_nS = np.linspace(12, 13, 21)
        results = generate_fitted_apic20_three_spike_g_nmda_max_sweep_outputs(
            g_nmda_max_values_nS=g_nmda_max_values_nS,
            show_plot=False,
        )

        _assert_three_spike_g_nmda_max_sweep_outputs(
            self,
            results,
            g_nmda_max_values_nS,
            {
                "t67ms",
                "t89ms",
                "t97ms",
                "t98ms",
                "t140ms",
                "max_reference_empirical_dvdt",
                "vmax_after_spike_1",
                "vmax_after_spike_2",
                "vmax_after_spike_3",
            },
        )

    def test_fitted_apic20_three_spike_g_nmda_max_sweep_protocol_names_are_unique(self):
        g_nmda_max_values_nS = np.asarray([0.5, 1.0, 1.25, 10.0], dtype=float)
        protocol_names = [
            fitted_apic20_three_spike_g_nmda_max_sweep_protocol_name(g_nmda_max_nS)
            for g_nmda_max_nS in g_nmda_max_values_nS
        ]

        self.assertEqual(len(g_nmda_max_values_nS), len(set(protocol_names)))
        self.assertEqual(
            "fitted_apic20/three_protocol_sweep/g_max_1p25",
            fitted_apic20_three_spike_g_nmda_max_sweep_protocol_name(1.25),
        )


def _assert_three_spike_g_nmda_max_sweep_outputs(
        test_case,
        results,
        g_nmda_max_values_nS,
        expected_snapshot_names,
):
    expected_labels = {
        g_nmda_max_label(g_nmda_max_nS)
        for g_nmda_max_nS in g_nmda_max_values_nS
    }
    test_case.assertEqual(expected_labels, set(results))
    generated_paths = {}
    for label, (gif_path, snapshots, metrics_path) in results.items():
        test_case.assertTrue(gif_path.exists(), str(gif_path))
        test_case.assertGreater(gif_path.stat().st_size, 0, str(gif_path))
        test_case.assertTrue(metrics_path.exists(), str(metrics_path))
        test_case.assertGreater(metrics_path.stat().st_size, 0, str(metrics_path))
        test_case.assertTrue(expected_snapshot_names.issubset(snapshots))
        metrics = json.loads(metrics_path.read_text())
        protocol_name = metrics["protocol_name"]
        _assert_protocol_csv_bundle(test_case, protocol_name)
        test_case.assertEqual(
            fitted_apic20_three_spike_g_nmda_max_sweep_protocol_name(
                metrics["g_nmda_max_nS"]
            ),
            protocol_name,
        )
        generated_paths[label] = {
            "gif_path": str(gif_path),
            "metrics_path": str(metrics_path),
            "trace_path": metrics["trace_path"],
            "metadata_path": metrics["metadata_path"],
            "snapshot_paths": [
                str(snapshot["plot_path"]) for snapshot in snapshots.values()
            ],
        }
        for snapshot in snapshots.values():
            plot_path = Path(snapshot["plot_path"])
            test_case.assertTrue(plot_path.exists(), str(plot_path))
            test_case.assertGreater(plot_path.stat().st_size, 0, str(plot_path))
            test_case.assertTrue(np.isfinite(snapshot["snapshot_v_mV"]))
            test_case.assertTrue(np.isfinite(snapshot["snapshot_g_nmda_nS"]))

    print(json.dumps(generated_paths, indent=2, sort_keys=True))


class SimulateFittedApic20SaturatedGNMDAMaxAmplificationScriptTestCases(unittest.TestCase):
    """Manual runnable for saturated 3-spike g_NMDA,max amplification sweeps."""

    def test_generate_saturated_fitted_apic20_g_nmda_max_amplification_sweep(self):
        amplifications = np.linspace(2, 3, 11)
        results = generate_saturated_fitted_apic20_g_nmda_amplification_sweep_outputs(
            amplifications=amplifications,
            show_plot=False,
        )

        _assert_g_nmda_amplification_sweep_outputs(
            self,
            results,
            amplifications,
            {
                "t67ms",
                "t89ms",
                "t97ms",
                "t98ms",
                "t140ms",
                "max_reference_empirical_dvdt",
                "vmax_after_spike_1",
                "vmax_after_spike_2",
                "vmax_after_spike_3",
            },
        )

    def test_generate_saturated_fitted_apic20_two_spike_g_nmda_max_amplification_sweep(self):
        amplifications = np.linspace(2, 3, 11)
        results = generate_saturated_fitted_apic20_g_nmda_amplification_sweep_outputs(
            amplifications=amplifications,
            show_plot=False,
            two_spike=True,
        )

        _assert_g_nmda_amplification_sweep_outputs(
            self,
            results,
            amplifications,
            {
                "spike_1_t50ms",
                "t67ms",
                "spike_2_t70ms",
                "t89ms",
                "t97ms",
                "t98ms",
                "t140ms",
                "vmax_after_spike_1",
                "vmax_after_spike_2",
            },
        )


def _assert_g_nmda_amplification_sweep_outputs(
        test_case,
        results,
        amplifications,
        expected_snapshot_names,
):
    test_case.assertEqual(
        {amplification_label(amplification) for amplification in amplifications},
        set(results),
    )
    generated_paths = {}
    for amp_label, (gif_path, snapshots, metrics_path) in results.items():
        test_case.assertTrue(gif_path.exists(), str(gif_path))
        test_case.assertGreater(gif_path.stat().st_size, 0, str(gif_path))
        test_case.assertTrue(metrics_path.exists(), str(metrics_path))
        test_case.assertGreater(metrics_path.stat().st_size, 0, str(metrics_path))
        test_case.assertTrue(expected_snapshot_names.issubset(snapshots))
        generated_paths[amp_label] = {
            "gif_path": str(gif_path),
            "metrics_path": str(metrics_path),
            "snapshot_paths": [
                str(snapshot["plot_path"]) for snapshot in snapshots.values()
            ],
        }
        for snapshot in snapshots.values():
            plot_path = Path(snapshot["plot_path"])
            test_case.assertTrue(plot_path.exists(), str(plot_path))
            test_case.assertGreater(plot_path.stat().st_size, 0, str(plot_path))
            test_case.assertTrue(np.isfinite(snapshot["snapshot_v_mV"]))
            test_case.assertTrue(np.isfinite(snapshot["snapshot_g_nmda_nS"]))

    print(json.dumps(generated_paths, indent=2, sort_keys=True))


FITTED_APIC20_PARALLEL_SCRIPT_METHODS = (
    (
        "standard_three_spike",
        "test_generate_fitted_apic20_moving_nullcline_animation_and_snapshots",
    ),
    (
        "current_three_spike",
        "test_generate_fitted_apic20_current_balance_animation_and_snapshots",
    ),
    (
        "standard_two_spike",
        "test_generate_fitted_apic20_two_spike_nullcline_protocol",
    ),
    (
        "current_two_spike",
        "test_generate_fitted_apic20_current_balance_two_spike_protocol",
    ),
)


def fitted_apic20_existing_plot_regeneration_specs():
    return (
        {
            "protocol_name": "standard_three_spike",
            "output_dir": FITTED_NULLCLINE_OUTPUT_DIR,
            "metrics_filename": "fitted_apic20_moving_nullcline_protocol.json",
            "gif_filename": "fitted_apic20_moving_nullcline.gif",
            "snapshot_prefix": "fitted_apic20_moving_nullcline",
        },
        {
            "protocol_name": "standard_two_spike",
            "output_dir": FITTED_NULLCLINE_SPIKE_SNAPSHOT_OUTPUT_DIR,
            "metrics_filename": "fitted_apic20_spike_time_snapshot_protocol.json",
            "gif_filename": "fitted_apic20_two_spike_moving_nullcline.gif",
            "snapshot_prefix": "fitted_apic20_two_spike_moving_nullcline",
        },
        {
            "protocol_name": "standard_three_spike_with_s_saturation",
            "output_dir": FITTED_NULLCLINE_SATURATING_OUTPUT_DIR,
            "metrics_filename": "fitted_apic20_moving_nullcline_protocol.json",
            "gif_filename": "fitted_apic20_moving_nullcline.gif",
            "snapshot_prefix": "fitted_apic20_moving_nullcline",
        },
        {
            "protocol_name": "standard_two_spike_with_s_saturation",
            "output_dir": FITTED_NULLCLINE_SATURATING_TWO_SPIKE_OUTPUT_DIR,
            "metrics_filename": "fitted_apic20_spike_time_snapshot_protocol.json",
            "gif_filename": "fitted_apic20_two_spike_moving_nullcline.gif",
            "snapshot_prefix": "fitted_apic20_two_spike_moving_nullcline",
        },
        {
            "protocol_name": "current_balance_three_spike",
            "output_dir": DEFAULT_OUTPUT_DIR / "fitted_apic20_current_nullcline",
            "metrics_filename": "fitted_apic20_current_balance_nullcline_protocol.json",
            "gif_filename": "fitted_apic20_current_balance_nullcline.gif",
            "snapshot_prefix": "fitted_apic20_current_balance_nullcline",
        },
        {
            "protocol_name": "current_balance_two_spike",
            "output_dir": DEFAULT_OUTPUT_DIR / "fitted_apic20_current_nullcline_spike_snapshots",
            "metrics_filename": "fitted_apic20_current_balance_two_spike_protocol.json",
            "gif_filename": "fitted_apic20_two_spike_current_balance_nullcline.gif",
            "snapshot_prefix": "fitted_apic20_two_spike_current_balance_nullcline",
        },
        {
            "protocol_name": "current_balance_three_spike_with_s_saturation",
            "output_dir": DEFAULT_OUTPUT_DIR / "fitted_apic20_current_nullcline_with_s_saturation",
            "metrics_filename": "fitted_apic20_current_balance_nullcline_protocol.json",
            "gif_filename": "fitted_apic20_current_balance_nullcline.gif",
            "snapshot_prefix": "fitted_apic20_current_balance_nullcline",
        },
        {
            "protocol_name": "current_balance_two_spike_with_s_saturation",
            "output_dir": DEFAULT_OUTPUT_DIR / "fitted_apic20_current_nullcline_with_s_saturation_two_spike",
            "metrics_filename": "fitted_apic20_current_balance_two_spike_protocol.json",
            "gif_filename": "fitted_apic20_two_spike_current_balance_nullcline.gif",
            "snapshot_prefix": "fitted_apic20_two_spike_current_balance_nullcline",
        },
    )


def existing_snapshot_times_for_regeneration(spec):
    metrics_path = Path(spec["output_dir"]) / spec["metrics_filename"]
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        if "snapshot_times_ms" in metrics:
            return metrics["snapshot_times_ms"]
    metadata, trace = read_fitted_apic20_protocol_csv_bundle(spec["protocol_name"])
    if len(metadata["spike_times_ms"]) == 2:
        snapshot_times = fitted_apic20_snapshot_times_up_to_ms(
            TWO_SPIKE_T_MAX_MS,
            spike_times_ms=metadata["spike_times_ms"],
        )
        snapshot_times.update(
            peak_voltage_snapshot_times_between_spikes_ms(trace, TWO_SPIKE_T_MAX_MS)
        )
        return snapshot_times
    snapshot_times = {
        "t67ms": 67.0,
        "t89ms": 89.0,
        "t97ms": 97.0,
        "t98ms": 98.0,
        "t140ms": 140.0,
    }
    snapshot_times.update(peak_voltage_snapshot_times_between_spikes_ms(trace, T_MAX_MS))
    return snapshot_times


def regenerate_fitted_apic20_plot_output_from_existing_csv(spec, show_plot=False):
    plt.switch_backend("Agg")
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, trace = read_fitted_apic20_protocol_csv_bundle(spec["protocol_name"])
    parameters = FittedApic20NullclineParameters(**metadata["parameters"])
    gif_path = generate_fitted_apic20_moving_nullcline_animation_from_trace(
        metadata=metadata,
        trace=trace,
        parameters=parameters,
        output_gif_path=output_dir / spec["gif_filename"],
        show_plot=show_plot,
    )
    snapshot_times = existing_snapshot_times_for_regeneration(spec)
    snapshots = {}
    for label, snapshot_time_ms in snapshot_times.items():
        snapshots[label] = generate_fitted_apic20_moving_nullcline_snapshot_from_trace(
            metadata=metadata,
            trace=trace,
            parameters=parameters,
            snapshot_time_ms=snapshot_time_ms,
            output_png_path=output_dir / f"{spec['snapshot_prefix']}_{label}.png",
            show_plot=show_plot,
        )
    metrics_path = output_dir / spec["metrics_filename"]
    metrics_path.write_text(
        json.dumps(
            {
                "protocol_name": spec["protocol_name"],
                "gif_path": str(gif_path),
                "snapshots": snapshots,
                "snapshot_times_ms": snapshot_times,
                "regenerated_from_csv": True,
                "trace_path": metadata.get("trace_csv"),
                "metadata_path": str(
                    FITTED_NULLCLINE_INTERACTIVE_CSV_DIR
                    / spec["protocol_name"]
                    / "metadata.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return {
        "protocol_name": spec["protocol_name"],
        "gif_path": str(gif_path),
        "metrics_path": str(metrics_path),
        "snapshot_count": len(snapshots),
    }


def regenerate_all_fitted_apic20_plot_outputs_from_existing_csv(show_plot=False):
    return [
        regenerate_fitted_apic20_plot_output_from_existing_csv(spec, show_plot=show_plot)
        for spec in fitted_apic20_existing_plot_regeneration_specs()
    ]


def _make_parallel_script_job(test_class, method_name):
    return lambda _: getattr(test_class(), method_name)()


def fitted_apic20_parallel_script_jobs():
    test_classes = (
        ("linear", SimulateFittedApic20WithoutSaturation),
        ("saturated", SimulateFittedApic20WithSaturation),
    )
    return [
        (
            f"{prefix}_{job_name}",
            _make_parallel_script_job(test_class, method_name),
        )
        for prefix, test_class in test_classes
        for job_name, method_name in FITTED_APIC20_PARALLEL_SCRIPT_METHODS
    ]


def run_fitted_apic20_script_job(job_name, job_function):
    plt.switch_backend("Agg")
    job_function(None)
    return {"job_name": job_name, "paths": [], "snapshot_count": None}


def regenerate_all_fitted_apic20_script_outputs_in_parallel(
        n_jobs=PARALLEL_SCRIPT_N_JOBS,
):
    jobs = fitted_apic20_existing_plot_regeneration_specs()
    results = Parallel(n_jobs=n_jobs)(
        delayed(regenerate_fitted_apic20_plot_output_from_existing_csv)(spec)
        for spec in jobs
    )
    return results


class SimulateFittedApic20ParallelScriptTestCases(unittest.TestCase):
    """Manual runnable that regenerates all non-GUI fitted APIC20 artifacts."""

    def test_regenerate_all_non_gui_fitted_apic20_scripts_in_parallel(self):
        results = regenerate_all_fitted_apic20_script_outputs_in_parallel()

        print(json.dumps(results, indent=2, sort_keys=True))


class SimulateFittedApic20NullclineGuiScriptTestCases(unittest.TestCase):
    """Manual GUI runnables for CSV-backed fitted APIC20 snapshot sliders."""

    def test_launch_fitted_apic20_csv_snapshot_slider(self):
        _run_csv_slider_protocol(self, INTERACTIVE_CSV_PROTOCOL_NAME)

    def test_launch_fitted_apic20_with_s_saturation_csv_snapshot_slider(self):
        _run_csv_slider_protocol(
            self,
            "current_balance_three_spike_with_s_saturation",
        )

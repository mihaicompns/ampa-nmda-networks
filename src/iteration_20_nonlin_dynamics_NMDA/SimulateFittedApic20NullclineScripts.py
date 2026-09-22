import csv
import json
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import RadioButtons, Slider
import numpy as np

from src.Plotting import show_plots_non_blocking
from src.iteration_20_nonlin_dynamics_NMDA.FitApic20NMDAStateScripts import (
    compute_nmda_sigma_from_animation,
    compute_voltage_error_summary,
    fit_capacitance_and_g_leak_for_voltage_replay,
    fit_nmda_s_state_to_apic20_reference,
    simulate_nmda_x_s_state,
    simulate_voltage_from_nmda_state,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"
FITTED_NULLCLINE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "fitted_apic20_nullcline"
FITTED_NULLCLINE_SPIKE_SNAPSHOT_OUTPUT_DIR = (
    DEFAULT_OUTPUT_DIR / "fitted_apic20_nullcline_spike_snapshots"
)
FITTED_NULLCLINE_SATURATING_OUTPUT_DIR = (
    DEFAULT_OUTPUT_DIR / "fitted_apic20_nullcline_with_s_saturation"
)
FITTED_NULLCLINE_SATURATING_TWO_SPIKE_OUTPUT_DIR = (
    DEFAULT_OUTPUT_DIR / "fitted_apic20_nullcline_with_s_saturation_two_spike"
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
PARALLEL_SCRIPT_N_JOBS = 8
FITTED_APIC20_INTERACTIVE_PROTOCOL_NAMES = (
    "standard_three_spike",
    "current_balance_three_spike",
    "standard_two_spike",
    "current_balance_two_spike",
    "standard_three_spike_with_s_saturation",
    "current_balance_three_spike_with_s_saturation",
    "standard_two_spike_with_s_saturation",
    "current_balance_two_spike_with_s_saturation",
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
    return FIRST_SPIKE_TIME_MS + SPIKE_DT_MS * np.arange(N_SPIKES)


def fitted_apic20_spike_times_up_to_ms(max_time_ms):
    return np.asarray(
        [
            spike_time_ms
            for spike_time_ms in fitted_apic20_spike_times_ms()
            if spike_time_ms <= float(max_time_ms)
        ],
        dtype=float,
    )


def fitted_apic20_spike_snapshot_times_up_to_ms(max_time_ms=89.0):
    return {
        f"spike_{spike_index}_t{spike_time_ms:g}ms": float(spike_time_ms)
        for spike_index, spike_time_ms in enumerate(
            fitted_apic20_spike_times_up_to_ms(max_time_ms),
            start=1,
        )
        if spike_time_ms <= float(max_time_ms)
    }


def fitted_apic20_snapshot_times_up_to_ms(max_time_ms=89.0):
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
    snapshot_times.update(fitted_apic20_spike_snapshot_times_up_to_ms(max_time_ms))
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


def read_fit_and_simulate_fitted_apic20_reference(spike_times_ms=None, with_s_saturation=False):
    from src.iteration_20_nonlin_dynamics_NMDA.single_compartment_equivalence import (
        read_standard_apic20_reference,
    )

    reference = read_standard_apic20_reference()
    parameters = read_fitted_apic20_parameters_from_csv()
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
    sigma = compute_nmda_sigma_from_animation(v_mV)
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
    sigma = compute_nmda_sigma_from_animation(v_values_mV)
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
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(14, 8),
    )
    return {
        "fig": fig,
        "voltage_ax": axes[0, 0],
        "g_nmda_ax": axes[0, 1],
        "s_ax": axes[0, 2],
        "nullcline_ax": axes[1, 0],
        "dvdt_ax": axes[1, 1],
        "d2vdt2_ax": axes[1, 2],
    }


def configure_fitted_nullcline_axes(axes, trace, parameters, t_min_ms, t_max_ms):
    t_ms = trace["t_ms"]
    time_mask = (t_ms >= t_min_ms) & (t_ms <= t_max_ms)
    axes["voltage_ax"].set_xlim(t_min_ms, t_max_ms)
    axes["g_nmda_ax"].set_xlim(t_min_ms, t_max_ms)
    axes["s_ax"].set_xlim(t_min_ms, t_max_ms)
    axes["dvdt_ax"].set_xlim(t_min_ms, t_max_ms)
    axes["d2vdt2_ax"].set_xlim(t_min_ms, t_max_ms)

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
    axes["dvdt_ax"].set_ylim(
        derivative_ylim(trace["empirical_dvdt_mV_per_ms"][time_mask])
    )
    axes["d2vdt2_ax"].set_ylim(
        derivative_ylim(trace["empirical_d2vdt2_mV_per_ms2"][time_mask])
    )

    axes["voltage_ax"].set_title(r"$V(t)$")
    axes["g_nmda_ax"].set_title(r"$g_{\mathrm{NMDA}}(t)$")
    axes["s_ax"].set_title(r"fitted $s(t)$")
    axes["nullcline_ax"].set_title("local fitted nullcline")
    axes["dvdt_ax"].set_title(r"simulated $dV/dt$")
    axes["d2vdt2_ax"].set_title(r"simulated curvature $d^2V/dt^2$")

    axes["voltage_ax"].set_ylabel("V [mV]")
    axes["g_nmda_ax"].set_ylabel(r"$g_{\mathrm{NMDA}}$ [nS]")
    axes["s_ax"].set_ylabel("s")
    axes["nullcline_ax"].set_ylabel(r"$dV/dt$ [mV/ms]")
    axes["dvdt_ax"].set_ylabel(r"$dV/dt$ [mV/ms]")
    axes["d2vdt2_ax"].set_ylabel(r"$d^2V/dt^2$ [mV/ms$^2$]")
    for ax in axes.values():
        if ax is axes["fig"]:
            continue
        ax.set_xlabel("t [ms]" if ax is not axes["nullcline_ax"] else "V [mV]")
    axes["nullcline_ax"].axhline(0.0, color="black", linewidth=1.0)
    axes["dvdt_ax"].axhline(0.0, color="black", linewidth=1.0)
    axes["d2vdt2_ax"].axhline(0.0, color="black", linewidth=1.0)


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
):
    reference, parameters, trace = read_fit_and_simulate_fitted_apic20_reference(
        spike_times_ms=spike_times_ms,
        with_s_saturation=with_s_saturation,
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
    axes["dvdt_ax"].plot(
        t_ms,
        trace["empirical_dvdt_mV_per_ms"],
        color="0.75",
        linewidth=1.0,
    )
    axes["d2vdt2_ax"].plot(
        t_ms,
        trace["empirical_d2vdt2_mV_per_ms2"],
        color="0.75",
        linewidth=1.0,
    )
    axes["s_ax"].legend(fontsize=7, loc="best")

    active_voltage_line, = axes["voltage_ax"].plot([], [], color="tab:blue", linewidth=2.0)
    active_g_line, = axes["g_nmda_ax"].plot([], [], color="tab:green", linewidth=2.0)
    active_s_line, = axes["s_ax"].plot([], [], color="tab:purple", linewidth=2.0)
    active_dvdt_line, = axes["dvdt_ax"].plot([], [], color="black", linewidth=2.0)
    active_d2vdt2_line, = axes["d2vdt2_ax"].plot([], [], color="tab:brown", linewidth=2.0)
    nullcline_line, = axes["nullcline_ax"].plot([], [], color="tab:orange", linewidth=2.0)
    voltage_point, = axes["voltage_ax"].plot([], [], "o", color="tab:blue")
    g_point, = axes["g_nmda_ax"].plot([], [], "o", color="tab:green")
    s_point, = axes["s_ax"].plot([], [], "o", color="tab:purple")
    dvdt_point, = axes["dvdt_ax"].plot([], [], "o", color="black")
    d2vdt2_point, = axes["d2vdt2_ax"].plot([], [], "o", color="tab:brown")
    nullcline_point, = axes["nullcline_ax"].plot([], [], "o", color="tab:orange")
    dvdt_time_line = axes["dvdt_ax"].axvline(0.0, color="0.55", linewidth=1.0)
    d2vdt2_time_line = axes["d2vdt2_ax"].axvline(0.0, color="0.55", linewidth=1.0)
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
        current_dvdt = float(trace["empirical_dvdt_mV_per_ms"][index])
        current_d2vdt2 = float(trace["empirical_d2vdt2_mV_per_ms2"][index])
        active_voltage_line.set_data(t_ms[: index + 1], trace["v_local_mV"][: index + 1])
        active_g_line.set_data(t_ms[: index + 1], trace["g_nmda_nS"][: index + 1])
        active_s_line.set_data(t_ms[: index + 1], trace["s"][: index + 1])
        active_dvdt_line.set_data(
            t_ms[: index + 1],
            trace["empirical_dvdt_mV_per_ms"][: index + 1],
        )
        active_d2vdt2_line.set_data(
            t_ms[: index + 1],
            trace["empirical_d2vdt2_mV_per_ms2"][: index + 1],
        )
        nullcline_line.set_data(v_values_mV, nullcline_dvdt[frame_number])
        nullcline_value = float(np.interp(current_v_mV, v_values_mV, nullcline_dvdt[frame_number]))
        voltage_point.set_data([current_t_ms], [current_v_mV])
        g_point.set_data([current_t_ms], [current_g_nmda_nS])
        s_point.set_data([current_t_ms], [current_s])
        dvdt_point.set_data([current_t_ms], [current_dvdt])
        d2vdt2_point.set_data([current_t_ms], [current_d2vdt2])
        nullcline_point.set_data([current_v_mV], [nullcline_value])
        dvdt_time_line.set_xdata([current_t_ms, current_t_ms])
        d2vdt2_time_line.set_xdata([current_t_ms, current_t_ms])
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
            f"dV/dt={current_dvdt:.3g} mV/ms, "
            f"d2V/dt2={current_d2vdt2:.3g} mV/ms^2, "
            f"nullcline dV/dt={nullcline_value:.3g} mV/ms"
        )
        return (
            active_voltage_line,
            active_g_line,
            active_s_line,
            active_dvdt_line,
            active_d2vdt2_line,
            nullcline_line,
            voltage_point,
            g_point,
            s_point,
            dvdt_point,
            d2vdt2_point,
            nullcline_point,
            dvdt_time_line,
            d2vdt2_time_line,
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
):
    reference, parameters, trace = read_fit_and_simulate_fitted_apic20_reference(
        spike_times_ms=spike_times_ms,
        with_s_saturation=with_s_saturation,
    )
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    t_ms = trace["t_ms"]
    index = int(np.searchsorted(t_ms, snapshot_time_ms, side="left"))
    index = int(np.clip(index, 1, len(t_ms) - 2))
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
    axes["nullcline_ax"].axvline(current_v_mV, color="tab:blue", linewidth=1.0)

    axes["dvdt_ax"].plot(t_ms, trace["empirical_dvdt_mV_per_ms"], color="0.75", linewidth=1.0)
    axes["dvdt_ax"].plot(
        t_ms[: index + 1],
        trace["empirical_dvdt_mV_per_ms"][: index + 1],
        color="black",
    )
    axes["dvdt_ax"].plot(current_t_ms, trace["empirical_dvdt_mV_per_ms"][index], "o", color="black")
    axes["dvdt_ax"].axvline(current_t_ms, color="0.55", linewidth=1.0)

    axes["d2vdt2_ax"].plot(
        t_ms,
        trace["empirical_d2vdt2_mV_per_ms2"],
        color="0.75",
        linewidth=1.0,
    )
    axes["d2vdt2_ax"].plot(
        t_ms[: index + 1],
        trace["empirical_d2vdt2_mV_per_ms2"][: index + 1],
        color="tab:brown",
    )
    axes["d2vdt2_ax"].plot(
        current_t_ms,
        trace["empirical_d2vdt2_mV_per_ms2"][index],
        "o",
        color="tab:brown",
    )
    axes["d2vdt2_ax"].axvline(current_t_ms, color="0.55", linewidth=1.0)


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
    spike_times_ms = fitted_apic20_spike_times_up_to_ms(max_time_ms=89.0)
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
        t_max_ms=89.0,
        show_plot=show_plot,
        spike_times_ms=spike_times_ms,
        with_s_saturation=with_s_saturation,
    )
    snapshot_times = fitted_apic20_snapshot_times_up_to_ms(max_time_ms=89.0)
    snapshot_times.update(peak_voltage_snapshot_times_between_spikes_ms(trace, 89.0))
    snapshots = {}
    for label, snapshot_time_ms in snapshot_times.items():
        snapshots[label] = generate_fitted_apic20_moving_nullcline_snapshot(
            snapshot_time_ms=snapshot_time_ms,
            output_png_path=output_dir / f"fitted_apic20_two_spike_moving_nullcline_{label}.png",
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
            "spike_times_ms": fitted_apic20_spike_times_up_to_ms(89.0),
            "t_min_ms": T_MIN_MS,
            "t_max_ms": 89.0,
            "with_s_saturation": False,
        },
        "current_balance_two_spike": {
            "label": "current-balance nullcline, 2 spikes",
            "spike_times_ms": fitted_apic20_spike_times_up_to_ms(89.0),
            "t_min_ms": T_MIN_MS,
            "t_max_ms": 89.0,
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
            "spike_times_ms": fitted_apic20_spike_times_up_to_ms(89.0),
            "t_min_ms": T_MIN_MS,
            "t_max_ms": 89.0,
            "with_s_saturation": True,
        },
        "current_balance_two_spike_with_s_saturation": {
            "label": "current-balance nullcline with s saturation, 2 spikes",
            "spike_times_ms": fitted_apic20_spike_times_up_to_ms(89.0),
            "t_min_ms": T_MIN_MS,
            "t_max_ms": 89.0,
            "with_s_saturation": True,
        },
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

    _reference, parameters, trace = read_fit_and_simulate_fitted_apic20_reference(
        spike_times_ms=spec["spike_times_ms"],
        with_s_saturation=spec["with_s_saturation"],
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
    trace_path = bundle_dir / "trace.csv"
    metadata = json.loads(metadata_path.read_text())
    data = np.genfromtxt(trace_path, delimiter=",", names=True)
    trace = {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}
    individual_names = sorted(
        name for name in trace if name.startswith("individual_s_")
    )
    trace["individual_s"] = np.asarray([trace[name] for name in individual_names])
    trace["spike_times_ms"] = np.asarray(metadata["spike_times_ms"], dtype=float)
    return metadata, trace


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
    )
    if metadata_is_stale:
        return save_fitted_apic20_protocol_csv_bundle(protocol_name, output_dir)
    return trace_path, metadata_path


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
    ensure_fitted_apic20_protocol_csv_bundle(protocol_name, output_dir)
    metadata, trace = read_fitted_apic20_protocol_csv_bundle(protocol_name, output_dir)
    parameters = FittedApic20NullclineParameters(**metadata["parameters"])
    return create_fitted_apic20_csv_snapshot_slider_figure(
        metadata=metadata,
        trace=trace,
        parameters=parameters,
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
    from src.iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20CurrentNullclineScripts import (
        compute_current_balance_nullcline_pA,
        estimate_current_balance_intersections_mV,
    )

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
    radio_ax = fig.add_axes([0.01, 0.16, 0.10, 0.68])
    radio = RadioButtons(
        radio_ax,
        [_interactive_protocol_menu_label(name) for name in protocols],
        active=protocols.index(metadata["protocol_name"]),
    )
    radio_ax.set_title("CSV protocol", fontsize=9)

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
            current_cache=state["current_cache"],
            compute_current_balance_nullcline_pA=compute_current_balance_nullcline_pA,
            estimate_current_balance_intersections_mV=estimate_current_balance_intersections_mV,
            artists=state["artists"],
        )
        fig.canvas.draw_idle()

    def load_protocol(protocol_name):
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
        sampled_current_curves = [
            compute_current_balance_nullcline_pA(
                next_parameters,
                next_trace["s"][index],
                v_values_mV,
            )
            for index in sample_indexes
        ]
        for name, ax in axes.items():
            if name != "fig":
                ax.cla()
        configure_interactive_csv_axes(
            axes,
            next_trace,
            next_metadata,
            sampled_residual_curves,
            sampled_current_curves,
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
                "current_cache": {},
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

    def select_protocol(label):
        load_protocol(protocols[radio.labels.index(label)])

    slider.on_changed(update_from_slider)
    radio.on_clicked(select_protocol)
    load_protocol(metadata["protocol_name"])
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.09, top=0.88, hspace=0.45)
    fig._fitted_apic20_csv_slider_widgets = {
        "slider": slider,
        "radio": radio,
    }
    if show_plot:
        plt.show(block=True)
    return fig, axes, slider


def _interactive_protocol_menu_label(protocol_name):
    return {
        "standard_three_spike": "standard\n3 spikes",
        "current_balance_three_spike": "current\n3 spikes",
        "standard_two_spike": "standard\n2 spikes",
        "current_balance_two_spike": "current\n2 spikes",
        "standard_three_spike_with_s_saturation": "standard sat.\n3 spikes",
        "current_balance_three_spike_with_s_saturation": "current sat.\n3 spikes",
        "standard_two_spike_with_s_saturation": "standard sat.\n2 spikes",
        "current_balance_two_spike_with_s_saturation": "current sat.\n2 spikes",
    }.get(protocol_name, protocol_name)


def create_interactive_csv_artists(axes, fig, initial_t_ms):
    active_voltage_line, = axes["voltage_ax"].plot([], [], color="tab:blue", linewidth=2.0)
    active_g_line, = axes["g_nmda_ax"].plot([], [], color="tab:green", linewidth=2.0)
    active_s_line, = axes["s_ax"].plot([], [], color="tab:purple", linewidth=2.0)
    active_dvdt_line, = axes["dvdt_ax"].plot([], [], color="black", linewidth=2.0)
    active_d2vdt2_line, = axes["d2vdt2_ax"].plot([], [], color="tab:brown", linewidth=2.0)
    residual_line, = axes["nullcline_ax"].plot([], [], color="tab:orange", linewidth=2.0)
    nmda_line, = axes["current_nullcline_ax"].plot([], [], color="tab:red", linewidth=2.0)
    leak_line, = axes["current_nullcline_ax"].plot([], [], color="tab:blue", linewidth=2.0)
    time_lines = [
        axes[name].axvline(initial_t_ms, color="0.45", linewidth=1.0)
        for name in ["voltage_ax", "g_nmda_ax", "s_ax", "dvdt_ax", "d2vdt2_ax"]
    ]
    residual_v_line = axes["nullcline_ax"].axvline(initial_t_ms, color="tab:blue", linewidth=1.0)
    current_v_line = axes["current_nullcline_ax"].axvline(initial_t_ms, color="0.35", linewidth=1.0)
    voltage_point, = axes["voltage_ax"].plot([], [], "o", color="tab:blue")
    g_point, = axes["g_nmda_ax"].plot([], [], "o", color="tab:green")
    s_point, = axes["s_ax"].plot([], [], "o", color="tab:purple")
    dvdt_point, = axes["dvdt_ax"].plot([], [], "o", color="black")
    d2vdt2_point, = axes["d2vdt2_ax"].plot([], [], "o", color="tab:brown")
    residual_point, = axes["nullcline_ax"].plot([], [], "o", color="tab:orange")
    current_nmda_point, = axes["current_nullcline_ax"].plot([], [], "o", color="tab:red")
    current_leak_point, = axes["current_nullcline_ax"].plot([], [], "o", color="tab:blue")
    intersection_dots, = axes["current_nullcline_ax"].plot(
        [], [], "o", color="black", markerfacecolor="black", markersize=5, linestyle="None"
    )
    intersection_crosshairs, = axes["current_nullcline_ax"].plot(
        [], [], "+", color="white", markersize=9, linestyle="None", markeredgewidth=1.2
    )
    return {
        "active_voltage_line": active_voltage_line,
        "active_g_line": active_g_line,
        "active_s_line": active_s_line,
        "active_dvdt_line": active_dvdt_line,
        "active_d2vdt2_line": active_d2vdt2_line,
        "residual_line": residual_line,
        "nmda_line": nmda_line,
        "leak_line": leak_line,
        "time_lines": time_lines,
        "residual_v_line": residual_v_line,
        "current_v_line": current_v_line,
        "voltage_point": voltage_point,
        "g_point": g_point,
        "s_point": s_point,
        "dvdt_point": dvdt_point,
        "d2vdt2_point": d2vdt2_point,
        "residual_point": residual_point,
        "current_nmda_point": current_nmda_point,
        "current_leak_point": current_leak_point,
        "intersection_dots": intersection_dots,
        "intersection_crosshairs": intersection_crosshairs,
        "title": fig.suptitle(""),
    }


def create_current_like_interactive_figure():
    fig, axes = plt.subplot_mosaic(
        [
            ["voltage_ax", "g_nmda_ax", "s_ax"],
            ["nullcline_ax", "dvdt_ax", "d2vdt2_ax"],
            ["current_nullcline_ax", "current_nullcline_ax", "current_nullcline_ax"],
        ],
        figsize=(14, 11),
    )
    axes["fig"] = fig
    return axes


def configure_interactive_csv_axes(axes, trace, metadata, residual_curves, current_curves):
    t_ms = trace["t_ms"]
    time_mask = (t_ms >= metadata["t_min_ms"]) & (t_ms <= metadata["t_max_ms"])
    for name in ["voltage_ax", "g_nmda_ax", "s_ax", "dvdt_ax", "d2vdt2_ax"]:
        axes[name].set_xlim(metadata["t_min_ms"], metadata["t_max_ms"])
    axes["nullcline_ax"].set_xlim(-90.0, 20.0)
    axes["current_nullcline_ax"].set_xlim(-90.0, 20.0)
    axes["voltage_ax"].set_ylim(
        float(np.min(trace["v_local_mV"][time_mask])) - 3.0,
        float(np.max(trace["v_local_mV"][time_mask])) + 3.0,
    )
    axes["g_nmda_ax"].set_ylim(0.0, float(np.max(trace["g_nmda_nS"][time_mask])) * 1.1)
    axes["s_ax"].set_ylim(
        0.0,
        float(np.max([np.max(trace["s"][time_mask]), np.max(trace["individual_s"][:, time_mask])])) * 1.1,
    )
    axes["dvdt_ax"].set_ylim(derivative_ylim(trace["empirical_dvdt_mV_per_ms"][time_mask]))
    axes["d2vdt2_ax"].set_ylim(derivative_ylim(trace["empirical_d2vdt2_mV_per_ms2"][time_mask]))
    axes["nullcline_ax"].set_ylim(derivative_ylim(residual_curves))
    current_values = np.concatenate(
        [currents["nmda_pA"] for currents in current_curves]
        + [currents["leak_balance_pA"] for currents in current_curves]
    )
    axes["current_nullcline_ax"].set_ylim(derivative_ylim(current_values))
    axes["voltage_ax"].set_title(r"$V(t)$")
    axes["g_nmda_ax"].set_title(r"$g_{\mathrm{NMDA}}(t)$")
    axes["s_ax"].set_title(r"fitted $s(t)$")
    axes["nullcline_ax"].set_title(r"$dV/dt(V; s(t))$")
    axes["dvdt_ax"].set_title(r"simulated $dV/dt$")
    axes["d2vdt2_ax"].set_title(r"simulated $d^2V/dt^2$")
    axes["current_nullcline_ax"].set_title(r"$I_{\mathrm{NMDA}}(V,t)=-I_L(V)$")
    axes["voltage_ax"].set_ylabel("V [mV]")
    axes["g_nmda_ax"].set_ylabel(r"$g_{\mathrm{NMDA}}$ [nS]")
    axes["s_ax"].set_ylabel("s")
    axes["nullcline_ax"].set_ylabel(r"$dV/dt$ [mV/ms]")
    axes["dvdt_ax"].set_ylabel(r"$dV/dt$ [mV/ms]")
    axes["d2vdt2_ax"].set_ylabel(r"$d^2V/dt^2$ [mV/ms$^2$]")
    axes["current_nullcline_ax"].set_ylabel("current [pA]")
    for name, ax in axes.items():
        if name == "fig":
            continue
        ax.set_xlabel("V [mV]" if "nullcline" in name else "t [ms]")
    axes["nullcline_ax"].axhline(0.0, color="black", linewidth=1.0)
    axes["dvdt_ax"].axhline(0.0, color="black", linewidth=1.0)
    axes["d2vdt2_ax"].axhline(0.0, color="black", linewidth=1.0)
    axes["current_nullcline_ax"].axhline(0.0, color="0.6", linewidth=1.0)


def plot_interactive_static_axes(axes, trace):
    t_ms = trace["t_ms"]
    axes["voltage_ax"].plot(t_ms, trace["v_local_mV"], color="0.75", linewidth=1.0)
    axes["g_nmda_ax"].plot(t_ms, trace["g_nmda_nS"], color="0.75", linewidth=1.0)
    axes["s_ax"].plot(t_ms, trace["s"], color="0.75", linewidth=1.0)
    for spike_index, individual_s in enumerate(trace["individual_s"], start=1):
        axes["s_ax"].plot(t_ms, individual_s, linestyle=":", linewidth=1.0, alpha=0.7)
    axes["dvdt_ax"].plot(t_ms, trace["empirical_dvdt_mV_per_ms"], color="0.75", linewidth=1.0)
    axes["d2vdt2_ax"].plot(
        t_ms,
        trace["empirical_d2vdt2_mV_per_ms2"],
        color="0.75",
        linewidth=1.0,
    )


def update_interactive_csv_snapshot(
    index,
    metadata,
    trace,
    parameters,
    v_values_mV,
    residual_cache,
    current_cache,
    compute_current_balance_nullcline_pA,
    estimate_current_balance_intersections_mV,
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
    if index not in current_cache:
        current_cache[index] = compute_current_balance_nullcline_pA(
            parameters,
            trace["s"][index],
            v_values_mV,
        )
    residual_curve = residual_cache[index]
    currents = current_cache[index]
    artists["active_voltage_line"].set_data(t_ms[: index + 1], trace["v_local_mV"][: index + 1])
    artists["active_g_line"].set_data(t_ms[: index + 1], trace["g_nmda_nS"][: index + 1])
    artists["active_s_line"].set_data(t_ms[: index + 1], trace["s"][: index + 1])
    artists["active_dvdt_line"].set_data(t_ms[: index + 1], trace["empirical_dvdt_mV_per_ms"][: index + 1])
    artists["active_d2vdt2_line"].set_data(
        t_ms[: index + 1],
        trace["empirical_d2vdt2_mV_per_ms2"][: index + 1],
    )
    artists["residual_line"].set_data(v_values_mV, residual_curve)
    artists["nmda_line"].set_data(v_values_mV, currents["nmda_pA"])
    artists["leak_line"].set_data(v_values_mV, currents["leak_balance_pA"])
    for line in artists["time_lines"]:
        line.set_xdata([current_t_ms, current_t_ms])
    artists["residual_v_line"].set_xdata([current_v_mV, current_v_mV])
    artists["current_v_line"].set_xdata([current_v_mV, current_v_mV])
    artists["voltage_point"].set_data([current_t_ms], [current_v_mV])
    artists["g_point"].set_data([current_t_ms], [trace["g_nmda_nS"][index]])
    artists["s_point"].set_data([current_t_ms], [trace["s"][index]])
    artists["dvdt_point"].set_data([current_t_ms], [trace["empirical_dvdt_mV_per_ms"][index]])
    artists["d2vdt2_point"].set_data([current_t_ms], [trace["empirical_d2vdt2_mV_per_ms2"][index]])
    artists["residual_point"].set_data(
        [current_v_mV],
        [np.interp(current_v_mV, v_values_mV, residual_curve)],
    )
    nmda_at_v = float(np.interp(current_v_mV, v_values_mV, currents["nmda_pA"]))
    leak_at_v = float(np.interp(current_v_mV, v_values_mV, currents["leak_balance_pA"]))
    artists["current_nmda_point"].set_data([current_v_mV], [nmda_at_v])
    artists["current_leak_point"].set_data([current_v_mV], [leak_at_v])
    intersections_mV = estimate_current_balance_intersections_mV(currents)
    intersections_pA = np.interp(intersections_mV, v_values_mV, currents["nmda_pA"])
    artists["intersection_dots"].set_data(intersections_mV, intersections_pA)
    artists["intersection_crosshairs"].set_data(intersections_mV, intersections_pA)
    artists["title"].set_text(
        f"{metadata['label']} | t={current_t_ms:.3f} ms, "
        f"V={current_v_mV:.3f} mV, s={trace['s'][index]:.4g}, "
        f"I_NMDA={nmda_at_v:.4g} pA, -I_L={leak_at_v:.4g} pA"
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
    test_case.assertEqual(
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
        set(snapshots),
    )
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
    test_case.assertEqual(
        {
            "spike_1_t50ms",
            "t67ms",
            "spike_2_t70ms",
            "t89ms",
            "vmax_after_spike_1",
            "vmax_after_spike_2",
        },
        set(snapshots),
    )
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
    parameter_csv_path, _reference, _parameters, _trace = fit_and_save_fitted_apic20_parameter_csv()
    jobs = fitted_apic20_parallel_script_jobs()
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_fitted_apic20_script_job)(job_name, job_function)
        for job_name, job_function in jobs
    )
    return [
        {
            "job_name": "parameter_fit_csv",
            "paths": [str(parameter_csv_path)],
            "snapshot_count": 0,
        },
        *results,
    ]


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

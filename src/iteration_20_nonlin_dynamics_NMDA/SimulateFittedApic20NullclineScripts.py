import json
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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


def fitted_apic20_spike_snapshot_times_up_to_ms(max_time_ms=89.0):
    return {
        f"spike_{spike_index}_t{spike_time_ms:g}ms": float(spike_time_ms)
        for spike_index, spike_time_ms in enumerate(fitted_apic20_spike_times_ms(), start=1)
        if spike_time_ms <= float(max_time_ms)
    }


def read_fit_and_simulate_fitted_apic20_reference():
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
    local_trace = simulate_fitted_apic20_trace(reference.t_ms, parameters)
    local_trace["reference_v_local_mV"] = reference.v_local_mV
    local_trace["passive_reference_fit_v_mV"] = fitted_v_mV
    local_trace["fit_capacitance_and_g_leak_v_mV"] = (
        fitted_v_with_fitted_capacitance_and_g_leak_mV
    )
    return reference, parameters, local_trace


def simulate_fitted_apic20_trace(t_ms, parameters):
    state = simulate_nmda_x_s_state(
        t_ms=t_ms,
        spike_times_ms=fitted_apic20_spike_times_ms(),
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
    individual_s = simulate_individual_s_kernels(t_ms, parameters)
    return {
        "condition": "fitted single compartment",
        "n_spikes": N_SPIKES,
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


def simulate_individual_s_kernels(t_ms, parameters):
    kernels = []
    for spike_time_ms in fitted_apic20_spike_times_ms():
        state = simulate_nmda_x_s_state(
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
):
    reference, parameters, trace = read_fit_and_simulate_fitted_apic20_reference()
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
):
    reference, parameters, trace = read_fit_and_simulate_fitted_apic20_reference()
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


def generate_fitted_apic20_standard_protocol_outputs(show_plot=True):
    reference, _parameters, _trace = read_fit_and_simulate_fitted_apic20_reference()
    output_dir = FITTED_NULLCLINE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = generate_fitted_apic20_moving_nullcline_animation(
        output_gif_path=output_dir / "fitted_apic20_moving_nullcline.gif",
        show_plot=show_plot,
    )
    snapshot_times = {
        "t67ms": 67.0,
        "t89ms": 89.0,
        "t97ms": 97.0,
        "t98ms": 98.0,
        "t140ms": 140.0,
        "max_reference_empirical_dvdt": max_reference_empirical_dvdt_time_ms(reference),
    }
    snapshots = {}
    for label, snapshot_time_ms in snapshot_times.items():
        snapshots[label] = generate_fitted_apic20_moving_nullcline_snapshot(
            snapshot_time_ms=snapshot_time_ms,
            output_png_path=output_dir / f"fitted_apic20_moving_nullcline_{label}.png",
            show_plot=show_plot,
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


def generate_fitted_apic20_spike_time_snapshot_protocol_outputs(show_plot=True):
    output_dir = FITTED_NULLCLINE_SPIKE_SNAPSHOT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_times = fitted_apic20_spike_snapshot_times_up_to_ms(max_time_ms=89.0)
    snapshots = {}
    for label, snapshot_time_ms in snapshot_times.items():
        snapshots[label] = generate_fitted_apic20_moving_nullcline_snapshot(
            snapshot_time_ms=snapshot_time_ms,
            output_png_path=output_dir / f"fitted_apic20_moving_nullcline_{label}.png",
            show_plot=show_plot,
        )
    metrics_path = output_dir / "fitted_apic20_spike_time_snapshot_protocol.json"
    metrics_path.write_text(
        json.dumps(
            {
                "snapshots": snapshots,
                "snapshot_times_ms": snapshot_times,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return snapshots, metrics_path


class SimulateFittedApic20NullclineScriptTestCases(unittest.TestCase):
    """Manual runnables for the fitted APIC20 single-compartment protocol."""

    def test_generate_fitted_apic20_moving_nullcline_animation_and_snapshots(self):
        gif_path, snapshots, metrics_path = (
            generate_fitted_apic20_standard_protocol_outputs(show_plot=True)
        )


        print(gif_path)
        self.assertTrue(gif_path.exists())
        self.assertGreater(gif_path.stat().st_size, 0)
        self.assertTrue(metrics_path.exists())
        self.assertGreater(metrics_path.stat().st_size, 0)
        self.assertEqual(
            {
                "t67ms",
                "t89ms",
                "t97ms",
                "t98ms",
                "t140ms",
                "max_reference_empirical_dvdt",
            },
            set(snapshots),
        )
        for snapshot in snapshots.values():
            plot_path = Path(snapshot["plot_path"])
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)
            self.assertTrue(np.isfinite(snapshot["snapshot_v_mV"]))
            self.assertTrue(np.isfinite(snapshot["snapshot_g_nmda_nS"]))

        show_plots_non_blocking()

    def test_generate_fitted_apic20_current_balance_animation_and_snapshots(self):
        from src.iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20CurrentNullclineScripts import (
            generate_fitted_apic20_current_balance_protocol_outputs,
        )

        gif_path, snapshots, metrics_path = (
            generate_fitted_apic20_current_balance_protocol_outputs(show_plot=True)
        )

        print(gif_path)
        self.assertTrue(gif_path.exists())
        self.assertGreater(gif_path.stat().st_size, 0)
        self.assertTrue(metrics_path.exists())
        self.assertGreater(metrics_path.stat().st_size, 0)
        required_snapshots = {
            "t67ms",
            "t89ms",
            "t97ms",
            "t98ms",
            "t140ms",
            "max_reference_empirical_dvdt",
        }
        self.assertTrue(required_snapshots.issubset(snapshots))
        for snapshot in snapshots.values():
            plot_path = Path(snapshot["plot_path"])
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)
            self.assertTrue(np.isfinite(snapshot["snapshot_v_mV"]))
            self.assertTrue(np.isfinite(snapshot["snapshot_current_nmda_pA_at_current_v"]))

        show_plots_non_blocking()

    def test_generate_fitted_apic20_current_balance_spike_time_snapshots(self):
        from src.iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20CurrentNullclineScripts import (
            generate_fitted_apic20_current_balance_spike_time_snapshot_protocol_outputs,
        )

        snapshots, metrics_path = (
            generate_fitted_apic20_current_balance_spike_time_snapshot_protocol_outputs(
                show_plot=True
            )
        )

        self.assertTrue(metrics_path.exists())
        self.assertGreater(metrics_path.stat().st_size, 0)
        self.assertEqual({"spike_1_t50ms", "spike_2_t70ms"}, set(snapshots))
        for snapshot in snapshots.values():
            plot_path = Path(snapshot["plot_path"])
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)
            self.assertTrue(np.isfinite(snapshot["snapshot_v_mV"]))
            self.assertTrue(np.isfinite(snapshot["snapshot_current_nmda_pA_at_current_v"]))

        show_plots_non_blocking()

    def test_generate_fitted_apic20_spike_time_nullcline_snapshots(self):
        snapshots, metrics_path = (
            generate_fitted_apic20_spike_time_snapshot_protocol_outputs(show_plot=True)
        )

        self.assertTrue(metrics_path.exists())
        self.assertGreater(metrics_path.stat().st_size, 0)
        self.assertEqual({"spike_1_t50ms", "spike_2_t70ms"}, set(snapshots))
        for snapshot in snapshots.values():
            plot_path = Path(snapshot["plot_path"])
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)
            self.assertTrue(np.isfinite(snapshot["snapshot_v_mV"]))
            self.assertTrue(np.isfinite(snapshot["snapshot_g_nmda_nS"]))

        show_plots_non_blocking()

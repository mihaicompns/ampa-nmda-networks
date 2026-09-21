import json
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from src.Plotting import show_plots_non_blocking
from src.iteration_20_nonlin_dynamics_NMDA.single_compartment_equivalence import (
    read_standard_apic20_reference,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"


@dataclass(frozen=True)
class NMDASStateFitResult:
    spike_weight: float
    tau_rise_ms: float
    tau_decay_ms: float
    alpha_per_ms: float
    fitted_g_nmda_max_nS: float
    reference_g_nmda_max_nS: float
    capacitance_nF: float
    g_leak_nS: float
    e_leak_mV: float
    e_excitatory_mV: float
    s_mse: float
    s_rmse: float
    g_nmda_mse_nS2: float
    g_nmda_rmse_nS: float
    voltage_mse_mV2: float
    voltage_rmse_mV: float
    optimizer_success: bool
    optimizer_message: str


def compute_nmda_sigma_from_animation(v_mV):
    return 1.0 / (1.0 + 0.25 * np.exp(-0.08 * np.asarray(v_mV, dtype=float)))


def reconstruct_normalized_s_from_g_nmda(reference):
    reference_g_nmda_max_nS = float(reference.metadata["gmaxnmda"])
    sigma = compute_nmda_sigma_from_animation(reference.v_local_mV)
    return reference.g_nmda_nS / (reference_g_nmda_max_nS * sigma)


def simulate_nmda_x_s_state(
    t_ms,
    spike_times_ms,
    spike_weight=1.0,
    tau_rise_ms=2.0,
    tau_decay_ms=100.0,
    alpha_per_ms=0.5,
):
    t_ms = np.asarray(t_ms, dtype=float)
    dt_ms = _uniform_dt_ms(t_ms)
    spike_steps = _spike_steps_for_grid(t_ms, spike_times_ms)

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
            + float(alpha_per_ms) * x_current
        )
        x_current += dx
        s_current += ds

    return {"x": x, "s": s}


def fit_nmda_s_state_to_apic20_reference(
    reference=None,
    spike_weight=1.0,
    capacitance_nF=0.0007312381418499121,
    g_leak_nS=0.024374604728330404,
    e_leak_mV=-70.0,
    e_excitatory_mV=0.0,
    fit_start_ms=None,
    initial_tau_rise_ms=2.0,
    initial_tau_decay_ms=100.0,
    initial_alpha_per_ms=0.5,
):
    if reference is None:
        reference = read_standard_apic20_reference()
    if fit_start_ms is None:
        fit_start_ms = float(reference.spike_times_ms[0])

    target_s = reconstruct_normalized_s_from_g_nmda(reference)
    fit_mask = reference.t_ms >= float(fit_start_ms)

    def unpack(log_values):
        tau_rise_ms, tau_decay_gap_ms, alpha_per_ms = np.exp(log_values)
        return tau_rise_ms, tau_rise_ms + tau_decay_gap_ms, alpha_per_ms

    def residuals(log_values):
        tau_rise_ms, tau_decay_ms, alpha_per_ms = unpack(log_values)
        simulation = simulate_nmda_x_s_state(
            t_ms=reference.t_ms,
            spike_times_ms=reference.spike_times_ms,
            spike_weight=spike_weight,
            tau_rise_ms=tau_rise_ms,
            tau_decay_ms=tau_decay_ms,
            alpha_per_ms=alpha_per_ms,
        )
        return simulation["s"][fit_mask] - target_s[fit_mask]

    initial_tau_decay_gap_ms = max(
        initial_tau_decay_ms - initial_tau_rise_ms,
        1e-6,
    )
    optimizer_result = least_squares(
        residuals,
        x0=np.log(
            [
                initial_tau_rise_ms,
                initial_tau_decay_gap_ms,
                initial_alpha_per_ms,
            ]
        ),
        bounds=(
            np.log([0.1, 1e-6, 1e-6]),
            np.log([100.0, 1000.0, 100.0]),
        ),
    )
    tau_rise_ms, tau_decay_ms, alpha_per_ms = unpack(optimizer_result.x)
    simulation = simulate_nmda_x_s_state(
        t_ms=reference.t_ms,
        spike_times_ms=reference.spike_times_ms,
        spike_weight=spike_weight,
        tau_rise_ms=tau_rise_ms,
        tau_decay_ms=tau_decay_ms,
        alpha_per_ms=alpha_per_ms,
    )
    sigma = compute_nmda_sigma_from_animation(reference.v_local_mV)
    fitted_g_nmda_max_nS = _least_squares_scale(
        basis=sigma[fit_mask] * simulation["s"][fit_mask],
        target=reference.g_nmda_nS[fit_mask],
    )
    fitted_g_nmda_nS = fitted_g_nmda_max_nS * sigma * simulation["s"]
    fitted_v_mV = simulate_voltage_from_nmda_state(
        t_ms=reference.t_ms,
        s=simulation["s"],
        initial_v_mV=float(reference.v_local_mV[0]),
        capacitance_nF=capacitance_nF,
        g_leak_nS=g_leak_nS,
        e_leak_mV=e_leak_mV,
        e_excitatory_mV=e_excitatory_mV,
        g_nmda_max_nS=fitted_g_nmda_max_nS,
    )
    s_error = simulation["s"][fit_mask] - target_s[fit_mask]
    g_error = fitted_g_nmda_nS[fit_mask] - reference.g_nmda_nS[fit_mask]
    voltage_error = fitted_v_mV[fit_mask] - reference.v_local_mV[fit_mask]

    fit_result = NMDASStateFitResult(
        spike_weight=float(spike_weight),
        tau_rise_ms=float(tau_rise_ms),
        tau_decay_ms=float(tau_decay_ms),
        alpha_per_ms=float(alpha_per_ms),
        fitted_g_nmda_max_nS=float(fitted_g_nmda_max_nS),
        reference_g_nmda_max_nS=float(reference.metadata["gmaxnmda"]),
        capacitance_nF=float(capacitance_nF),
        g_leak_nS=float(g_leak_nS),
        e_leak_mV=float(e_leak_mV),
        e_excitatory_mV=float(e_excitatory_mV),
        s_mse=float(np.mean(s_error**2)),
        s_rmse=float(np.sqrt(np.mean(s_error**2))),
        g_nmda_mse_nS2=float(np.mean(g_error**2)),
        g_nmda_rmse_nS=float(np.sqrt(np.mean(g_error**2))),
        voltage_mse_mV2=float(np.mean(voltage_error**2)),
        voltage_rmse_mV=float(np.sqrt(np.mean(voltage_error**2))),
        optimizer_success=bool(optimizer_result.success),
        optimizer_message=str(optimizer_result.message),
    )
    return reference, target_s, simulation, fitted_g_nmda_nS, fitted_v_mV, fit_result


def simulate_voltage_from_nmda_state(
    t_ms,
    s,
    initial_v_mV,
    capacitance_nF,
    g_leak_nS,
    e_leak_mV,
    e_excitatory_mV,
    g_nmda_max_nS,
):
    t_ms = np.asarray(t_ms, dtype=float)
    s = np.asarray(s, dtype=float)
    dt_ms = _uniform_dt_ms(t_ms)
    v_mV = np.zeros_like(t_ms, dtype=float)
    v_current_mV = float(initial_v_mV)

    for step in range(len(t_ms)):
        v_mV[step] = v_current_mV
        if step == len(t_ms) - 1:
            break

        sigma = float(compute_nmda_sigma_from_animation(v_current_mV))
        leak_pA = float(g_leak_nS) * (v_current_mV - float(e_leak_mV))
        nmda_pA = (
            float(g_nmda_max_nS)
            * sigma
            * float(s[step])
            * (v_current_mV - float(e_excitatory_mV))
        )
        dvdt_mV_per_ms = -1e-3 * (leak_pA + nmda_pA) / float(capacitance_nF)
        v_current_mV += dt_ms * dvdt_mV_per_ms

    return v_mV


def fit_g_leak_for_voltage_replay(
    reference,
    s,
    g_nmda_max_nS,
    capacitance_nF,
    e_leak_mV,
    e_excitatory_mV,
    initial_g_leak_nS,
    fit_start_ms=50.0,
):
    fit_mask = reference.t_ms >= float(fit_start_ms)

    def residuals(log_g_leak):
        g_leak_nS = float(np.exp(log_g_leak[0]))
        fitted_v_mV = simulate_voltage_from_nmda_state(
            t_ms=reference.t_ms,
            s=s,
            initial_v_mV=float(reference.v_local_mV[0]),
            capacitance_nF=capacitance_nF,
            g_leak_nS=g_leak_nS,
            e_leak_mV=e_leak_mV,
            e_excitatory_mV=e_excitatory_mV,
            g_nmda_max_nS=g_nmda_max_nS,
        )
        if not np.all(np.isfinite(fitted_v_mV)):
            return np.full(np.count_nonzero(fit_mask), 1e9)
        return fitted_v_mV[fit_mask] - reference.v_local_mV[fit_mask]

    optimizer_result = least_squares(
        residuals,
        x0=np.log([float(initial_g_leak_nS)]),
        bounds=(np.log([1e-6]), np.log([50.0])),
    )
    fitted_g_leak_nS = float(np.exp(optimizer_result.x[0]))
    fitted_v_mV = simulate_voltage_from_nmda_state(
        t_ms=reference.t_ms,
        s=s,
        initial_v_mV=float(reference.v_local_mV[0]),
        capacitance_nF=capacitance_nF,
        g_leak_nS=fitted_g_leak_nS,
        e_leak_mV=e_leak_mV,
        e_excitatory_mV=e_excitatory_mV,
        g_nmda_max_nS=g_nmda_max_nS,
    )
    voltage_rmse_mV = compute_voltage_rmse_mV(
        reference=reference,
        fitted_v_mV=fitted_v_mV,
        fit_start_ms=fit_start_ms,
    )
    return fitted_g_leak_nS, fitted_v_mV, voltage_rmse_mV, optimizer_result


def fit_capacitance_and_g_leak_for_voltage_replay(
    reference,
    s,
    g_nmda_max_nS,
    e_leak_mV,
    e_excitatory_mV,
    initial_capacitance_nF,
    initial_g_leak_nS,
    fit_start_ms=50.0,
):
    fit_mask = reference.t_ms >= float(fit_start_ms)

    def residuals(log_values):
        capacitance_nF, g_leak_nS = np.exp(log_values)
        fitted_v_mV = simulate_voltage_from_nmda_state(
            t_ms=reference.t_ms,
            s=s,
            initial_v_mV=float(reference.v_local_mV[0]),
            capacitance_nF=float(capacitance_nF),
            g_leak_nS=float(g_leak_nS),
            e_leak_mV=e_leak_mV,
            e_excitatory_mV=e_excitatory_mV,
            g_nmda_max_nS=g_nmda_max_nS,
        )
        if not np.all(np.isfinite(fitted_v_mV)):
            return np.full(np.count_nonzero(fit_mask), 1e9)
        return fitted_v_mV[fit_mask] - reference.v_local_mV[fit_mask]

    optimizer_result = least_squares(
        residuals,
        x0=np.log([float(initial_capacitance_nF), float(initial_g_leak_nS)]),
        bounds=(
            np.log([1e-6, 1e-6]),
            np.log([1.0, 50.0]),
        ),
    )
    fitted_capacitance_nF, fitted_g_leak_nS = np.exp(optimizer_result.x)
    fitted_v_mV = simulate_voltage_from_nmda_state(
        t_ms=reference.t_ms,
        s=s,
        initial_v_mV=float(reference.v_local_mV[0]),
        capacitance_nF=float(fitted_capacitance_nF),
        g_leak_nS=float(fitted_g_leak_nS),
        e_leak_mV=e_leak_mV,
        e_excitatory_mV=e_excitatory_mV,
        g_nmda_max_nS=g_nmda_max_nS,
    )
    voltage_rmse_mV = compute_voltage_rmse_mV(
        reference=reference,
        fitted_v_mV=fitted_v_mV,
        fit_start_ms=fit_start_ms,
    )
    return (
        float(fitted_capacitance_nF),
        float(fitted_g_leak_nS),
        fitted_v_mV,
        voltage_rmse_mV,
        optimizer_result,
    )


def read_recorded_axial_current_out_total_nA(reference):
    rows = np.genfromtxt(
        reference.trace_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    if "axial_current_out_total_nA" not in rows.dtype.names:
        raise ValueError(
            "Reference CSV must include axial_current_out_total_nA: "
            f"{reference.trace_path}"
        )
    return np.asarray(rows["axial_current_out_total_nA"], dtype=float)


def simulate_voltage_from_nmda_state_subtracting_axial_current(
    t_ms,
    s,
    recorded_axial_current_out_total_nA,
    initial_v_mV,
    capacitance_nF,
    g_leak_nS,
    e_leak_mV,
    e_excitatory_mV,
    g_nmda_max_nS,
):
    t_ms = np.asarray(t_ms, dtype=float)
    s = np.asarray(s, dtype=float)
    recorded_axial_current_out_total_nA = np.asarray(
        recorded_axial_current_out_total_nA,
        dtype=float,
    )
    dt_ms = _uniform_dt_ms(t_ms)
    v_mV = np.zeros_like(t_ms, dtype=float)
    v_current_mV = float(initial_v_mV)

    for step in range(len(t_ms)):
        v_mV[step] = v_current_mV
        if step == len(t_ms) - 1:
            break

        sigma = float(compute_nmda_sigma_from_animation(v_current_mV))
        leak_pA = float(g_leak_nS) * (v_current_mV - float(e_leak_mV))
        nmda_pA = (
            float(g_nmda_max_nS)
            * sigma
            * float(s[step])
            * (v_current_mV - float(e_excitatory_mV))
        )
        local_dvdt_mV_per_ms = (
            -1e-3 * (leak_pA + nmda_pA) / float(capacitance_nF)
        )
        axial_dvdt_mV_per_ms = (
            -recorded_axial_current_out_total_nA[step] / float(capacitance_nF)
        )
        v_current_mV += dt_ms * (
            local_dvdt_mV_per_ms + axial_dvdt_mV_per_ms
        )

    return v_mV


def compute_voltage_rmse_mV(reference, fitted_v_mV, fit_start_ms=50.0):
    fit_mask = reference.t_ms >= float(fit_start_ms)
    error = np.asarray(fitted_v_mV, dtype=float)[fit_mask] - reference.v_local_mV[fit_mask]
    return float(np.sqrt(np.mean(error**2)))


def compute_voltage_error_summary(reference, fitted_v_mV, fit_start_ms=50.0):
    fit_mask = reference.t_ms >= float(fit_start_ms)
    error = (
        np.asarray(fitted_v_mV, dtype=float)[fit_mask]
        - reference.v_local_mV[fit_mask]
    )
    return {
        "fit_start_ms": float(fit_start_ms),
        "mse_mV2": float(np.mean(error**2)),
        "rmse_mV": float(np.sqrt(np.mean(error**2))),
        "mae_mV": float(np.mean(np.abs(error))),
        "max_abs_error_mV": float(np.max(np.abs(error))),
    }


def plot_nmda_s_state_fit(
    reference,
    target_s,
    simulation,
    fitted_g_nmda_nS,
    fitted_v_mV,
    fit_result,
):
    alpha = 0.6
    fig, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(9, 12),
        sharex=True,
    )
    axes[0].plot(
        reference.t_ms,
        simulation["x"],
        color="#9467bd",
        linewidth=2,
        alpha=alpha,
        label="fitted x(t)",
    )
    axes[1].plot(
        reference.t_ms,
        target_s,
        color="black",
        linewidth=2,
        alpha=alpha,
        label="CSV reconstructed s(t)",
    )
    axes[1].plot(
        reference.t_ms,
        simulation["s"],
        color="#1f77b4",
        linewidth=2,
        alpha=alpha,
        label="fitted s(t)",
    )
    axes[2].plot(
        reference.t_ms,
        reference.g_nmda_nS,
        color="black",
        linewidth=2,
        alpha=alpha,
        label="CSV g_NMDA(t)",
    )
    axes[2].plot(
        reference.t_ms,
        fitted_g_nmda_nS,
        color="#2ca02c",
        linewidth=3,
        alpha=alpha,
        linestyle=":",
        label="gmax * sigma(V) * fitted s(t)",
    )
    axes[3].axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    axes[3].plot(
        reference.t_ms,
        fitted_g_nmda_nS - reference.g_nmda_nS,
        color="#d62728",
        linewidth=1.5,
        label="fit - CSV",
    )
    axes[4].plot(
        reference.t_ms,
        reference.v_local_mV,
        color="black",
        linewidth=2,
        alpha=alpha,
        label="CSV V(t)",
    )
    axes[4].plot(
        reference.t_ms,
        fitted_v_mV,
        color="#ff7f0e",
        linewidth=2,
        alpha=alpha,
        label="passive + fitted NMDA V(t)",
    )
    for ax in axes:
        for spike_time_ms in reference.spike_times_ms:
            ax.axvline(spike_time_ms, color="0.75", linestyle=":", linewidth=1.0)
        ax.legend(loc="best")

    axes[0].set_ylabel("x")
    axes[1].set_ylabel("s")
    axes[2].set_ylabel("g_NMDA [nS]")
    axes[3].set_ylabel("residual [nS]")
    axes[4].set_ylabel("V [mV]")
    axes[4].set_xlabel("t [ms]")
    fig.suptitle(
        "APIC20 NMDA state fit from generated CSV\n"
        f"w_x={fit_result.spike_weight:.4g}, "
        f"tau_r={fit_result.tau_rise_ms:.4g} ms, "
        f"tau_d={fit_result.tau_decay_ms:.4g} ms, "
        f"alpha={fit_result.alpha_per_ms:.4g} /ms, "
        f"gmax={fit_result.fitted_g_nmda_max_nS:.4g} nS, "
        f"V RMSE={fit_result.voltage_rmse_mV:.4g} mV"
    )
    fig.tight_layout()
    return fig, axes


def run_standard_apic20_nmda_s_state_fit(
    output_dir=DEFAULT_OUTPUT_DIR,
    show_plot=True,
):
    reference, target_s, simulation, fitted_g_nmda_nS, fitted_v_mV, fit_result = (
        fit_nmda_s_state_to_apic20_reference()
    )
    fig, axes = plot_nmda_s_state_fit(
        reference=reference,
        target_s=target_s,
        simulation=simulation,
        fitted_g_nmda_nS=fitted_g_nmda_nS,
        fitted_v_mV=fitted_v_mV,
        fit_result=fit_result,
    )

    output_dir = Path(output_dir)
    output_path = output_dir / "standard_apic20_nmda_s_state_fit.png"
    show_plots_non_blocking(
        show=show_plot,
        save_name="standard_apic20_nmda_s_state_fit",
        out_dir=output_dir,
    )
    metrics_path = output_dir / "standard_apic20_nmda_s_state_fit.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "fit": asdict(fit_result),
                "source_trace": str(reference.trace_path),
                "source_metadata": str(reference.metadata_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (
        reference,
        target_s,
        simulation,
        fitted_g_nmda_nS,
        fitted_v_mV,
        fit_result,
        fig,
        axes,
        output_path,
        metrics_path,
    )


def run_standard_apic20_nmda_s_state_fit_subtracting_recorded_axial_current(
    output_dir=DEFAULT_OUTPUT_DIR,
    show_plot=True,
):
    (
        reference,
        target_s,
        simulation,
        fitted_g_nmda_nS,
        fitted_v_mV,
        fit_result,
    ) = fit_nmda_s_state_to_apic20_reference()
    recorded_axial_current_out_total_nA = (
        read_recorded_axial_current_out_total_nA(reference)
    )
    fitted_v_subtracting_axial_mV = (
        simulate_voltage_from_nmda_state_subtracting_axial_current(
            t_ms=reference.t_ms,
            s=simulation["s"],
            recorded_axial_current_out_total_nA=recorded_axial_current_out_total_nA,
            initial_v_mV=float(reference.v_local_mV[0]),
            capacitance_nF=fit_result.capacitance_nF,
            g_leak_nS=fit_result.g_leak_nS,
            e_leak_mV=fit_result.e_leak_mV,
            e_excitatory_mV=fit_result.e_excitatory_mV,
            g_nmda_max_nS=fit_result.fitted_g_nmda_max_nS,
        )
    )
    voltage_rmse_mV = compute_voltage_rmse_mV(
        reference=reference,
        fitted_v_mV=fitted_v_mV,
    )
    voltage_subtracting_axial_rmse_mV = compute_voltage_rmse_mV(
        reference=reference,
        fitted_v_mV=fitted_v_subtracting_axial_mV,
    )
    fig, axes = plot_nmda_s_state_fit(
        reference=reference,
        target_s=target_s,
        simulation=simulation,
        fitted_g_nmda_nS=fitted_g_nmda_nS,
        fitted_v_mV=fitted_v_mV,
        fit_result=fit_result,
    )
    axes[4].plot(
        reference.t_ms,
        fitted_v_subtracting_axial_mV,
        color="#1f77b4",
        linewidth=2,
        alpha=0.7,
        label=(
            "passive + fitted NMDA - recorded axial "
            f"RMSE={voltage_subtracting_axial_rmse_mV:.3g} mV"
        ),
    )
    axes[4].legend(loc="best")
    fig.suptitle(
        "APIC20 NMDA state fit with recorded axial current subtracted from V replay\n"
        f"without axial RMSE={voltage_rmse_mV:.4g} mV, "
        f"subtract axial RMSE={voltage_subtracting_axial_rmse_mV:.4g} mV"
    )

    output_dir = Path(output_dir)
    output_path = output_dir / "standard_apic20_nmda_s_state_fit_subtract_axial_voltage.png"
    show_plots_non_blocking(
        show=show_plot,
        save_name="standard_apic20_nmda_s_state_fit_subtract_axial_voltage",
        out_dir=output_dir,
    )
    metrics_path = output_dir / "standard_apic20_nmda_s_state_fit_subtract_axial_voltage.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "voltage_rmse_mV": voltage_rmse_mV,
                "voltage_subtracting_axial_rmse_mV": (
                    voltage_subtracting_axial_rmse_mV
                ),
                "source_trace": str(reference.trace_path),
                "fit_g_nmda_max_nS": fit_result.fitted_g_nmda_max_nS,
                "fit_tau_rise_ms": fit_result.tau_rise_ms,
                "fit_tau_decay_ms": fit_result.tau_decay_ms,
                "fit_alpha_per_ms": fit_result.alpha_per_ms,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (
        reference,
        fitted_v_mV,
        fitted_v_subtracting_axial_mV,
        recorded_axial_current_out_total_nA,
        voltage_rmse_mV,
        voltage_subtracting_axial_rmse_mV,
        fig,
        axes,
        output_path,
        metrics_path,
    )


def run_standard_apic20_nmda_s_state_fit_g_leak_only(
    output_dir=DEFAULT_OUTPUT_DIR,
    show_plot=True,
):
    (
        reference,
        target_s,
        simulation,
        fitted_g_nmda_nS,
        fitted_v_mV,
        fit_result,
    ) = fit_nmda_s_state_to_apic20_reference()
    (
        fitted_g_leak_nS,
        fitted_v_with_fitted_g_leak_mV,
        voltage_g_leak_rmse_mV,
        optimizer_result,
    ) = fit_g_leak_for_voltage_replay(
        reference=reference,
        s=simulation["s"],
        g_nmda_max_nS=fit_result.fitted_g_nmda_max_nS,
        capacitance_nF=fit_result.capacitance_nF,
        e_leak_mV=fit_result.e_leak_mV,
        e_excitatory_mV=fit_result.e_excitatory_mV,
        initial_g_leak_nS=fit_result.g_leak_nS,
    )
    voltage_rmse_mV = compute_voltage_rmse_mV(
        reference=reference,
        fitted_v_mV=fitted_v_mV,
    )
    fig, axes = plot_nmda_s_state_fit(
        reference=reference,
        target_s=target_s,
        simulation=simulation,
        fitted_g_nmda_nS=fitted_g_nmda_nS,
        fitted_v_mV=fitted_v_mV,
        fit_result=fit_result,
    )
    axes[4].plot(
        reference.t_ms,
        fitted_v_with_fitted_g_leak_mV,
        color="#1f77b4",
        linewidth=2,
        alpha=0.7,
        label=(
            f"fit only g_L={fitted_g_leak_nS:.4g} nS, "
            f"RMSE={voltage_g_leak_rmse_mV:.3g} mV"
        ),
    )
    axes[4].legend(loc="best")
    fig.suptitle(
        "APIC20 NMDA state fit with voltage replay fitting only g_L\n"
        f"C_m={fit_result.capacitance_nF:.6g} nF, "
        f"original g_L={fit_result.g_leak_nS:.6g} nS, "
        f"fitted g_L={fitted_g_leak_nS:.6g} nS, "
        f"fixed-g_L RMSE={voltage_rmse_mV:.4g} mV, "
        f"fit-g_L RMSE={voltage_g_leak_rmse_mV:.4g} mV"
    )

    output_dir = Path(output_dir)
    output_path = output_dir / "standard_apic20_nmda_s_state_fit_g_leak_only.png"
    show_plots_non_blocking(
        show=show_plot,
        save_name="standard_apic20_nmda_s_state_fit_g_leak_only",
        out_dir=output_dir,
    )
    metrics_path = output_dir / "standard_apic20_nmda_s_state_fit_g_leak_only.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "voltage_rmse_mV": voltage_rmse_mV,
                "voltage_fit_g_leak_rmse_mV": voltage_g_leak_rmse_mV,
                "capacitance_nF": fit_result.capacitance_nF,
                "original_g_leak_nS": fit_result.g_leak_nS,
                "fitted_g_leak_nS": fitted_g_leak_nS,
                "source_trace": str(reference.trace_path),
                "fit_g_nmda_max_nS": fit_result.fitted_g_nmda_max_nS,
                "fit_tau_rise_ms": fit_result.tau_rise_ms,
                "fit_tau_decay_ms": fit_result.tau_decay_ms,
                "fit_alpha_per_ms": fit_result.alpha_per_ms,
                "optimizer_success": bool(optimizer_result.success),
                "optimizer_message": str(optimizer_result.message),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (
        reference,
        fitted_v_mV,
        fitted_v_with_fitted_g_leak_mV,
        fitted_g_leak_nS,
        voltage_rmse_mV,
        voltage_g_leak_rmse_mV,
        fig,
        axes,
        output_path,
        metrics_path,
    )


def run_standard_apic20_nmda_s_state_fit_capacitance_and_g_leak(
    output_dir=DEFAULT_OUTPUT_DIR,
    show_plot=True,
):
    (
        reference,
        target_s,
        simulation,
        fitted_g_nmda_nS,
        fitted_v_mV,
        fit_result,
    ) = fit_nmda_s_state_to_apic20_reference()
    (
        fitted_capacitance_nF,
        fitted_g_leak_nS,
        fitted_v_with_fitted_capacitance_and_g_leak_mV,
        voltage_capacitance_and_g_leak_rmse_mV,
        optimizer_result,
    ) = fit_capacitance_and_g_leak_for_voltage_replay(
        reference=reference,
        s=simulation["s"],
        g_nmda_max_nS=fit_result.fitted_g_nmda_max_nS,
        e_leak_mV=fit_result.e_leak_mV,
        e_excitatory_mV=fit_result.e_excitatory_mV,
        initial_capacitance_nF=fit_result.capacitance_nF,
        initial_g_leak_nS=fit_result.g_leak_nS,
    )
    voltage_rmse_mV = compute_voltage_rmse_mV(
        reference=reference,
        fitted_v_mV=fitted_v_mV,
    )
    fig, axes = plot_nmda_s_state_fit(
        reference=reference,
        target_s=target_s,
        simulation=simulation,
        fitted_g_nmda_nS=fitted_g_nmda_nS,
        fitted_v_mV=fitted_v_mV,
        fit_result=fit_result,
    )
    axes[4].plot(
        reference.t_ms,
        fitted_v_with_fitted_capacitance_and_g_leak_mV,
        color="#1f77b4",
        linewidth=2,
        alpha=0.7,
        label=(
            f"fit C_m={fitted_capacitance_nF:.4g} nF, "
            f"g_L={fitted_g_leak_nS:.4g} nS, "
            f"RMSE={voltage_capacitance_and_g_leak_rmse_mV:.3g} mV"
        ),
    )
    axes[4].legend(loc="best")
    fig.suptitle(
        "APIC20 NMDA state fit with voltage replay fitting C_m and g_L\n"
        f"original C_m={fit_result.capacitance_nF:.6g} nF, "
        f"fitted C_m={fitted_capacitance_nF:.6g} nF, "
        f"original g_L={fit_result.g_leak_nS:.6g} nS, "
        f"fitted g_L={fitted_g_leak_nS:.6g} nS, "
        f"fixed RMSE={voltage_rmse_mV:.4g} mV, "
        f"fit RMSE={voltage_capacitance_and_g_leak_rmse_mV:.4g} mV"
    )

    output_dir = Path(output_dir)
    output_path = (
        output_dir
        / "standard_apic20_nmda_s_state_fit_capacitance_and_g_leak.png"
    )
    show_plots_non_blocking(
        show=show_plot,
        save_name="standard_apic20_nmda_s_state_fit_capacitance_and_g_leak",
        out_dir=output_dir,
    )
    metrics_path = (
        output_dir
        / "standard_apic20_nmda_s_state_fit_capacitance_and_g_leak.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "voltage_rmse_mV": voltage_rmse_mV,
                "voltage_fit_capacitance_and_g_leak_rmse_mV": (
                    voltage_capacitance_and_g_leak_rmse_mV
                ),
                "original_capacitance_nF": fit_result.capacitance_nF,
                "fitted_capacitance_nF": fitted_capacitance_nF,
                "original_g_leak_nS": fit_result.g_leak_nS,
                "fitted_g_leak_nS": fitted_g_leak_nS,
                "source_trace": str(reference.trace_path),
                "fit_g_nmda_max_nS": fit_result.fitted_g_nmda_max_nS,
                "fit_tau_rise_ms": fit_result.tau_rise_ms,
                "fit_tau_decay_ms": fit_result.tau_decay_ms,
                "fit_alpha_per_ms": fit_result.alpha_per_ms,
                "optimizer_success": bool(optimizer_result.success),
                "optimizer_message": str(optimizer_result.message),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (
        reference,
        fitted_v_mV,
        fitted_v_with_fitted_capacitance_and_g_leak_mV,
        fitted_capacitance_nF,
        fitted_g_leak_nS,
        voltage_rmse_mV,
        voltage_capacitance_and_g_leak_rmse_mV,
        fig,
        axes,
        output_path,
        metrics_path,
    )


def _least_squares_scale(basis, target):
    basis = np.asarray(basis, dtype=float)
    target = np.asarray(target, dtype=float)
    denominator = float(np.dot(basis, basis))
    if denominator <= 0.0:
        raise ValueError("Cannot fit scale for an all-zero basis")
    return float(np.dot(basis, target) / denominator)


def _uniform_dt_ms(t_ms):
    t_ms = np.asarray(t_ms, dtype=float)
    dts = np.diff(t_ms)
    if len(dts) == 0:
        raise ValueError("t_ms must contain at least two samples")
    dt_ms = float(dts[0])
    if not np.allclose(dts, dt_ms):
        raise ValueError("t_ms must be uniformly sampled")
    return dt_ms


def _spike_steps_for_grid(t_ms, spike_times_ms):
    dt_ms = _uniform_dt_ms(t_ms)
    first_t_ms = float(t_ms[0])
    spike_steps = {}
    for spike_time_ms in spike_times_ms:
        step = int(round((float(spike_time_ms) - first_t_ms) / dt_ms))
        if 0 <= step < len(t_ms):
            spike_steps[step] = spike_steps.get(step, 0) + 1
    return spike_steps


class FitApic20NMDAStateScriptTestCases(unittest.TestCase):
    """Manual PyCharm runnables for fitting APIC20 NMDA x/s state dynamics."""

    def test_fit_standard_apic20_nmda_s_state_from_generated_csv(self):
        (
            reference,
            target_s,
            simulation,
            fitted_g_nmda_nS,
            fitted_v_mV,
            fit_result,
            fig,
            axes,
            output_path,
            metrics_path,
        ) = run_standard_apic20_nmda_s_state_fit(show_plot=True)

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
        self.assertTrue(metrics_path.exists())
        self.assertGreater(metrics_path.stat().st_size, 0)
        self.assertEqual(3, reference.n_spikes)
        self.assertGreater(fit_result.tau_decay_ms, fit_result.tau_rise_ms)
        self.assertGreater(fit_result.alpha_per_ms, 0.0)
        self.assertGreater(fit_result.fitted_g_nmda_max_nS, 0.0)
        self.assertLess(fit_result.s_rmse, 0.2)
        self.assertLess(fit_result.g_nmda_rmse_nS, 2.0)
        self.assertEqual(reference.v_local_mV.shape, fitted_v_mV.shape)
        self.assertTrue(np.all(np.isfinite(fitted_v_mV)))

        show_plots_non_blocking()

    def test_fit_standard_apic20_nmda_s_state_subtract_recorded_axial_current(self):
        (
            reference,
            fitted_v_mV,
            fitted_v_subtracting_axial_mV,
            recorded_axial_current_out_total_nA,
            voltage_rmse_mV,
            voltage_subtracting_axial_rmse_mV,
            fig,
            axes,
            output_path,
            metrics_path,
        ) = run_standard_apic20_nmda_s_state_fit_subtracting_recorded_axial_current(
            show_plot=True
        )

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
        self.assertTrue(metrics_path.exists())
        self.assertGreater(metrics_path.stat().st_size, 0)
        self.assertEqual(reference.v_local_mV.shape, fitted_v_mV.shape)
        self.assertEqual(reference.v_local_mV.shape, fitted_v_subtracting_axial_mV.shape)
        self.assertEqual(
            reference.v_local_mV.shape,
            recorded_axial_current_out_total_nA.shape,
        )
        self.assertTrue(np.all(np.isfinite(fitted_v_subtracting_axial_mV)))
        self.assertLess(voltage_subtracting_axial_rmse_mV, voltage_rmse_mV)

        show_plots_non_blocking()

    def test_fit_standard_apic20_nmda_s_state_fit_g_leak_only(self):
        (
            reference,
            fitted_v_mV,
            fitted_v_with_fitted_g_leak_mV,
            fitted_g_leak_nS,
            voltage_rmse_mV,
            voltage_g_leak_rmse_mV,
            fig,
            axes,
            output_path,
            metrics_path,
        ) = run_standard_apic20_nmda_s_state_fit_g_leak_only(show_plot=True)

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
        self.assertTrue(metrics_path.exists())
        self.assertGreater(metrics_path.stat().st_size, 0)
        self.assertEqual(reference.v_local_mV.shape, fitted_v_mV.shape)
        self.assertEqual(
            reference.v_local_mV.shape,
            fitted_v_with_fitted_g_leak_mV.shape,
        )
        self.assertTrue(np.all(np.isfinite(fitted_v_with_fitted_g_leak_mV)))
        self.assertGreater(fitted_g_leak_nS, 0.0)
        self.assertLess(voltage_g_leak_rmse_mV, voltage_rmse_mV)

        show_plots_non_blocking()

    def test_fit_standard_apic20_nmda_s_state_fit_capacitance_and_g_leak(self):
        (
            reference,
            fitted_v_mV,
            fitted_v_with_fitted_capacitance_and_g_leak_mV,
            fitted_capacitance_nF,
            fitted_g_leak_nS,
            voltage_rmse_mV,
            voltage_capacitance_and_g_leak_rmse_mV,
            fig,
            axes,
            output_path,
            metrics_path,
        ) = run_standard_apic20_nmda_s_state_fit_capacitance_and_g_leak(
            show_plot=True
        )

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
        self.assertTrue(metrics_path.exists())
        self.assertGreater(metrics_path.stat().st_size, 0)
        self.assertEqual(reference.v_local_mV.shape, fitted_v_mV.shape)
        self.assertEqual(
            reference.v_local_mV.shape,
            fitted_v_with_fitted_capacitance_and_g_leak_mV.shape,
        )
        self.assertTrue(
            np.all(np.isfinite(fitted_v_with_fitted_capacitance_and_g_leak_mV))
        )
        self.assertGreater(fitted_capacitance_nF, 0.0)
        self.assertGreater(fitted_g_leak_nS, 0.0)
        self.assertLess(voltage_capacitance_and_g_leak_rmse_mV, voltage_rmse_mV)

        show_plots_non_blocking()

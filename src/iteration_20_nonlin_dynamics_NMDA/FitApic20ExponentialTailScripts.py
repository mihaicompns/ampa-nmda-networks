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
class Apic20ExponentialTailFitResult:
    t0_ms: float
    tau_rise_ms: float
    amplitude_mV: float
    tau_decay_ms: float
    mse_mV2: float
    rmse_mV: float
    optimizer_cost: float
    optimizer_success: bool
    optimizer_message: str


def double_exponential_tail(t_ms, t0_ms, amplitude_mV, tau_decay_ms, tau_rise_ms=2.0):
    t_ms = np.asarray(t_ms, dtype=float)
    elapsed_ms = t_ms - float(t0_ms)
    tail = np.zeros_like(elapsed_ms)
    after_t0 = elapsed_ms >= 0.0
    tail[after_t0] = float(amplitude_mV) * (
        np.exp(-elapsed_ms[after_t0] / float(tau_decay_ms))
        - np.exp(-elapsed_ms[after_t0] / float(tau_rise_ms))
    )
    return tail


def split_reference_trace_from_t0(reference, t0_ms=90.0):
    t0_ms = float(t0_ms)
    tail_mask = reference.t_ms >= t0_ms
    if not np.any(tail_mask):
        raise ValueError("t0_ms excludes the whole reference trace")

    v_at_t0_mV = float(
        np.interp(t0_ms, reference.t_ms, reference.v_local_mV)
    )
    spike_times_from_t0_ms = reference.spike_times_ms[
        reference.spike_times_ms >= t0_ms
    ]
    return {
        "t_ms": reference.t_ms[tail_mask],
        "v_local_mV": reference.v_local_mV[tail_mask],
        "delta_v_mV": reference.v_local_mV[tail_mask] - v_at_t0_mV,
        "v_at_t0_mV": v_at_t0_mV,
        "spike_times_ms": spike_times_from_t0_ms,
    }


def fit_double_exponential_to_apic20_tail(
    reference=None,
    t0_ms=90.0,
    initial_amplitude_mV=None,
    initial_tau_rise_ms=2.0,
    initial_tau_decay_ms=90.0,
):
    if reference is None:
        reference = read_standard_apic20_reference()

    tail = split_reference_trace_from_t0(reference=reference, t0_ms=t0_ms)
    target_delta_v_mV = tail["delta_v_mV"]
    if initial_amplitude_mV is None:
        initial_amplitude_mV = max(float(np.max(target_delta_v_mV)), 1e-3)

    def unpack(log_values):
        amplitude_mV, tau_rise_ms, tau_decay_gap_ms = np.exp(log_values)
        tau_decay_ms = tau_rise_ms + tau_decay_gap_ms
        return float(amplitude_mV), float(tau_rise_ms), float(tau_decay_ms)

    def residuals(log_values):
        amplitude_mV, tau_rise_ms, tau_decay_ms = unpack(log_values)
        fitted_delta_v_mV = double_exponential_tail(
            t_ms=tail["t_ms"],
            t0_ms=t0_ms,
            amplitude_mV=amplitude_mV,
            tau_decay_ms=tau_decay_ms,
            tau_rise_ms=tau_rise_ms,
        )
        return fitted_delta_v_mV - target_delta_v_mV

    initial_tau_decay_gap_ms = max(
        initial_tau_decay_ms - initial_tau_rise_ms,
        1e-6,
    )
    lower_bounds = np.log([1e-6, 0.1, 1e-6])
    upper_bounds = np.log([1e6, 1e3, 1e3])
    x0 = np.log(
        [
            initial_amplitude_mV,
            initial_tau_rise_ms,
            initial_tau_decay_gap_ms,
        ]
    )
    optimizer_result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
    )
    amplitude_mV, tau_rise_ms, tau_decay_ms = unpack(optimizer_result.x)
    residual = residuals(optimizer_result.x)
    mse_mV2 = float(np.mean(residual**2))
    fit_result = Apic20ExponentialTailFitResult(
        t0_ms=float(t0_ms),
        tau_rise_ms=tau_rise_ms,
        amplitude_mV=amplitude_mV,
        tau_decay_ms=tau_decay_ms,
        mse_mV2=mse_mV2,
        rmse_mV=float(np.sqrt(mse_mV2)),
        optimizer_cost=float(optimizer_result.cost),
        optimizer_success=bool(optimizer_result.success),
        optimizer_message=str(optimizer_result.message),
    )
    return reference, tail, fit_result


def plot_apic20_exponential_tail_fit(reference, tail, fit_result):
    fitted_delta_v_mV = double_exponential_tail(
        t_ms=tail["t_ms"],
        t0_ms=fit_result.t0_ms,
        amplitude_mV=fit_result.amplitude_mV,
        tau_decay_ms=fit_result.tau_decay_ms,
        tau_rise_ms=fit_result.tau_rise_ms,
    )
    residual_mV = fitted_delta_v_mV - tail["delta_v_mV"]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(9, 8),
        sharex=False,
    )
    axes[0].plot(
        reference.t_ms,
        reference.v_local_mV,
        color="0.25",
        linewidth=1.5,
        label="full APIC20 reference",
    )
    axes[0].axvline(
        fit_result.t0_ms,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="fit start",
    )
    axes[0].axvspan(
        fit_result.t0_ms,
        float(reference.t_ms[-1]),
        color="#d62728",
        alpha=0.08,
    )
    axes[0].set_ylabel(r"$V_{\mathrm{local}}$ [mV]")
    axes[0].legend(loc="best")

    axes[1].plot(
        tail["t_ms"],
        tail["delta_v_mV"],
        color="black",
        linewidth=2,
        label=(
            r"APIC20 tail "
            f"V(t) - V({fit_result.t0_ms:g} ms)"
        ),
    )
    axes[1].plot(
        tail["t_ms"],
        fitted_delta_v_mV,
        color="#1f77b4",
        linewidth=2,
        label=(
            r"$A(e^{-(t-t_0)/\tau_d} - e^{-(t-t_0)/\tau_r})$"
        ),
    )
    for spike_time_ms in tail["spike_times_ms"]:
        axes[1].axvline(
            spike_time_ms,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.5,
        )
    axes[1].set_ylabel(r"$\Delta V$ [mV]")
    axes[1].legend(loc="best")

    axes[2].axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    axes[2].plot(
        tail["t_ms"],
        residual_mV,
        color="#d62728",
        linewidth=1.5,
        label="fit - APIC20 tail",
    )
    axes[2].set_xlabel(r"$t$ [ms]")
    axes[2].set_ylabel("residual [mV]")
    axes[2].legend(loc="best")

    fig.suptitle(
        "APIC20 voltage tail fit from 90 ms\n"
        f"A={fit_result.amplitude_mV:.4g} mV, "
        f"tau_decay={fit_result.tau_decay_ms:.4g} ms, "
        f"tau_rise={fit_result.tau_rise_ms:.4g} ms, "
        f"MSE={fit_result.mse_mV2:.4g} mV^2"
    )
    fig.tight_layout()
    return fig, axes


def run_standard_apic20_exponential_tail_fit(
    output_dir=DEFAULT_OUTPUT_DIR,
    show_plot=True,
    t0_ms=90.0,
    initial_tau_rise_ms=2.0,
):
    reference, tail, fit_result = fit_double_exponential_to_apic20_tail(
        t0_ms=t0_ms,
        initial_tau_rise_ms=initial_tau_rise_ms,
    )
    fig, axes = plot_apic20_exponential_tail_fit(
        reference=reference,
        tail=tail,
        fit_result=fit_result,
    )

    output_dir = Path(output_dir)
    output_path = output_dir / "standard_apic20_tail_90ms_double_exponential_fit.png"
    show_plots_non_blocking(
        show=show_plot,
        save_name="standard_apic20_tail_90ms_double_exponential_fit",
        out_dir=output_dir,
    )
    metrics_path = output_dir / "standard_apic20_tail_90ms_double_exponential_fit.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "fit": asdict(fit_result),
                "tail": {
                    "v_at_t0_mV": tail["v_at_t0_mV"],
                    "spike_times_ms": tail["spike_times_ms"].tolist(),
                },
                "source_trace": str(reference.trace_path),
                "source_metadata": str(reference.metadata_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return reference, tail, fit_result, fig, axes, output_path, metrics_path


class FitApic20ExponentialTailScriptTestCases(unittest.TestCase):
    """Manual PyCharm runnables for fitting APIC20 voltage tails with exponentials."""

    def test_fit_standard_apic20_reference_trace_tail_from_90ms(self):
        reference, tail, fit_result, fig, axes, output_path, metrics_path = (
            run_standard_apic20_exponential_tail_fit(show_plot=True)
        )

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
        self.assertTrue(metrics_path.exists())
        self.assertGreater(metrics_path.stat().st_size, 0)
        self.assertEqual(3, reference.n_spikes)
        self.assertAlmostEqual(90.0, tail["spike_times_ms"][0])
        self.assertGreater(fit_result.amplitude_mV, 0.0)
        self.assertGreater(fit_result.tau_rise_ms, 0.0)
        self.assertGreater(fit_result.tau_decay_ms, fit_result.tau_rise_ms)
        self.assertTrue(np.isfinite(fit_result.mse_mV2))

        show_plots_non_blocking()

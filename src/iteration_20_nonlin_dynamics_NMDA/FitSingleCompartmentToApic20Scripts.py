import json
import unittest
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from src.Plotting import show_plots_non_blocking
from src.iteration_20_nonlin_dynamics_NMDA.single_compartment_equivalence import (
    SingleCompartmentFitParameters,
    fit_single_compartment_to_reference_with_fixed_passive_parameters,
    fit_single_compartment_to_reference,
    plot_reference_and_simulation_two_panels,
    plot_reference_and_fit_two_panels,
    plot_standard_apic20_reference_trace,
    read_standard_apic20_reference,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"


def run_standard_apic20_reference_plot(
    output_dir=DEFAULT_OUTPUT_DIR,
    show_plot=True,
):
    reference = read_standard_apic20_reference()
    fig, ax, reference = plot_standard_apic20_reference_trace(reference=reference)

    output_dir = Path(output_dir)
    output_path = output_dir / "standard_apic20_reference_trace.png"
    show_plots_non_blocking(
        show=show_plot,
        save_name="standard_apic20_reference_trace",
        out_dir=output_dir,
    )
    return reference, fig, ax, output_path


def run_standard_apic20_single_compartment_fit(
    output_dir=DEFAULT_OUTPUT_DIR,
    show_plot=True,
    max_nfev=80,
    save_intermediate_plots=True,
):
    reference = read_standard_apic20_reference()
    fit_result = fit_single_compartment_to_reference(
        reference=reference,
        initial=SingleCompartmentFitParameters(
            g_nmda_max_nS=float(reference.metadata["gmaxnmda"]),
            capacitance_nF=0.5,
            g_leak_nS=0.01,
            tau_rise_ms=2.0,
            tau_decay_ms=100.0,
            alpha_per_ms=0.5,
            presynaptic_spike_weight=1.0,
        ),
        max_nfev=max_nfev,
        fit_start_ms=45.0,
        collect_history=save_intermediate_plots,
    )
    fig, axes = plot_reference_and_fit_two_panels(reference, fit_result)

    output_dir = Path(output_dir)
    if save_intermediate_plots:
        save_fit_history_plots(
            reference=reference,
            fit_result=fit_result,
            output_dir=output_dir / "single_compartment_apic20_fit_iterations",
            show_plot=True,
        )

    output_path = output_dir / "single_compartment_apic20_fit_four_panel.png"
    show_plots_non_blocking(
        show=show_plot,
        save_name="single_compartment_apic20_fit_four_panel",
        out_dir=output_dir,
    )
    metrics_path = output_dir / "single_compartment_apic20_fit_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "fit_parameters": asdict(fit_result.fit_params),
                "metrics": fit_result.metrics,
                "history_length": len(fit_result.history),
                "source_trace": str(reference.trace_path),
                "source_metadata": str(reference.metadata_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return reference, fit_result, fig, axes, output_path, metrics_path


def run_standard_apic20_single_compartment_fit_with_fixed_passive_parameters(
    fixed_capacitance_nF,
    fixed_g_leak_nS,
    fixed_tau_rise_ms,
    fixed_tau_decay_ms,
    fixed_alpha_per_ms,
    output_dir=DEFAULT_OUTPUT_DIR,
    show_plot=True,
    max_nfev=80,
    save_intermediate_plots=True,
):
    reference = read_standard_apic20_reference()
    fit_result = fit_single_compartment_to_reference_with_fixed_passive_parameters(
        reference=reference,
        initial=SingleCompartmentFitParameters(
            g_nmda_max_nS=0.1,
            capacitance_nF=fixed_capacitance_nF,
            g_leak_nS=fixed_g_leak_nS,
            tau_rise_ms=fixed_tau_rise_ms,
            tau_decay_ms=fixed_tau_decay_ms,
            alpha_per_ms=fixed_alpha_per_ms,
            presynaptic_spike_weight=1.0,
        ),
        fixed_capacitance_nF=fixed_capacitance_nF,
        fixed_g_leak_nS=fixed_g_leak_nS,
        fixed_tau_rise_ms=fixed_tau_rise_ms,
        fixed_tau_decay_ms=fixed_tau_decay_ms,
        fixed_alpha_per_ms=fixed_alpha_per_ms,
        max_nfev=max_nfev,
        fit_start_ms=45.0,
        collect_history=save_intermediate_plots,
    )
    fig, axes = plot_reference_and_fit_two_panels(reference, fit_result)

    output_dir = Path(output_dir)
    if save_intermediate_plots:
        save_fit_history_plots(
            reference=reference,
            fit_result=fit_result,
            output_dir=(
                output_dir
                / "single_compartment_apic20_fixed_passive_fit_iterations"
            ),
            show_plot=True,
        )

    output_path = output_dir / "single_compartment_apic20_fixed_passive_fit_four_panel.png"
    show_plots_non_blocking(
        show=show_plot,
        save_name="single_compartment_apic20_fixed_passive_fit_four_panel",
        out_dir=output_dir,
    )
    metrics_path = output_dir / "single_compartment_apic20_fixed_passive_fit_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "fit_parameters": asdict(fit_result.fit_params),
                "metrics": fit_result.metrics,
                "history_length": len(fit_result.history),
                "source_trace": str(reference.trace_path),
                "source_metadata": str(reference.metadata_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return reference, fit_result, fig, axes, output_path, metrics_path


def save_fit_history_plots(reference, fit_result, output_dir, show_plot=False):
    output_dir = Path(output_dir)
    output_paths = []
    for history_frame in fit_result.history:
        fig, axes = plot_reference_and_simulation_two_panels(
            reference=reference,
            simulation=history_frame.simulation,
            simulation_label=(
                "single compartment "
                f"iteration {history_frame.iteration_index}, "
                f"RMSE={history_frame.metrics['rmse_mV']:.3g} mV"
            ),
            title=(
                "Intermediate single-compartment fit "
                f"iteration {history_frame.iteration_index}"
            ),
        )
        fig.set_size_inches(9, 12)
        fig.subplots_adjust(bottom=0.22, top=0.93, hspace=0.25)
        fig.text(
            0.01,
            0.02,
            _format_fit_history_text(history_frame),
            ha="left",
            va="bottom",
            fontsize=8,
            family="monospace",
        )
        save_name = f"single_compartment_fit_iteration_{history_frame.iteration_index:03d}"
        output_path = output_dir / f"{save_name}.png"
        show_plots_non_blocking(
            show=show_plot,
            save_name=save_name,
            out_dir=output_dir,
        )
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def _format_fit_history_text(history_frame):
    fit_params = history_frame.fit_params
    return "\n".join(
        [
            f"Iteration: {history_frame.iteration_index}",
            "Parameters:",
            f"  g_nmda_max_nS              = {fit_params.g_nmda_max_nS:.6g}",
            f"  capacitance_nF             = {fit_params.capacitance_nF:.6g}",
            f"  g_leak_nS                  = {fit_params.g_leak_nS:.6g}",
            f"  tau_rise_ms                = {fit_params.tau_rise_ms:.6g}",
            f"  tau_decay_ms               = {fit_params.tau_decay_ms:.6g}",
            f"  alpha_per_ms               = {fit_params.alpha_per_ms:.6g}",
            f"  presynaptic_spike_weight   = {fit_params.presynaptic_spike_weight:.6g}",
        ]
    )


class FitSingleCompartmentToApic20ScriptTestCases(unittest.TestCase):
    """Manual PyCharm runnables for fitting the single-compartment equivalent."""

    def test_plot_standard_apic20_reference_trace(self):
        reference, fig, ax, output_path = run_standard_apic20_reference_plot(
            show_plot=True
        )

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
        self.assertEqual(3, reference.n_spikes)
        self.assertGreater(reference.v_local_mV.max(), -35.0)

        show_plots_non_blocking()

    def test_fit_standard_apic20_csv_trace_and_plot_two_panels(self):
        reference, fit_result, fig, axes, output_path, metrics_path = (
            run_standard_apic20_single_compartment_fit(show_plot=True)
        )

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
        self.assertTrue(metrics_path.exists())
        self.assertGreater(metrics_path.stat().st_size, 0)
        self.assertEqual(3, reference.n_spikes)
        self.assertGreaterEqual(reference.t_ms[-1], 399.0)
        self.assertLess(fit_result.metrics["rmse_mV"], 30.0)

        show_plots_non_blocking()

    def test_fit_standard_apic20_csv_trace_with_fixed_passive_parameters(self):
        section_area_um2 = 73.1238141849912
        cm_density_uF_per_cm2 = 1.0
        g_l_density_S_per_cm2 = 3.3333333333333335e-05

        c_pF = cm_density_uF_per_cm2 * section_area_um2 * 0.01
        fixed_capacitance_nF = c_pF * 1e-3
        fixed_g_leak_nS = 10 * section_area_um2 * g_l_density_S_per_cm2
        fixed_tau_rise_ms = 5.0
        fixed_tau_decay_ms = 90.0
        fixed_alpha_per_ms = (
            fixed_tau_decay_ms - fixed_tau_rise_ms
        ) / (fixed_tau_rise_ms * fixed_tau_decay_ms)

        print("Area:", section_area_um2, "um²")
        print("cm density:", cm_density_uF_per_cm2, "uF/cm²")
        print("Total capacitance:", c_pF, "pF")
        print("g L density:", g_l_density_S_per_cm2, "S/cm²")
        print("Total g L:", fixed_g_leak_nS, "nS")
        print("tau", c_pF / fixed_g_leak_nS, "ms")
        print("tau rise:", fixed_tau_rise_ms, "ms")
        print("tau decay:", fixed_tau_decay_ms, "ms")
        print("alpha:", fixed_alpha_per_ms, "1/ms")

        reference, fit_result, fig, axes, output_path, metrics_path = (
            run_standard_apic20_single_compartment_fit_with_fixed_passive_parameters(
                fixed_capacitance_nF=fixed_capacitance_nF,
                fixed_g_leak_nS=fixed_g_leak_nS,
                fixed_tau_rise_ms=fixed_tau_rise_ms,
                fixed_tau_decay_ms=fixed_tau_decay_ms,
                fixed_alpha_per_ms=fixed_alpha_per_ms,
                show_plot=True,
            )
        )

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
        self.assertTrue(metrics_path.exists())
        self.assertGreater(metrics_path.stat().st_size, 0)
        self.assertEqual(3, reference.n_spikes)
        self.assertAlmostEqual(
            fixed_capacitance_nF,
            fit_result.fit_params.capacitance_nF,
        )
        self.assertAlmostEqual(fixed_g_leak_nS, fit_result.fit_params.g_leak_nS)
        self.assertAlmostEqual(fixed_tau_rise_ms, fit_result.fit_params.tau_rise_ms)
        self.assertAlmostEqual(fixed_tau_decay_ms, fit_result.fit_params.tau_decay_ms)
        self.assertAlmostEqual(fixed_alpha_per_ms, fit_result.fit_params.alpha_per_ms)
        self.assertLess(fit_result.metrics["rmse_mV"], 100.0)

        show_plots_non_blocking()

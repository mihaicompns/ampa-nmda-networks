import json
import unittest
from dataclasses import asdict
from pathlib import Path

from src.Plotting import show_plots_non_blocking
from src.iteration_20_nonlin_dynamics_NMDA.single_compartment_equivalence import (
    SingleCompartmentFitParameters,
    compute_trace_fit_metrics,
    plot_reference_and_simulation_two_panels,
    read_standard_apic20_reference,
    simulate_single_compartment_trace,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"
BEST_APIC20_FIT_PARAMETERS = SingleCompartmentFitParameters(
    g_nmda_max_nS=20.4916,
    capacitance_nF=0.136678,
    g_leak_nS=12.9435,
    tau_rise_ms=2.0,
    tau_decay_ms=44.2467,
    alpha_per_ms=0.687119,
    presynaptic_spike_weight=1.3417,
)


def run_best_apic20_single_compartment_comparison(
    output_dir=DEFAULT_OUTPUT_DIR,
    show_plot=True,
    fit_params=BEST_APIC20_FIT_PARAMETERS,
):
    reference = read_standard_apic20_reference()
    simulation = simulate_single_compartment_trace(
        fit_params=fit_params,
        target_t_ms=reference.t_ms,
        spike_times_ms=reference.spike_times_ms,
        initial_v_mV=float(reference.v_local_mV[0]),
    )
    metrics = compute_trace_fit_metrics(
        target_t_ms=reference.t_ms,
        target_v_mV=reference.v_local_mV,
        fitted_t_ms=simulation["t_ms"],
        fitted_v_mV=simulation["V_mV"],
    )
    fig, axes = plot_reference_and_simulation_two_panels(
        reference=reference,
        simulation=simulation,
        simulation_label="best-fit single compartment",
        title="APIC20 baseline CSV vs best-fit single-compartment simulation",
    )

    output_dir = Path(output_dir)
    output_path = output_dir / "single_compartment_apic20_best_fit_comparison.png"
    show_plots_non_blocking(
        show=show_plot,
        save_name="single_compartment_apic20_best_fit_comparison",
        out_dir=output_dir,
    )
    metrics_path = output_dir / "single_compartment_apic20_best_fit_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "fit_parameters": asdict(fit_params),
                "metrics": metrics,
                "source_trace": str(reference.trace_path),
                "source_metadata": str(reference.metadata_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return reference, simulation, metrics, fig, axes, output_path, metrics_path


class CompareBestSingleCompartmentToApic20ScriptTestCases(unittest.TestCase):
    """Manual PyCharm runnables for comparing a fixed best fit to APIC20."""

    def test_compare_best_fit_to_standard_apic20_csv_trace(self):
        reference, simulation, metrics, fig, axes, output_path, metrics_path = (
            run_best_apic20_single_compartment_comparison(show_plot=True)
        )

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
        self.assertTrue(metrics_path.exists())
        self.assertGreater(metrics_path.stat().st_size, 0)
        self.assertEqual(3, reference.n_spikes)
        self.assertEqual(4, len(axes))
        self.assertEqual(len(reference.t_ms), len(simulation["t_ms"]))
        self.assertLess(metrics["rmse_mV"], 30.0)

        show_plots_non_blocking()


if __name__ == "__main__":
    unittest.main()

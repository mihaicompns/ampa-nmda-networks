import os
import unittest
from pathlib import Path

from brian2 import ms, mV, nS

from src.Plotting import show_plots_non_blocking
from src.iteration_20_nonlin_dynamics_NMDA.nmda_spike_model import NMDASpikeParameters, simulate_presynaptic_spike_train, \
    incremental_peak_depolarizations, plot_one_two_three_spike_voltage_overlay

DEFAULT_SPIKE_TIMES = [5.0 * ms, 7.0 * ms, 9.0 * ms]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"


def run_nmda_spike_overlay(
    backend="plain",
    spike_times=DEFAULT_SPIKE_TIMES,
    params=NMDASpikeParameters(
        dt=0.05 * ms,
        t_stop=500.0 * ms,
        g_leak=0.0 * nS,
        g_nmda_max=5.0 * nS,
        initial_v=-70.0 * mV,
    ),
    output_dir=DEFAULT_OUTPUT_DIR,
    show_plot=True,
):
    simulations = simulate_presynaptic_spike_train(
        spike_times=spike_times,
        params=params,
        backend=backend,
    )
    increments = incremental_peak_depolarizations(
        simulations=simulations,
        spike_times=spike_times,
    )
    fig, axes = plot_one_two_three_spike_voltage_overlay(
        simulations=simulations,
        params=params,
        spike_times=spike_times,
        title=(
            "NMDA-only local model without leak, "
            f"{backend}, g_NMDA,max={float(params.g_nmda_max / nS):.3g} nS"
        ),
    )

    output_dir = Path(output_dir)
    output_path = output_dir / f"overlay_{backend}_simulation.png"
    show_plots_non_blocking(
        show=show_plot,
        save_name=f"overlay_{backend}_simulation",
        out_dir=output_dir,
    )
    return fig, axes, simulations, increments, output_path


def run_plain_nmda_spike_overlay(show_plot=True):
    return run_nmda_spike_overlay(backend="plain", show_plot=show_plot)


def run_brian2_nmda_spike_overlay(show_plot=True):
    return run_nmda_spike_overlay(backend="brian2", show_plot=show_plot)


class PlotNMDASpikeOverlayScriptTestCases(unittest.TestCase):
    """Manual PyCharm runnables for long/research NMDA spike plots."""

    def test_plain_nmda_spike_overlay(self):
        fig, axes, simulations, increments, output_path = run_plain_nmda_spike_overlay(
            show_plot=True
        )

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
        self.assertEqual(2, len(axes))
        self.assertEqual([1, 2, 3], list(simulations))
        self.assertGreaterEqual(simulations[3]["t_ms"][-1], 500.0)

        show_plots_non_blocking()

    def test_brian2_nmda_spike_overlay(self):
        fig, axes, simulations, increments, output_path = run_brian2_nmda_spike_overlay(
            show_plot=True
        )

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
        self.assertEqual(2, len(axes))
        self.assertEqual([1, 2, 3], list(simulations))
        self.assertGreaterEqual(simulations[3]["t_ms"][-1], 500.0)

        show_plots_non_blocking()

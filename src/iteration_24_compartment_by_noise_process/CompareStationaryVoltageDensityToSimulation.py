"""Compare the stationary voltage density with simulated voltage samples.

The manual unittest runnable at the bottom is intended for PyCharm.  It runs
the iteration-24 Stratonovich--Heun white-noise model and the Brian2 summed
NMDA-input reference, then displays their post-burn-in densities together
with the normalized analytical stationary density.
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import Hz, nS

from iteration_21_nmda_spikes_under_ei_balance.SimulateSummedNMDASpikesUnderEIBalanceScripts import (
    AmpaNmdaExperimentConfig,
    simulate_ampa_nmda_single_compartment_ensemble_with_summed_nmda_input,
)
from iteration_24_compartment_by_noise_process.NoiseProcessNMDASpikesScripts import (
    simulate_stratonovich_heun,
)
from iteration_24_compartment_by_noise_process.PlotStationaryVoltageDensity import (
    compute_stationary_voltage_density,
)


HISTOGRAM_BINS = 100
COMPARTMENT_INDEX = 0


def compute_stationary_density_simulation_comparison(experiment):
    """Compute the analytical density and both post-burn-in simulations."""
    stationary_result = compute_stationary_voltage_density(experiment)
    noise_result = simulate_stratonovich_heun(experiment)
    brian2_result = (
        simulate_ampa_nmda_single_compartment_ensemble_with_summed_nmda_input(
            experiment
        )
    )
    return {
        "stationary": stationary_result,
        "noise": noise_result,
        "brian2": brian2_result,
    }


def _plot_empirical_and_stationary(
    axis,
    samples_mV,
    stationary_result,
    bins,
    *,
    empirical_label,
    empirical_color,
    title,
):
    samples_mV = np.asarray(samples_mV, dtype=float).ravel()
    axis.hist(
        samples_mV,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=empirical_color,
        edgecolor=empirical_color,
        alpha=0.38,
        label=f"{empirical_label} (n={samples_mV.size:,})",
    )
    axis.plot(
        stationary_result["voltage_mV"],
        stationary_result["density_per_mV"],
        color="black",
        linewidth=2.4,
        label=r"stationary $r_0(V)$",
    )
    axis.set(title=title, xlabel="V [mV]", ylabel=r"density [mV$^{-1}$]")
    axis.grid(alpha=0.18)
    axis.legend(loc="best", fontsize=8)


def plot_stationary_density_simulation_comparison(
    comparison,
    *,
    compartment_index=COMPARTMENT_INDEX,
    histogram_bins=HISTOGRAM_BINS,
    show_plot=True,
):
    """Display analytical-versus-empirical density comparisons."""
    stationary = comparison["stationary"]
    noise_voltage = np.asarray(comparison["noise"]["voltage_mV"], dtype=float)
    brian2_voltage = np.asarray(
        comparison["brian2"]["voltage_mV"], dtype=float
    )
    compartment_index = int(compartment_index)
    if not 0 <= compartment_index < noise_voltage.shape[0]:
        raise IndexError("compartment_index must refer to a simulated unit")

    voltage_min = min(
        float(stationary["voltage_mV"].min()),
        float(noise_voltage.min()),
        float(brian2_voltage.min()),
    )
    voltage_max = max(
        float(stationary["voltage_mV"].max()),
        float(noise_voltage.max()),
        float(brian2_voltage.max()),
    )
    bins = np.linspace(voltage_min, voltage_max, int(histogram_bins) + 1)

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    _plot_empirical_and_stationary(
        axes[0, 0],
        noise_voltage.ravel(),
        stationary,
        bins,
        empirical_label="local Stratonovich--Heun population",
        empirical_color="tab:green",
        title="Stationary density vs local-noise population",
    )
    _plot_empirical_and_stationary(
        axes[0, 1],
        noise_voltage[compartment_index],
        stationary,
        bins,
        empirical_label=f"local compartment {compartment_index}",
        empirical_color="tab:orange",
        title=f"Stationary density vs compartment {compartment_index}",
    )
    _plot_empirical_and_stationary(
        axes[1, 0],
        brian2_voltage.ravel(),
        stationary,
        bins,
        empirical_label="Brian2 summed-input population",
        empirical_color="tab:blue",
        title="Stationary density vs Brian2 shot noise",
    )

    combined_axis = axes[1, 1]
    combined_axis.plot(
        stationary["voltage_mV"],
        stationary["density_per_mV"],
        color="black",
        linewidth=2.6,
        label=r"stationary $r_0(V)$",
    )
    for values, color, label in (
        (noise_voltage.ravel(), "tab:green", "local population"),
        (noise_voltage[compartment_index], "tab:orange", f"compartment {compartment_index}"),
        (brian2_voltage.ravel(), "tab:blue", "Brian2 population"),
    ):
        density, edges = np.histogram(values, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        combined_axis.step(centers, density, where="mid", color=color, label=label)
    combined_axis.set(
        title="All normalized densities",
        xlabel="V [mV]",
        ylabel=r"density [mV$^{-1}$]",
    )
    combined_axis.grid(alpha=0.18)
    combined_axis.legend(loc="best", fontsize=8)

    figure.suptitle(
        "Analytical stationary density vs post-burn-in simulation\n"
        rf"$\sigma={stationary['sigma_sqrt_ms']:.4g}\sqrt{{\mathrm{{ms}}}}$, "
        rf"$s_0={stationary['s_0']:.4g}$"
    )
    figure.tight_layout()
    if show_plot:
        plt.show()
    return figure, axes


def compute_and_plot_default_comparison(*, show_plot=True):
    """Run the fitted APIC20 configuration used by the reference figure."""
    experiment = AmpaNmdaExperimentConfig.from_fitted_apic20().with_property(
        r_e=25 * Hz,
        w_ampa=0 * nS,
    )
    comparison = compute_stationary_density_simulation_comparison(experiment)
    figure, axes = plot_stationary_density_simulation_comparison(
        comparison,
        show_plot=show_plot,
    )
    return comparison, figure, axes


class CompareStationaryVoltageDensityToSimulationRunnables(unittest.TestCase):
    """Manual PyCharm runnable which computes and shows the comparison."""

    def test_compute_and_show_stationary_density_simulation_comparison(self):
        comparison, _, _ = compute_and_plot_default_comparison(show_plot=True)
        stationary = comparison["stationary"]
        normalization = np.trapezoid(
            stationary["density_per_mV"], stationary["voltage_mV"]
        )
        self.assertAlmostEqual(1.0, float(normalization), places=10)
        self.assertGreater(comparison["noise"]["voltage_mV"].size, 0)
        self.assertGreater(comparison["brian2"]["voltage_mV"].size, 0)


if __name__ == "__main__":
    unittest.main()

"""Scale the first stationary-density hump and compare Gaussian-fit peaks.

Edit ``K_FIRST_HUMP`` below, then run the unittest method at the bottom from
PyCharm.  The density to the left of the valley between its two modes is
multiplied by k and the complete curve is subsequently renormalized.
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import Hz, nS
from scipy.signal import find_peaks

from iteration_21_nmda_spikes_under_ei_balance.SimulateSummedNMDASpikesUnderEIBalanceScripts import (
    AmpaNmdaExperimentConfig,
    _evaluate_gaussian_mixture_density,
    _fit_gaussian_mixture,
)
from iteration_24_compartment_by_noise_process.CompareStationaryVoltageDensityToSimulation import (
    compute_stationary_density_simulation_comparison,
)


# Change this value by hand.  k > 1 increases the probability mass assigned
# to the first (lower-voltage) hump; k=1 leaves the density unchanged.
HISTOGRAM_BINS = 100
GAUSSIAN_FIT_SAMPLE_COUNT = 1_500_000


def scale_first_stationary_hump(stationary_result, k):
    """Multiply the first hump by ``k`` and return a normalized copy.

    The split is the minimum between the first two voltage-ordered modes.
    It is found from the analytical curve rather than hard-coded in mV.
    """
    k = float(k)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("k must be finite and greater than zero")

    voltage_mV = np.asarray(stationary_result["voltage_mV"], dtype=float)
    density = np.asarray(stationary_result["density_per_mV"], dtype=float)
    minimum_peak_distance = max(1, density.size // 20)
    peak_indices, properties = find_peaks(
        density,
        prominence=max(float(density.max()) * 1e-5, np.finfo(float).eps),
        distance=minimum_peak_distance,
    )
    if peak_indices.size < 2:
        raise ValueError("the stationary density does not contain two detectable humps")

    # Retain the two most prominent modes, then put them in voltage order.
    prominence_order = np.argsort(properties["prominences"])[-2:]
    first_peak, second_peak = np.sort(peak_indices[prominence_order])
    valley_index = first_peak + int(
        np.argmin(density[first_peak : second_peak + 1])
    )

    scaled_density = density.copy()
    scaled_density[: valley_index + 1] *= k
    normalization = float(np.trapezoid(scaled_density, voltage_mV))
    scaled_density /= normalization

    result = dict(stationary_result)
    result.update(
        density_per_mV=scaled_density,
        original_density_per_mV=density,
        first_hump_k=k,
        first_peak_mV=float(voltage_mV[first_peak]),
        second_peak_mV=float(voltage_mV[second_peak]),
        valley_mV=float(voltage_mV[valley_index]),
    )
    return result


def fit_population_gaussian_mixtures(comparison):
    """Fit two-Gaussian mixtures and return their peak-height ratio."""
    brian2_values = np.asarray(
        comparison["brian2"]["voltage_mV"], dtype=float
    ).ravel()
    heun_values = np.asarray(
        comparison["noise"]["voltage_mV"], dtype=float
    ).ravel()
    evaluation_grid_mV = np.linspace(
        min(float(brian2_values.min()), float(heun_values.min())),
        max(float(brian2_values.max()), float(heun_values.max())),
        1200,
    )
    brian2_fit = _fit_gaussian_mixture(
        brian2_values,
        2,
        maximum_sample_count=GAUSSIAN_FIT_SAMPLE_COUNT,
    )
    heun_fit = _fit_gaussian_mixture(
        heun_values,
        2,
        maximum_sample_count=GAUSSIAN_FIT_SAMPLE_COUNT,
    )
    _, brian2_mixture = _evaluate_gaussian_mixture_density(
        brian2_fit, evaluation_grid_mV
    )
    _, heun_mixture = _evaluate_gaussian_mixture_density(
        heun_fit, evaluation_grid_mV
    )
    brian2_max = float(np.max(brian2_mixture))
    heun_max = float(np.max(heun_mixture))
    return {
        "grid_mV": evaluation_grid_mV,
        "brian2_density_per_mV": brian2_mixture,
        "heun_density_per_mV": heun_mixture,
        "brian2_max_per_mV": brian2_max,
        "heun_max_per_mV": heun_max,
        "brian2_to_heun_max_ratio": brian2_max / heun_max,
    }


def _plot_histogram_with_stationary_curves(
    axis, values, bins, adjusted_stationary, *, color, label, title
):
    values = np.asarray(values, dtype=float).ravel()
    axis.hist(
        values,
        bins=bins,
        density=True,
        color=color,
        edgecolor=color,
        alpha=0.35,
        label=f"{label} (n={values.size:,})",
    )
    axis.plot(
        adjusted_stationary["voltage_mV"],
        adjusted_stationary["original_density_per_mV"],
        "--",
        color="0.35",
        linewidth=1.8,
        label=r"original $r_0(V)$",
    )
    axis.plot(
        adjusted_stationary["voltage_mV"],
        adjusted_stationary["density_per_mV"],
        color="black",
        linewidth=2.5,
        label=rf"adjusted $r_0(V)$, $k={adjusted_stationary['first_hump_k']:g}$",
    )
    axis.axvline(
        adjusted_stationary["valley_mV"],
        color="tab:red",
        linestyle=":",
        linewidth=1.2,
        label=rf"hump split: {adjusted_stationary['valley_mV']:.2f} mV",
    )
    axis.set(title=title, xlabel="V [mV]", ylabel=r"density [mV$^{-1}$]")
    axis.grid(alpha=0.18)
    axis.legend(loc="best", fontsize=8)


def plot_scaled_hump_comparison(
    comparison,
    adjusted_stationary,
    gaussian_fits,
    *,
    histogram_bins=HISTOGRAM_BINS,
    show_plot=True,
):
    """Plot the scaled density and Brian2/Heun Gaussian-fit peak ratio."""
    noise_values = np.asarray(comparison["noise"]["voltage_mV"], dtype=float)
    brian2_values = np.asarray(comparison["brian2"]["voltage_mV"], dtype=float)
    voltage_min = min(
        float(adjusted_stationary["voltage_mV"].min()),
        float(noise_values.min()),
        float(brian2_values.min()),
    )
    voltage_max = max(
        float(adjusted_stationary["voltage_mV"].max()),
        float(noise_values.max()),
        float(brian2_values.max()),
    )
    bins = np.linspace(voltage_min, voltage_max, int(histogram_bins) + 1)

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    _plot_histogram_with_stationary_curves(
        axes[0, 0],
        noise_values.ravel(),
        bins,
        adjusted_stationary,
        color="tab:green",
        label="Stratonovich--Heun population",
        title="Adjusted stationary density vs Stratonovich--Heun",
    )
    _plot_histogram_with_stationary_curves(
        axes[0, 1],
        brian2_values.ravel(),
        bins,
        adjusted_stationary,
        color="tab:blue",
        label="Brian2 population",
        title="Adjusted stationary density vs Brian2",
    )

    overlay_axis = axes[1, 0]
    overlay_axis.plot(
        adjusted_stationary["voltage_mV"],
        adjusted_stationary["original_density_per_mV"],
        "--",
        color="0.45",
        linewidth=1.8,
        label="original stationary",
    )
    overlay_axis.plot(
        adjusted_stationary["voltage_mV"],
        adjusted_stationary["density_per_mV"],
        color="black",
        linewidth=2.5,
        label=rf"adjusted stationary ($k={adjusted_stationary['first_hump_k']:g}$)",
    )
    for values, color, label in (
        (noise_values.ravel(), "tab:green", "Stratonovich--Heun"),
        (brian2_values.ravel(), "tab:blue", "Brian2"),
    ):
        density, edges = np.histogram(values, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        overlay_axis.step(centers, density, where="mid", color=color, label=label)
    overlay_axis.set(
        title="All normalized densities",
        xlabel="V [mV]",
        ylabel=r"density [mV$^{-1}$]",
    )
    overlay_axis.grid(alpha=0.18)
    overlay_axis.legend(loc="best", fontsize=8)

    fit_axis = axes[1, 1]
    fit_axis.plot(
        gaussian_fits["grid_mV"],
        gaussian_fits["brian2_density_per_mV"],
        color="tab:blue",
        linewidth=2.5,
        label="Brian2 2-Gaussian fit",
    )
    fit_axis.plot(
        gaussian_fits["grid_mV"],
        gaussian_fits["heun_density_per_mV"],
        color="tab:green",
        linewidth=2.5,
        label="Stratonovich--Heun 2-Gaussian fit",
    )
    ratio = gaussian_fits["brian2_to_heun_max_ratio"]
    fit_axis.text(
        0.03,
        0.96,
        "Peak-height ratio\n"
        rf"$\max p_{{Brian2}} / \max p_{{Heun}} = {ratio:.4f}$\n"
        rf"$\max p_{{Brian2}}={gaussian_fits['brian2_max_per_mV']:.5f}$\n"
        rf"$\max p_{{Heun}}={gaussian_fits['heun_max_per_mV']:.5f}$",
        transform=fit_axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.5"},
    )
    fit_axis.set(
        title="Gaussian-mixture peak-height ratio",
        xlabel="V [mV]",
        ylabel=r"density [mV$^{-1}$]",
    )
    fit_axis.grid(alpha=0.18)
    fit_axis.legend(loc="best", fontsize=8)

    figure.suptitle(
        "First-hump scaling of the stationary density\n"
        rf"$k={adjusted_stationary['first_hump_k']:g}$; "
        rf"valley $={adjusted_stationary['valley_mV']:.2f}$ mV; "
        rf"Brian2/Heun fitted peak ratio $={ratio:.4f}$"
    )
    figure.tight_layout()
    if show_plot:
        plt.show()
    return figure, axes


def compute_and_plot_default_scaled_hump(k, show_plot=True):
    """Run the full fitted APIC20 comparison with a hand-selected ``k``."""
    experiment = AmpaNmdaExperimentConfig.from_fitted_apic20().with_property(
        r_e=25 * Hz,
        w_ampa=0 * nS,
    )
    comparison = compute_stationary_density_simulation_comparison(experiment)
    adjusted = scale_first_stationary_hump(comparison["stationary"], k)
    gaussian_fits = fit_population_gaussian_mixtures(comparison)
    figure, axes = plot_scaled_hump_comparison(
        comparison,
        adjusted,
        gaussian_fits,
        show_plot=show_plot,
    )
    return comparison, adjusted, gaussian_fits, figure, axes


class ScaleFirstStationaryDensityHumpRunnables(unittest.TestCase):
    """Manual PyCharm runnable for hand-tuning ``K_FIRST_HUMP``."""

    def test_compute_and_show_scaled_first_hump_comparison(self):
        k = 5
        comparison, adjusted, fits, _, _ = (
            compute_and_plot_default_scaled_hump(k = k, show_plot=True)
        )
        normalization = np.trapezoid(
            adjusted["density_per_mV"], adjusted["voltage_mV"]
        )
        self.assertAlmostEqual(1.0, float(normalization), places=10)
        self.assertGreater(fits["brian2_to_heun_max_ratio"], 0.0)
        self.assertGreater(comparison["noise"]["voltage_mV"].size, 0)


if __name__ == "__main__":
    unittest.main()

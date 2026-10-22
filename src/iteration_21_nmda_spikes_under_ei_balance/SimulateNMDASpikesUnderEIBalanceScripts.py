import json
import csv
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from joblib import Parallel, delayed
from scipy.special import logsumexp, ndtr
from brian2 import (
    Hz,
    Clock,
    Network,
    NeuronGroup,
    PoissonInput,
    StateMonitor,
    kHz,
    ms,
    mV,
    nF,
    nS,
    seed,
    start_scope,
)

from Plotting import show_plots_non_blocking

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iteration_16.SolveCubicEquation import (  # noqa: E402
    E_0,
    compute_siegert_firing_rate_for_fitted_solution,
    evaluate_sympy_solution,
    polynomial_r,
    sigma_sq,
)
from iteration_16.model import (  # noqa: E402
    ConductanceDiffusionSimulationConfig,
    chapter1Results,
    config_with_intermediate_synapses,
)
from iteration_16.simpy import bind_config_to_sympy_values  # noqa: E402
from iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20NullclineScripts import (  # noqa: E402
    read_fitted_apic20_parameters_from_csv,
)

ITERATION_16_SOLUTIONS_PATH = SRC_DIR / "iteration_16" / "solutions" / "solutions.txt"
OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"
DEBUG_DISTRIBUTION_DIR = Path(__file__).resolve().parent / "debug_distributions"

# Keep the stored names identical to the simulation result keys.
DEBUG_DISTRIBUTION_VARIABLES = {
    "effective_g_nmda_nS": "g_NMDA [nS]",
    "s_nmda": "s_NMDA [1]",
    "x_nmda": "x_NMDA [1]",
    "g_ampa_nS": "g_AMPA [nS]",
    "voltage_mV": "voltage [mV]",
}


@dataclass(frozen=True)
class DebugHistogramSaveConfiguration:
    """Select the compact histogram debug files produced by the long run."""

    enabled: bool = False
    save_g_nmda: bool = False
    save_s_nmda: bool = False
    save_x_nmda: bool = False
    save_g_ampa: bool = False
    save_voltage: bool = False
    save_csv: bool = True
    save_json: bool = False
    bins: int = 200

    def selected_result_keys(self):
        flags = {
            "effective_g_nmda_nS": self.save_g_nmda,
            "s_nmda": self.save_s_nmda,
            "x_nmda": self.save_x_nmda,
            "g_ampa_nS": self.save_g_ampa,
            "voltage_mV": self.save_voltage,
        }
        return [key for key, selected in flags.items() if selected]


# Edit this configuration before running the 10,000-compartment runnable.
# The camelCase name is retained because this is the requested runnable flag.
saveDebugFile = DebugHistogramSaveConfiguration(
    enabled=False,
    save_g_nmda=False,
    save_s_nmda=False,
    save_x_nmda=False,
    save_g_ampa=False,
    save_voltage=True,
    save_csv=True,
    save_json=False,
    bins=200,
)

saveAMPAFile = DebugHistogramSaveConfiguration(
    enabled=True,
    save_g_nmda=False,
    save_s_nmda=False,
    save_x_nmda=False,
    save_g_ampa=True,
    save_voltage=False,
    save_csv=True,
    save_json=False,
    bins=500,
)


# Kept as a named, inspectable Brian2 equation string.  Every dynamic variable
# and subexpression declares its physical unit.
AMPA_NMDA_SINGLE_COMPARTMENT_MODEL = """
dV/dt = (-I_leak - I_AMPA - I_NMDA) / C_m : volt

dg_e/dt = -g_e / tau_AMPA : siemens
dx_NMDA/dt = -x_NMDA / tau_NMDA_rise : 1
ds_NMDA/dt = -s_NMDA / tau_NMDA_decay
             + alpha_NMDA * x_NMDA : 1

sigma_Mg = 1 / (1 + exp(-0.08 * V / mV) * Mg / 3.57) : 1
g_NMDA_open = g_NMDA_max * s_NMDA : siemens
g_NMDA = g_NMDA_open * sigma_Mg : siemens

I_leak = g_L * (V - E_L) : amp
I_AMPA = g_e * (V - E_E) : amp
I_NMDA = g_NMDA * (V - E_E) : amp
"""


def simulate_ampa_nmda_single_compartment_ensemble(
    *,
    n_models,
    simulation_time,
    burn_in,
    dt,
    random_seed,
    C_m,
    g_L,
    E_L,
    E_E,
    tau_AMPA,
    w_AMPA,
    r_e,
    n_e,
    tau_NMDA_rise,
    tau_NMDA_decay,
    alpha_NMDA,
    g_NMDA_max,
    n_NMDA,
    w_NMDA,
    Mg,
):
    """Simulate independent copies and return voltage samples and statistics.

    ``n_e * r_e`` and ``n_NMDA * r_e`` are superposed Poisson rates for each
    compartment.  Brian2's ``PoissonInput`` gives every compartment an
    independent realization, so all 1,000 copies can be integrated together.
    """
    start_scope()
    seed(random_seed)
    clock = Clock(dt=dt)
    g_NMDA = g_NMDA_max / n_NMDA
    namespace = {
        "C_m": C_m,
        "g_L": g_L,
        "E_L": E_L,
        "E_E": E_E,
        "tau_AMPA": tau_AMPA,
        "tau_NMDA_rise": tau_NMDA_rise,
        "tau_NMDA_decay": tau_NMDA_decay,
        "alpha_NMDA": alpha_NMDA,
        "g_NMDA_max": g_NMDA,
        "Mg": Mg,
    }
    compartments = NeuronGroup(
        n_models,
        AMPA_NMDA_SINGLE_COMPARTMENT_MODEL,
        method="euler",
        namespace=namespace,
        clock=clock,
        name="single_compartments",
    )
    compartments.V = E_L
    compartments.g_e = 0 * nS
    compartments.x_NMDA = 0
    compartments.s_NMDA = 0

    ampa_input = PoissonInput(
        compartments, "g_e", N=n_e, rate=r_e, weight=w_AMPA
    )
    nmda_input = PoissonInput(
        compartments, "x_NMDA", N=n_NMDA, rate=r_e, weight=w_NMDA
    )
    state = StateMonitor(
        compartments,
        ["V", "g_e", "g_NMDA", "s_NMDA", "x_NMDA"],
        record=True,
        clock=clock,
    )
    Network(compartments, ampa_input, nmda_input, state).run(simulation_time)

    sample_mask = state.t >= burn_in
    voltage_mV = np.asarray(state.V[:, sample_mask] / mV)
    g_ampa_nS = np.asarray(state.g_e[:, sample_mask] / nS)
    effective_g_nmda_nS = np.asarray(state.g_NMDA[:, sample_mask] / nS)
    s_nmda = np.asarray(state.s_NMDA[:, sample_mask])
    x_nmda = np.asarray(state.x_NMDA[:, sample_mask])
    per_model_mean_mV = np.mean(voltage_mV, axis=1)
    per_model_variance_mV2 = np.var(voltage_mV, axis=1)
    return {
        "model_equations": AMPA_NMDA_SINGLE_COMPARTMENT_MODEL,
        "time_ms": np.asarray(state.t[sample_mask] / ms),
        "voltage_mV": voltage_mV,
        "g_ampa_nS": g_ampa_nS,
        "effective_g_nmda_nS": effective_g_nmda_nS,
        "s_nmda": s_nmda,
        "x_nmda": x_nmda,
        "per_model_mean_mV": per_model_mean_mV,
        "per_model_variance_mV2": per_model_variance_mV2,
        "ensemble_mean_mV": float(np.mean(voltage_mV)),
        "ensemble_variance_mV2": float(np.var(voltage_mV)),
    }


def estimate_full_text_json_size(result, sample_size=10_000):
    """Estimate full-precision JSON size without materializing huge lists.

    JSON has no native float array representation, so its size depends on the
    decimal spellings of the values.  We measure those spellings on an evenly
    spaced sample and add one comma per value.  ``binary_bytes`` is exact for
    the five NumPy arrays and is the useful comparison for the debug files.
    """
    number_of_float_values = 0
    binary_bytes = 0
    estimated_json_bytes = 2  # outer braces
    by_variable = {}
    for key, label in DEBUG_DISTRIBUTION_VARIABLES.items():
        values = np.asarray(result[key])
        flat = values.ravel()
        count = int(flat.size)
        number_of_float_values += count
        binary_bytes += int(values.nbytes)
        if count:
            indices = np.linspace(
                0, count - 1, min(count, int(sample_size)), dtype=np.int64
            )
            encoded_lengths = [len(json.dumps(float(value))) for value in flat[indices]]
            average_float_bytes = float(np.mean(encoded_lengths))
        else:
            average_float_bytes = 0.0
        # Includes commas and a modest allowance for row brackets and the key.
        array_bytes = int(round(count * (average_float_bytes + 1.0)))
        array_bytes += int(values.shape[0] * 2 if values.ndim == 2 else 2)
        array_bytes += len(json.dumps(key)) + 2
        estimated_json_bytes += array_bytes
        by_variable[key] = {
            "label": label,
            "shape": list(values.shape),
            "number_of_float_values": count,
            "binary_bytes": int(values.nbytes),
            "estimated_json_bytes": array_bytes,
        }
    return {
        "number_of_float_values": number_of_float_values,
        "binary_bytes": binary_bytes,
        "estimated_json_bytes": estimated_json_bytes,
        "variables": by_variable,
    }


def format_storage_size(number_of_bytes):
    """Return a compact decimal/GiB storage-size description."""
    return f"{number_of_bytes / 1e9:.2f} GB ({number_of_bytes / 2**30:.2f} GiB)"


def save_debug_histograms(result, output_dir, configuration):
    """Save selected histogram counts/edges as separate CSV and/or JSON files."""
    if not configuration.enabled:
        return {}
    if not configuration.save_csv and not configuration.save_json:
        raise ValueError("Enable save_csv, save_json, or both")
    if configuration.bins < 2:
        raise ValueError("Histogram bins must be at least 2")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    for key in configuration.selected_result_keys():
        label = DEBUG_DISTRIBUTION_VARIABLES[key]
        counts, bin_edges = np.histogram(
            np.asarray(result[key]).ravel(), bins=configuration.bins
        )
        paths = []
        if configuration.save_json:
            json_path = output_dir / f"{key}_histogram.json"
            json_path.write_text(
                json.dumps(
                    {
                        "variable": key,
                        "label": label,
                        "counts": counts.tolist(),
                        "bin_edges": bin_edges.tolist(),
                    },
                    indent=2,
                )
            )
            paths.append(json_path)
        if configuration.save_csv:
            csv_path = output_dir / f"{key}_histogram.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["bin_left", "bin_right", "count"])
                writer.writerows(zip(bin_edges[:-1], bin_edges[1:], counts))
            paths.append(csv_path)
        saved_paths[key] = paths
    return saved_paths


def load_debug_histogram(histogram_path, variable_name):
    """Load one variable's histogram without touching the other distributions."""
    histogram_path = Path(histogram_path)
    if variable_name not in DEBUG_DISTRIBUTION_VARIABLES:
        raise ValueError(
            f"Unknown variable {variable_name!r}; choose from "
            f"{tuple(DEBUG_DISTRIBUTION_VARIABLES)}"
        )
    if histogram_path.suffix.lower() == ".json":
        histogram = json.loads(histogram_path.read_text())
        if histogram["variable"] != variable_name:
            raise ValueError(
                f"File contains {histogram['variable']!r}, not {variable_name!r}"
            )
        counts = np.asarray(histogram["counts"], dtype=np.int64)
        bin_edges = np.asarray(histogram["bin_edges"], dtype=float)
        label = histogram["label"]
    elif histogram_path.suffix.lower() == ".csv":
        table = np.genfromtxt(histogram_path, delimiter=",", names=True)
        table = np.atleast_1d(table)
        counts = np.asarray(table["count"], dtype=np.int64)
        bin_edges = np.r_[table["bin_left"], table["bin_right"][-1]].astype(float)
        label = DEBUG_DISTRIBUTION_VARIABLES[variable_name]
    else:
        raise ValueError("Histogram file must have a .csv or .json suffix")
    return {
        "label": label,
        "counts": counts,
        "bin_edges": bin_edges,
    }


def fit_saved_debug_histogram(
    histogram_path,
    variable_name,
    output_path=None,
    max_fit_samples=20_000,
):
    """Fit and plot one saved histogram without rerunning the simulation."""
    histogram = load_debug_histogram(histogram_path, variable_name)
    counts = histogram["counts"]
    bin_edges = histogram["bin_edges"]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total_count = int(np.sum(counts))
    if total_count < 5:
        raise ValueError("At least five histogram observations are required")

    # Deterministically approximate the weighted histogram with at most 20k
    # bin-centre observations, which keeps each exploratory fit quick.
    fit_sample_count = min(total_count, int(max_fit_samples))
    exact_repetitions = counts.astype(float) * fit_sample_count / total_count
    repetitions = np.floor(exact_repetitions).astype(int)
    missing = fit_sample_count - int(np.sum(repetitions))
    if missing:
        residual_order = np.argsort(exact_repetitions - repetitions)[::-1]
        repetitions[residual_order[:missing]] += 1
    fit_values = np.repeat(bin_centers, repetitions)
    fit = fit_two_gaussian_mixture(fit_values, max_samples=max_fit_samples)

    if output_path is None:
        output_path = Path(histogram_path).with_name(f"fit_{variable_name}.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    widths = np.diff(bin_edges)
    density = counts / (total_count * widths)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.stairs(density, bin_edges, fill=True, color="0.85", label="saved histogram")
    for component_index in range(2):
        axis.plot(
            fit["grid"],
            fit["component_density"][:, component_index],
            linestyle=":",
            label=f"Gaussian {component_index + 1}",
        )
    axis.plot(fit["grid"], fit["mixture_density"], linewidth=2, label="mixture")
    axis.set_xlabel(histogram["label"])
    axis.set_ylabel("density")
    axis.set_title(f"Two-Gaussian fit: {histogram['label']}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path, fit


def fit_two_gaussian_mixture(values, random_seed=21, max_samples=20_000):
    """Fit a 1-D two-Gaussian mixture and score it on held-out samples.

    Several quantile-based initializations are tried because the distributions
    produced here are usually one broad Gaussian with a small lower shoulder.
    A single EM initialization can split the broad peak instead of describing
    that shoulder, even though both solutions are local likelihood maxima.
    """
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size < 5:
        raise ValueError("At least five finite samples are required")
    rng = np.random.default_rng(random_seed)
    if values.size > max_samples:
        values = values[rng.choice(values.size, max_samples, replace=False)]
    rng.shuffle(values)
    split = max(2, int(0.8 * values.size))
    train, test = values[:split], values[split:]
    overall_std = max(float(np.std(train)), np.finfo(float).eps)
    minimum_std = max(overall_std * 1e-3, np.finfo(float).eps)
    best_fit = None
    for percentiles in ((15.0, 60.0), (25.0, 75.0), (40.0, 85.0)):
        means = np.percentile(train, percentiles)
        standard_deviations = np.full(2, overall_std)
        weights = np.full(2, 0.5)

        for _ in range(200):
            old_parameters = np.r_[weights, means, standard_deviations]
            log_components = (
                np.log(np.maximum(weights, np.finfo(float).eps))[None, :]
                - np.log(standard_deviations)[None, :]
                - 0.5 * np.log(2 * np.pi)
                - 0.5
                * ((train[:, None] - means[None, :]) / standard_deviations[None, :]) ** 2
            )
            responsibilities = np.exp(
                log_components - logsumexp(log_components, axis=1)[:, None]
            )
            component_mass = np.maximum(
                np.sum(responsibilities, axis=0), np.finfo(float).eps
            )
            weights = component_mass / np.sum(component_mass)
            means = np.sum(responsibilities * train[:, None], axis=0) / component_mass
            variances = (
                np.sum(responsibilities * (train[:, None] - means) ** 2, axis=0)
                / component_mass
            )
            standard_deviations = np.maximum(np.sqrt(variances), minimum_std)
            if np.allclose(
                old_parameters,
                np.r_[weights, means, standard_deviations],
                rtol=1e-7,
                atol=1e-9,
            ):
                break

        train_log_density = logsumexp(
            np.log(np.maximum(weights, np.finfo(float).eps))[None, :]
            - np.log(standard_deviations)[None, :]
            - 0.5 * np.log(2 * np.pi)
            - 0.5
            * ((train[:, None] - means[None, :]) / standard_deviations[None, :]) ** 2,
            axis=1,
        )
        train_log_likelihood = float(np.sum(train_log_density))
        if best_fit is None or train_log_likelihood > best_fit[0]:
            best_fit = (
                train_log_likelihood,
                weights.copy(),
                means.copy(),
                standard_deviations.copy(),
            )

    train_log_likelihood, weights, means, standard_deviations = best_fit
    component_order = np.argsort(means)
    weights = weights[component_order]
    means = means[component_order]
    standard_deviations = standard_deviations[component_order]

    def mixture_log_density(samples):
        log_components = (
            np.log(weights)[None, :]
            - np.log(standard_deviations)[None, :]
            - 0.5 * np.log(2 * np.pi)
            - 0.5
            * ((samples[:, None] - means[None, :]) / standard_deviations[None, :]) ** 2
        )
        return logsumexp(log_components, axis=1)

    held_out_mean_log_likelihood = float(np.mean(mixture_log_density(test)))
    sorted_test = np.sort(test)
    fitted_cdf = np.sum(
        weights[None, :]
        * ndtr(
            (sorted_test[:, None] - means[None, :])
            / standard_deviations[None, :]
        ),
        axis=1,
    )
    empirical_cdf = (np.arange(sorted_test.size) + 0.5) / sorted_test.size
    cdf_max_error = float(np.max(np.abs(empirical_cdf - fitted_cdf)))
    grid = np.linspace(float(np.min(values)), float(np.max(values)), 400)
    component_density = (
        weights[None, :]
        / (standard_deviations[None, :] * np.sqrt(2 * np.pi))
        * np.exp(
            -0.5
            * ((grid[:, None] - means[None, :]) / standard_deviations[None, :]) ** 2
        )
    )
    return {
        "weights": weights,
        "means": means,
        "standard_deviations": standard_deviations,
        "grid": grid,
        "component_density": component_density,
        "mixture_density": np.sum(component_density, axis=1),
        "held_out_mean_log_likelihood": held_out_mean_log_likelihood,
        "bic": -2 * train_log_likelihood + 5 * np.log(train.size),
        "cdf_max_error": cdf_max_error,
    }


def fit_single_exponential(values, random_seed=21, max_samples=20_000):
    """Fit an exponential distribution on non-negative values by MLE."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values) & (values >= 0.0)]
    if values.size < 3:
        raise ValueError("At least three non-negative finite samples are required")

    rng = np.random.default_rng(random_seed)
    if values.size > max_samples:
        values = values[rng.choice(values.size, max_samples, replace=False)]
    rng.shuffle(values)
    split = max(2, int(0.8 * values.size))
    train, test = values[:split], values[split:]
    scale = max(float(np.mean(train)), np.finfo(float).eps)
    rate = 1.0 / scale
    grid = np.linspace(0.0, float(np.max(values)), 500)
    density = rate * np.exp(-rate * grid)
    train_log_likelihood = float(train.size * np.log(rate) - rate * np.sum(train))
    held_out_mean_log_likelihood = float(np.mean(np.log(rate) - rate * test))
    sorted_test = np.sort(test)
    fitted_cdf = 1.0 - np.exp(-rate * sorted_test)
    empirical_cdf = np.arange(1, test.size + 1) / test.size

    return {
        "scale": scale,
        "rate": rate,
        "grid": grid,
        "density": density,
        "held_out_mean_log_likelihood": held_out_mean_log_likelihood,
        "bic": -2 * train_log_likelihood + np.log(train.size),
        "cdf_max_error": float(np.max(np.abs(empirical_cdf - fitted_cdf))),
    }


def plot_chosen_ampa_nmda_traces(
    result,
    chosen_model_indices=(0, 1, 2),
    output_path=OUTPUT_DIR / "ampa_nmda_three_chosen_traces.png",
    figure_title=None,
):
    """Plot V, AMPA, and Mg-blocked NMDA conductance for chosen models."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_ms = result["time_ms"]
    fig, axes = plt.subplots(
        5,
        3,
        figsize=(15, 14),
        sharex=False,
        sharey=False,
        gridspec_kw={"width_ratios": [4, 1, 1.35]},
    )
    trace_axes = axes[:, 0]
    (
        voltage_histogram_ax,
        ampa_histogram_ax,
        nmda_histogram_ax,
        s_nmda_histogram_ax,
        x_nmda_histogram_ax,
    ) = axes[:, 1]
    fit_axes = axes[:, 2]
    for model_index in chosen_model_indices:
        label = f"model {model_index}"
        trace_axes[0].plot(time_ms, result["voltage_mV"][model_index], label=label)
        trace_axes[1].plot(time_ms, result["g_ampa_nS"][model_index], label=label)
        trace_axes[2].plot(
            time_ms,
            result["effective_g_nmda_nS"][model_index],
            label=label,
        )
        trace_axes[3].plot(time_ms, result["s_nmda"][model_index], label=label)
        trace_axes[4].plot(time_ms, result["x_nmda"][model_index], label=label)

    # Use all recorded samples from all compartments for the distributions.
    voltage_histogram_ax.hist(
        result["voltage_mV"].ravel(),
        bins=50,
        density=True,
        orientation="horizontal",
        color="tab:green",
        alpha=0.75,
    )
    ampa_histogram_ax.hist(
        result["g_ampa_nS"].ravel(),
        bins=50,
        density=True,
        orientation="horizontal",
        color="tab:blue",
        alpha=0.75,
    )
    nmda_histogram_ax.hist(
        result["effective_g_nmda_nS"].ravel(),
        bins=50,
        density=True,
        orientation="horizontal",
        color="tab:orange",
        alpha=0.75,
    )
    s_nmda_histogram_ax.hist(
        result["s_nmda"].ravel(),
        bins=50,
        density=True,
        orientation="horizontal",
        color="tab:red",
        alpha=0.75,
    )
    x_nmda_histogram_ax.hist(
        result["x_nmda"].ravel(),
        bins=50,
        density=True,
        orientation="horizontal",
        color="tab:purple",
        alpha=0.75,
    )
    distribution_values = (
        result["voltage_mV"],
        result["g_ampa_nS"],
        result["effective_g_nmda_nS"],
        result["s_nmda"],
        result["x_nmda"],
    )
    distribution_colors = ("tab:green", "tab:blue", "tab:orange", "tab:red", "tab:purple")
    fit_kinds = ("gaussian_mixture",) * 5
    for fit_axis, values, color, fit_kind in zip(
        fit_axes, distribution_values, distribution_colors, fit_kinds
    ):
        fit_axis.hist(
            np.asarray(values).ravel(),
            bins=50,
            density=True,
            orientation="horizontal",
            color="0.85",
        )
        if fit_kind == "exponential":
            fit = fit_single_exponential(values)
            fit_axis.plot(fit["density"], fit["grid"], color=color, linewidth=2)
            parameter_text = f"scale={fit['scale']:.3g}"
        else:
            fit = fit_two_gaussian_mixture(values)
            for component_index in range(2):
                fit_axis.plot(
                    fit["component_density"][:, component_index],
                    fit["grid"],
                    linestyle=":",
                    color=color,
                    alpha=0.7,
                )
            fit_axis.plot(
                fit["mixture_density"], fit["grid"], color=color, linewidth=2
            )
            parameter_text = "2-Gaussian mixture"
        fit_axis.set_xlabel("density")
        fit_axis.text(
            0.98,
            0.98,
            f"{parameter_text}\nheld-out LL={fit['held_out_mean_log_likelihood']:.3g}\n"
            f"BIC={fit['bic']:.3g}\nD={fit['cdf_max_error']:.3g}",
            transform=fit_axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
    trace_axes[0].set_ylabel("V [mV]")
    trace_axes[1].set_ylabel(r"$g_{AMPA}$ [nS]")
    trace_axes[2].set_ylabel(r"$g_{NMDA,max}s\sigma_{Mg}(V)$ [nS]")
    trace_axes[3].set_ylabel(r"$s_{NMDA}$ [1]")
    trace_axes[4].set_ylabel(r"$x_{NMDA}$ [1]")
    trace_axes[4].set_xlabel("time [ms]")
    trace_axes[0].set_title("Three single-compartment realizations")
    voltage_histogram_ax.set_title("distribution", fontsize=10)
    fit_axes[0].set_title("distribution fit", fontsize=10)
    voltage_histogram_ax.set_xlabel("density")
    ampa_histogram_ax.set_xlabel("density")
    nmda_histogram_ax.set_xlabel("density")
    s_nmda_histogram_ax.set_xlabel("density")
    x_nmda_histogram_ax.set_xlabel("density")
    for axis in trace_axes:
        axis.legend(loc="best")
    if figure_title:
        fig.suptitle(figure_title, fontsize=10, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def fitted_apic20_ei_balance_config(
    base_config: ConductanceDiffusionSimulationConfig = config_with_intermediate_synapses,
):
    fitted_parameters = read_fitted_apic20_parameters_from_csv()
    return base_config.with_property(
        membrane_capacitance=float(fitted_parameters.capacitance_nF) * nF,
        g_L=float(fitted_parameters.g_leak_nS) * nS,
        g_nmda_max=float(fitted_parameters.g_nmda_max_nS) * nS,
        e_L=float(fitted_parameters.e_leak_mV) * mV,
        resting_voltage=float(fitted_parameters.e_leak_mV) * mV,
        e_ampa=float(fitted_parameters.e_excitatory_mV) * mV,
        e_nmda=float(fitted_parameters.e_excitatory_mV) * mV,
        label="APIC20 fitted numbers under EI balance",
    )


def read_sympy_cubic_solution_expressions(
    solutions_file_name=ITERATION_16_SOLUTIONS_PATH,
):
    return sp.sympify(Path(solutions_file_name).read_text())


def assert_solution_symbols_are_bound(solutions, config):
    bound_names = set(bind_config_to_sympy_values(config))
    solution_symbols = {
        str(symbol)
        for solution in solutions
        for symbol in getattr(solution, "free_symbols", set())
    }
    unbound_names = sorted(solution_symbols - bound_names)
    if unbound_names:
        raise ValueError(f"Unbound SymPy solution symbols: {unbound_names}")
    return {
        name: bind_config_to_sympy_values(config)[name]
        for name in sorted(solution_symbols)
    }


def read_solutions(
    config: ConductanceDiffusionSimulationConfig,
    solutions_file_name=ITERATION_16_SOLUTIONS_PATH,
):
    sols = []
    solutions = read_sympy_cubic_solution_expressions(solutions_file_name)
    binding_values = assert_solution_symbols_are_bound(solutions, config)
    for solution in solutions:
        x_sol, y_sol = evaluate_sympy_solution(
            solution,
            config,
            mu_v_target=chapter1Results.mu_v,
            sigma_target=chapter1Results.sigma_v,
        )
        rate_sol = x_sol / (config.g() / nS)
        if rate_sol > 0 and y_sol > 0:
            sols.append((rate_sol, y_sol))
    return sols, binding_values


def plot_fitted_apic20_cubic_solutions(
    config: ConductanceDiffusionSimulationConfig | None = None,
    solutions_file_name=ITERATION_16_SOLUTIONS_PATH,
    output_png_path=OUTPUT_DIR / "fitted_apic20_ei_balance_cubic_solutions.png",
    output_json_path=OUTPUT_DIR / "fitted_apic20_ei_balance_cubic_solutions.json",
    r_max_hz=10_000.0,
    polynomial_full_min_hz=-500.0,
    polynomial_full_max_hz=65_000.0,
    polynomial_zoom_min_hz=-400.0,
    polynomial_zoom_max_hz=100.0,
):
    if config is None:
        config = fitted_apic20_ei_balance_config()
    sols, binding_values = read_solutions(config, solutions_file_name)
    output_png_path = Path(output_png_path)
    output_json_path = Path(output_json_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    r = np.linspace(0.1, float(r_max_hz), 3000) * Hz
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    mean_ax = axes[0, 0]
    variance_ax = axes[0, 1]
    rate_ax = axes[1, 0]
    unused_ax = axes[1, 1]
    polynomial_full_ax = axes[2, 0]
    polynomial_zoom_ax = axes[2, 1]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    polynomial_roots = []
    polynomial_full_rates = np.linspace(
        float(polynomial_full_min_hz),
        float(polynomial_full_max_hz),
        6000,
    ) * Hz
    polynomial_zoom_rates = np.linspace(
        float(polynomial_zoom_min_hz),
        float(polynomial_zoom_max_hz),
        3000,
    ) * Hz
    for index, (rate_sol, gamma) in enumerate(sols):
        color = colors[index % len(colors)]
        mean_vm = E_0(r, config, gamma)
        variance_vm = sigma_sq(r, config, gamma)
        polynomial_full_values, roots = polynomial_r(
            polynomial_full_rates,
            config,
            gamma,
            mu_v=chapter1Results.mu_v,
            sigma_v=chapter1Results.sigma_v,
        )
        polynomial_zoom_values, _zoom_roots = polynomial_r(
            polynomial_zoom_rates,
            config,
            gamma,
            mu_v=chapter1Results.mu_v,
            sigma_v=chapter1Results.sigma_v,
        )
        mean_ax.plot(r / Hz, mean_vm / mV, color=color, label=rf"$\gamma={gamma:.3g}$")
        variance_ax.plot(r / Hz, variance_vm / (mV**2), color=color, label=rf"$\gamma={gamma:.3g}$")
        rate_ax.plot(
            r / Hz,
            compute_siegert_firing_rate_for_fitted_solution(
                config,
                gamma,
                r,
                means_vm=mean_vm,
                vars_vm=variance_vm,
            )
            / Hz,
            color=color,
            label=rf"$\gamma={gamma:.3g}$",
        )
        polynomial_full_scale = float(np.max(np.abs(polynomial_full_values)))
        scaled_polynomial_full_values = (
            polynomial_full_values / polynomial_full_scale
            if polynomial_full_scale > 0.0
            else polynomial_full_values
        )
        polynomial_zoom_scale = float(np.max(np.abs(polynomial_zoom_values)))
        scaled_polynomial_zoom_values = (
            polynomial_zoom_values / polynomial_zoom_scale
            if polynomial_zoom_scale > 0.0
            else polynomial_zoom_values
        )
        polynomial_full_ax.plot(
            polynomial_full_rates[polynomial_full_rates >= 0 * Hz] / Hz,
            scaled_polynomial_full_values[polynomial_full_rates >= 0 * Hz],
            color=color,
            label=rf"$\gamma={gamma:.3g}$",
        )
        polynomial_full_ax.plot(
            polynomial_full_rates[polynomial_full_rates < 0 * Hz] / Hz,
            scaled_polynomial_full_values[polynomial_full_rates < 0 * Hz],
            color="tab:red",
            linestyle=":",
            linewidth=1.4,
            label="negative input rate",
        )
        polynomial_zoom_ax.plot(
            polynomial_zoom_rates[polynomial_zoom_rates >= 0 * Hz] / Hz,
            scaled_polynomial_zoom_values[polynomial_zoom_rates >= 0 * Hz],
            color=color,
            label=rf"$\gamma={gamma:.3g}$",
        )
        polynomial_zoom_ax.plot(
            polynomial_zoom_rates[polynomial_zoom_rates < 0 * Hz] / Hz,
            scaled_polynomial_zoom_values[polynomial_zoom_rates < 0 * Hz],
            color="tab:red",
            linestyle=":",
            linewidth=1.4,
            label="negative input rate",
        )
        root_rates_hz = root_rates_from_conductance_roots(roots, config)
        polynomial_roots.append(
            {
                "gamma": float(gamma),
                "polynomial_full_scale": polynomial_full_scale,
                "polynomial_zoom_scale": polynomial_zoom_scale,
                "root_rates_hz": root_rates_hz,
            }
        )
        full_visible_root_rates_hz = [
            root_rate_hz
            for root_rate_hz in root_rates_hz
            if float(polynomial_full_min_hz)
            <= root_rate_hz
            <= float(polynomial_full_max_hz)
        ]
        zoom_visible_root_rates_hz = [
            root_rate_hz
            for root_rate_hz in root_rates_hz
            if float(polynomial_zoom_min_hz)
            <= root_rate_hz
            <= float(polynomial_zoom_max_hz)
        ]
        for ax, visible_root_rates_hz in [
            (polynomial_full_ax, full_visible_root_rates_hz),
            (polynomial_zoom_ax, zoom_visible_root_rates_hz),
        ]:
            if not visible_root_rates_hz:
                continue
            ax.scatter(
                visible_root_rates_hz,
                np.zeros(len(visible_root_rates_hz)),
                color=color,
                edgecolors="white",
                marker="o",
                s=42,
                zorder=5,
            )
            for root_rate_hz in visible_root_rates_hz:
                ax.annotate(
                    f"{root_rate_hz:.3g} Hz",
                    (root_rate_hz, 0.0),
                    xytext=(5, 8),
                    textcoords="offset points",
                    fontsize=7,
                )

    mean_ax.axhline(chapter1Results.mu_v / mV, color="black", linestyle="--", linewidth=1.0)
    variance_ax.axhline(
        (chapter1Results.sigma_v / mV) ** 2,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    unused_ax.axis("off")
    polynomial_full_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    polynomial_zoom_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    mean_ax.set_ylabel("E0 [mV]")
    variance_ax.set_ylabel("sigma_v^2 [mV^2]")
    rate_ax.set_ylabel("Siegert rate [Hz]")
    polynomial_full_ax.set_ylabel("normalized P(r)")
    polynomial_zoom_ax.set_ylabel("normalized P(r)")
    mean_ax.set_xlabel("input rate [Hz]")
    variance_ax.set_xlabel("input rate [Hz]")
    rate_ax.set_xlabel("input rate [Hz]")
    polynomial_full_ax.set_xlabel("input rate [Hz]")
    polynomial_zoom_ax.set_xlabel("input rate [Hz]")
    mean_ax.set_title("Mean voltage from cubic EI-balance solution")
    variance_ax.set_title("Voltage variance from cubic EI-balance solution")
    rate_ax.set_title("Output firing-rate check")
    polynomial_full_ax.set_title("Cubic polynomial over full root range")
    polynomial_zoom_ax.set_title("Cubic polynomial zoom around local roots")
    polynomial_full_ax.set_xlim(float(polynomial_full_min_hz), float(polynomial_full_max_hz))
    polynomial_zoom_ax.set_xlim(float(polynomial_zoom_min_hz), float(polynomial_zoom_max_hz))
    for ax in [mean_ax, variance_ax, rate_ax, polynomial_full_ax, polynomial_zoom_ax]:
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(config.label)
    fig.tight_layout()
    fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    output_json_path.write_text(
        json.dumps(
            {
                "plot_path": str(output_png_path),
                "solutions_file_name": str(solutions_file_name),
                "binding_values": binding_values,
                "solutions": [
                    {"rate_hz": float(rate / Hz), "gamma": float(gamma)}
                    for rate, gamma in sols
                ],
                "polynomial_roots": polynomial_roots,
                "r_max_hz": float(r_max_hz),
                "polynomial_full_min_hz": float(polynomial_full_min_hz),
                "polynomial_full_max_hz": float(polynomial_full_max_hz),
                "polynomial_zoom_min_hz": float(polynomial_zoom_min_hz),
                "polynomial_zoom_max_hz": float(polynomial_zoom_max_hz),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return output_png_path, output_json_path, sols


def root_rates_from_conductance_roots(roots, config):
    root_rates_hz = []
    for root in roots:
        root_rate = root / config.g()
        root_rate_value = complex(root_rate / Hz)
        if abs(root_rate_value.imag) < 1e-8:
            root_rates_hz.append(float(root_rate_value.real))
    return sorted(root_rates_hz)


class SimulateNMDASpikesUnderEIBalanceScriptTestCases(unittest.TestCase):
    """Runnables for iteration 21: NMDA spikes under EI balance."""

    def test_ampa_nmda_brian2_model_runs_for_100_ms(self):
        """Fast smoke test: Brian2 accepts the units and integrates the model."""
        simulation_time = 100 * ms
        n_models = 4

        result = simulate_ampa_nmda_single_compartment_ensemble(
            n_models=n_models,
            simulation_time=simulation_time,
            burn_in=0 * ms,
            dt=0.1 * ms,
            random_seed=21,
            C_m=0.5 * nF,
            g_L=20 * nS,
            E_L=-65 * mV,
            E_E=0 * mV,
            tau_AMPA=2 * ms,
            w_AMPA=0.05 * nS,
            r_e=5 * Hz,
            n_e=8,
            tau_NMDA_rise=2 * ms,
            tau_NMDA_decay=100 * ms,
            alpha_NMDA=0.5 * kHz,
            g_NMDA_max=4 * nS,
            n_NMDA=8,
            w_NMDA=1.0,
            Mg=1.0,
        )

        self.assertEqual(result["voltage_mV"].shape, (n_models, 1000))
        self.assertTrue(np.all(np.isfinite(result["voltage_mV"])))
        self.assertAlmostEqual(result["time_ms"][0], 0.0)
        self.assertAlmostEqual(result["time_ms"][-1], 99.9)

    def test_read_fitted_apic20_cubic_solutions(self):
        config = fitted_apic20_ei_balance_config()
        sols, binding_values = read_solutions(config)

        self.assertGreater(len(binding_values), 0)
        self.assertIsInstance(sols, list)
        self.assertAlmostEqual(
            float(read_fitted_apic20_parameters_from_csv().capacitance_nF),
            float(config.membrane_capacitance / nF),
        )

    def test_plot_fitted_apic20_cubic_solutions(self):
        plot_path, json_path, _sols = plot_fitted_apic20_cubic_solutions()

        self.assertTrue(plot_path.exists(), str(plot_path))
        self.assertGreater(plot_path.stat().st_size, 0, str(plot_path))
        self.assertTrue(json_path.exists(), str(json_path))
        self.assertGreater(json_path.stat().st_size, 0, str(json_path))

    def test_simulate_10_000_ampa_nmda_single_compartments(
        self,
        r_e=6 * Hz,
        show_plot=True,
    ):

        # Principal experiment inputs are intentionally plain and local: edit
        # these values, then run this one test from the IDE.
        n_models = 10_000
        n_e = 800
        n_NMDA = 10
        simulation_time = 2_000 * ms
        burn_in = 100 * ms
        dt = 0.1 * ms
        random_seed = 21
        chosen_model_indices = (0, 1, 2)

        # Parameters fitted in iteration 20.  The individual assignments stay
        # here so it is obvious which fitted values drive this experiment.
        fitted = read_fitted_apic20_parameters_from_csv()
        C_m = fitted.capacitance_nF * nF
        g_L = fitted.g_leak_nS * nS
        E_L = fitted.e_leak_mV * mV
        E_E = fitted.e_excitatory_mV * mV
        tau_NMDA_rise = fitted.tau_rise_ms * ms
        tau_NMDA_decay = fitted.tau_decay_ms * ms
        alpha_NMDA = fitted.alpha_per_ms / ms
        g_NMDA_max = fitted.g_nmda_max_nS * nS
        w_NMDA = fitted.spike_weight

        # These were not fitted in iteration 20 and remain explicit inputs.
        tau_AMPA = 5 * ms
        w_AMPA = 0.05 * nS
        Mg = 1.0

        result = simulate_ampa_nmda_single_compartment_ensemble(
            n_models=n_models,
            simulation_time=simulation_time,
            burn_in=burn_in,
            dt=dt,
            random_seed=random_seed,
            C_m=C_m,
            g_L=g_L,
            E_L=E_L,
            E_E=E_E,
            tau_AMPA=tau_AMPA,
            w_AMPA=w_AMPA,
            r_e=r_e,
            n_e=n_e,
            tau_NMDA_rise=tau_NMDA_rise,
            tau_NMDA_decay=tau_NMDA_decay,
            alpha_NMDA=alpha_NMDA,
            g_NMDA_max=g_NMDA_max,
            n_NMDA=n_NMDA,
            w_NMDA=w_NMDA,
            Mg=Mg,
        )

        debug_output_dir = (
            DEBUG_DISTRIBUTION_DIR / f"{float(r_e / Hz):g}_Hz"
        )
        if saveDebugFile.enabled:
            storage = estimate_full_text_json_size(result)
            print(
                "Estimated storage for all five complete arrays: "
                f"binary={format_storage_size(storage['binary_bytes'])}, "
                f"full-text JSON≈{format_storage_size(storage['estimated_json_bytes'])}"
            )
            saved_debug_paths = save_debug_histograms(
                result,
                debug_output_dir,
                saveDebugFile,
            )
            for variable_name, paths in saved_debug_paths.items():
                for path in paths:
                    print(
                        f"Saved {variable_name} histogram: {path} "
                        f"({format_storage_size(path.stat().st_size)})"
                    )

        print(
            f"V = {result['ensemble_mean_mV']:.3f} mV, "
            f"variance = {result['ensemble_variance_mV2']:.3f} mV^2"
        )
        figure_title = (
            f"AMPA–NMDA ensemble: N={n_models:,}, r_e={float(r_e / Hz):g} Hz, "
            f"N_e={n_e}, N_NMDA={n_NMDA}, T={float(simulation_time / ms):g} ms, "
            f"burn-in={float(burn_in / ms):g} ms, dt={float(dt / ms):g} ms, seed={random_seed}\n"
            f"Membrane: C_m={float(C_m / nF):.4g} nF, g_L={float(g_L / nS):.4g} nS, "
            f"E_L={float(E_L / mV):.4g} mV, E_E={float(E_E / mV):.4g} mV; "
            f"AMPA: tau={float(tau_AMPA / ms):.4g} ms, w={float(w_AMPA / nS):.4g} nS\n"
            f"NMDA: tau_r={float(tau_NMDA_rise / ms):.4g} ms, "
            f"tau_d={float(tau_NMDA_decay / ms):.4g} ms, "
            f"alpha={float(alpha_NMDA / kHz):.4g} kHz, "
            f"g_max={float(g_NMDA_max / nS):.4g} nS, w={w_NMDA:g}, "
            f"Mg={Mg:g}; distributions: two-Gaussian mixture"
        )
        plot_path = plot_chosen_ampa_nmda_traces(
            result,
            chosen_model_indices=chosen_model_indices,
            output_path=(
                OUTPUT_DIR
                / f"ampa_nmda_three_chosen_traces_{float(r_e / Hz):g}_Hz.png"
            ),
            figure_title=figure_title,
        )

        self.assertEqual(result["voltage_mV"].shape[0], n_models)
        self.assertTrue(np.all(np.isfinite(result["voltage_mV"])))
        self.assertGreater(result["ensemble_variance_mV2"], 0.0)
        self.assertTrue(plot_path.exists(), str(plot_path))
        self.assertGreater(plot_path.stat().st_size, 0, str(plot_path))
        if show_plot:
            show_plots_non_blocking()
            return None
        plt.close("all")
        return {
            "rate_hz": float(r_e / Hz),
            "ensemble_mean_mV": result["ensemble_mean_mV"],
            "ensemble_variance_mV2": result["ensemble_variance_mV2"],
            "plot_path": str(plot_path),
        }

    def _fit_one_saved_debug_distribution(
        self,
        variable_name,
        rate_hz=6.0,
        file_format="csv",
        show_plot=True,
    ):
        histogram_path = (
            DEBUG_DISTRIBUTION_DIR
            / f"{rate_hz:g}_Hz"
            / f"{variable_name}_histogram.{file_format}"
        )
        plot_path, fit = fit_saved_debug_histogram(
            histogram_path,
            variable_name,
        )
        self.assertTrue(plot_path.exists(), str(plot_path))
        self.assertTrue(np.all(np.isfinite(fit["means"])))
        if show_plot:
            show_plots_non_blocking()
        else:
            plt.close("all")
        return plot_path, fit

    def test_fit_saved_g_nmda_distribution(self):
        return self._fit_one_saved_debug_distribution("effective_g_nmda_nS")

    def test_fit_saved_s_nmda_distribution(self):
        return self._fit_one_saved_debug_distribution("s_nmda")

    def test_fit_saved_x_nmda_distribution(self):
        return self._fit_one_saved_debug_distribution("x_nmda")

    def test_fit_saved_g_ampa_distribution(self):
        return self._fit_one_saved_debug_distribution("g_ampa_nS")

    def test_fit_saved_voltage_distribution(self):
        return self._fit_one_saved_debug_distribution("voltage_mV")

    def test_simulate_1000_ampa_nmda_single_compartments_multiple_r_es(self):
        rates_hz = np.arange(1, 11, dtype=float)
        # Every rate gets its own process and therefore its own independent
        # Brian2 scope.  n_jobs=-1 uses all CPUs available to joblib.
        completed_jobs = Parallel(
            n_jobs=2,
            backend="loky",
            return_as="generator_unordered",
        )(
            delayed(run_ampa_nmda_rate_in_worker)(rate_hz)
            for rate_hz in rates_hz
        )

        # Plotting stays in the parent process (GUI backends are not reliably
        # process-safe), but results are yielded and shown as soon as each
        # Brian2 worker finishes.  Other rates keep simulating in parallel.
        results = []
        for result in completed_jobs:
            results.append(result)
            plot_image = plt.imread(result["plot_path"])
            figure, axis = plt.subplots(figsize=(15, 14))
            axis.imshow(plot_image)
            axis.axis("off")
            axis.set_title(f"Completed Brian2 run: r_e={result['rate_hz']:g} Hz")
            figure.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

        results.sort(key=lambda result: result["rate_hz"])
        self.assertEqual([result["rate_hz"] for result in results], rates_hz.tolist())
        for result in results:
            self.assertTrue(Path(result["plot_path"]).exists(), result["plot_path"])


def run_ampa_nmda_rate_in_worker(rate_hz):
    """Pickle-safe joblib entry point for one independent Brian2 rate run."""
    test_case = SimulateNMDASpikesUnderEIBalanceScriptTestCases(
        methodName="test_simulate_10_000_ampa_nmda_single_compartments"
    )
    return test_case.test_simulate_10_000_ampa_nmda_single_compartments(
        r_e=float(rate_hz) * Hz,
        show_plot=False,
    )


if __name__ == "__main__":
    unittest.main()

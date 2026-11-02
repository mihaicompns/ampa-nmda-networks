"""Filtered-Poisson NMDA diffusion series built on the iteration-21 model.

For each finite positive scaling parameter ``M`` this module simulates

    eta_M(t) = integral H_s(t-u) [dN_M(u) - M*nu*du] / sqrt(M),

where ``N_M`` has rate ``M*nu``, ``nu = N_NMDA*r_e``, and ``H_s`` is the
fitted double-exponential NMDA kernel.  The voltage equation is

    dV = [f(V) + h(V)*eta_M(t)] dt.

Thus ``eta_M`` is a centered fluctuation of the summed NMDA gating state.  It
is not a raw Poisson increment and it is not silently replaced by white noise.
"""

import csv
import gzip
import json
import unittest
from dataclasses import dataclass
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from brian2 import Hz, ms, mV, nF, nS

from iteration_21_nmda_spikes_under_ei_balance.SimulateSummedNMDASpikesUnderEIBalanceScripts import (
    AmpaNmdaExperimentConfig,
    _evaluate_gaussian_mixture_density,
    _fit_gaussian_mixture,
    simulate_ampa_nmda_single_compartment_ensemble_with_summed_nmda_input,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output_nmda_diffusion_series"
SAVED_SCAN_DIR = OUTPUT_DIR / "saved_scan"
SAVED_SCAN_MANIFEST = SAVED_SCAN_DIR / "manifest.json"  # legacy 25 Hz scan
M_SCAN_VALUES = (0.1, 1, 10, 100, 1_000, 10_000)
VOLTAGE_HISTOGRAM_DENSITY_LIMITS = (0.0, 0.13)


def _positive_finite_M(M):
    """Return ``M`` as a positive finite float."""
    M = float(M)
    if not np.isfinite(M) or M <= 0:
        raise ValueError("M must be a positive finite number")
    return M


def _number_label(value):
    return f"{float(value):.17g}".replace("-", "m").replace(".", "p").replace("+", "")


def diffusion_scan_manifest_path(r_e, *, saved_scan_root=SAVED_SCAN_DIR):
    """Return the rate-specific manifest path for one diffusion-series scan."""
    rate_hz = float(r_e / Hz) if hasattr(r_e, "dim") else float(r_e)
    return Path(saved_scan_root) / f"r_e_{rate_hz:g}_Hz" / "manifest.json"


def _M_from_diffusion_csv_path(csv_path):
    label = Path(csv_path).name.removeprefix("diffusion_M_").removesuffix(".csv.gz")
    return _positive_finite_M(label.replace("m", "-").replace("p", "."))


def _limits_with_padding(arrays, padding_fraction=0.05):
    lower = min(float(np.min(values)) for values in arrays)
    upper = max(float(np.max(values)) for values in arrays)
    padding = (upper - lower) * padding_fraction
    if padding == 0:
        padding = max(abs(lower) * padding_fraction, 1e-12)
    return lower - padding, upper + padding


def nmda_kernel_value(time, experiment):
    r"""Evaluate the fitted NMDA impulse-response kernel ``H_s(t)``.

    ``H_s(t) = alpha*w*tau_r*tau_D/(tau_D-tau_r)
                * [exp(-t/tau_D) - exp(-t/tau_r)]`` for ``t >= 0``.
    """
    time = np.asarray(time)
    tau_r = experiment.tau_nmda_rise
    tau_d = experiment.tau_nmda_decay
    coefficient = (
        experiment.alpha_nmda
        * experiment.w_nmda
        * tau_r
        * tau_d
        / (tau_d - tau_r)
    )
    values = coefficient * (
        np.exp(-time / tau_d) - np.exp(-time / tau_r)
    )
    return np.where(time >= 0 * ms, values, 0.0)


def _nmda_kernel_power_integral_ms(experiment, order):
    r"""Return ``integral_0^inf H_s(t)^order dt`` as a Brian2 time."""
    if int(order) != order or order < 1:
        raise ValueError("order must be a positive integer")
    order = int(order)
    tau_r = experiment.tau_nmda_rise
    tau_d = experiment.tau_nmda_decay
    coefficient = (
        experiment.alpha_nmda
        * experiment.w_nmda
        * tau_r
        * tau_d
        / (tau_d - tau_r)
    )
    exponential_integral = sum(
        (-1.0) ** inner_index
        * comb(order, inner_index)
        / (
            (order - inner_index) / tau_d
            + inner_index / tau_r
        )
        for inner_index in range(order + 1)
    )
    return coefficient**order * exponential_integral


def filtered_poisson_eta_theoretical_cumulant(experiment, M, order):
    r"""Return the stationary marginal cumulant of ``eta_M``.

    For ``p >= 2``:

    ``kappa_p(eta_M) = M**(1-p/2) * nu * integral H_s(t)**p dt``.

    Centering makes the first cumulant zero.  Consequently the variance is
    independent of ``M``, while every cumulant above order two vanishes along
    the diffusion series.
    """
    M = _positive_finite_M(M)
    if int(order) != order or order < 1:
        raise ValueError("order must be a positive integer")
    order = int(order)
    if order == 1:
        return 0.0
    nu = experiment.n_nmda * experiment.r_e
    # nu * integral(H**p dt) is dimensionless; conversion happens only at
    # this scalar result boundary.
    return float(
        float(M) ** (1.0 - order / 2.0)
        * nu
        * _nmda_kernel_power_integral_ms(experiment, order)
    )


def _sample_cumulants(values, maximum_order=10):
    """Estimate cumulants through order ten from centered sample moments."""
    values = np.asarray(values, dtype=np.longdouble).ravel()
    mean = np.mean(values, dtype=np.longdouble)
    centered = values - mean
    moments = np.ones(maximum_order + 1, dtype=np.longdouble)
    moments[1] = 0.0
    for order in range(2, maximum_order + 1):
        moments[order] = np.mean(centered**order, dtype=np.longdouble)
    cumulants = np.zeros(maximum_order + 1, dtype=np.longdouble)
    cumulants[1] = mean
    for order in range(2, maximum_order + 1):
        cumulants[order] = moments[order] - sum(
            comb(order - 1, lower_order - 1)
            * cumulants[lower_order]
            * moments[order - lower_order]
            for lower_order in range(2, order)
        )
    return np.asarray(cumulants[1:], dtype=float)


def _voltage_drift_terms(voltage, eta, experiment, g_ampa=0):
    r"""Return the full AMPA--NMDA voltage drift with Brian2 units.

    ``eta`` is the centered NMDA gating fluctuation, so the NMDA gating
    variable used here is ``s_0 + eta``.  ``g_ampa`` is the summed AMPA
    conductance from all ``n_e`` excitatory sources.
    """
    sigma_mg = 1.0 / (
        1.0
        + (experiment.magnesium / 3.57)
        * np.exp(-0.08 * voltage / mV)
    )
    c_x = experiment.alpha_nmda * experiment.w_nmda * experiment.tau_nmda_rise
    nu = experiment.n_nmda * experiment.r_e
    s0 = nu * c_x * experiment.tau_nmda_decay
    nmda_scale = experiment.g_nmda_max / experiment.n_nmda / experiment.C_m
    voltage_from_excitation = voltage - experiment.E_E
    f = (
        -experiment.g_L / experiment.C_m * (voltage - experiment.E_L)
        -g_ampa / experiment.C_m * voltage_from_excitation
        -nmda_scale * sigma_mg * s0 * voltage_from_excitation
    )
    h = -nmda_scale * sigma_mg * voltage_from_excitation
    return f + h * eta


def simulate_filtered_poisson_nmda_diffusion_member(
    experiment,
    M,
    *,
    count_increments=None,
    ampa_only_count_increments=None,
    shared_ampa_count_increments=None,
):
    r"""Simulate one finite-``M`` member for all compartments in vector form.

    The double-exponential convolution is represented by its two exact
    between-event recursions.  Centered count increments enter the ``x`` state
    with amplitude ``w_NMDA/sqrt(M)``.  Brian2 units are retained until the
    result dictionary is assembled.
    """
    M = _positive_finite_M(M)
    number_of_steps_float = float(experiment.simulation_time / experiment.dt)
    number_of_steps = int(round(number_of_steps_float))
    if not np.isclose(number_of_steps_float, number_of_steps):
        raise ValueError("simulation_time must be an integer multiple of dt")

    rng = np.random.default_rng(experiment.random_seed)
    initial_uniform = rng.uniform(size=experiment.n_models)
    voltage = experiment.initial_voltage_min + initial_uniform * (
        experiment.initial_voltage_max - experiment.initial_voltage_min
    )
    nu = experiment.n_nmda * experiment.r_e
    count_mean_per_bin = float(M * nu * experiment.dt)
    expected_shape = (number_of_steps, experiment.n_models)
    if count_increments is None:
        count_increments = rng.poisson(count_mean_per_bin, size=expected_shape)
    else:
        count_increments = np.asarray(count_increments)
        if count_increments.shape != expected_shape:
            raise ValueError(
                f"count_increments must have shape {expected_shape}, "
                f"got {count_increments.shape}"
            )
        if np.any(count_increments < 0):
            raise ValueError("count_increments must be nonnegative")

    # Match the iteration-21 AMPA input exactly in distribution: n_e-n_nmda
    # AMPA-only sources plus n_nmda sources shared with NMDA.  For M=1 the
    # shared AMPA events are the very same events that drive NMDA.  For M>1,
    # the accelerated diffusion-series count cannot also drive AMPA (its rate
    # is M times too large), so a base-rate shared AMPA stream is generated.
    ampa_only_rate = (experiment.n_e - experiment.n_nmda) * experiment.r_e
    ampa_only_mean_per_bin = float(ampa_only_rate * experiment.dt)
    shared_ampa_mean_per_bin = float(nu * experiment.dt)

    def _validated_or_sampled_counts(values, mean_per_bin, name):
        if values is None:
            return rng.poisson(mean_per_bin, size=expected_shape)
        values = np.asarray(values)
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {values.shape}"
            )
        if np.any(values < 0):
            raise ValueError(f"{name} must be nonnegative")
        return values

    ampa_only_count_increments = _validated_or_sampled_counts(
        ampa_only_count_increments,
        ampa_only_mean_per_bin,
        "ampa_only_count_increments",
    )
    if shared_ampa_count_increments is None and M == 1:
        shared_ampa_count_increments = count_increments
    else:
        shared_ampa_count_increments = _validated_or_sampled_counts(
            shared_ampa_count_increments,
            shared_ampa_mean_per_bin,
            "shared_ampa_count_increments",
        )

    x_centered = np.zeros(experiment.n_models, dtype=float)
    eta = np.zeros(experiment.n_models, dtype=float)
    g_ampa = np.zeros(experiment.n_models) * experiment.w_ampa
    voltage_trajectory = np.empty((experiment.n_models, number_of_steps + 1)) * mV
    eta_trajectory = np.empty((experiment.n_models, number_of_steps + 1))
    g_ampa_trajectory = (
        np.empty((experiment.n_models, number_of_steps + 1)) * nS
    )
    voltage_trajectory[:, 0] = voltage
    eta_trajectory[:, 0] = eta
    g_ampa_trajectory[:, 0] = g_ampa

    rise_decay = np.exp(-experiment.dt / experiment.tau_nmda_rise)
    decay_decay = np.exp(-experiment.dt / experiment.tau_nmda_decay)
    ampa_decay = np.exp(-experiment.dt / experiment.tau_ampa)
    coupling_time = (
        (rise_decay - decay_decay)
        / (
            1.0 / experiment.tau_nmda_decay
            - 1.0 / experiment.tau_nmda_rise
        )
    )
    alpha_coupling = float(experiment.alpha_nmda * coupling_time)
    scaled_jump = float(experiment.w_nmda) / np.sqrt(M)

    for step_index in range(number_of_steps):
        voltage = voltage + _voltage_drift_terms(
            voltage, eta, experiment, g_ampa
        ) * experiment.dt

        # Exact propagation of the double-exponential kernel over one bin:
        #   x(t+dt)   = exp(-dt/tau_r) x(t)
        #   eta(t+dt) = exp(-dt/tau_D) eta(t)
        #             + alpha*x(t)*(e^-dt/tau_r-e^-dt/tau_D)
        #               /(1/tau_D-1/tau_r).
        eta = decay_decay * eta + alpha_coupling * x_centered
        centered_counts = count_increments[step_index] - count_mean_per_bin
        x_centered = rise_decay * x_centered + scaled_jump * centered_counts
        total_ampa_counts = (
            ampa_only_count_increments[step_index]
            + shared_ampa_count_increments[step_index]
        )
        g_ampa = ampa_decay * g_ampa + experiment.w_ampa * total_ampa_counts

        voltage_trajectory[:, step_index + 1] = voltage
        eta_trajectory[:, step_index + 1] = eta
        g_ampa_trajectory[:, step_index + 1] = g_ampa

    times = np.arange(number_of_steps + 1) * experiment.dt
    sample_mask = times >= experiment.burn_in
    voltage_mV = np.asarray(voltage_trajectory[:, sample_mask] / mV, dtype=float)
    eta_samples = np.asarray(eta_trajectory[:, sample_mask], dtype=float)
    theoretical_cumulants = np.asarray(
        [
            filtered_poisson_eta_theoretical_cumulant(experiment, M, order)
            for order in range(1, 11)
        ]
    )
    return {
        "M": M,
        "rate_hz": float(experiment.r_e / Hz),
        "N_NMDA": int(experiment.n_nmda),
        "poisson_rate_hz": float(M * nu / Hz),
        "count_mean_per_bin": count_mean_per_bin,
        "time_ms": np.asarray(times[sample_mask] / ms, dtype=float),
        "voltage_mV": voltage_mV,
        "eta": eta_samples,
        "g_ampa_nS": np.asarray(g_ampa_trajectory[:, sample_mask] / nS, dtype=float),
        "initial_voltage_mV": np.asarray(voltage_trajectory[:, 0] / mV, dtype=float),
        "eta_empirical_cumulants": _sample_cumulants(eta_samples, 10),
        "eta_theoretical_cumulants": theoretical_cumulants,
        "ensemble_mean_mV": float(np.mean(voltage_mV)),
        "ensemble_variance_mV2": float(np.var(voltage_mV)),
        "dt_ms": float(experiment.dt / ms),
        "random_seed": int(experiment.random_seed),
    }


def plot_filtered_poisson_nmda_diffusion_member(
    result,
    experiment,
    *,
    selected_unit=0,
    output_path=None,
    show_plot=True,
    voltage_limits_mV=None,
    eta_limits=None,
    cumulant_limits=None,
):
    """Plot post-burn-in voltage, filtered noise, and cumulants 1--10."""
    selected_unit = int(selected_unit)
    if not 0 <= selected_unit < experiment.n_models:
        raise IndexError("selected_unit must refer to a simulated compartment")
    if output_path is None:
        output_path = OUTPUT_DIR / (
            f"nmda_diffusion_series_M_{result['M']}_"
            f"rate_{result['rate_hz']:.2f}_Hz.png"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(18, 13))
    grid = figure.add_gridspec(
        3, 2, width_ratios=(5.0, 1.35), height_ratios=(1.2, 1.2, 1.2),
        hspace=0.40, wspace=0.08,
    )
    voltage_axis = figure.add_subplot(grid[0, 0])
    voltage_distribution_axis = figure.add_subplot(grid[0, 1], sharey=voltage_axis)
    eta_axis = figure.add_subplot(grid[1, 0])
    eta_distribution_axis = figure.add_subplot(grid[1, 1], sharey=eta_axis)
    cumulant_axis = figure.add_subplot(grid[2, :])

    voltage_axis.plot(
        result["time_ms"], result["voltage_mV"][selected_unit], linewidth=0.8
    )
    voltage_axis.set(
        title=f"Unit {selected_unit}: voltage after burn-in",
        xlabel="time [ms]", ylabel="V [mV]",
    )
    if voltage_limits_mV is not None:
        voltage_axis.set_ylim(voltage_limits_mV)
    voltage_values = result["voltage_mV"].ravel()
    voltage_distribution_axis.hist(
        voltage_values, bins=60, density=True, orientation="horizontal",
        color="0.85",
    )
    if np.ptp(voltage_values) > 0:
        fit = _fit_gaussian_mixture(voltage_values, 2)
        for component in fit["component_density"].T:
            voltage_distribution_axis.plot(component, fit["grid"], ":", linewidth=1.2)
        voltage_distribution_axis.plot(
            fit["mixture_density"], fit["grid"], color="black", linewidth=2
        )
    voltage_distribution_axis.set(
        title="Population V",
        xlabel="density",
        xlim=VOLTAGE_HISTOGRAM_DENSITY_LIMITS,
    )
    voltage_distribution_axis.tick_params(axis="y", labelleft=False)

    eta_axis.plot(
        result["time_ms"], result["eta"][selected_unit],
        color="tab:purple", linewidth=0.8,
    )
    eta_axis.set(
        title=rf"Filtered centered Poisson process $\eta^{{({result['M']})}}(t)$",
        xlabel="time [ms]", ylabel=r"$\eta^{(M)}$",
    )
    if eta_limits is not None:
        eta_axis.set_ylim(eta_limits)
    eta_distribution_axis.hist(
        result["eta"].ravel(), bins=60, density=True, orientation="horizontal",
        color="tab:purple", alpha=0.45,
    )
    eta_distribution_axis.set(title=r"Population $\eta^{(M)}$", xlabel="density")
    eta_distribution_axis.tick_params(axis="y", labelleft=False)

    orders = np.arange(1, 11)
    width = 0.36
    cumulant_axis.bar(
        orders - width / 2, result["eta_theoretical_cumulants"], width,
        color="black", alpha=0.72, label="theoretical",
    )
    cumulant_axis.bar(
        orders + width / 2, result["eta_empirical_cumulants"], width,
        color="tab:purple", alpha=0.78, label="empirical",
    )
    if np.any(result["eta_empirical_cumulants"] <= 0):
        cumulant_axis.set_yscale("symlog", linthresh=1e-6)
    cumulant_axis.set(
        title=r"Stationary marginal cumulants of $\eta^{(M)}$",
        xlabel="cumulant order p", ylabel=r"$\kappa_p$", xticks=orders,
    )
    if cumulant_limits is not None:
        cumulant_axis.set_ylim(cumulant_limits)
    cumulant_axis.legend()
    figure.suptitle(
        rf"NMDA filtered-Poisson diffusion series: $r_e={result['rate_hz']:.2f}$ Hz, "
        rf"$N_{{NMDA}}={result['N_NMDA']}$, $M={result['M']}$, "
        rf"$M\nu={result['poisson_rate_hz']:.2f}$ Hz\n"
        rf"$\eta^{{(M)}}=H_s * (dN^{{(M)}}-M\nu dt)/\sqrt{{M}}$, "
        rf"$dt={result['dt_ms']:.2f}$ ms"
    )
    for axis in figure.axes:
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    figure.subplots_adjust(top=0.90)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(figure)
    return output_path


def plot_shot_noise_vs_poisson_voltage_distributions(
    shot_noise_result,
    poisson_noise_result,
    experiment,
    *,
    output_path=None,
    show_plot=True,
    bins=80,
    voltage_limits_mV=None,
):
    """Compare post-burn-in population voltage distributions in a 2x2 figure.

    The first row shows each simulation separately with its histogram and
    two-Gaussian fit.  The second row overlays the histograms and then the
    fitted Gaussian components and mixtures.  Both comparisons use common
    bins/grids, so visible differences are not plotting-grid artifacts.
    """
    shot_values = np.asarray(shot_noise_result["voltage_mV"], dtype=float).ravel()
    poisson_values = np.asarray(
        poisson_noise_result["voltage_mV"], dtype=float
    ).ravel()
    shot_values = shot_values[np.isfinite(shot_values)]
    poisson_values = poisson_values[np.isfinite(poisson_values)]
    if shot_values.size < 2 or poisson_values.size < 2:
        raise ValueError("Both simulations need at least two finite voltage samples")

    if voltage_limits_mV is None:
        voltage_min = float(min(shot_values.min(), poisson_values.min()))
        voltage_max = float(max(shot_values.max(), poisson_values.max()))
    else:
        voltage_min, voltage_max = map(float, voltage_limits_mV)
    histogram_edges = np.linspace(voltage_min, voltage_max, int(bins) + 1)
    common_grid = np.linspace(voltage_min, voltage_max, 600)

    def fitted_densities(values):
        fit = _fit_gaussian_mixture(
            values, 2, maximum_sample_count=1_500_000
        )
        components, mixture = _evaluate_gaussian_mixture_density(
            fit, common_grid
        )
        return fit, components, mixture

    shot_fit, shot_components, shot_mixture = fitted_densities(shot_values)
    poisson_fit, poisson_components, poisson_mixture = fitted_densities(
        poisson_values
    )

    if output_path is None:
        output_path = OUTPUT_DIR / (
            f"shot_noise_vs_poisson_M_{poisson_noise_result['M']}_"
            f"rate_{poisson_noise_result['rate_hz']:.2f}_Hz.png"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    styles = (
        (axes[0, 0], shot_values, shot_components, shot_mixture,
         "tab:blue", "Brian2 shot noise"),
        (axes[0, 1], poisson_values, poisson_components, poisson_mixture,
         "tab:orange", rf"Poisson diffusion member ($M={poisson_noise_result['M']}$)"),
    )
    for axis, values, components, mixture, color, label in styles:
        axis.hist(
            values, bins=histogram_edges, density=True,
            color=color, edgecolor=color, alpha=0.50,
            label=f"histogram (n={values.size:,})",
        )
        for component_index, component in enumerate(components.T, start=1):
            axis.plot(
                common_grid, component, ":", color=color, linewidth=1.4,
                label=f"Gaussian {component_index}",
            )
        axis.plot(
            common_grid, mixture, color=color, linewidth=2.5,
            label="2-Gaussian mixture",
        )
        axis.set(title=label, xlabel="V [mV]", ylabel="density")
        axis.set_xlim(voltage_min, voltage_max)
        axis.legend(fontsize=8)

    histogram_axis = axes[1, 0]
    for values, color, label in (
        (shot_values, "tab:blue", "Brian2 shot noise"),
        (poisson_values, "tab:orange", "Poisson noise"),
    ):
        histogram_axis.hist(
            values, bins=histogram_edges, density=True,
            color=color, edgecolor=color, alpha=0.50, label=label,
        )
    histogram_axis.set(
        title="Histogram overlap", xlabel="V [mV]", ylabel="density"
    )
    histogram_axis.set_xlim(voltage_min, voltage_max)
    histogram_axis.legend(fontsize=8)

    fit_axis = axes[1, 1]
    for components, mixture, color, label in (
        (shot_components, shot_mixture, "tab:blue", "Brian2 shot noise"),
        (poisson_components, poisson_mixture, "tab:orange", "Poisson noise"),
    ):
        fit_axis.fill_between(common_grid, 0, mixture, color=color, alpha=0.50)
        for component_index, component in enumerate(components.T, start=1):
            fit_axis.plot(
                common_grid, component, ":", color=color, linewidth=1.4,
                label=f"{label} G{component_index}",
            )
        fit_axis.plot(
            common_grid, mixture, color=color, linewidth=2.5,
            label=f"{label} mixture",
        )
    fit_axis.set(
        title="Gaussian-fit overlap", xlabel="V [mV]", ylabel="density"
    )
    fit_axis.set_xlim(voltage_min, voltage_max)
    fit_axis.legend(fontsize=8)

    figure.suptitle(
        rf"NMDA-only voltage comparison: $r_e={float(experiment.r_e / Hz):.2f}$ Hz, "
        rf"$N_{{NMDA}}={experiment.n_nmda}$, $M={poisson_noise_result['M']}$, "
        rf"$dt={float(experiment.dt / ms):.2f}$ ms"
    )
    for axis in axes.ravel():
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(figure)
    return {
        "output_path": output_path,
        "histogram_edges": histogram_edges,
        "common_grid_mV": common_grid,
        "shot_noise_fit": shot_fit,
        "poisson_noise_fit": poisson_fit,
    }


def _experiment_to_json(experiment):
    """Serialize every experiment setting with explicit units."""
    return {
        "n_models": int(experiment.n_models),
        "n_e": int(experiment.n_e),
        "n_nmda": int(experiment.n_nmda),
        "simulation_time_ms": float(experiment.simulation_time / ms),
        "burn_in_ms": float(experiment.burn_in / ms),
        "dt_ms": float(experiment.dt / ms),
        "random_seed": int(experiment.random_seed),
        "r_e_hz": float(experiment.r_e / Hz),
        "C_m_nF": float(experiment.C_m / nF),
        "g_L_nS": float(experiment.g_L / nS),
        "E_L_mV": float(experiment.E_L / mV),
        "E_E_mV": float(experiment.E_E / mV),
        "tau_ampa_ms": float(experiment.tau_ampa / ms),
        "w_ampa_nS": float(experiment.w_ampa / nS),
        "tau_nmda_rise_ms": float(experiment.tau_nmda_rise / ms),
        "tau_nmda_decay_ms": float(experiment.tau_nmda_decay / ms),
        "alpha_nmda_per_ms": float(experiment.alpha_nmda * ms),
        "g_nmda_max_nS": float(experiment.g_nmda_max / nS),
        "w_nmda": float(experiment.w_nmda),
        "magnesium": float(experiment.magnesium),
        "initial_voltage_min_mV": float(experiment.initial_voltage_min / mV),
        "initial_voltage_max_mV": float(experiment.initial_voltage_max / mV),
        "magnesium_block_tau_ms": (
            None
            if experiment.magnesium_block_tau is None
            else float(experiment.magnesium_block_tau / ms)
        ),
        "with_nullcline": bool(experiment.with_nullcline),
    }


def _experiment_from_json(settings):
    magnesium_block_tau = settings["magnesium_block_tau_ms"]
    return AmpaNmdaExperimentConfig(
        n_models=int(settings["n_models"]),
        n_e=int(settings["n_e"]),
        n_nmda=int(settings["n_nmda"]),
        simulation_time=float(settings["simulation_time_ms"]) * ms,
        burn_in=float(settings["burn_in_ms"]) * ms,
        dt=float(settings["dt_ms"]) * ms,
        random_seed=int(settings["random_seed"]),
        r_e=float(settings["r_e_hz"]) * Hz,
        C_m=float(settings["C_m_nF"]) * nF,
        g_L=float(settings["g_L_nS"]) * nS,
        E_L=float(settings["E_L_mV"]) * mV,
        E_E=float(settings["E_E_mV"]) * mV,
        tau_ampa=float(settings["tau_ampa_ms"]) * ms,
        w_ampa=float(settings["w_ampa_nS"]) * nS,
        tau_nmda_rise=float(settings["tau_nmda_rise_ms"]) * ms,
        tau_nmda_decay=float(settings["tau_nmda_decay_ms"]) * ms,
        alpha_nmda=float(settings["alpha_nmda_per_ms"]) / ms,
        g_nmda_max=float(settings["g_nmda_max_nS"]) * nS,
        w_nmda=float(settings["w_nmda"]),
        magnesium=float(settings["magnesium"]),
        initial_voltage_min=float(settings["initial_voltage_min_mV"]) * mV,
        initial_voltage_max=float(settings["initial_voltage_max_mV"]) * mV,
        magnesium_block_tau=(
            None if magnesium_block_tau is None else float(magnesium_block_tau) * ms
        ),
        with_nullcline=bool(settings["with_nullcline"]),
    )


def _write_wide_csv_gz(output_path, time_ms, named_arrays):
    """Write time-major arrays without constructing a second full-size matrix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(values, dtype=float) for name, values in named_arrays.items()}
    number_of_units = next(iter(arrays.values())).shape[0]
    number_of_samples = len(time_ms)
    if any(values.shape != (number_of_units, number_of_samples) for values in arrays.values()):
        raise ValueError("all saved arrays must have shape (n_models, n_samples)")
    header = ["time_ms"] + [
        f"{name}_unit_{unit_index}"
        for name in arrays
        for unit_index in range(number_of_units)
    ]
    with gzip.open(output_path, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for sample_index, time_value in enumerate(time_ms):
            writer.writerow(
                [float(time_value)]
                + [
                    float(value)
                    for values in arrays.values()
                    for value in values[:, sample_index]
                ]
            )
    return output_path


def _read_wide_csv_gz(input_path, number_of_units, array_names):
    values = np.loadtxt(input_path, delimiter=",", skiprows=1)
    values = np.atleast_2d(values)
    result = {"time_ms": values[:, 0]}
    start = 1
    for name in array_names:
        stop = start + number_of_units
        result[name] = values[:, start:stop].T
        start = stop
    return result


def save_filtered_poisson_nmda_diffusion_scan(
    experiment,
    shot_noise_result,
    scan_results,
    *,
    output_dir=SAVED_SCAN_DIR,
):
    """Save a supplied scan sequentially as compressed CSV plus JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_result = next(iter(scan_results.values()))["result"]
    shot_time_ms = shot_noise_result.get("time_ms", first_result["time_ms"])
    shot_csv = _write_wide_csv_gz(
        output_dir / "shot_noise_voltage.csv.gz",
        shot_time_ms,
        {"voltage_mV": shot_noise_result["voltage_mV"]},
    )
    manifest = {
        "format_version": 1,
        "experiment": _experiment_to_json(experiment),
        "shot_noise_csv": shot_csv.name,
        "axis_extrema": {
            "voltage_mV": [
                float(np.min(shot_noise_result["voltage_mV"])),
                float(np.max(shot_noise_result["voltage_mV"])),
            ],
            "eta": [None, None],
            "cumulants": [None, None],
        },
        "members": [],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for M, scan_entry in scan_results.items():
        _append_filtered_poisson_nmda_diffusion_member(
            manifest_path, scan_entry["result"]
        )
    return manifest_path


def _append_filtered_poisson_nmda_diffusion_member(manifest_path, result):
    """Save one M member and update scalar scan-wide bounds."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    M = _positive_finite_M(result["M"])
    csv_path = _write_wide_csv_gz(
        manifest_path.parent / f"diffusion_M_{_number_label(M)}.csv.gz",
        result["time_ms"],
        {"voltage_mV": result["voltage_mV"], "eta": result["eta"]},
    )
    member = {
        "M": M,
        "csv": csv_path.name,
        "shape": list(result["voltage_mV"].shape),
        "rate_hz": float(result["rate_hz"]),
        "N_NMDA": int(result["N_NMDA"]),
        "poisson_rate_hz": float(result["poisson_rate_hz"]),
        "count_mean_per_bin": float(result["count_mean_per_bin"]),
        "eta_empirical_cumulants": np.asarray(
            result["eta_empirical_cumulants"], dtype=float
        ).tolist(),
        "eta_theoretical_cumulants": np.asarray(
            result["eta_theoretical_cumulants"], dtype=float
        ).tolist(),
        "ensemble_mean_mV": float(result["ensemble_mean_mV"]),
        "ensemble_variance_mV2": float(result["ensemble_variance_mV2"]),
        "dt_ms": float(result["dt_ms"]),
        "random_seed": int(result["random_seed"]),
    }
    manifest["members"] = [
        existing
        for existing in manifest["members"]
        if not np.isclose(float(existing["M"]), M, rtol=0, atol=1e-15)
    ]
    manifest["members"].append(member)
    manifest["members"].sort(key=lambda existing: float(existing["M"]))

    def extend_extrema(name, values):
        old_minimum, old_maximum = manifest["axis_extrema"][name]
        new_minimum = float(np.min(values))
        new_maximum = float(np.max(values))
        manifest["axis_extrema"][name] = [
            new_minimum if old_minimum is None else min(old_minimum, new_minimum),
            new_maximum if old_maximum is None else max(old_maximum, new_maximum),
        ]

    extend_extrema("voltage_mV", result["voltage_mV"])
    extend_extrema("eta", result["eta"])
    extend_extrema(
        "cumulants",
        np.concatenate(
            [result["eta_empirical_cumulants"], result["eta_theoretical_cumulants"]]
        ),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return member


def reconcile_filtered_poisson_nmda_diffusion_manifest(manifest_path):
    """Add saved diffusion CSVs missing after an interrupted/stale manifest write."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = _experiment_from_json(manifest["experiment"])
    known_M = {float(member["M"]) for member in manifest["members"]}
    recovered_M = []
    for csv_path in sorted(manifest_path.parent.glob("diffusion_M_*.csv.gz")):
        M = _M_from_diffusion_csv_path(csv_path)
        if any(np.isclose(M, value, rtol=0, atol=1e-15) for value in known_M):
            continue
        eta_theoretical_cumulants = np.asarray([
            filtered_poisson_eta_theoretical_cumulant(experiment, M, order)
            for order in range(1, 11)
        ])
        nu = experiment.n_nmda * experiment.r_e
        number_of_samples = int(round(float(
            (experiment.simulation_time - experiment.burn_in) / experiment.dt
        ))) + 1
        member = {
            "M": M,
            "csv": csv_path.name,
            "shape": [experiment.n_models, number_of_samples],
            "rate_hz": float(experiment.r_e / Hz),
            "N_NMDA": int(experiment.n_nmda),
            "poisson_rate_hz": float(M * nu / Hz),
            "count_mean_per_bin": float(M * nu * experiment.dt),
            # These summaries were lost with the stale manifest. NaNs make
            # that explicit without rereading multi-hundred-MB CSVs merely
            # to make their voltage samples discoverable again.
            "eta_empirical_cumulants": [None] * 10,
            "eta_theoretical_cumulants": eta_theoretical_cumulants.tolist(),
            "ensemble_mean_mV": None,
            "ensemble_variance_mV2": None,
            "dt_ms": float(experiment.dt / ms),
            "random_seed": int(experiment.random_seed),
        }
        manifest["members"].append(member)
        known_M.add(M)
        recovered_M.append(M)
    if recovered_M:
        manifest["members"].sort(key=lambda member: float(member["M"]))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return tuple(sorted(recovered_M))


def load_filtered_poisson_nmda_diffusion_member(
    manifest_path=SAVED_SCAN_MANIFEST,
    *,
    M,
    load_shot_noise=True,
):
    """Load one M member, never the complete diffusion series."""
    manifest_path = Path(manifest_path)
    reconcile_filtered_poisson_nmda_diffusion_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = _experiment_from_json(manifest["experiment"])
    requested_M = _positive_finite_M(M)
    member = next(
        (item for item in manifest["members"] if item["M"] == requested_M),
        None,
    )
    if member is None:
        raise KeyError(f"M={requested_M} is not present in {manifest_path}")
    arrays = _read_wide_csv_gz(
        manifest_path.parent / member["csv"],
        experiment.n_models,
        ("voltage_mV", "eta"),
    )
    result = arrays | {
        key: np.asarray(member[key], dtype=float)
        if key.endswith("cumulants")
        else member[key]
        for key in member
        if key not in {"csv", "shape"}
    }
    shot_noise_result = None
    if load_shot_noise:
        shot_noise_result = _read_wide_csv_gz(
            manifest_path.parent / manifest["shot_noise_csv"],
            experiment.n_models,
            ("voltage_mV",),
        )
    return experiment, shot_noise_result, result


def plot_filtered_poisson_nmda_diffusion_scan_from_saved(
    manifest_path=SAVED_SCAN_MANIFEST,
    *,
    output_dir=OUTPUT_DIR,
    show_plot=True,
):
    """Plot one saved M at a time while retaining one shared shot reference."""
    manifest_path = Path(manifest_path)
    reconcile_filtered_poisson_nmda_diffusion_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = _experiment_from_json(manifest["experiment"])
    shot_noise_result = _read_wide_csv_gz(
        manifest_path.parent / manifest["shot_noise_csv"],
        experiment.n_models,
        ("voltage_mV",),
    )

    def padded_extrema(name):
        lower, upper = manifest["axis_extrema"][name]
        return _limits_with_padding((np.asarray([lower, upper]),))

    fixed_axis_limits = manifest.get("fixed_axis_limits")
    if fixed_axis_limits is None:
        voltage_limits_mV = padded_extrema("voltage_mV")
        eta_limits = padded_extrema("eta")
        cumulant_limits = padded_extrema("cumulants")
    else:
        voltage_limits_mV = tuple(fixed_axis_limits["voltage_mV"])
        eta_limits = tuple(fixed_axis_limits["eta"])
        cumulant_limits = tuple(fixed_axis_limits["cumulants"])
    output_dir = Path(output_dir)
    plotted_results = {}
    for member in manifest["members"]:
        M = float(member["M"])
        _, _, result = load_filtered_poisson_nmda_diffusion_member(
            manifest_path, M=M, load_shot_noise=False
        )
        label = _number_label(M)
        output_path = plot_filtered_poisson_nmda_diffusion_member(
            result,
            experiment,
            output_path=output_dir / f"nmda_diffusion_series_M_{label}.png",
            show_plot=show_plot,
            voltage_limits_mV=voltage_limits_mV,
            eta_limits=eta_limits,
            cumulant_limits=cumulant_limits,
        )
        comparison = plot_shot_noise_vs_poisson_voltage_distributions(
            shot_noise_result,
            result,
            experiment,
            output_path=output_dir / f"shot_noise_vs_poisson_M_{label}.png",
            show_plot=show_plot,
            voltage_limits_mV=voltage_limits_mV,
        )
        plotted_results[M] = {
            "output_path": output_path,
            "comparison_output_path": comparison["output_path"],
            "axis_limits": {
                "voltage_mV": voltage_limits_mV,
                "eta": eta_limits,
                "cumulants": cumulant_limits,
            },
        }
        del result
    return plotted_results


def simulate_and_plot_filtered_poisson_nmda_diffusion_scan(
    experiment,
    M_values=M_SCAN_VALUES,
    *,
    saved_output_dir=None,
    plot_output_dir=OUTPUT_DIR,
    show_plot=True,
):
    """Complete simulate-save-reload-plot for each M before starting the next."""
    M_values = tuple(_positive_finite_M(M) for M in M_values)
    if not M_values:
        raise ValueError("M_values must contain at least one value")

    voltage_limits_mV = _limits_with_padding(
        (
            np.asarray(
                [
                    float(experiment.initial_voltage_min / mV),
                    float(experiment.initial_voltage_max / mV),
                    float(experiment.E_L / mV),
                    float(experiment.E_E / mV),
                ]
            ),
        )
    )
    eta_standard_deviation = np.sqrt(
        filtered_poisson_eta_theoretical_cumulant(experiment, 1, 2)
    )
    eta_half_range = 12 * eta_standard_deviation / np.sqrt(min(1.0, min(M_values)))
    eta_limits = (-eta_half_range, eta_half_range)
    theoretical_cumulants = np.asarray(
        [
            filtered_poisson_eta_theoretical_cumulant(experiment, M, order)
            for M in M_values
            for order in range(1, 11)
        ]
    )
    cumulant_limits = _limits_with_padding((theoretical_cumulants,))
    fixed_axis_limits = {
        "voltage_mV": voltage_limits_mV,
        "eta": eta_limits,
        "cumulants": cumulant_limits,
    }

    shot_noise_result = (
        simulate_ampa_nmda_single_compartment_ensemble_with_summed_nmda_input(
            experiment
        )
    )
    if saved_output_dir is None:
        saved_output_dir = diffusion_scan_manifest_path(experiment.r_e).parent
    saved_output_dir = Path(saved_output_dir)
    saved_output_dir.mkdir(parents=True, exist_ok=True)
    shot_csv = _write_wide_csv_gz(
        saved_output_dir / "shot_noise_voltage.csv.gz",
        shot_noise_result["time_ms"],
        {"voltage_mV": shot_noise_result["voltage_mV"]},
    )
    manifest_path = saved_output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "experiment": _experiment_to_json(experiment),
                "shot_noise_csv": shot_csv.name,
                "axis_extrema": {
                    "voltage_mV": [
                        float(np.min(shot_noise_result["voltage_mV"])),
                        float(np.max(shot_noise_result["voltage_mV"])),
                    ],
                    "eta": [None, None],
                    "cumulants": [None, None],
                },
                "fixed_axis_limits": {
                    name: list(limits) for name, limits in fixed_axis_limits.items()
                },
                "members": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del shot_noise_result
    saved_shot_noise_result = _read_wide_csv_gz(
        shot_csv,
        experiment.n_models,
        ("voltage_mV",),
    )

    runtime_results = {}
    for M in M_values:
        result = simulate_filtered_poisson_nmda_diffusion_member(experiment, M=M)
        member = _append_filtered_poisson_nmda_diffusion_member(
            manifest_path, result
        )
        result_shape = tuple(member["shape"])
        ensemble_mean_mV = member["ensemble_mean_mV"]
        ensemble_variance_mV2 = member["ensemble_variance_mV2"]
        del result

        _, _, saved_result = load_filtered_poisson_nmda_diffusion_member(
            manifest_path,
            M=M,
            load_shot_noise=False,
        )
        label = _number_label(M)
        output_path = plot_filtered_poisson_nmda_diffusion_member(
            saved_result,
            experiment,
            output_path=Path(plot_output_dir) / f"nmda_diffusion_series_M_{label}.png",
            show_plot=show_plot,
            voltage_limits_mV=voltage_limits_mV,
            eta_limits=eta_limits,
            cumulant_limits=cumulant_limits,
        )
        comparison = plot_shot_noise_vs_poisson_voltage_distributions(
            saved_shot_noise_result,
            saved_result,
            experiment,
            output_path=Path(plot_output_dir) / f"shot_noise_vs_poisson_M_{label}.png",
            show_plot=show_plot,
            voltage_limits_mV=voltage_limits_mV,
        )
        runtime_results[M] = {
            "manifest_path": manifest_path,
            "csv_path": saved_output_dir / member["csv"],
            "result_shape": result_shape,
            "ensemble_mean_mV": ensemble_mean_mV,
            "ensemble_variance_mV2": ensemble_variance_mV2,
            "output_path": output_path,
            "comparison_output_path": comparison["output_path"],
            "axis_limits": fixed_axis_limits,
        }
        del saved_result
    return runtime_results


def plot_M_series_toward_diffusion_approximation(
    manifest_path=SAVED_SCAN_MANIFEST,
    *,
    noise_method="stratonovich_heun",
    noise_result_csv=None,
    noise_dt=None,
    output_path=None,
    show_plot=False,
):
    """Plot finite-M fits against a simulated or saved diffusion limit.

    Euler-Maruyama methods default to the converged ``0.01 ms`` solver step
    used by the iteration-24 runnables.  The finite-M manifest timestep is an
    output/sampling setting and must not silently become the Euler solver step.

    ``noise_result_csv`` may instead name a saved wide CSV containing
    ``time_ms`` followed by one ``voltage_mV`` column per model.  When it is
    supplied, no diffusion-limit simulation is run.
    """
    from iteration_24_compartment_by_noise_process.NoiseProcessNMDASpikesScripts import (
        NoiseLimitMethod,
        simulate_ito_corrected_euler_maruyama,
        simulate_stratonovich_as_ito_euler_maruyama,
        simulate_stratonovich_heun,
    )

    noise_method = NoiseLimitMethod(noise_method)
    manifest_path = Path(manifest_path)
    reconcile_filtered_poisson_nmda_diffusion_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = _experiment_from_json(manifest["experiment"])
    if output_path is None:
        output_path = OUTPUT_DIR / (
            f"M_series_toward_{noise_method.value}.png"
        )
    shot_noise_result = _read_wide_csv_gz(
        manifest_path.parent / manifest["shot_noise_csv"],
        experiment.n_models,
        ("voltage_mV",),
    )
    voltage_limits = manifest.get("fixed_axis_limits", {}).get("voltage_mV")
    if voltage_limits is None:
        voltage_limits = manifest["axis_extrema"]["voltage_mV"]
    common_grid = np.linspace(*map(float, voltage_limits), 800)

    figure, axis = plt.subplots(figsize=(14, 8))

    def plot_gaussian_mixture(values, *, label, color, linewidth, linestyle="-"):
        fit = _fit_gaussian_mixture(
            np.asarray(values, dtype=float).ravel(),
            2,
            maximum_sample_count=1_500_000,
        )
        _, density = _evaluate_gaussian_mixture_density(fit, common_grid)
        axis.plot(
            common_grid,
            density,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )

    plot_gaussian_mixture(
        shot_noise_result["voltage_mV"],
        label="Brian2 shot-noise reference",
        color="black",
        linewidth=3.2,
    )

    included_members = sorted(
        (
            member
            for member in manifest["members"]
            if not np.isclose(float(member["M"]), 0.01)
        ),
        key=lambda member: float(member["M"]),
    )
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(included_members)))
    included_M_values = []
    for member, color in zip(included_members, colors):
        M = float(member["M"])
        _, _, saved_result = load_filtered_poisson_nmda_diffusion_member(
            manifest_path,
            M=M,
            load_shot_noise=False,
        )
        plot_gaussian_mixture(
            saved_result["voltage_mV"],
            label=rf"finite $M={M:g}$",
            color=color,
            linewidth=1.8,
        )
        included_M_values.append(M)
        del saved_result

    if noise_result_csv is not None:
        noise_result_csv = Path(noise_result_csv)
        noise_result = _read_wide_csv_gz(
            noise_result_csv,
            experiment.n_models,
            ("voltage_mV",),
        )
        noise_source = str(noise_result_csv)
        noise_dt_ms = None
    else:
        euler_methods = {
            NoiseLimitMethod.ITO_CORRECTED_EULER_MARUYAMA,
            NoiseLimitMethod.STRATONOVICH_AS_ITO_EULER_MARUYAMA,
        }
        if noise_dt is None:
            noise_dt = 0.01 * ms if noise_method in euler_methods else experiment.dt
        noise_experiment = experiment.with_property(dt=noise_dt)
        simulators = {
            NoiseLimitMethod.STRATONOVICH_HEUN: simulate_stratonovich_heun,
            NoiseLimitMethod.ITO_CORRECTED_EULER_MARUYAMA:
                simulate_ito_corrected_euler_maruyama,
            NoiseLimitMethod.STRATONOVICH_AS_ITO_EULER_MARUYAMA:
                simulate_stratonovich_as_ito_euler_maruyama,
        }
        noise_result = simulators[noise_method](noise_experiment)
        noise_source = "simulation"
        noise_dt_ms = float(noise_experiment.dt / ms)
    plot_gaussian_mixture(
        noise_result["voltage_mV"],
        label=f"diffusion limit: {noise_method.value.replace('_', ' ')}",
        color="tab:red",
        linewidth=3.2,
        linestyle="--",
    )
    del noise_result

    axis.set(
        title="Voltage-distribution convergence toward the NMDA diffusion approximation",
        xlabel="V [mV]",
        ylabel="two-Gaussian mixture density",
        xlim=tuple(voltage_limits),
    )
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axis.legend(fontsize=9, ncol=2)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(figure)
    return {
        "output_path": output_path,
        "included_M_values": tuple(included_M_values),
        "excluded_M_values": (0.01,),
        "noise_method": noise_method.value,
        "noise_source": noise_source,
        "noise_dt_ms": noise_dt_ms,
    }


def run_diffusion_approximation_no_AMPA(experiment: AmpaNmdaExperimentConfig, r_e = 25 * Hz, show_plot=True):
    experiment = experiment.with_property(w_ampa=0 * nS, r_e=r_e)

    return simulate_and_plot_filtered_poisson_nmda_diffusion_scan(
        experiment,
        M_values=M_SCAN_VALUES,
        show_plot=show_plot,
    )

class NMDADiffusionSeriesSimulationRunnables(unittest.TestCase):
    """Long manually launched simulations for PyCharm."""

    def test_simulate_one_rate_and_M(self, show_plot=True):
        r_e = 25 * Hz
        M = 100
        experiment = AmpaNmdaExperimentConfig.from_fitted_apic20(
            r_e=r_e
        ).with_property(w_ampa=0 * nS)
        scan_entry = simulate_and_plot_filtered_poisson_nmda_diffusion_scan(
            experiment,
            M_values=(M,),
            show_plot=show_plot,
        )[float(M)]
        expected_samples = int(
            round(float((experiment.simulation_time - experiment.burn_in) / experiment.dt))
        ) + 1
        self.assertEqual(
            scan_entry["result_shape"],
            (experiment.n_models, expected_samples),
        )
        self.assertTrue(scan_entry["manifest_path"].exists())
        self.assertTrue(scan_entry["output_path"].exists())
        self.assertTrue(scan_entry["comparison_output_path"].exists())
        return scan_entry

    def test_simulate_difussion_M_scan_25_hz(self):
        scan_results = run_diffusion_approximation_no_AMPA(experiment= AmpaNmdaExperimentConfig.from_fitted_apic20(), r_e = 25* Hz, show_plot=True)

        self.assertEqual(tuple(scan_results), M_SCAN_VALUES)
        for scan_entry in scan_results.values():
            self.assertTrue(scan_entry["manifest_path"].exists())
            self.assertTrue(scan_entry["output_path"].exists())
            self.assertTrue(scan_entry["comparison_output_path"].exists())
        return scan_results

    def test_simulate_difussion_M_scan_28_hz(self):
        scan_results = run_diffusion_approximation_no_AMPA(experiment= AmpaNmdaExperimentConfig.from_fitted_apic20(), r_e = 28* Hz, show_plot=True)

        self.assertEqual(tuple(scan_results), M_SCAN_VALUES)
        for scan_entry in scan_results.values():
            self.assertTrue(scan_entry["manifest_path"].exists())
            self.assertTrue(scan_entry["output_path"].exists())
            self.assertTrue(scan_entry["comparison_output_path"].exists())
        return scan_results

    def test_replot_saved_M_scan(self, show_plot=True):
        plotted_results = plot_filtered_poisson_nmda_diffusion_scan_from_saved(
            SAVED_SCAN_MANIFEST,
            show_plot=show_plot,
        )

        self.assertTrue(plotted_results)
        for scan_entry in plotted_results.values():
            self.assertTrue(scan_entry["output_path"].exists())
            self.assertTrue(scan_entry["comparison_output_path"].exists())
        return plotted_results

    def test_plot_M_series_toward_diffusion_approximation(self, show_plot=True):
        comparison = plot_M_series_toward_diffusion_approximation(
            SAVED_SCAN_MANIFEST,
            noise_method="stratonovich_heun",
            show_plot=show_plot,
        )
        comparison_2 = plot_M_series_toward_diffusion_approximation(
            SAVED_SCAN_MANIFEST,
            noise_method="ito_corrected_euler_maruyama",
            show_plot=show_plot,
        )

        comparison_3 = plot_M_series_toward_diffusion_approximation(
            SAVED_SCAN_MANIFEST,
            noise_method="stratonovich_as_ito_euler_maruyama",
            show_plot=show_plot,
        )

        self.assertEqual(comparison["included_M_values"], M_SCAN_VALUES)
        self.assertEqual(comparison_2["included_M_values"], M_SCAN_VALUES)
        self.assertEqual(comparison_3["included_M_values"], M_SCAN_VALUES)
        self.assertTrue(comparison["output_path"].exists())
        return comparison


if __name__ == "__main__":
    unittest.main()

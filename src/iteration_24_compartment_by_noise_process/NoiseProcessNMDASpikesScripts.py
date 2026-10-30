"""Vector simulations of the white-noise limit of the NMDA voltage model.

The scientific parameters and default protocol come from iteration 21.  All
calculations retain Brian2 quantities; units are stripped only when constructing
the result dictionaries used for analysis and plotting.
"""

import unittest
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from brian2 import ms, mV, nF, nS
from brian2.units import Hz

from iteration_21_nmda_spikes_under_ei_balance.SimulateSummedNMDASpikesUnderEIBalanceScripts import (
    AmpaNmdaExperimentConfig,
    _fit_gaussian_mixture,
    simulate_ampa_nmda_single_compartment_ensemble_with_summed_nmda_input,
)
from iteration_21_nmda_spikes_under_ei_balance.GeneralizedLambert import (
    find_fixed_points,
    fixed_point_current_slope,
)
OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output_noise_limit"
GAUSSIAN_FIT_SAMPLE_COUNT = 1_500_000


class NoiseLimitMethod(str, Enum):
    """The three requested interpretations/integration schemes."""

    STRATONOVICH_HEUN = "stratonovich_heun"
    ITO_CORRECTED_EULER_MARUYAMA = "ito_corrected_euler_maruyama"
    STRATONOVICH_AS_ITO_EULER_MARUYAMA = "stratonovich_as_ito_euler_maruyama"


def _noise_statistics(experiment):
    r"""Return ``s0`` and one-sided covariance integral ``D_s`` with units."""
    c_x = experiment.alpha_nmda * experiment.w_nmda * experiment.tau_nmda_rise
    total_rate = experiment.n_nmda * experiment.r_e
    s0 = total_rate * c_x * experiment.tau_nmda_decay
    d_s = total_rate * c_x**2 * experiment.tau_nmda_decay**2 / 2
    return s0, d_s


def _deterministic_drift_at_s(voltage, s_value, experiment):
    """Evaluate f(V; s) while retaining Brian2 units."""
    magnesium_slope = 0.08 / mV
    sigma_mg = 1 / (
        1
        + (experiment.magnesium / 3.57)
        * np.exp(-magnesium_slope * voltage)
    )
    leak_scale = experiment.g_L / experiment.C_m
    nmda_scale = experiment.g_nmda_max / experiment.n_nmda / experiment.C_m
    return (
        -leak_scale * (voltage - experiment.E_L)
        -nmda_scale * sigma_mg * s_value * (voltage - experiment.E_E)
    )


def evaluate_noise_limit_terms(voltage, experiment):
    r"""Evaluate ``f(V)``, ``h(V)``, and ``h'(V)`` as Brian2 quantities.

    The implemented Stratonovich equation is
    ``dV=f(V)dt + sqrt(2 D_s) h(V) o dW`` with the user's definitions of
    ``f`` and ``h``.  The summed NMDA state uses the iteration-21 per-input
    normalization ``g_nmda_max / n_nmda``.
    """
    s0, d_s = _noise_statistics(experiment)
    magnesium_slope = 0.08 / mV
    sigma_mg = 1 / (
        1
        + (experiment.magnesium / 3.57)
        * np.exp(-magnesium_slope * voltage)
    )
    nmda_scale = experiment.g_nmda_max / experiment.n_nmda / experiment.C_m
    voltage_from_excitation = voltage - experiment.E_E
    f = _deterministic_drift_at_s(voltage, s0, experiment)
    h = -nmda_scale * sigma_mg * voltage_from_excitation
    h_prime = -nmda_scale * sigma_mg * (
        1
        + magnesium_slope
        * voltage_from_excitation
        * (1 - sigma_mg)
    )
    return {
        "f": f,
        "h": h,
        "h_prime": h_prime,
        "ito_correction": d_s * h * h_prime,
        "sigma_mg": sigma_mg,
        "s0": s0,
        "d_s": d_s,
    }


def _step(voltage, dt, d_w, experiment, method):
    terms = evaluate_noise_limit_terms(voltage, experiment)
    noise_scale = np.sqrt(2 * terms["d_s"])

    if method is NoiseLimitMethod.STRATONOVICH_HEUN:
        # Stratonovich SDE:
        #   dV = f(V) dt + sqrt(2 D_s) h(V) o dW.
        #
        # Stochastic Heun predictor-corrector, with Delta W_n ~ N(0, dt):
        #   V_tilde = V_n + f(V_n) dt
        #                   + sqrt(2 D_s) h(V_n) Delta W_n,
        #   V_{n+1} = V_n
        #       + 1/2 [f(V_n) + f(V_tilde)] dt
        #       + 1/2 sqrt(2 D_s) [h(V_n) + h(V_tilde)] Delta W_n.
        # Averaging h at the two states is what gives this scheme its
        # Stratonovich interpretation; this is not an Euler-Maruyama step.
        predictor = voltage + terms["f"] * dt + noise_scale * terms["h"] * d_w
        predicted_terms = evaluate_noise_limit_terms(predictor, experiment)
        return voltage + 0.5 * (terms["f"] + predicted_terms["f"]) * dt + (
            0.5 * noise_scale * (terms["h"] + predicted_terms["h"]) * d_w
        )

    if method is NoiseLimitMethod.ITO_CORRECTED_EULER_MARUYAMA:
        # Itô equation corresponding to the Stratonovich SDE:
        #   dV = [f(V) + D_s h(V) h'(V)] dt
        #          + sqrt(2 D_s) h(V) dW.
        #
        # Euler-Maruyama discretization:
        #   V_{n+1} = V_n
        #       + [f(V_n) + D_s h(V_n) h'(V_n)] dt
        #       + sqrt(2 D_s) h(V_n) Delta W_n.
        drift = terms["f"] + terms["ito_correction"]
    elif method is NoiseLimitMethod.STRATONOVICH_AS_ITO_EULER_MARUYAMA:
        # Debug/control case: take the coefficients written in Stratonovich
        # form but apply an Itô Euler-Maruyama step without the conversion:
        #   V_{n+1} = V_n + f(V_n) dt
        #                   + sqrt(2 D_s) h(V_n) Delta W_n.
        # This deliberately omits D_s h(V_n) h'(V_n), so it does not simulate
        # the same stochastic process as either of the two methods above.
        drift = terms["f"]
    else:
        raise ValueError(f"unknown noise-limit method: {method!r}")
    return voltage + drift * dt + noise_scale * terms["h"] * d_w


def simulate_nmda_noise_limit(experiment, method, *, normal_increments=None):
    """Simulate all units simultaneously using the selected stochastic method.

    ``normal_increments`` may be supplied as ``(n_steps, n_models)`` standard
    normal variates to compare methods using exactly the same Wiener path.
    The returned trajectory includes the initial state at time zero.
    """
    method = NoiseLimitMethod(method)
    number_of_steps_float = float(experiment.simulation_time / experiment.dt)
    number_of_steps = int(round(number_of_steps_float))
    if not np.isclose(number_of_steps_float, number_of_steps):
        raise ValueError("simulation_time must be an integer multiple of dt")
    rng = np.random.default_rng(experiment.random_seed)
    initial_uniform = rng.uniform(size=experiment.n_models)
    voltage = experiment.initial_voltage_min + initial_uniform * (
        experiment.initial_voltage_max - experiment.initial_voltage_min
    )
    if normal_increments is None:
        normal_increments = rng.standard_normal(
            (number_of_steps, experiment.n_models)
        )
    else:
        normal_increments = np.asarray(normal_increments, dtype=float)
        expected_shape = (number_of_steps, experiment.n_models)
        if normal_increments.shape != expected_shape:
            raise ValueError(
                f"normal_increments must have shape {expected_shape}, "
                f"got {normal_increments.shape}"
            )

    trajectory = np.empty((experiment.n_models, number_of_steps + 1)) * mV
    trajectory[:, 0] = voltage
    sqrt_dt = np.sqrt(experiment.dt)
    for step_index in range(number_of_steps):
        d_w = normal_increments[step_index] * sqrt_dt
        voltage = _step(voltage, experiment.dt, d_w, experiment, method)
        trajectory[:, step_index + 1] = voltage

    times = np.arange(number_of_steps + 1) * experiment.dt
    sample_mask = times >= experiment.burn_in
    # This is the results boundary: analysis arrays intentionally have no units.
    voltage_mV = np.asarray(trajectory[:, sample_mask] / mV, dtype=float)
    s0, d_s = _noise_statistics(experiment)
    return {
        "method": method.value,
        "time_ms": np.asarray(times[sample_mask] / ms, dtype=float),
        "voltage_mV": voltage_mV,
        "initial_voltage_mV": np.asarray(trajectory[:, 0] / mV, dtype=float),
        "per_model_mean_mV": np.mean(voltage_mV, axis=1),
        "per_model_variance_mV2": np.var(voltage_mV, axis=1),
        "ensemble_mean_mV": float(np.mean(voltage_mV)),
        "ensemble_variance_mV2": float(np.var(voltage_mV)),
        "s0": float(s0),
        "d_s_ms": float(d_s / ms),
        "dt_ms": float(experiment.dt / ms),
        "random_seed": int(experiment.random_seed),
    }


def simulate_stratonovich_heun(experiment, *, normal_increments=None):
    """Simulate the Stratonovich equation with stochastic Heun."""
    return simulate_nmda_noise_limit(
        experiment,
        NoiseLimitMethod.STRATONOVICH_HEUN,
        normal_increments=normal_increments,
    )


def simulate_ito_corrected_euler_maruyama(experiment, *, normal_increments=None):
    """Simulate the equivalent Itô equation with Euler-Maruyama."""
    return simulate_nmda_noise_limit(
        experiment,
        NoiseLimitMethod.ITO_CORRECTED_EULER_MARUYAMA,
        normal_increments=normal_increments,
    )


def simulate_stratonovich_as_ito_euler_maruyama(
    experiment, *, normal_increments=None
):
    """Apply Euler-Maruyama to the uncorrected Stratonovich coefficients."""
    return simulate_nmda_noise_limit(
        experiment,
        NoiseLimitMethod.STRATONOVICH_AS_ITO_EULER_MARUYAMA,
        normal_increments=normal_increments,
    )


def simulate_all_noise_limit_methods(experiment):
    """Run the three methods with one common vector of Wiener increments."""
    number_of_steps = int(round(float(experiment.simulation_time / experiment.dt)))
    rng = np.random.default_rng(experiment.random_seed)
    # Consume the same initial-voltage draw used inside each simulation.
    rng.uniform(size=experiment.n_models)
    increments = rng.standard_normal((number_of_steps, experiment.n_models))
    return {
        NoiseLimitMethod.STRATONOVICH_HEUN.value: simulate_stratonovich_heun(
            experiment, normal_increments=increments
        ),
        NoiseLimitMethod.ITO_CORRECTED_EULER_MARUYAMA.value:
            simulate_ito_corrected_euler_maruyama(
                experiment, normal_increments=increments
            ),
        NoiseLimitMethod.STRATONOVICH_AS_ITO_EULER_MARUYAMA.value:
            simulate_stratonovich_as_ito_euler_maruyama(
                experiment, normal_increments=increments
            ),
    }


def evaluate_s0_voltage_nullcline(
    experiment, *, voltage_min=-75 * mV, voltage_max=0 * mV, number_of_points=2001
):
    """Return the deterministic ``s=s0`` voltage nullcline and fixed points."""
    voltage = np.linspace(0.0, 1.0, int(number_of_points)) * (
        voltage_max - voltage_min
    ) + voltage_min
    terms = evaluate_noise_limit_terms(voltage, experiment)
    voltage_mV = np.asarray(voltage / mV, dtype=float)
    dv_dt_mV_per_ms = np.asarray(terms["f"] / (mV / ms), dtype=float)

    # For the exponential white-noise reduction,
    #   Var[s] = D_s(0) = integral_0^inf Cov[s(t),s(t+tau)]dtau / tau_D.
    # These are process-noise envelopes caused by s fluctuations, not
    # confidence/credible intervals for an estimated nullcline.
    s_variance = terms["d_s"] / experiment.tau_nmda_decay
    s_standard_deviation = np.sqrt(s_variance)
    drift_envelopes = {}
    for sigma_count in (1, 2):
        for direction_name, sign in (("minus", -1), ("plus", 1)):
            s_value = terms["s0"] + sign * sigma_count * s_standard_deviation
            drift = _deterministic_drift_at_s(voltage, s_value, experiment)
            drift_envelopes[
                f"{direction_name}_{sigma_count}_sigma_dv_dt_mV_per_ms"
            ] = np.asarray(drift / (mV / ms), dtype=float)

    # At fixed s0, the open NMDA conductance is gbar_NMDA*s0/N.
    open_nmda_conductance_nS = float(
        experiment.g_nmda_max / experiment.n_nmda * terms["s0"] / nS
    )
    g_leak_nS = float(experiment.g_L / nS)
    e_leak_mV = float(experiment.E_L / mV)
    e_excitatory_mV = float(experiment.E_E / mV)
    magnesium_coefficient = float(experiment.magnesium) / 3.57
    fixed_points = np.asarray(
        find_fixed_points(
            open_nmda_conductance_nS,
            g_l=g_leak_nS,
            e_l=e_leak_mV,
            e_e=e_excitatory_mV,
            a=magnesium_coefficient,
            b=0.08,
        ),
        dtype=float,
    )
    voltage_min_mV = float(voltage_min / mV)
    voltage_max_mV = float(voltage_max / mV)
    fixed_points = fixed_points[
        (fixed_points >= voltage_min_mV) & (fixed_points <= voltage_max_mV)
    ]
    stable_mask = np.asarray(
        [
            fixed_point_current_slope(
                fixed_point,
                open_nmda_conductance_nS,
                g_l=g_leak_nS,
                e_e=e_excitatory_mV,
                a=magnesium_coefficient,
                b=0.08,
            )
            > 0
            for fixed_point in fixed_points
        ],
        dtype=bool,
    )
    return {
        "voltage_mV": voltage_mV,
        "dv_dt_mV_per_ms": dv_dt_mV_per_ms,
        "s0": float(terms["s0"]),
        "s_variance": float(s_variance),
        "s_standard_deviation": float(s_standard_deviation),
        "fixed_points_mV": fixed_points,
        "stable_fixed_points_mV": fixed_points[stable_mask],
        "unstable_fixed_points_mV": fixed_points[~stable_mask],
        **drift_envelopes,
    }


def format_noise_limit_simulation_details(experiment):
    """Return fitted cell, input, and numerical settings for the suptitle."""
    return (
        rf"$r_e={float(experiment.r_e / Hz):.2f}\,\mathrm{{Hz}}$, "
        rf"$N_{{NMDA}}={experiment.n_nmda}$, "
        rf"$N_{{units}}={experiment.n_models}$, "
        rf"$C_m={float(experiment.C_m / nF):.2f}\,\mathrm{{nF}}$, "
        rf"$g_L={float(experiment.g_L / nS):.2f}\,\mathrm{{nS}}$, "
        rf"$\bar g_{{NMDA}}={float(experiment.g_nmda_max / nS):.2f}\,\mathrm{{nS}}$, "
        rf"$E_L={float(experiment.E_L / mV):.2f}\,\mathrm{{mV}}$, "
        rf"$E_E={float(experiment.E_E / mV):.2f}\,\mathrm{{mV}}$"
        "\n"
        rf"$\tau_r={float(experiment.tau_nmda_rise / ms):.2f}\,\mathrm{{ms}}$, "
        rf"$\tau_D={float(experiment.tau_nmda_decay / ms):.2f}\,\mathrm{{ms}}$, "
        rf"$\alpha={float(experiment.alpha_nmda * ms):.2f}\,\mathrm{{ms}}^{{-1}}$, "
        rf"$w={float(experiment.w_nmda):.2f}$, "
        rf"$Mg={float(experiment.magnesium):.2f}$, "
        rf"$dt={float(experiment.dt / ms):.2f}\,\mathrm{{ms}}$, "
        rf"$T={float(experiment.simulation_time / ms):.2f}\,\mathrm{{ms}}$, "
        rf"$T_{{burn}}={float(experiment.burn_in / ms):.2f}\,\mathrm{{ms}}$, "
        rf"seed $={experiment.random_seed}$"
    )


def format_s_nullcline_formula(experiment, nullcline):
    """Return the shot-noise formulas and evaluated values used by the bands."""
    c_x = float(
        experiment.alpha_nmda
        * experiment.w_nmda
        * experiment.tau_nmda_rise
    )
    return (
        r"$C_x=\alpha w\tau_r=" rf"{c_x:.2f}$,"
        r"$s_0=N_{NMDA}r_eC_x\tau_D="
        rf"{nullcline['s0']:.2f}$,"
        r"$\sigma_s^2=D_s(0)="
        r"\frac{N_{NMDA}r_eC_x^2\tau_D}{2}="
        rf"{nullcline['s_variance']:.2f}$,"
        rf"$\sigma_s={nullcline['s_standard_deviation']:.2f}$"
    )


def plot_noise_limit_method_comparison(
    results,
    experiment,
    *,
    chosen_model_indices=(0, 1),
    output_path=OUTPUT_DIR / "noise_limit_method_comparison.png",
    show_plot=False,
):
    """Plot matched-path traces and post-burn-in voltage distributions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    for column, method in enumerate(NoiseLimitMethod):
        result = results[method.value]
        for model_index in chosen_model_indices:
            if model_index < experiment.n_models:
                axes[0, column].plot(
                    result["time_ms"],
                    result["voltage_mV"][model_index],
                    linewidth=0.8,
                    label=f"unit {model_index}",
                )
        axes[0, column].set_title(method.value.replace("_", " "))
        axes[0, column].set_xlabel("time [ms]")
        axes[0, column].set_ylabel("V [mV]")
        axes[0, column].legend(fontsize=8)
        axes[1, column].hist(result["voltage_mV"].ravel(), bins=60, density=True)
        axes[1, column].set_xlabel("V [mV]")
        axes[1, column].set_ylabel("density")
    figure.suptitle(
        "NMDA white-noise limit with matched Wiener paths\n"
        f"dt={float(experiment.dt / ms):.2f} ms, seed={experiment.random_seed}, "
        f"s0={next(iter(results.values()))['s0']:.2f}, "
        f"D_s={next(iter(results.values()))['d_s_ms']:.2f} ms"
    )
    for axis in figure.axes:
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show(block=False)
    else:
        plt.close(figure)
    return output_path


def plot_single_noise_limit_method(
    result,
    experiment,
    *,
    brian2_result,
    chosen_model_indices=(0, 1),
    output_path=None,
    show_plot=False,
):
    """Plot population/unit diagnostics and the ``s0`` voltage nullcline."""
    method = NoiseLimitMethod(result["method"])
    chosen_model_indices = tuple(chosen_model_indices)
    if len(chosen_model_indices) != 2:
        raise ValueError("chosen_model_indices must contain exactly two units")
    if any(index < 0 or index >= experiment.n_models for index in chosen_model_indices):
        raise IndexError("chosen_model_indices must refer to simulated units")
    if output_path is None:
        output_path = OUTPUT_DIR / f"{method.value}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with_nullcline = bool(experiment.with_nullcline)
    figure = plt.figure(figsize=(18, 17 if with_nullcline else 13))
    grid = figure.add_gridspec(
        4 if with_nullcline else 3,
        2,
        width_ratios=(5.0, 1.35),
        height_ratios=(1.4, 1.4, 1.0, 1.8) if with_nullcline else (1.4, 1.4, 1.0),
        hspace=0.42,
        wspace=0.08,
    )

    def plot_horizontal_distribution(axis, values, color):
        """Match iteration 21: horizontal histogram plus two-Gaussian fit."""
        values = np.asarray(values, dtype=float).ravel()
        axis.hist(
            values,
            bins=60,
            density=True,
            orientation="horizontal",
            color="0.88",
            edgecolor="none",
        )
        if values.size >= 5 and np.ptp(values) > 0:
            fit = _fit_gaussian_mixture(
                values,
                2,
                maximum_sample_count=GAUSSIAN_FIT_SAMPLE_COUNT,
            )
            for component in fit["component_density"].T:
                axis.plot(component, fit["grid"], ":", color=color, linewidth=1.3)
            axis.plot(
                fit["mixture_density"],
                fit["grid"],
                color=color,
                linewidth=2,
                label="2-Gaussian fit",
            )
            axis.legend(loc="best", fontsize=7)
        axis.set_xlabel("density")
        axis.tick_params(axis="y", labelleft=False)

    # Row 0: the full Brian2 model from iteration 21, using the same experiment.
    brian2_trace_axis = figure.add_subplot(grid[0, 0])
    brian2_distribution_axis = figure.add_subplot(
        grid[0, 1], sharey=brian2_trace_axis
    )
    for model_index in chosen_model_indices:
        brian2_trace_axis.plot(
            brian2_result["time_ms"],
            brian2_result["voltage_mV"][model_index],
            linewidth=0.8,
            label=f"unit {model_index}",
        )
    brian2_trace_axis.set(
        title="Brian2 summed-input model (iteration 21; after burn-in)",
        xlabel="time [ms]",
        ylabel="V [mV]",
    )
    brian2_trace_axis.legend(loc="best", fontsize=8)
    plot_horizontal_distribution(
        brian2_distribution_axis,
        brian2_result["voltage_mV"],
        "tab:blue",
    )
    brian2_distribution_axis.set_title(
        "Brian2 population V + 2-Gaussian fit", fontsize=9
    )

    # Row 1: like iteration 21, both traces and fitted distributions use only
    # samples retained after burn-in.
    population_trace_axis = figure.add_subplot(grid[1, 0])
    population_distribution_axis = figure.add_subplot(
        grid[1, 1], sharey=population_trace_axis
    )
    for model_index in chosen_model_indices:
        population_trace_axis.plot(
            result["time_ms"],
            result["voltage_mV"][model_index],
            linewidth=0.8,
            label=f"unit {model_index}",
        )
    population_trace_axis.set(
        title="Noise-limit voltage time course after burn-in",
        xlabel="time [ms]",
        ylabel="V [mV]",
    )
    population_trace_axis.legend(loc="best", fontsize=8)
    plot_horizontal_distribution(
        population_distribution_axis,
        result["voltage_mV"],
        "tab:green",
    )
    population_distribution_axis.set_title("Population V after burn-in", fontsize=9)

    # Row 2: only unit 0 gets its own trace/distribution row.  Unit 1 remains
    # visible in the shared Brian2 and noise-limit panels above.
    for row, model_index in enumerate(chosen_model_indices[:1], start=2):
        trace_axis = figure.add_subplot(grid[row, 0])
        distribution_axis = figure.add_subplot(grid[row, 1], sharey=trace_axis)
        color = f"C{row - 2}"
        trace_axis.plot(
            result["time_ms"],
            result["voltage_mV"][model_index],
            color=color,
            linewidth=0.9,
        )
        trace_axis.set(
            title=f"Unit {model_index}",
            xlabel="time [ms]",
            ylabel="V [mV]",
        )
        plot_horizontal_distribution(
            distribution_axis,
            result["voltage_mV"][model_index],
            color,
        )
        distribution_axis.set_title("V after burn-in", fontsize=9)

    if not with_nullcline:
        figure.suptitle(
            f"NMDA white-noise limit: {method.value.replace('_', ' ')}\n"
            f"{format_noise_limit_simulation_details(experiment)}\n"
            f"$s_0={result['s0']:.2f}$, "
            rf"$\mathcal{{D}}_s={result['d_s_ms']:.2f}\,\mathrm{{ms}}$",
            fontsize=11,
        )
        for axis in figure.axes:
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        figure.subplots_adjust(top=0.88)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        if show_plot:
            plt.show(block=False)
        else:
            plt.close(figure)
        return output_path

    # Last row: one standard axis spanning both layout columns.
    nullcline_axis = figure.add_subplot(grid[3, :])
    nullcline = evaluate_s0_voltage_nullcline(experiment)
    nullcline_axis.axhline(0.0, color="black", linewidth=1)
    minus_2_sigma = nullcline["minus_2_sigma_dv_dt_mV_per_ms"]
    plus_2_sigma = nullcline["plus_2_sigma_dv_dt_mV_per_ms"]
    nullcline_axis.fill_between(
        nullcline["voltage_mV"],
        np.minimum(minus_2_sigma, plus_2_sigma),
        np.maximum(minus_2_sigma, plus_2_sigma),
        color="gold",
        alpha=0.80,
        linewidth=0,
        label=rf"$s_0\pm2\sigma_s$ ($\sigma_s={nullcline['s_standard_deviation']:.2f}$)",
        zorder=1,
    )
    minus_1_sigma = nullcline["minus_1_sigma_dv_dt_mV_per_ms"]
    plus_1_sigma = nullcline["plus_1_sigma_dv_dt_mV_per_ms"]
    nullcline_axis.fill_between(
        nullcline["voltage_mV"],
        np.minimum(minus_1_sigma, plus_1_sigma),
        np.maximum(minus_1_sigma, plus_1_sigma),
        color="darkorange",
        alpha=0.90,
        linewidth=0,
        label=r"$s_0\pm1\sigma_s$",
        zorder=2,
    )
    nullcline_axis.plot(
        nullcline["voltage_mV"],
        nullcline["dv_dt_mV_per_ms"],
        color="tab:red",
        linewidth=2,
        label=rf"$s=s_0={nullcline['s0']:.2f}$",
        zorder=4,
    )
    nullcline_axis.plot(
        nullcline["stable_fixed_points_mV"],
        np.zeros_like(nullcline["stable_fixed_points_mV"]),
        "o",
        color="black",
        linestyle="None",
        label="stable fixed point",
        zorder=5,
    )
    if nullcline["unstable_fixed_points_mV"].size:
        nullcline_axis.plot(
            nullcline["unstable_fixed_points_mV"],
            np.zeros_like(nullcline["unstable_fixed_points_mV"]),
            "o",
            markerfacecolor="none",
            markeredgecolor="black",
            linestyle="None",
            label="unstable fixed point",
            zorder=5,
        )
    nullcline_axis.set_xlabel("V [mV]")
    nullcline_axis.set_ylabel("dV/dt [mV/ms]")
    nullcline_axis.set_xlim(-75.0, 0.0)
    nullcline_axis.set_title(
        r"Voltage nullcline at $s=s_0$"
        "\n"
        f"{format_s_nullcline_formula(experiment, nullcline)}",
        fontsize=10,
        pad=12,
    )
    nullcline_axis.legend(fontsize=8, ncol=2)
    figure.suptitle(
        f"NMDA white-noise limit: {method.value.replace('_', ' ')}\n"
        f"{format_noise_limit_simulation_details(experiment)}\n"
        f"$s_0={result['s0']:.2f}$, "
        rf"$\mathcal{{D}}_s={result['d_s_ms']:.2f}\,\mathrm{{ms}}$",
        fontsize=11,
    )
    for axis in figure.axes:
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    figure.subplots_adjust(top=0.90)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show(block=False)
    else:
        plt.close(figure)
    return output_path


def plot_population_vs_compartment_histogram(
    result,
    experiment,
    *,
    compartment_index=0,
    output_path=None,
    show_plot=False,
):
    """Compare one compartment with the population in a three-row figure."""
    method = NoiseLimitMethod(result["method"])
    compartment_index = int(compartment_index)
    if not 0 <= compartment_index < experiment.n_models:
        raise IndexError("compartment_index must refer to a simulated unit")
    if output_path is None:
        output_path = OUTPUT_DIR / (
            f"{method.value}_population_vs_compartment_{compartment_index}.png"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    population_values = np.asarray(result["voltage_mV"], dtype=float).ravel()
    compartment_values = np.asarray(
        result["voltage_mV"][compartment_index], dtype=float
    ).ravel()
    voltage_min = float(min(population_values.min(), compartment_values.min()))
    voltage_max = float(max(population_values.max(), compartment_values.max()))
    bins = np.linspace(voltage_min, voltage_max, 81)

    population_fit = _fit_gaussian_mixture(
        population_values, 2, maximum_sample_count=GAUSSIAN_FIT_SAMPLE_COUNT
    )
    compartment_fit = _fit_gaussian_mixture(
        compartment_values, 2, maximum_sample_count=GAUSSIAN_FIT_SAMPLE_COUNT
    )

    figure = plt.figure(figsize=(14, 17))
    grid = figure.add_gridspec(3, 2, height_ratios=(1, 1, 1.15), hspace=0.35)
    compartment_axis = figure.add_subplot(grid[0, :])
    population_axis = figure.add_subplot(grid[1, :], sharex=compartment_axis)
    histogram_overlap_axis = figure.add_subplot(grid[2, 0])
    gaussian_overlap_axis = figure.add_subplot(grid[2, 1], sharex=histogram_overlap_axis)

    def plot_histogram_and_fit(axis, values, fit, color, label):
        axis.hist(
            values,
            bins=bins,
            density=True,
            color=color,
            alpha=0.35,
            label=f"{label} histogram (n={values.size})",
        )
        for component_number, component_density in enumerate(
            fit["component_density"].T, start=1
        ):
            axis.plot(
                fit["grid"],
                component_density,
                ":",
                color=color,
                linewidth=1.2,
                label=f"Gaussian {component_number}",
            )
        axis.plot(
            fit["grid"],
            fit["mixture_density"],
            color=color,
            linewidth=2.5,
            label=f"{label} 2-Gaussian fit",
        )
        axis.set(ylabel="density", title=label)
        axis.legend(loc="best", fontsize=8)

    plot_histogram_and_fit(
        compartment_axis,
        compartment_values,
        compartment_fit,
        "tab:orange",
        f"Compartment {compartment_index}",
    )
    plot_histogram_and_fit(
        population_axis,
        population_values,
        population_fit,
        "0.35",
        f"All {experiment.n_models} compartments",
    )

    population_histogram, edges = np.histogram(
        population_values, bins=bins, density=True
    )
    compartment_histogram, _ = np.histogram(
        compartment_values, bins=bins, density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    histogram_overlap_axis.step(
        centers, population_histogram, where="mid", color="black", label="population"
    )
    histogram_overlap_axis.step(
        centers,
        compartment_histogram,
        where="mid",
        color="tab:orange",
        label=f"compartment {compartment_index}",
    )
    # Shade the full support of both histograms, not only their difference.
    # Opposing hatches keep both distributions identifiable in overlap regions.
    histogram_overlap_axis.fill_between(
        centers,
        0,
        population_histogram,
        step="mid",
        facecolor="0.65",
        edgecolor="black",
        hatch="///",
        linewidth=0.0,
        alpha=0.40,
        label="population area",
    )
    histogram_overlap_axis.fill_between(
        centers,
        0,
        compartment_histogram,
        step="mid",
        facecolor="tab:orange",
        edgecolor="tab:orange",
        hatch="\\\\\\",
        linewidth=0.0,
        alpha=0.40,
        label=f"compartment {compartment_index} area",
    )
    histogram_overlap_axis.fill_between(
        centers,
        population_histogram,
        compartment_histogram,
        step="mid",
        color="tab:red",
        alpha=0.28,
        label="histogram difference",
    )
    histogram_overlap_axis.set(
        title="Histogram overlap and difference",
        xlabel="V [mV]",
        ylabel="density",
    )
    histogram_overlap_axis.legend(fontsize=8)

    common_grid = np.linspace(voltage_min, voltage_max, 600)
    population_density = np.interp(
        common_grid, population_fit["grid"], population_fit["mixture_density"]
    )
    compartment_density = np.interp(
        common_grid, compartment_fit["grid"], compartment_fit["mixture_density"]
    )
    gaussian_overlap_axis.plot(
        common_grid, population_density, color="black", linewidth=2.5,
        label="population fit",
    )
    gaussian_overlap_axis.plot(
        common_grid, compartment_density, color="tab:orange", linewidth=2.5,
        label=f"compartment {compartment_index} fit",
    )
    gaussian_overlap_axis.fill_between(
        common_grid,
        population_density,
        compartment_density,
        color="tab:red",
        alpha=0.28,
        label="fitted-density difference",
    )
    differences = []
    for component_index in range(2):
        delta_mean = (
            compartment_fit["means"][component_index]
            - population_fit["means"][component_index]
        )
        delta_variance = (
            compartment_fit["standard_deviations"][component_index] ** 2
            - population_fit["standard_deviations"][component_index] ** 2
        )
        delta_weight = (
            compartment_fit["weights"][component_index]
            - population_fit["weights"][component_index]
        )
        differences.append(
            f"G{component_index + 1}: Δμ={delta_mean:.2f}, "
            f"Δvar={delta_variance:.2f}, Δw={delta_weight:.2f}"
        )
    gaussian_overlap_axis.text(
        0.02,
        0.98,
        "\n".join(differences),
        transform=gaussian_overlap_axis.transAxes,
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "tab:red"},
    )
    gaussian_overlap_axis.set(
        title="Gaussian-fit overlap and parameter differences",
        xlabel="V [mV]",
        ylabel="density",
    )
    gaussian_overlap_axis.legend(fontsize=8)

    figure.suptitle(
        f"Post-burn-in voltage comparison: {method.value.replace('_', ' ')}",
        fontsize=13,
    )
    for axis in figure.axes:
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    figure.subplots_adjust(top=0.95)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show(block=False)
    else:
        plt.close(figure)
    return output_path


def plot_gaussian_component_mean_variance_scatter(
    result,
    experiment,
    *,
    output_path=None,
    show_plot=False,
):
    """Scatter compartment Gaussian means versus variances by component."""
    method = NoiseLimitMethod(result["method"])
    if output_path is None:
        output_path = OUTPUT_DIR / f"{method.value}_gaussian_mean_variance.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    population_values = np.asarray(result["voltage_mV"], dtype=float).ravel()
    population_fit = _fit_gaussian_mixture(
        population_values, 2, maximum_sample_count=GAUSSIAN_FIT_SAMPLE_COUNT
    )
    compartment_parameters = []
    for compartment_values in np.asarray(result["voltage_mV"], dtype=float):
        try:
            fit = _fit_gaussian_mixture(
                compartment_values,
                2,
                maximum_sample_count=GAUSSIAN_FIT_SAMPLE_COUNT,
            )
        except ValueError:
            continue
        compartment_parameters.append(
            (
                fit["means"],
                fit["standard_deviations"] ** 2,
                fit["weights"],
            )
        )
    if not compartment_parameters:
        raise ValueError("No compartment had enough samples for Gaussian fitting")
    means, variances, weights = (
        np.asarray(values) for values in zip(*compartment_parameters)
    )

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    for component_index, axis in enumerate(axes):
        scatter = axis.scatter(
            means[:, component_index],
            variances[:, component_index],
            c=weights[:, component_index],
            s=18 + 90 * weights[:, component_index],
            cmap="viridis",
            alpha=0.55,
            edgecolors="none",
            label="individual compartments",
        )
        population_mean = population_fit["means"][component_index]
        population_variance = (
            population_fit["standard_deviations"][component_index] ** 2
        )
        axis.scatter(
            [population_mean],
            [population_variance],
            marker="*",
            s=360,
            color="red",
            edgecolor="black",
            linewidth=1.8,
            label=(
                f"population: μ={population_mean:.2f}, "
                f"var={population_variance:.2f}, "
                f"w={population_fit['weights'][component_index]:.2f}"
            ),
            zorder=5,
        )
        axis.set(
            title=f"Gaussian {component_index + 1}",
            xlabel="component mean [mV]",
            ylabel="component variance [mV²]",
        )
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.legend(loc="best", fontsize=8)
        figure.colorbar(scatter, ax=axis, label="component weight")
    figure.suptitle(
        f"Across-compartment Gaussian parameters: {method.value.replace('_', ' ')}",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show(block=False)
    else:
        plt.close(figure)
    return output_path


class NoiseProcessNMDASpikesScriptTestCases(unittest.TestCase):
    """Manual long-running entry point using the iteration-21 fitted protocol."""

    def _plot_and_check(self, result, brian2_result, experiment, show_plot):
        method = NoiseLimitMethod(result["method"])
        output_path = plot_single_noise_limit_method(
            result,
            experiment,
            brian2_result=brian2_result,
            output_path=OUTPUT_DIR / f"{method.value}.png",
            show_plot=show_plot,
        )
        self.assertTrue(output_path.exists())
        histogram_comparison_path = plot_population_vs_compartment_histogram(
            result,
            experiment,
            compartment_index=0,
            output_path=(
                OUTPUT_DIR / f"{method.value}_population_vs_compartment_0.png"
            ),
            show_plot=show_plot,
        )
        self.assertTrue(histogram_comparison_path.exists())
        gaussian_scatter_path = plot_gaussian_component_mean_variance_scatter(
            result,
            experiment,
            output_path=OUTPUT_DIR / f"{method.value}_gaussian_mean_variance.png",
            show_plot=show_plot,
        )
        self.assertTrue(gaussian_scatter_path.exists())
        return result, output_path

    def test_simulate_and_plot_stratonovich_heun(self, show_plot=True):
        """Create settings, run stochastic Heun, then plot its result."""
        experiment = (
            AmpaNmdaExperimentConfig.from_fitted_apic20()
            .with_property(r_e=25 * Hz)
        )
        brian2_result = (
            simulate_ampa_nmda_single_compartment_ensemble_with_summed_nmda_input(
                experiment
            )
        )
        result = simulate_stratonovich_heun(experiment)
        return self._plot_and_check(result, brian2_result, experiment, show_plot)

        """Run only the Stratonovich stochastic-Heun simulation."""
        self.first_sim(show_plot)

    def test_simulate_and_plot_ito_corrected_euler_maruyama(
        self, show_plot=True
    ):
        """Create settings, run corrected Itô Euler-Maruyama, then plot."""
        experiment = (
            AmpaNmdaExperimentConfig.from_fitted_apic20()
            .with_property(r_e=25 * Hz, dt=0.01 * ms)
        )
        brian2_result = (
            simulate_ampa_nmda_single_compartment_ensemble_with_summed_nmda_input(
                experiment
            )
        )
        result = simulate_ito_corrected_euler_maruyama(experiment)
        return self._plot_and_check(result, brian2_result, experiment, show_plot)

    def test_simulate_and_plot_stratonovich_as_ito_euler_maruyama(
        self, show_plot=True
    ):
        """Create settings, run uncorrected Euler-Maruyama, then plot."""
        experiment = (
            AmpaNmdaExperimentConfig.from_fitted_apic20()
            .with_property(r_e=25 * Hz, dt=0.01 * ms)
        )
        brian2_result = (
            simulate_ampa_nmda_single_compartment_ensemble_with_summed_nmda_input(
                experiment
            )
        )
        result = simulate_stratonovich_as_ito_euler_maruyama(experiment)
        return self._plot_and_check(result, brian2_result, experiment, show_plot)

    def test_simulate_and_plot_all_methods(self, show_plot=True):
        """Aggregate the three concrete simulation/plot workflows."""
        self.test_simulate_and_plot_stratonovich_heun(show_plot)
        self.test_simulate_and_plot_ito_corrected_euler_maruyama(show_plot)
        self.simulate_ito_corrected_euler_maruyama(show_plot)


if __name__ == "__main__":
    unittest.main()

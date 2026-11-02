"""Shared theoretical/empirical shot-noise cumulants and plot text."""

import math

import numpy as np


def exponential_shot_noise_cumulants(rate_hz, tau_ms, amplitude):
    """Return stationary cumulants 1..4 for exponential Poisson shot noise."""
    rate_tau = float(rate_hz) * float(tau_ms) / 1000.0
    return np.asarray(
        [rate_tau * float(amplitude) ** order / order for order in range(1, 5)]
    )


def filtered_nmda_shot_noise_cumulants(
    rate_hz, tau_rise_ms, tau_decay_ms, alpha_per_ms, weight
):
    """Return cumulants 1..4 of s for linear Poisson-driven NMDA kinetics."""
    tau_rise = float(tau_rise_ms) / 1000.0
    tau_decay = float(tau_decay_ms) / 1000.0
    alpha = float(alpha_per_ms) * 1000.0
    amplitude_scale = (
        alpha * float(weight) * tau_rise * tau_decay / (tau_decay - tau_rise)
    )
    return np.asarray(
        [
            float(rate_hz)
            * amplitude_scale**order
            * sum(
                (-1) ** index
                * math.comb(order, index)
                / ((order - index) / tau_decay + index / tau_rise)
                for index in range(order + 1)
            )
            for order in range(1, 5)
        ]
    )


def standardized_filtered_nmda_cumulants(rate_hz, tau_rise_ms, tau_decay_ms):
    """Return the closed-form stationary skewness and excess kurtosis of s."""
    rate_per_ms = float(rate_hz) / 1000.0
    tau_rise = float(tau_rise_ms)
    tau_decay = float(tau_decay_ms)
    skewness = (
        2.0 ** 2.5 * (tau_rise + tau_decay) ** 1.5
        / (
            3.0
            * np.sqrt(rate_per_ms)
            * (tau_decay + 2.0 * tau_rise)
            * (2.0 * tau_decay + tau_rise)
        )
    )
    excess_kurtosis = (
        3.0 * (tau_rise + tau_decay)
        / (
            rate_per_ms
            * (tau_decay + 3.0 * tau_rise)
            * (3.0 * tau_decay + tau_rise)
        )
    )
    return float(skewness), float(excess_kurtosis)


def estimated_cumulants(values, max_samples=500_000, random_seed=0):
    """Estimate cumulants 1..4 from a bounded deterministic sample."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size > max_samples:
        indices = np.random.default_rng(random_seed).choice(
            values.size, size=max_samples, replace=False
        )
        values = values[indices]
    centered = values - np.mean(values)
    variance = float(np.mean(centered**2))
    return np.asarray(
        [
            float(np.mean(values)),
            variance,
            float(np.mean(centered**3)),
            float(np.mean(centered**4) - 3.0 * variance**2),
        ]
    )


def summarize_cumulants(theoretical, empirical):
    """Bundle raw and standardized theoretical and empirical cumulants."""
    theoretical = np.asarray(theoretical, dtype=float)
    empirical = np.asarray(empirical, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "theoretical_cumulants": theoretical,
            "empirical_cumulants": empirical,
            "theoretical_skewness": float(theoretical[2] / theoretical[1] ** 1.5),
            "empirical_skewness": float(empirical[2] / empirical[1] ** 1.5),
            "theoretical_excess_kurtosis": float(theoretical[3] / theoretical[1] ** 2),
            "empirical_excess_kurtosis": float(empirical[3] / empirical[1] ** 2),
        }


def compute_shot_noise_gaussianity_summary(
    result,
    *,
    ampa_input_rate_hz,
    nmda_input_rate_hz,
    ampa_tau_ms,
    ampa_weight_nS,
    nmda_tau_rise_ms,
    nmda_tau_decay_ms,
    nmda_alpha_per_ms,
    nmda_weight,
):
    """Compare theory and data for x, s, effective g_NMDA, and g_AMPA."""
    x_theory = exponential_shot_noise_cumulants(
        nmda_input_rate_hz, nmda_tau_rise_ms, nmda_weight
    )
    s_theory = filtered_nmda_shot_noise_cumulants(
        nmda_input_rate_hz,
        nmda_tau_rise_ms,
        nmda_tau_decay_ms,
        nmda_alpha_per_ms,
        nmda_weight,
    )
    # Evaluate the supplied closed forms explicitly as a contract on s theory.
    s_skewness, s_excess = standardized_filtered_nmda_cumulants(
        nmda_input_rate_hz, nmda_tau_rise_ms, nmda_tau_decay_ms
    )
    pairs = {
        "x": (x_theory, result["x_nmda"]),
        "s": (s_theory, result["s_nmda"]),
        "g_AMPA": (
            exponential_shot_noise_cumulants(
                ampa_input_rate_hz, ampa_tau_ms, ampa_weight_nS
            ),
            result["g_ampa_nS"],
        ),
    }
    summary = {
        name: summarize_cumulants(theory, estimated_cumulants(values))
        for name, (theory, values) in pairs.items()
    }
    g_nmda_empirical = estimated_cumulants(result["effective_g_nmda_nS"])
    with np.errstate(divide="ignore", invalid="ignore"):
        summary["g_NMDA"] = {
            "empirical_cumulants": g_nmda_empirical,
            "empirical_skewness": float(
                g_nmda_empirical[2] / g_nmda_empirical[1] ** 1.5
            ),
            "empirical_excess_kurtosis": float(
                g_nmda_empirical[3] / g_nmda_empirical[1] ** 2
            ),
        }
    summary["s"]["theoretical_skewness"] = s_skewness
    summary["s"]["theoretical_excess_kurtosis"] = s_excess
    return summary


def format_shot_noise_gaussianity_rows(summary):
    """Return common display text aligned with V, g_AMPA, g_NMDA, s, x."""
    rows = ["not evaluated for V"]
    for name in ("g_AMPA", "g_NMDA", "s", "x"):
        values = summary[name]
        symbol = {
            "g_AMPA": r"g_{AMPA}",
            "g_NMDA": r"g_{NMDA}",
            "s": "s",
            "x": "x",
        }[name]
        if name == "g_NMDA":
            third_values = f"data {values['empirical_skewness']:>8.3g}"
            fourth_values = (
                f"data {values['empirical_excess_kurtosis']:>8.3g}"
            )
        else:
            third_values = (
                f"theory {values['theoretical_skewness']:>8.3g}  "
                f"data {values['empirical_skewness']:>8.3g}"
            )
            fourth_values = (
                f"theory {values['theoretical_excess_kurtosis']:>8.3g}  "
                f"data {values['empirical_excess_kurtosis']:>8.3g}"
            )
        rows.append(
            rf"$\hat{{\kappa}}_3[{symbol}]$ (skewness)" "\n"
            f"  {third_values}    \n"
            rf"$\hat{{\kappa}}_4[{symbol}]$ (excess kurtosis)" "\n"
            f"  {fourth_values}"
        )
    return rows

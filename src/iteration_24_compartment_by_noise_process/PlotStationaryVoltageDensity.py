"""Plot the normalized stationary NMDA voltage density.

The manual unittest runnable at the bottom is intended to be launched from
PyCharm.  All voltage calculations use mV and all time calculations use ms.
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import Hz, kHz, ms, mV, nF, nS
from scipy.integrate import cumulative_trapezoid

from iteration_21_nmda_spikes_under_ei_balance.SimulateSummedNMDASpikesUnderEIBalanceScripts import (
    AmpaNmdaExperimentConfig,
)


# The lower bound approximates -infinity.  Move it left and increase the point
# count to check convergence of the normalization and the visible density.
VOLTAGE_MIN_MV = -120.0
VOLTAGE_POINTS = 20_001
EXCITATORY_REVERSAL_MARGIN_MV = 0.01


def stationary_density_terms(voltage_mV, experiment):
    """Return f(V), g(V), sigma^2 and the magnesium unblock factor.

    This uses
        dV = f(V) dt + sigma g(V) o dW,
    where sigma^2 = 2 D_s.  Thus f is in mV/ms, g is in mV/ms and sigma^2
    is in ms when the supplied voltage is expressed in mV.
    """
    voltage_mV = np.asarray(voltage_mV, dtype=float)
    e_l_mV = float(experiment.E_L / mV)
    e_e_mV = float(experiment.E_E / mV)
    capacitance_nF = float(experiment.C_m / nF)
    leak_per_ms = float(experiment.g_L / nS) / capacitance_nF / 1000.0
    nmda_per_ms = (
        float(experiment.g_nmda_max / nS)
        / int(experiment.n_nmda)
        / capacitance_nF
        / 1000.0
    )

    magnesium_unblock = 1.0 / (
        1.0
        + (float(experiment.magnesium) / 3.57)
        * np.exp(-0.08 * voltage_mV)
    )
    total_rate_per_ms = float(experiment.n_nmda * experiment.r_e / kHz)
    c_x = (
        float(experiment.alpha_nmda / kHz)
        * float(experiment.w_nmda)
        * float(experiment.tau_nmda_rise / ms)
    )
    s_0 = total_rate_per_ms * c_x * float(experiment.tau_nmda_decay / ms)
    tau_decay_ms = float(experiment.tau_nmda_decay / ms)
    # sigma = C_x sqrt(N r_e) tau_D, exactly as in the derivation.
    sigma_sqrt_ms = c_x * np.sqrt(total_rate_per_ms) * tau_decay_ms
    sigma_squared_ms = sigma_sqrt_ms**2

    f_mV_per_ms = (
        -leak_per_ms * (voltage_mV - e_l_mV)
        -nmda_per_ms
        * magnesium_unblock
        * s_0
        * (voltage_mV - e_e_mV)
    )
    g_mV_per_ms = (
        -nmda_per_ms * magnesium_unblock * (voltage_mV - e_e_mV)
    )
    return {
        "f_mV_per_ms": f_mV_per_ms,
        "g_mV_per_ms": g_mV_per_ms,
        "sigma_sqrt_ms": sigma_sqrt_ms,
        "sigma_squared_ms": sigma_squared_ms,
        "s_0": s_0,
        "magnesium_unblock": magnesium_unblock,
    }


def compute_stationary_voltage_density(
    experiment,
    *,
    voltage_min_mV=VOLTAGE_MIN_MV,
    number_of_points=VOLTAGE_POINTS,
    reversal_margin_mV=EXCITATORY_REVERSAL_MARGIN_MV,
):
    r"""Compute the normalized density ``r_0(V)`` on a voltage grid.

    The exponent is evaluated relative to finite ``E_L``:

        2/sigma^2 * integral(E_L, V) f(u)/g(u)^2 du.

    Replacing the formally written lower limit ``-infinity`` by ``E_L`` only
    adds a V-independent constant, which cancels during normalization.
    """
    e_l_mV = float(experiment.E_L / mV)
    e_e_mV = float(experiment.E_E / mV)
    voltage_max_mV = e_e_mV - float(reversal_margin_mV)
    if not voltage_min_mV < e_l_mV < voltage_max_mV:
        raise ValueError("the grid must contain E_L and remain below E_E")
    if number_of_points < 3:
        raise ValueError("number_of_points must be at least 3")

    # Including E_L exactly prevents interpolation error in the reference.
    voltage_mV = np.unique(
        np.append(
            np.linspace(voltage_min_mV, voltage_max_mV, number_of_points),
            e_l_mV,
        )
    )
    terms = stationary_density_terms(voltage_mV, experiment)
    f_over_g_squared = terms["f_mV_per_ms"] / terms["g_mV_per_ms"] ** 2
    primitive_ms = cumulative_trapezoid(
        f_over_g_squared, voltage_mV, initial=0.0
    )
    reference_index = int(np.searchsorted(voltage_mV, e_l_mV))
    integral_from_e_l_ms = primitive_ms - primitive_ms[reference_index]

    log_unnormalized = (
        -np.log(np.abs(terms["g_mV_per_ms"]))
        + 2.0 * integral_from_e_l_ms / terms["sigma_squared_ms"]
    )
    log_shift = float(np.max(log_unnormalized))
    shifted_density = np.exp(log_unnormalized - log_shift)
    shifted_z_mV = float(np.trapezoid(shifted_density, voltage_mV))
    density_per_mV = shifted_density / shifted_z_mV

    return {
        "voltage_mV": voltage_mV,
        "density_per_mV": density_per_mV,
        "log_unnormalized": log_unnormalized,
        "log_Z": log_shift + np.log(shifted_z_mV),
        "normalization_N": np.exp(-log_shift) / shifted_z_mV,
        "sigma_sqrt_ms": terms["sigma_sqrt_ms"],
        "sigma_squared_ms": terms["sigma_squared_ms"],
        "s_0": terms["s_0"],
    }


def plot_stationary_voltage_density(result, *, show_plot=True):
    """Plot the normalized density and, by default, display the window."""
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        result["voltage_mV"],
        result["density_per_mV"],
        linewidth=2.0,
        color="tab:blue",
    )
    axis.set(
        xlabel="V [mV]",
        ylabel=r"$r_0(V)$ [mV$^{-1}$]",
        title=(
            "Normalized stationary voltage density\n"
            rf"$\sigma={result['sigma_sqrt_ms']:.4g}\sqrt{{\mathrm{{ms}}}}$, "
            rf"$s_0={result['s_0']:.4g}$"
        ),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    if show_plot:
        plt.show()
    return figure, axis


def compute_and_plot_default_density(*, show_plot=True):
    """Use the iteration-24 fitted APIC20 setup and plot its density."""
    experiment = AmpaNmdaExperimentConfig.from_fitted_apic20().with_property(
        r_e=25 * Hz,
        w_ampa=0 * nS,
    )
    result = compute_stationary_voltage_density(experiment)
    figure, axis = plot_stationary_voltage_density(result, show_plot=show_plot)
    return result, figure, axis


class PlotStationaryVoltageDensityRunnables(unittest.TestCase):
    """Manual PyCharm runnable which computes and shows the density plot."""

    def test_compute_and_show_stationary_voltage_density(self):
        result, _, _ = compute_and_plot_default_density(show_plot=True)
        normalization = np.trapezoid(
            result["density_per_mV"], result["voltage_mV"]
        )
        self.assertAlmostEqual(1.0, float(normalization), places=10)
        self.assertTrue(np.all(np.isfinite(result["density_per_mV"])))
        self.assertTrue(np.all(result["density_per_mV"] >= 0.0))


if __name__ == "__main__":
    unittest.main()

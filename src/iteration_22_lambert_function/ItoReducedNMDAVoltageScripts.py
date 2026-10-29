"""Plot the one-dimensional Itô reduction of NMDA-driven membrane voltage."""

import unittest
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from brian2 import Hz, ms, mV, nF, nS

from iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20NullclineScripts import (
    read_fitted_apic20_parameters_from_csv,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"

# Editable scientific configuration for the manual runnable.
INPUT_RATE_HZ = 25.0
NUMBER_OF_NMDA_INPUTS = 10
MAGNESIUM_BLOCK_A = 0.25
MAGNESIUM_SLOPE_PER_MV = 0.08
VOLTAGE_MIN_MV = -90.0
VOLTAGE_MAX_MV = 10.0
DRIFT_Y_LIMITS_MV_PER_MS = (-20.0, 20.0)


@dataclass(frozen=True)
class ItoReducedNMDAParameters:
    """Parameters of the reduced Itô voltage equation in ms, mV, nS, nF."""

    capacitance_nF: float
    g_leak_nS: float
    e_leak_mV: float
    e_excitatory_mV: float
    g_nmda_max_nS: float
    number_of_nmda_inputs: int
    magnesium_block_a: float
    magnesium_slope_per_mV: float
    s0: float
    d_s_ms: float
    input_rate_hz: float
    tau_rise_ms: float
    tau_decay_ms: float
    alpha_per_ms: float
    spike_weight: float
    s_variance: float
    c_x: float
    s_autocovariance_zero_lag: float

    @classmethod
    def from_fitted_apic20(
        cls,
        *,
        input_rate_hz=INPUT_RATE_HZ,
        number_of_nmda_inputs=NUMBER_OF_NMDA_INPUTS,
        magnesium_block_a=MAGNESIUM_BLOCK_A,
        magnesium_slope_per_mV=MAGNESIUM_SLOPE_PER_MV,
    ):
        """Use the fitted APIC20 cell and filtered-shot-noise s statistics."""
        fitted = read_fitted_apic20_parameters_from_csv()
        input_rate = float(input_rate_hz) * Hz
        tau_rise = 2.0 * ms
        tau_decay = 50.0 * ms
        alpha = 0.46 / ms
        spike_weight = 1.0
        c_x = alpha * spike_weight * tau_rise
        s0 = number_of_nmda_inputs * input_rate * c_x * tau_decay
        s_autocovariance_zero_lag = (
            number_of_nmda_inputs * input_rate * c_x**2 * tau_decay / 2.0
        )
        # The Itô white-noise intensity is the one-sided integral of D_s(tau):
        # integral_0^infinity D_s(tau) d tau = D_s(0)*tau_D.
        d_s = s_autocovariance_zero_lag * tau_decay
        return cls(
            capacitance_nF=float(fitted.capacitance_nF) * nF,
            g_leak_nS=3.056 * nS,
            e_leak_mV=-70.0 * mV,
            e_excitatory_mV=0.0 * mV,
            g_nmda_max_nS=13.92 * nS,
            number_of_nmda_inputs=int(number_of_nmda_inputs),
            magnesium_block_a=float(magnesium_block_a),
            magnesium_slope_per_mV=float(magnesium_slope_per_mV) / mV,
            s0=float(s0),
            d_s_ms=d_s,
            input_rate_hz=input_rate,
            tau_rise_ms=tau_rise,
            tau_decay_ms=tau_decay,
            alpha_per_ms=alpha,
            spike_weight=spike_weight,
            s_variance=float(s_autocovariance_zero_lag),
            c_x=float(c_x),
            s_autocovariance_zero_lag=float(s_autocovariance_zero_lag),
        )


def compute_s_autocovariance(lag_ms, parameters):
    r"""Return $D_s(\tau)=Nr_e C_x^2\tau_D e^{-|\tau|/\tau_D}/2$."""
    lag_ms = np.asarray(lag_ms, dtype=float)
    total_rate_per_ms = (
        parameters.number_of_nmda_inputs * float(parameters.input_rate_hz / (1 / ms))
    )
    return (
        total_rate_per_ms
        * parameters.c_x**2
        * float(parameters.tau_decay_ms / ms)
        / 2.0
        * np.exp(-np.abs(lag_ms) / float(parameters.tau_decay_ms / ms))
    )


def compute_ito_voltage_terms(voltage_mV, parameters):
    """Evaluate the displayed Itô drift and signed noise amplitude."""
    # Strip Brian2 units once, at the numerical boundary. From here onward the
    # normalized system is mV, ms, nS and nF.
    capacitance_nF = float(parameters.capacitance_nF / nF)
    g_leak_nS = float(parameters.g_leak_nS / nS)
    e_leak_mV = float(parameters.e_leak_mV / mV)
    e_excitatory_mV = float(parameters.e_excitatory_mV / mV)
    g_nmda_max_nS = float(parameters.g_nmda_max_nS / nS)
    magnesium_slope_per_mV = float(parameters.magnesium_slope_per_mV * mV)
    tau_rise_ms = float(parameters.tau_rise_ms / ms)
    tau_decay_ms = float(parameters.tau_decay_ms / ms)
    alpha_per_ms = float(parameters.alpha_per_ms * ms)
    voltage_mV = np.asarray(voltage_mV, dtype=float)
    c_x = alpha_per_ms * parameters.spike_weight * tau_rise_ms
    d_s_zero_lag = float(compute_s_autocovariance(0.0, parameters))
    ito_noise_intensity_ms = d_s_zero_lag * tau_decay_ms
    sigma_mg = 1.0 / (
        1.0
        + parameters.magnesium_block_a
        * np.exp(-magnesium_slope_per_mV * voltage_mV)
    )
    # nS/nF = 1/s; divide by 1000 to express all voltage rates per ms.
    leak_scale_per_ms = g_leak_nS / capacitance_nF / 1000.0
    nmda_scale_per_ms = (
        g_nmda_max_nS
        / parameters.number_of_nmda_inputs
        / capacitance_nF
        / 1000.0
    )
    voltage_from_excitation_mV = voltage_mV - e_excitatory_mV
    deterministic_drift = (
        -leak_scale_per_ms * (voltage_mV - e_leak_mV)
        - nmda_scale_per_ms
        * sigma_mg
        * parameters.s0
        * voltage_from_excitation_mV
    )
    ito_correction = (
        ito_noise_intensity_ms
        * nmda_scale_per_ms**2
        * sigma_mg**2
        * voltage_from_excitation_mV
        * (
            1.0
            + magnesium_slope_per_mV
            * voltage_from_excitation_mV
            * (1.0 - sigma_mg)
        )
    )
    noise_amplitude = (
        -np.sqrt(2.0 * ito_noise_intensity_ms)
        * nmda_scale_per_ms
        * sigma_mg
        * voltage_from_excitation_mV
    )
    return {
        "voltage_mV": voltage_mV,
        "sigma_mg": sigma_mg,
        "c_x": c_x,
        "c_x_units": "dimensionless",
        "d_s_zero_lag": d_s_zero_lag,
        "d_s_zero_lag_units": "dimensionless^2",
        "ito_noise_intensity_ms": ito_noise_intensity_ms,
        "ito_noise_intensity_units": "ms",
        "deterministic_drift_mV_per_ms": deterministic_drift,
        "ito_correction_mV_per_ms": ito_correction,
        "total_drift_mV_per_ms": deterministic_drift + ito_correction,
        "noise_amplitude_mV_per_sqrt_ms": noise_amplitude,
    }


def format_s_process_formula_block(parameters):
    r"""Format the definitions and simulated values of the NMDA $s$ process."""
    tau_decay_ms = float(parameters.tau_decay_ms / ms)
    return (
        r"$C_x=\alpha w_x\tau_r="
        rf"{parameters.c_x:.5g}$"
        r"$\qquad s_0=N r_e C_x\tau_D="
        rf"{parameters.s0:.5g}$"
        "\n"
        r"$D_s(\tau)=\frac{N r_e C_x^2\tau_D}{2}"
        r"e^{-|\tau|/\tau_D}="
        rf"{parameters.s_autocovariance_zero_lag:.5g}"
        rf"e^{{-|\tau|/({tau_decay_ms:g}\,\mathrm{{ms}})}}$"
        "\n"
        r"$\mathcal{D}_s\equiv\int_0^\infty D_s(\tau)\,d\tau"
        r"=D_s(0)\tau_D="
        rf"{float(parameters.d_s_ms / ms):.5g}\,\mathrm{{ms}}$"
    )


def format_noise_amplitude_formula(parameters):
    r"""Format the exact implemented diffusion coefficient and its prefactor."""
    capacitance_nF = float(parameters.capacitance_nF / nF)
    g_nmda_max_nS = float(parameters.g_nmda_max_nS / nS)
    # nS/nF = 1/s, hence /1000 converts the conductance scale to 1/ms.
    nmda_scale_per_ms = (
        g_nmda_max_nS
        / parameters.number_of_nmda_inputs
        / capacitance_nF
        / 1000.0
    )
    diffusion_prefactor_per_sqrt_ms = (
        np.sqrt(2.0 * float(parameters.d_s_ms / ms)) * nmda_scale_per_ms
    )
    return (
        r"$b(V)=-\sqrt{2\mathcal{D}_s}\,"
        r"\frac{\bar g_{\mathrm{NMDA}}}{N C_m}\,"
        r"\sigma_{\mathrm{Mg}}(V)(V-E_E)$"
        "\n"
        rf"$=-{diffusion_prefactor_per_sqrt_ms:.4g}\,"
        r"\sigma_{\mathrm{Mg}}(V)(V-E_E)\;"
        r"[\mathrm{mV}/\sqrt{\mathrm{ms}}]$"
    )


def plot_ito_reduced_nmda_voltage(
    parameters,
    output_path,
    *,
    voltage_min_mV=VOLTAGE_MIN_MV,
    voltage_max_mV=VOLTAGE_MAX_MV,
    number_of_voltage_points=2_001,
    show_plot=False,
):
    """Plot drift and diffusion amplitude versus voltage."""
    voltage_mV = np.linspace(
        float(voltage_min_mV), float(voltage_max_mV), int(number_of_voltage_points)
    )
    terms = compute_ito_voltage_terms(voltage_mV, parameters)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    axes[0].plot(
        voltage_mV,
        terms["deterministic_drift_mV_per_ms"],
        label=r"deterministic drift $f(V)$",
        color="tab:blue",
        linewidth=2.0,
    )
    axes[0].plot(
        voltage_mV,
        terms["ito_correction_mV_per_ms"],
        label=r"Itô correction $a_{\mathrm{corr}}(V)$",
        color="tab:orange",
    )
    axes[0].plot(
        voltage_mV,
        terms["total_drift_mV_per_ms"],
        label=r"total drift $a_{\mathrm{It\hat{o}}}(V)=f(V)+a_{\mathrm{corr}}(V)$",
        color="black",
        linewidth=2.2,
    )
    axes[0].axhline(0.0, color="0.5", linewidth=1)
    axes[0].set_ylim(*DRIFT_Y_LIMITS_MV_PER_MS)
    axes[0].set_ylabel(r"drift $a(V)$ [mV/ms]", fontsize=14)
    axes[0].legend(fontsize=12)

    axes[1].plot(
        voltage_mV,
        terms["noise_amplitude_mV_per_sqrt_ms"],
        color="tab:purple",
        linewidth=2,
    )
    axes[1].axhline(0.0, color="0.5", linewidth=1)
    axes[1].set_title(
        format_noise_amplitude_formula(parameters),
        fontsize=13,
        pad=10,
    )
    axes[1].set_ylabel(
        r"noise amplitude $b(V)$ [mV/$\sqrt{\mathrm{ms}}$]", fontsize=14
    )
    axes[1].set_xlabel("membrane voltage V [mV]", fontsize=14)
    axes[1].set_xlim(float(voltage_min_mV), float(voltage_max_mV))
    for axis in axes:
        axis.tick_params(axis="both", labelsize=12)

    figure.suptitle("Reduced NMDA voltage Itô equation", y=0.985, fontsize=18)
    figure.text(
        0.5,
        0.945,
        rf"$r_e={float(parameters.input_rate_hz / Hz):g}\,\mathrm{{Hz}}$, "
        rf"$N={parameters.number_of_nmda_inputs}$, "
        rf"$A={parameters.magnesium_block_a:g}$, "
        rf"$B={float(parameters.magnesium_slope_per_mV * mV):g}\,\mathrm{{mV}}^{{-1}}$"
        "\n"
        rf"$C_m={float(parameters.capacitance_nF / nF):.5g}\,\mathrm{{nF}}$, "
        rf"$g_L={float(parameters.g_leak_nS / nS):.5g}\,\mathrm{{nS}}$, "
        rf"$\bar g_{{\mathrm{{NMDA}}}}={float(parameters.g_nmda_max_nS / nS):.5g}\,\mathrm{{nS}}$, "
        rf"$E_L={float(parameters.e_leak_mV / mV):g}\,\mathrm{{mV}}$, "
        rf"$E_E={float(parameters.e_excitatory_mV / mV):g}\,\mathrm{{mV}}$",
        ha="center",
        va="top",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.865,
        format_s_process_formula_block(parameters),
        ha="center",
        va="top",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.72))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show_plot:
        plt.show(block=False)
    else:
        plt.close(figure)
    return output_path


class PlotItoReducedNMDAVoltageScripts(unittest.TestCase):
    """Manual PyCharm runnable for the configured Itô voltage plot."""

    def test_plot_ito_reduced_nmda_voltage(self):
        parameters = ItoReducedNMDAParameters.from_fitted_apic20(
            input_rate_hz=INPUT_RATE_HZ,
            number_of_nmda_inputs=NUMBER_OF_NMDA_INPUTS,
            magnesium_block_a=MAGNESIUM_BLOCK_A,
            magnesium_slope_per_mV=MAGNESIUM_SLOPE_PER_MV,
        )
        output_path = plot_ito_reduced_nmda_voltage(
            parameters,
            OUTPUT_DIR / "ito_reduced_nmda_voltage_minus90_to_10_mV.png",
            voltage_min_mV=VOLTAGE_MIN_MV,
            voltage_max_mV=VOLTAGE_MAX_MV,
            show_plot=True,
        )
        self.assertTrue(output_path.exists(), output_path)
        self.assertGreater(output_path.stat().st_size, 0, output_path)


if __name__ == "__main__":
    unittest.main()

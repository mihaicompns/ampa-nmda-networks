import sys

import numpy as np
from loguru import logger

from Plotting import show_plots_non_blocking, prepare_bigger_fonts
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config, DiffusionLIFConfig

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

import unittest

from brian2 import Hz, mvolt, mV, Quantity
import matplotlib.pyplot as plt

from iteration_12_transfer_function_of_lif_neurons.LookForAllSolutionsMuSigma import (
    mu_to_sigma_for_constant_rate, mu_to_sigma_for_constant_gain,
    compute_sigma_necessary_for_given_rate_derivative_and_mean, MuToSigmaResult, )


def print_intersection_statistics(siegert_gradients: SiegertGradients, x: float, y: float,
                                  mk801_rate: Quantity, control_rate: Quantity, delta_mu: Quantity, gain:Quantity,
                                  label: str):

    r_0 = siegert_gradients.firing_rate(x, y)
    r_0_control = siegert_gradients.firing_rate(x + delta_mu / mV, y)
    dr_over_dmu = siegert_gradients.d_rate_d_mu(x * mV, y * mV)

    r_est = r_0 + dr_over_dmu * delta_mu
    taylor_error = np.abs(r_0_control - r_est)

    siegert_error_mk801 = np.abs(r_0 - mk801_rate)
    siegert_error_control = np.abs(r_0_control - control_rate)

    grad_error = np.abs(dr_over_dmu - gain)
    precision = 4
    logger.info(f'''
        {label}
        μ = {x: .{precision}f} mV, σ = {y: .{precision}f} mV.
    Rates:
    MK801:  Desired {mk801_rate / Hz: .{precision}f} Hz. Actual r(μ, σ) = {r_0 / Hz: .{precision}f} Hz  
            Estimation error: {siegert_error_mk801 / Hz: .{precision}f} Hz. Relative error {(siegert_error_mk801 /  mk801_rate): .{precision}f}
            
    Control:    Desired {control_rate / Hz: .{precision}f} Hz. Actual r(μ + Δμ, σ) = {r_0_control / Hz: .{precision}f} Hz
                Control Estimation error: {siegert_error_control / Hz: .{precision}f} Hz. Relative error {(siegert_error_control/ control_rate): .{precision}f}
    Taylor rate: Actual Control {r_0_control / Hz: .{precision}f}. Actual r_est(μ + Δμ) = r_0 + dr/dμ * Δμ = {r_est / Hz: .{precision}f} Hz
                 Taylor Error: {(taylor_error / Hz): .{precision}f} Hz. Relative Taylor Err: {(taylor_error / r_0_control) : .{precision}f}
                 
    Gain:   Desired {gain / Hz * mV : .{precision}f} Hz/mV. Actual: dr/dμ = {dr_over_dmu / Hz * mV : .{precision}f} Hz/mV)
            Gradient Error: {grad_error / Hz * mV : .{precision}f} Hz/mV. Relative error {(grad_error / gain): .{precision}f}
        
            ''')

def solve_for_two_rates(mk801_rate: Quantity = 0.05 * Hz, control_rate: Quantity = 0.18 * Hz,
                        delta_mu: Quantity = 0.7 * mV, lif_config: DiffusionLIFConfig = default_diffusion_lif_config):
    nmda_block_mu_to_sigma = mu_to_sigma_for_constant_rate(
        lif_config.with_label("MK-801"), r_target=mk801_rate
    )
    control_mu_to_sigma = mu_to_sigma_for_constant_rate(
        lif_config.with_label("Control"), r_target=control_rate
    )
    control_mu_to_sigma_translated = MuToSigmaResult(mus=control_mu_to_sigma.mus * mV - delta_mu,
                                                     sigmas=control_mu_to_sigma.sigmas * mV,
                                                     r_target=control_mu_to_sigma.r_target,
                                                     exp_label=control_mu_to_sigma.exp_label)
    x_intersect, y_intersect = compute_intersection(nmda_block_mu_to_sigma, control_mu_to_sigma_translated)
    return nmda_block_mu_to_sigma, control_mu_to_sigma, x_intersect, y_intersect

def compute_intersection(curve_1: MuToSigmaResult, curve_2: MuToSigmaResult, interval=(-59, -41)):
    # Find solution
    from scipy.interpolate import interp1d
    f1 = interp1d(curve_1.mus, curve_1.sigmas, kind='cubic')
    f2 = interp1d(curve_2.mus, curve_2.sigmas, kind='cubic')

    def f(x):
        return f1(x) - f2(x)

    left, right = interval

    interval = np.linspace(left, right, num=1001)
    f_ = f(interval)
    signs = np.sign(f_)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    from scipy.optimize import brentq

    res = [None] * len(sign_changes)
    for index, sign_change in enumerate(sign_changes):
        left = interval[sign_change]
        right = interval[sign_change + 1]
        x_intersect = brentq(f, left, right)
        y_intersect = f1(x_intersect)
        res[index] = (x_intersect, y_intersect)

    if len(sign_changes) == 1:
        return res[0]
    else:
        return res


def plot_two_rates_and_one_gain(mk_801_rate_computations, control_rate_computations, gain_computations,
                                siegert_gradients, plot_label: str, caller_test_case):
    prepare_bigger_fonts()
    try:
        intersects_control_gain = compute_intersection(control_rate_computations, gain_computations,
                                                       interval=(-59, -41))
        show_intersect_control_gain = len(intersects_control_gain) > 0
    except ValueError as e:
        logger.exception("Could not compute intersect rate Control = {} and gain = {}",
                         control_rate_computations.r_target / Hz, gain_computations.r_target / Hz * mV)
        show_intersect_control_gain = False

    try:
        x_intersect_rate, y_intersect_rate = compute_intersection(mk_801_rate_computations,
                                                                  control_rate_computations)
        show_intersect_mk801_control = True
        print(f"Intercept rares: ({x_intersect_rate, y_intersect_rate})")
    except ValueError as e:
        logger.exception("Could not compute intersect rate MK801 = {} and rate Control = {}",
                         mk_801_rate_computations.r_target / Hz, control_rate_computations.r_target / Hz)
        show_intersect_mk801_control = False

    try:
        x_intersect_mk801_gain, y_intersect_mk801_gain = compute_intersection(mk_801_rate_computations,
                                                                              gain_computations)
        show_intersect_mk801_gain = True
        print(f"Intercept MK801 and gain: ({x_intersect_mk801_gain, x_intersect_mk801_gain})")
    except ValueError as e:
        logger.exception("Could not compute intersect rate MK801 = {} and gain = {}",
                         mk_801_rate_computations.r_target / Hz, gain_computations.r_target / Hz * mV)
        show_intersect_mk801_gain = True

    prepare_bigger_fonts()
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[3, 1],
        hspace=0.4,
        wspace=0.7
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
    # Bottom row (spans all columns)
    ax4 = fig.add_subplot(gs[1, :])
    l1, = ax1.plot(mk_801_rate_computations.mus, mk_801_rate_computations.sigmas, color="orange",
                   label=r"MK 801: $\sigma = f_1(\mu)$ for $r(\mu, f_1(\mu)) = $"f" {mk_801_rate_computations.r_target / Hz :.4f} Hz")
    l2, = ax1.plot(control_rate_computations.mus, control_rate_computations.sigmas, color="black",
                   label=r"Control: $\sigma = f_2(\mu)$ for $r(\mu, f_2(\mu)) = $"f" {control_rate_computations.r_target / Hz :.4f} Hz")
    ax2.plot(mk_801_rate_computations.mus, mk_801_rate_computations.sigmas, color="orange")
    l3, = ax2.plot(gain_computations.mus, gain_computations.sigmas, color="purple",
                   label=f"{gain_computations.exp_label}"r": $\sigma = f_3(\mu)$ for $\frac{d r}{d \mu}(\mu, f_3(\mu)) = $"f" {gain_computations.r_target / Hz * mV: .4f} Hz/mV")
    ax3.plot(control_rate_computations.mus, control_rate_computations.sigmas, color="black",
             label=f"{control_rate_computations.exp_label}, r = {control_rate_computations.r_target / Hz} Hz")
    ax3.plot(gain_computations.mus, gain_computations.sigmas, color="purple",
             label=r"$\sigma = f_3(\mu)$ for $\frac{d r}{d \mu}(\mu, f_3(\mu)) = $"f" {gain_computations.r_target / Hz * mV: .4f} Hz/mV")

    dot_size = 80

    l7 = ax1.scatter([], [], color="gray", s=dot_size, alpha=0.6, label="Intersection points")
    handles = [l1, l2, l3, l7]
    if show_intersect_mk801_control:
        ax1.scatter(x_intersect_rate, y_intersect_rate, color="red", s=dot_size, zorder=3,
                    label="Solution for two rates", alpha=0.6)

    if show_intersect_mk801_gain:
        ax2.scatter(x_intersect_mk801_gain, y_intersect_mk801_gain, color="blue", s=dot_size, zorder=3,
                    label="Solution for rate and gain", alpha=0.6)

    if show_intersect_control_gain:
        ax3.scatter(*zip(*intersects_control_gain), color="green", s=dot_size, zorder=3,
                    label="Solution for rate and gain", alpha=0.6)

    if show_intersect_mk801_control:
        print_intersection_statistics(siegert_gradients,x = x_intersect_rate, y= y_intersect_rate,
                              mk801_rate=mk_801_rate_computations.r_target, control_rate=control_rate_computations.r_target,
                                      delta_mu=control_rate_computations.delta_mu, gain=gain_computations.r_target,
                                      label="Intersection of MK801 rate and Control rate (1 intersection)")

    if show_intersect_mk801_gain:
        print_intersection_statistics(siegert_gradients, x=x_intersect_mk801_gain, y=y_intersect_mk801_gain,
                                      mk801_rate=mk_801_rate_computations.r_target,
                                      control_rate=control_rate_computations.r_target,
                                      delta_mu=control_rate_computations.delta_mu, gain=gain_computations.r_target,
                                      label="Intersection of MK801 rate and gain (1 intersection)")
    if show_intersect_control_gain:
        for index, (x, y) in enumerate(intersects_control_gain):
            print_intersection_statistics(siegert_gradients, x=x, y=y,
                                          mk801_rate=mk_801_rate_computations.r_target,
                                          control_rate=control_rate_computations.r_target,
                                          delta_mu=control_rate_computations.delta_mu, gain=gain_computations.r_target,
                                          label=f"Intersection of Control rate and gain ({index+1}) intersection")

    ax1.set_title(
        r"$r(\mu, \sigma) =$" f"{mk_801_rate_computations.r_target / Hz : .2f} Hz \n"
        r"$r(\mu + \Delta \mu, \sigma) = $"f"{control_rate_computations.r_target / Hz : .2f} Hz")

    ax2.set_title(
        r"$r(\mu, \sigma) =$" f"{mk_801_rate_computations.r_target / Hz : .2f} Hz \n"
        r"$\frac{d r}{d\mu}(\mu, \sigma) = $"f"{gain_computations.r_target / Hz * mV : .3f} Hz/mV")

    ax3.set_title(
        r"$r(\mu + \Delta \mu, \sigma) = $ "f"{control_rate_computations.r_target / Hz : .2f} Hz \n"
        r"$\frac{d r}{d\mu}(\mu, \sigma) = $" f"{gain_computations.r_target / Hz * mV : .3f} Hz/mV")

    ax4.axis("off")
    ax4.legend(handles=handles, loc="center")
    ax1.set_ylabel(r"$\sigma$ [mV]")
    [ax.set_xlabel(r"$\mu$ [mV]") for ax in [ax1, ax2, ax3]]

    for ax, label in zip([ax1, ax2, ax3], ["A", "B", "C"]):
        ax.text(
            - 0.4, 1.2, f"({label})",
            transform=ax.transAxes,
            fontsize=20,
            fontweight=10,
            va="top",
            ha="left"
        )

    fig.suptitle(f"Model-based estimation of "r"$(\mu_{\mathrm{sol}}, \sigma_{\mathrm{sol}} )$" f"\n{plot_label}")
    fig.subplots_adjust(top=0.75)
    fig.tight_layout()
    show_plots_non_blocking(caller_test_case=caller_test_case)

    mus = np.linspace(-65, -41, 1000) * mV
    gains_sigma_2 = [siegert_gradients.d_rate_d_mu(mu_v=mu, sigma_v=2 * mV) / Hz * mV for mu in mus]

    prepare_bigger_fonts()
    plt.figure(figsize=(8, 6))
    plt.plot(gain_computations.mus, gain_computations.d_rate_d_mus_no_units(siegert_gradients), color="purple",
             label=r"$\sigma = f_3(\mu)$ for $\frac{d r}{d \mu}(\mu, f_3(\mu)) = $"f" {gain_computations.r_target / Hz * mV: .4f} Hz/mV",
             lw=2)

    plt.plot(mk_801_rate_computations.mus[:-100],
             mk_801_rate_computations.d_rate_d_mus_no_units(siegert_gradients)[:-100], color="orange",
             label=f"MK801 Rate {mk_801_rate_computations.r_target / Hz} Hz")
    plt.plot(mus / mV, gains_sigma_2, label=r'$\frac{d r}{d \mu}(\mu, \sigma_v$=2 mV)')

    plt.xlabel(r"$\mu$ [mV]")
    plt.ylabel(r"$\frac{d r}{d \mu}$""\n[Hz/mV]", rotation=0, labelpad=20)
    plt.tight_layout()
    plt.legend()
    plt.show()


class SolveByGraphicalSolutionScripts(unittest.TestCase):

    def test_solve_graphical_for_two_rates(self, lif_config = default_diffusion_lif_config, rates = (0.05 * Hz, 0.18 * Hz)):
        delta_mu = 0.7 * mV

        nmda_block_mu_to_sigma, control_mu_to_sigma, mu_sol, sigma_sol = solve_for_two_rates(delta_mu=delta_mu, lif_config=lif_config,
                                                                                             mk801_rate=rates[0], control_rate=rates[1])

        print("mu solution ", mu_sol)
        print("sigma solution ", sigma_sol)

        sg = SiegertGradients.for_lif_config(lif_config=lif_config)
        print(
            f"First rate: {sg.firing_rate(mu_sol, sigma_sol)}. Second rate {sg.firing_rate(mu_sol + delta_mu / mV, sigma_sol)}")
        print("Derivative at sol: ", sg.d_rate_d_mu(mu_sol * mV, sigma_sol * mV) / Hz * mV,
              ". Derivative translated ",
              sg.d_rate_d_mu((mu_sol + delta_mu / mV) * mV, sigma_sol * mV) / Hz * mV)
        # Solution found
        d_rate = control_mu_to_sigma.r_target - nmda_block_mu_to_sigma.r_target
        desired_gain = d_rate / delta_mu / Hz * mV
        actual_gain = sg.d_rate_d_mu(mu_sol * mV, sigma_sol * mV) / Hz * mV
        print("Desired gain: ", desired_gain)
        print("Actual gain: ", actual_gain)
        print("Ratio: ", desired_gain / actual_gain)

        print("RAW gain: ", sg.d_rate_d_mu_primitive(mu_sol * mV, sigma_sol * mV) / Hz * mV)

        print("Taylor expansion: ",
              nmda_block_mu_to_sigma.r_target + delta_mu * sg.d_rate_d_mu(mu_sol * mV, sigma_sol * mV))

        error_nmda_block = np.abs(
            (nmda_block_mu_to_sigma.r_target - sg.firing_rate(mu_sol * mV, sigma_sol * mV)) / Hz)
        error_control = np.abs(
            (control_mu_to_sigma.r_target - sg.firing_rate(mu_sol * mV + delta_mu, sigma_sol * mV)) / Hz)

        prepare_bigger_fonts()
        plt.figure(figsize=(9, 8))

        plt.plot(nmda_block_mu_to_sigma.mus, nmda_block_mu_to_sigma.sigmas, color="orange",
                 label=f"{nmda_block_mu_to_sigma.exp_label}, r = {nmda_block_mu_to_sigma.r_target / Hz} Hz",
                 lw=1.5)
        plt.plot(control_mu_to_sigma.mus - delta_mu / mV, control_mu_to_sigma.sigmas, color="black",
                 label=f"{control_mu_to_sigma.exp_label}, r = {control_mu_to_sigma.r_target / Hz} Hz",
                 lw=1.7)

        dot_size = 80
        plt.scatter(mu_sol, sigma_sol, color="red", s=dot_size, zorder=3,
                    label=f"Solution {{{mu_sol: .3f}, {sigma_sol: .3f}}} mV", alpha=0.6)

        plt.vlines(mu_sol, ymin=0, ymax=sigma_sol * 1.3, colors='b', linestyles='--', lw=0.9)
        plt.hlines(sigma_sol,
                   xmin=np.min((nmda_block_mu_to_sigma.mus[0], control_mu_to_sigma.mus[0] - delta_mu / mV)),
                   xmax=mu_sol + 2, colors='b', linestyles='--', lw=0.9)

        plt.xlabel(r"$\mu$ [mV]")
        plt.ylabel(r"$\sigma$ [mV]")
        plt.title(r"{($\mu$, $\sigma$) | $r(\mu, \sigma) = $"" constant} \n"
                  "predicted by first time passage formula \n"
                  r"$r(\mu_{\mathrm{sol}}, \sigma_{\mathrm{sol}}) =$" f"{sg.firing_rate(mu_sol * mV, sigma_sol * mV) / Hz : .3f} +{error_nmda_block: .0E} Hz, "
                  r"$r(\mu_{\mathrm{sol}} + \Delta \mu, \sigma_{\mathrm{sol}}) =$" f"{sg.firing_rate(mu_sol * mV + delta_mu, sigma_sol * mV) / Hz : .3f} +{error_control: .0E} Hz")

        plt.legend()
        show_plots_non_blocking(caller_test_case=self)

    def test_solve_graphical_for_rate_and_derivative(self):
        delta_mu = 0.7 * mvolt

        rate_mk801 = 0.05 * Hz
        rate_control = 0.18 * Hz

        gain = (rate_control - rate_mk801) / delta_mu

        siegert_gradients = SiegertGradients.default()

        gain_computations = mu_to_sigma_for_constant_gain(
            default_diffusion_lif_config.with_label(r"$\frac{\Delta r}{ \Delta \mu}$ from experiment"), gain=gain
        )

        mk_801_rate_computations = mu_to_sigma_for_constant_rate(
            default_diffusion_lif_config.with_label("MK-801"), r_target=rate_mk801
        )

        control_rate_computations = mu_to_sigma_for_constant_rate(
            default_diffusion_lif_config.with_label("Control"), r_target=rate_control
        )

        control_rate_computations = MuToSigmaResult(mus=control_rate_computations.mus * mV - delta_mu,
                                                    sigmas=control_rate_computations.sigmas * mV,
                                                    r_target=control_rate_computations.r_target,
                                                    exp_label=control_rate_computations.exp_label,
                                                    delta_mu=delta_mu)
        #control_rate_computations = control_rate_computations.with_delta_mu(delta_mu)

        plot_two_rates_and_one_gain(mk_801_rate_computations, control_rate_computations, gain_computations,
                                    siegert_gradients,
                                    plot_label=r"for $\frac{\Delta r}{\Delta \mu}$ based on experimental data",
                                    caller_test_case=self)

    def test_show_influence_of_v_r_on_mean_sigma_scan_VRs(self):

        V_Rs = np.linspace(-80, -41, num=1000)

        mus_s = [None] * len(V_Rs)
        sigmas_s = [None] * len(V_Rs)
        for index, V_R in enumerate(V_Rs):

            config = default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_V_R, V_R)

            _, _, mu, sigma = solve_for_two_rates(lif_config=config)
            mus_s[index] = mu
            sigmas_s[index] = sigma

        prepare_bigger_fonts()

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        ax1.plot(V_Rs, mus_s)
        ax1.set_title("$\mu$")

        ax2.plot(V_Rs, sigmas_s)
        ax2.set_title("$\sigma$")

        fig.suptitle(r"Scan over $V_R")
        show_plots_non_blocking(self)


    def test_show_influence_of_v_r_on_mean_sigma_scan_tau_m(self):

        tau_ms = np.linspace(1, 100, num=101)

        mus_s = [None] * len(tau_ms)
        sigmas_s = [None] * len(tau_ms)
        for index, tau_m in enumerate(tau_ms):

            config = default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_TAU_MEMBRANE, tau_m)

            _, _, mu, sigma = solve_for_two_rates(lif_config=config)
            mus_s[index] = mu
            sigmas_s[index] = sigma

        prepare_bigger_fonts()

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(8, 6))

        ax1.plot(tau_ms, mus_s)
        ax1.set_title("$\mu$")

        ax2.plot(tau_ms, sigmas_s)
        ax2.set_title("$\sigma$")

        ax1.set_xlabel(r"$\tau_m$ [ms]")
        ax1.set_ylabel("$\mu$ [mV]")

        ax2.set_xlabel(r"$\tau_m$ [ms]")
        ax2.set_ylabel("$\sigma$ [mV]")

        ax1.axvline(x=10, label="Default model \n 10 ms", color='b', linestyle='--',)
        ax2.axvline(x=10, label="Default model \n 10 ms", color='b', linestyle='--',)

        #ax1.legend()
        #ax2.legend()
        fig.legend()

        fig.tight_layout()

        fig.suptitle(r"Scan over $\tau_m$")

        show_plots_non_blocking(self)


    def test_check_components_of_binary_search_1(self):
        lif_config_mk801 = default_diffusion_lif_config.with_label("MK-801")

        siegert_gradients = SiegertGradients.for_lif_config(lif_config_mk801)
        siegert_gradients.d_rate_d_mu(mu_v=-50 * mvolt, sigma_v=0.1 * mvolt)
        print(siegert_gradients.d_rate_d_mu(mu_v=-50 * mvolt, sigma_v=0.1 * mvolt))
        target_gain = (0.3 - 0.05) * Hz / (0.7 * mV)

        print(target_gain)
        compute_sigma_necessary_for_given_rate_derivative_and_mean(mu=-50 * mvolt,
                                                                   target_gain=target_gain,
                                                                   lif_config=lif_config_mk801)

    def test_fit_new_rates(self):
        rate_mk801_real = 0.05 * Hz / 0.59
        rate_control_real = 0.18 * Hz / 0.66
        self.test_solve_graphical_for_two_rates(rates = [0.05* Hz, 0.18 * Hz])
        self.test_solve_graphical_for_two_rates(rates = [rate_mk801_real, rate_control_real])

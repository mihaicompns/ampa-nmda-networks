import unittest

import matplotlib.pyplot as plt
from brian2 import mV, Hz, Quantity, is_dimensionless, volt

from Plotting import prepare_bigger_fonts, show_plots_non_blocking
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_12_transfer_function_of_lif_neurons.config import DiffusionLIFConfig, default_diffusion_lif_config

import numpy as np
from scipy.optimize import fsolve

def with_units(vars: list) -> list:
    if vars[1] < 0:
        vars[1] *= 0
    if is_dimensionless(vars[0]):
        return [var * volt for var in vars]
    return vars

def solve_for_two_rates(lif_config: DiffusionLIFConfig, mk_801_rate: Quantity = 0.05 * Hz, control_rate: Quantity = 0.18 * Hz, delta_mu: Quantity = 0.7 * Hz):
    sg = SiegertGradients.for_lif_config(lif_config)

    def system(vars, rate_mk801, rate_control, delta_mu):
        mu, sigma = with_units(vars)

        return [
            sg.firing_rate(mu, sigma) - rate_mk801,
            sg.firing_rate(mu + delta_mu, sigma) - rate_control
        ]

    def jacobian(vars, rate_mk801, rate_control, delta_mu):
        mu, sigma = with_units(vars)

        dmu1, dsigma1 = sg.grad_rate_mu_sigma(mu_v=mu, sigma_v=sigma)
        dmu2, dsigma2 = sg.grad_rate_mu_sigma(mu_v=mu + delta_mu, sigma_v=sigma)

        return [
            [dmu1, dsigma1],
            [dmu2, dsigma2]
        ]

    x0 = [-55 * mV, 1.0 * mV]  # initial guess
    sol = fsolve(system, x0, args=(mk_801_rate, control_rate, delta_mu), fprime=jacobian)
    print(sol)
    mu_sol, sigma_sol = sol
    return mu_sol * volt, sigma_sol * volt

def scan_lif_configs(lif_configs=list[DiffusionLIFConfig], x_axis=np.empty((0,)), label=r"$V_R$", unit="mV", delta_mu: Quantity = 0.7 * mV, caller_test_case=None):

    mus_s = [None] * len(lif_configs)
    sigmas_s = [None] * len(lif_configs)
    rates_mk = [None] * len(lif_configs)
    rates_control = [None] * len(lif_configs)
    for index, config in enumerate(lif_configs):

        mu, sigma = solve_for_two_rates(lif_config=config, delta_mu=delta_mu)
        mus_s[index] = (mu + delta_mu) / mV
        sigmas_s[index] = sigma / mV
        rates_mk[index] = SiegertGradients.for_lif_config(config).firing_rate(mu_v=mu, sigma_v=sigma) / Hz
        rates_control[index] = SiegertGradients.for_lif_config(config).firing_rate(mu_v=mu + delta_mu,
                                                                                   sigma_v=sigma) / Hz

    prepare_bigger_fonts()

    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))

    ax_mu = ax[0, 0]
    ax_sigma = ax[0, 1]
    ax_mk_rates = ax[1, 0]
    ax_control_rates = ax[1, 1]
    ax_control_rates.sharey(ax_mk_rates)

    ax_mu.plot(x_axis, mus_s, lw=2)
    ax_mu.set_title("$\mu_{sol}$")
    ax_mu.set_ylabel("[mV]")

    ax_sigma.plot(x_axis, sigmas_s, lw=2, color="orange")
    ax_sigma.set_title("$\sigma{sol}$")
    ax_sigma.set_ylabel("[mV]")

    ax_mk_rates.plot(x_axis, rates_mk, lw=2)
    ax_mk_rates.set_title("$r_0(\mu, \sigma)$ MK801")
    ax_mk_rates.set_ylabel("[Hz]")

    ax_control_rates.plot(x_axis, rates_control, lw=2, color="black")
    ax_control_rates.set_title("$r_0(\mu + \Delta \mu, \sigma)$ Control")
    ax_control_rates.set_ylabel("[Hz]")

    for axs in ax.flatten():
        axs.set_xlabel(fr"{label} [{unit}]")

    fig.suptitle(f"Effect of {label} \n on parameter fitting")
    fig.tight_layout()
    show_plots_non_blocking(caller_test_case=caller_test_case)



class DirectSolutions(unittest.TestCase):
    def test_find_solution_directly(self, lif_config: DiffusionLIFConfig = default_diffusion_lif_config):

        delta_mu = 0.7 * mV
        mk_801_rate = 0.05 * Hz
        control_rate = 0.18 * Hz

        mu_sol, sigma_sol = solve_for_two_rates(lif_config, mk_801_rate, control_rate, delta_mu,)

        sg = SiegertGradients.for_lif_config(lif_config)

        print(sg.firing_rate(mu_sol, sigma_sol) / Hz)
        print(sg.firing_rate(mu_sol + delta_mu, sigma_sol) / Hz)


    def test_show_influence_of_v_r_on_mean_sigma_scan_VRs(self):
        V_Rs = np.linspace(-80, -41, num=1000)
        lif_configs = [default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_V_R, V_R) for V_R in V_Rs]

        scan_lif_configs(lif_configs=lif_configs, x_axis=V_Rs, label=r"$V_R$", unit="mV", caller_test_case=self)

if __name__ == '__main__':
    unittest.main()

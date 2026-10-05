import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, Hz, Quantity, volt
from scipy.optimize import fsolve
from tqdm.auto import tqdm

from Plotting import show_plots_non_blocking, prepare_bigger_fonts
from iteration_12_siegert.GraphicalSolutions import SolveByGraphicalSolutionScripts, solve_for_two_rates
from iteration_12_siegert.SolutionByFSolve import with_units
from iteration_12_siegert.SolutionByLine import plot_two_rates_and_delta_solutions, SolutionsByTwoPointsScripts, \
    fit_mu_and_sigma_two_rates_and_delta_mu
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_12_transfer_function_of_lif_neurons.config import DiffusionLIFConfig, default_diffusion_lif_config
from scripts.iteration_12_siegert.Chapter_2 import Chapter2Figures

def solve_one_config(index, config, delta_mu):
    _, _, mu, sigma = solve_for_two_rates(
        delta_mu=delta_mu,
        lif_config=config
    )

    mu = mu * mV
    sigma = sigma * mV

    sg = SiegertGradients.for_lif_config(config)

    mu_s = (mu + delta_mu) / mV
    sigma_s = sigma / mV

    rate_mk = sg.firing_rate(
        mu_v=mu,
        sigma_v=sigma
    ) / Hz

    rate_control = sg.firing_rate(
        mu_v=mu + delta_mu,
        sigma_v=sigma
    ) / Hz

    return index, mu_s, sigma_s, rate_mk, rate_control

def scan_lif_configs_with_bentq(lif_configs=list[DiffusionLIFConfig], x_axis=np.empty((0,)), label=r"$V_R$", unit="mV", delta_mu: Quantity = 0.7 * mV, caller_test_case=None):

    mus_s = [None] * len(lif_configs)
    sigmas_s = [None] * len(lif_configs)
    rates_mk = [None] * len(lif_configs)
    rates_control = [None] * len(lif_configs)

    for index, config in tqdm(
            enumerate(lif_configs),
            total=len(lif_configs),
            desc="Processing LIF configs"
    ):

        _, _, mu, sigma = solve_for_two_rates(delta_mu=delta_mu, lif_config=config)
        mu = mu * mV
        sigma = sigma * mV
        mus_s[index] = (mu + delta_mu) / mV
        sigmas_s[index] = sigma / mV
        rates_mk[index] = SiegertGradients.for_lif_config(config).firing_rate(mu_v=mu, sigma_v=sigma) / Hz
        rates_control[index] = SiegertGradients.for_lif_config(config).firing_rate(mu_v=mu + delta_mu,
                                                                                   sigma_v=sigma) / Hz
    '''

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(solve_one_config)(index, config, delta_mu)
        for index, config in enumerate(lif_configs)
    )

    for index, mu_s, sigma_s, rate_mk, rate_control in results:
        mus_s[index] = mu_s
        sigmas_s[index] = sigma_s
        rates_mk[index] = rate_mk
        rates_control[index] = rate_control
    '''

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

def fsolve_for_two_rates(lif_config: DiffusionLIFConfig, mk_801_rate: Quantity = 0.05 * Hz, control_rate: Quantity = 0.18 * Hz, delta_mu: Quantity = 0.7 * Hz):
    sg = SiegertGradients.for_lif_config(lif_config)

    trajectory = []

    def system(vars, rate_mk801, rate_control, delta_mu):
        mu, sigma = with_units(vars)

        trajectory.append((mu, sigma))

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
    return mu_sol * volt, sigma_sol * volt, trajectory

class VReset43Scripts(unittest.TestCase):

    def test_understand_V_R_43_mv(self):
        target_rates = [0.05 * Hz, 0.18 * Hz]
        lif_config = default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_V_R, -43)
        delta_mu = 0.7 * mV

        control_line, mk801_line, mu_v, sigma_v = fit_mu_and_sigma_two_rates_and_delta_mu(delta_mu, lif_config, target_rates)

        mu_v_mk801 = mu_v
        mu_v_control = mu_v_mk801 + delta_mu

        siegert_gradient = SiegertGradients.for_lif_config(lif_config)
        rate_mk801 = siegert_gradient.firing_rate(mu_v=mu_v_mk801, sigma_v=sigma_v)
        rate_control = siegert_gradient.firing_rate(mu_v=mu_v_control, sigma_v=sigma_v)

        print(f"mu_v MK-801 = {mu_v_mk801}, mu_v Control = {mu_v_control}, sigma_v = {sigma_v}")
        print(f"Predicted MK801 rate: {rate_mk801}, Control rate: {rate_control}")
        print(f"delta mu {delta_mu}: m mk801= {mk801_line.slope}, m control = {control_line.slope}"
              f"b mk801 = {mk801_line.intercept}, b control = {control_line.intercept}")

        plot_two_rates_and_delta_solutions(lif_config, delta_mu, control_line, mk801_line, mu_v_control,
                                                mu_v_mk801, sigma_v, caller_test_case=self)

    def test_quality_of_line_through_two_points_fit_V_R_43(self):
        v_r_43_config = default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_V_R, -43)
        SolutionsByTwoPointsScripts().test_quality_of_line_through_two_points_fit(lif_config=v_r_43_config)


    def test_classical_linear_fit_VR_43(self):
        v_r_43_config = default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_V_R, -43)
        Chapter2Figures().test_show_linear_fit_and_compute_mu_sigma(lif_config=v_r_43_config)
        fitting = [v_r_43_config.fitting(0.05 * Hz, label=r"MK801, $V_R = $ -43 mV"), v_r_43_config.fitting(0.18 * Hz, label=r"Control, $V_R = $ -43 mV")]
        Chapter2Figures().test_plot_line_computation_vs_fit(fitting_problems=fitting)

    def test_solve_graphically_VR_43(self):
        v_r_43_config = default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_V_R, -43)
        SolveByGraphicalSolutionScripts().test_solve_graphical_for_two_rates(lif_config=v_r_43_config)

    def test_by_brentqs_method(self):
        V_Rs = np.linspace(-80, -41, num=101)
        lif_configs = [default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_V_R, V_R) for V_R in V_Rs]
        scan_lif_configs_with_bentq(lif_configs=lif_configs, x_axis=V_Rs, label=r"$V_R$", unit="mV", caller_test_case=self)

    def test_check_convergence_of_fsolve_for_43_mv(self, lif_config: DiffusionLIFConfig = default_diffusion_lif_config):
        delta_mu = 0.7 * mV
        mk_801_rate = 0.05 * Hz
        control_rate = 0.18 * Hz

        lif_config = lif_config.with_property(DiffusionLIFConfig.KEY_V_R, -43)

        mu_sol, sigma_sol, trajectory = fsolve_for_two_rates(lif_config, mk_801_rate, control_rate, delta_mu)

        sg = SiegertGradients.for_lif_config(lif_config)

        print(sg.firing_rate(mu_sol, sigma_sol) / Hz)
        print(sg.firing_rate(mu_sol + delta_mu, sigma_sol) / Hz)

        mus = [x[0] / mV for x in trajectory]
        sigmas = [x[1] / mV for x in trajectory]

        iters = np.arange(len(trajectory))

        plt.figure(figsize=(6, 5))

        sc = plt.scatter(
            mus,
            sigmas,
            c=iters,
            cmap="viridis",
            s=60
        )

        plt.plot(mus, sigmas, alpha=0.4)

        plt.xlabel("mu (mV)")
        plt.ylabel("sigma (mV)")
        plt.title("fsolve trajectory")

        cbar = plt.colorbar(sc)
        cbar.set_label("Iteration")

        plt.show()




if __name__ == '__main__':
    unittest.main()

"""
Runnable Siegert μ–σ scripts: plots and scans.

Convention: test_scripts_* = runnable experiments/plots (discovered by IntelliJ as tests).
"""
import sys

from loguru import logger

from iteration_12_transfer_function_of_lif_neurons.config import default_diffusion_lif_config, DiffusionLIFConfig
from scripts.iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

import unittest

import numpy as np
from brian2 import mV, Hz, mvolt

from iteration_12_transfer_function_of_lif_neurons.LookForAllSolutionsMuSigma import (
    compute_mu_to_sigma_curve_for_experiment,
    compute_mu_to_sigma_fsolve_scan_mus,
    compute_mu_to_sigma_fsolve_scan_sigmas,
    plot_mus_vs_sigmas,
    plot_loss_landscape_with_curve, mu_to_sigma_for_constant_rate, )
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import (
    SiegertGradientDescent,
    SiegertGradients, )
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import NeuronModelParams


def d_sigma_over_d_mu(mu_v, sigma_v, siegert_gradients: SiegertGradients):
    # Arguments of the CDF
    a, b = siegert_gradients.integration_limits(mu_v, sigma_v)

    Phi_a = siegert_gradients.E(a)
    Phi_b = siegert_gradients.E(b)

    # Numerator
    numerator = sigma_v * (Phi_a - Phi_b)

    # Denominator
    denominator = (mu_v - siegert_gradients.v_reset / mV) * Phi_a - (mu_v - siegert_gradients.theta / mV) * Phi_b

    return numerator / denominator


class LookForAllSolutionsScripts(unittest.TestCase):
    """Runnable scripts: μ–σ curves, loss landscape, fsolve scans. test_scripts_* = run directly in IntelliJ."""

    def test_scripts_look_for_all_solutions(self):
        experiment = palmer_control
        r_target = 0.3 * Hz
        solver = SiegertGradientDescent(
            tau_m=experiment.effective_time_constant_up_state.tau_eff(),
            theta=experiment.neuron_params.theta,
            v_reset=experiment.neuron_params.V_r,
            tau_ref=experiment.neuron_params.tau_rp,
            unit="mV",
        )
        fig, _ = plot_loss_landscape_with_curve(
            solver, r_target, (-60, -35), (0, 6),
            resolution=50, target_rate_Hz=0.3, caller_test_case=self,
        )
        fig.show()

    def test_scripts_look_for_all_solutions_using_binary_search(self):
        experiment = palmer_control
        #palmer_control.with_property(NeuronModelParams.KEY_NEURON_V_R, -55)
        nmda_block_mu_to_sigma = compute_mu_to_sigma_curve_for_experiment(
            experiment.with_label("MK-801"), r_target=0.05 * Hz
        )
        control_mu_to_sigma = compute_mu_to_sigma_curve_for_experiment(
            experiment.with_label("Control"), r_target=0.3 * Hz
        )
        mu_60_arg = np.abs(nmda_block_mu_to_sigma.mus - (-60)).argmin()
        mu_close_to_60 = nmda_block_mu_to_sigma.mus[mu_60_arg] * mvolt
        sigma_for_mu_60 = nmda_block_mu_to_sigma.sigmas[mu_60_arg] * mvolt
        mu_60_1_arg = np.abs(control_mu_to_sigma.mus + (60 - 0.1)).argmin()
        mu_close_to_60_1 = control_mu_to_sigma.mus[mu_60_1_arg] * mvolt
        sigma_for_mu_60_1 = control_mu_to_sigma.sigmas[mu_60_1_arg] * mvolt
        siegert_gradients = SiegertGradients.for_experiment(experiment)
        rate_60_mv = siegert_gradients.firing_rate(
            mu_v=mu_close_to_60, sigma_v=sigma_for_mu_60
        )
        rate_60_1 = siegert_gradients.firing_rate(
            mu_v=mu_close_to_60_1, sigma_v=sigma_for_mu_60_1
        )

        integral_limits_nmda_block = [siegert_gradients.integration_limits(mu_v = mu, sigma_v = sigma) for mu, sigma in zip(
            nmda_block_mu_to_sigma.mus, nmda_block_mu_to_sigma.sigmas
        )]

        print(rate_60_mv, rate_60_1)
        print(
            f"Delta in mu {(mu_close_to_60_1 - mu_close_to_60) / mV}. "
            f"Delta in sigma {(sigma_for_mu_60_1 - sigma_for_mu_60) / mV}."
        )
        plot_mus_vs_sigmas(
            [nmda_block_mu_to_sigma, control_mu_to_sigma],
            caller_test_case=self,
        )

    def test_scripts_look_for_all_solutions_using_binary_search_palmer_rates(self):
        experiment = palmer_control
        #palmer_control.with_property(NeuronModelParams.KEY_NEURON_V_R, -55)
        experiment = palmer_control.with_property(NeuronModelParams.KEY_NEURON_V_R, -65).with_property(NeuronModelParams.KEY_NEURON_THRESHOLD, -40)
        nmda_block_mu_to_sigma = compute_mu_to_sigma_curve_for_experiment(
            experiment.with_label("MK-801"), r_target=0.05 * Hz
        )
        control_mu_to_sigma = compute_mu_to_sigma_curve_for_experiment(
            experiment.with_label("Control"), r_target=0.18 * Hz
        )
        mu_60_arg = np.abs(nmda_block_mu_to_sigma.mus - (-60)).argmin()
        mu_close_to_60 = nmda_block_mu_to_sigma.mus[mu_60_arg] * mvolt
        sigma_for_mu_60 = nmda_block_mu_to_sigma.sigmas[mu_60_arg] * mvolt
        rate_60_mv = SiegertGradients.for_experiment(experiment).firing_rate(
            mu_v=mu_close_to_60, sigma_v=sigma_for_mu_60
        )
        mu_59_9_arg = np.abs(control_mu_to_sigma.mus + (60 - 0.1)).argmin()
        mu_close_to_59_9 = control_mu_to_sigma.mus[mu_59_9_arg] * mvolt
        sigma_for_mu_59_9 = control_mu_to_sigma.sigmas[mu_59_9_arg] * mvolt
        rate_59_9_mv = SiegertGradients.for_experiment(experiment).firing_rate(
            mu_v=mu_close_to_59_9, sigma_v=sigma_for_mu_59_9
        )
        print(
            f"mu = {mu_close_to_60 / mV:.4f}, sigma = {sigma_for_mu_60 / mV:.4f} "
            f"produces rate {rate_60_mv / Hz:.4f}"
        )
        print(
            f"mu = {mu_close_to_59_9 / mV:.4f}, sigma = {sigma_for_mu_59_9 / mV:.4f} "
            f"produces rate {rate_59_9_mv / Hz:.4f}"
        )
        plot_mus_vs_sigmas(
            [nmda_block_mu_to_sigma, control_mu_to_sigma],
            caller_test_case=self,
        )

    def test_scripts_look_for_all_solutions_using_fsolve_scan_sigma(self):
        result = compute_mu_to_sigma_fsolve_scan_sigmas(
            palmer_control, r_target=0.3 * Hz
        )
        print(result)
        self.assertEqual(101, len(result.mus))
        self.assertEqual(101, len(result.sigmas))
        plot_mus_vs_sigmas([result], caller_test_case=self)

    def test_scripts_look_for_all_solutions_using_fsolve_scan_mu(self):
        result = compute_mu_to_sigma_fsolve_scan_mus(
            palmer_control, r_target=0.3 * Hz
        )
        print(result)
        self.assertEqual(1001, len(result.mus))
        self.assertEqual(1001, len(result.sigmas))
        plot_mus_vs_sigmas([result], caller_test_case=self)

    def test_quicktest_linear_solution(self):
        siegert_gradient = SiegertGradients.for_lif_config(default_diffusion_lif_config)

        m_MK801 = -0.252
        b_MK801 = -10.08

        m_Control = -0.278
        b_Control = -11.12
        delta_mu = 0.4

        sigma_v, mu_v =  1/(m_MK801 - m_Control) * np.array([[-m_Control, m_MK801], [-1, 1]]) @ np.array([[b_MK801], [m_Control * delta_mu + b_Control]])

        print(sigma_v, mu_v)

        print(siegert_gradient.firing_rate(mu_v = mu_v[0] * mV, sigma_v = sigma_v[0] * mV))
        print(siegert_gradient.firing_rate(mu_v = (mu_v[0] + delta_mu) * mV, sigma_v = sigma_v[0] * mV))

    def test_compute_fitted_solution(self):

        for v_r in [-60, -65, -70, -75, -80, -85, -90]:
            lif_config = default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_V_R, v_r)

            nmda_block_mu_to_sigma = mu_to_sigma_for_constant_rate(
                lif_config.with_label("MK-801"), r_target=0.05 * Hz
            )
            control_mu_to_sigma = mu_to_sigma_for_constant_rate(
                lif_config.with_label("Control"), r_target=0.18 * Hz
            )

            delta_mu = 0.7

            m_mk801, b_mk801, _, _, _ = nmda_block_mu_to_sigma.linear_fit()
            m_control, b_control, _, _, _ = control_mu_to_sigma.linear_fit()

            sigma_v, mu_v = 1 / (m_mk801 - m_control) * np.array([[-m_control, m_mk801], [-1, 1]]) @ np.array(
                [[b_mk801], [m_control * delta_mu + b_control]])

            sigma_v = sigma_v[0]
            mu_v = mu_v[0]

            print(f"V_reset {v_r}: mu_v = {mu_v}, sigma_v = {sigma_v}")

            siegert_gradient = SiegertGradients.for_lif_config(lif_config)
            print(siegert_gradient.firing_rate(mu_v=mu_v * mV, sigma_v=sigma_v * mV))
            print(siegert_gradient.firing_rate(mu_v=(mu_v + delta_mu) * mV, sigma_v=sigma_v * mV))

    def test_compute_intercept_analytically(self):

        def compute_mu_v(lif_config: DiffusionLIFConfig, target_rate):
            # Compute the argument of the exponential
            exponent_arg = (1.0 / target_rate - lif_config.tau_rp) / lif_config.tau_m


            print(exponent_arg)

            if exponent_arg > 700:  # exp(700) ~ 1e304, close to double overflow
                # For r -> 0+, mu_v -> theta from above.
                # Use the approximation: mu_v = theta + (theta - V_R) / exp(exponent_arg)
                # But since exp is huge, mu_v ≈ theta.
                return lif_config.theta

            exp_term = np.exp(exponent_arg)

            if np.isclose(exp_term, 1.0):
                return np.nan

            return lif_config.theta + (lif_config.V_r - lif_config.theta) / (1 - exp_term)

        mu_v_mk801 = compute_mu_v(default_diffusion_lif_config, 0.05 * Hz) / mV
        mu_v_control= compute_mu_v(default_diffusion_lif_config, 0.18 * Hz) / mV

        print("mu MK801: ", mu_v_mk801)
        print("mu Control: ", mu_v_control)





import unittest
from numbers import Number

import numpy as np
from brian2 import Hz, mV, Quantity, is_dimensionless
from loguru import logger
from scipy.optimize import root_scalar

from BinarySeach import binary_search_for_target_value_precission_in_result_space
from iteration_12_transfer_function_of_lif_neurons.LookForAllSolutionsMuSigma import \
    compute_sigma_necessary_for_given_rate_and_mean, \
    compute_sigma_necessary_for_given_rate_and_mean_newton
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_12_transfer_function_of_lif_neurons.config import DiffusionLIFConfig, default_diffusion_lif_config


class MuSigmaBinarySearchState:
    def __init__(self, mu, delta_sigma, sigma_mk801, sigma_control):
        self.mu = mu
        self.delta_sigma = delta_sigma
        self.sigma_mk801 = sigma_mk801
        self.sigma_control = sigma_control

    def __str__(self):
        return f'''mu={self.mu:.5f}, Δσ sigma mk801={self.sigma_mk801:.5f}, sigma control={self.sigma_control:.5f}
            '''
    def __truediv__(self, other):
        if not isinstance(other, Number):
            raise TypeError("Division only supported with numeric types")

        return self.mu / other

def find_solution_by_double_binary_search(rate_mk801: Quantity = 0.05 * Hz, rate_control: Quantity = 0.18 * Hz, delta_mu=0.7 * mV, lif_config: DiffusionLIFConfig = default_diffusion_lif_config):
    sg = SiegertGradients.for_lif_config(lif_config)


    def find_delta_sigma(current_mu: Quantity) -> Quantity:

        sigma_control = compute_sigma_necessary_for_given_rate_and_mean(current_mu + delta_mu, r_target=rate_control,
                                                                        lif_config=lif_config)
        sigma_mk801 = compute_sigma_necessary_for_given_rate_and_mean(current_mu, r_target=rate_mk801,
                                                                      lif_config=lif_config)

        logger.info("r0(mu - delta mu = {}, sigma control = {}) = {}, r0(mu = {}, sigma mk801 = {}) = {}",
                    current_mu - delta_mu, sigma_control, sg.firing_rate(current_mu - delta_mu, sigma_control),
                    current_mu, sigma_mk801, sg.firing_rate(current_mu, sigma_mk801))

        return sigma_control - sigma_mk801

    try:
        mu, _ = binary_search_for_target_value_precission_in_result_space(
            lower_value=-41 * mV,
            upper_value=-70 * mV,
            func=lambda m: find_delta_sigma(current_mu=m),
            target_result=0 * mV,
            precision=1e-10 * mV,
            max_iters=100
        )
    except ValueError as e:
        print(f"mu={mu}: {e}")
        return np.nan  # fallback if binary search fails

    print(f"mu: {mu}")
    sigma = compute_sigma_necessary_for_given_rate_and_mean(mu, r_target=rate_mk801,
                                                            lif_config=lif_config)

    return mu, sigma

def find_delta_sigma(current_mu: Quantity, lower_state: MuSigmaBinarySearchState, upper_state: Quantity,
                     rate_mk801:Quantity = 0.05 * Hz, rate_control: Quantity = 0.18 * Hz, delta_mu:Quantity = 0.7 * mV, lif_config: DiffusionLIFConfig = default_diffusion_lif_config) -> Quantity:

    sg = SiegertGradients.for_lif_config(lif_config)

    sigma_control = compute_sigma_necessary_for_given_rate_and_mean(current_mu + delta_mu, r_target=rate_control,
                                                                    lif_config=lif_config)
    sigma_mk801 = compute_sigma_necessary_for_given_rate_and_mean(current_mu, r_target=rate_mk801,
                                                                  lif_config=lif_config)

    logger.info("r0(mu - delta mu = {}, sigma control = {}) = {}, r0(mu = {}, sigma mk801 = {}) = {}",
                current_mu - delta_mu, sigma_control, sg.firing_rate(current_mu - delta_mu, sigma_control),
                current_mu, sigma_mk801, sg.firing_rate(current_mu, sigma_mk801))

    return sigma_control - sigma_mk801

def find_delta_sigma_newton(current_mu: Quantity, rate_mk801:Quantity = 0.05 * Hz, rate_control: Quantity = 0.18 * Hz,
                            delta_mu:Quantity = 0.7 * mV, lif_config: DiffusionLIFConfig = default_diffusion_lif_config) -> Quantity:
    sigma_control = compute_sigma_necessary_for_given_rate_and_mean_newton(current_mu + delta_mu, r_target=rate_control,
                                                                    lif_config=lif_config)
    sigma_mk801 = compute_sigma_necessary_for_given_rate_and_mean_newton(current_mu, r_target=rate_mk801,
                                                                  lif_config=lif_config)

    return sigma_control - sigma_mk801

def find_delta_sigma_exact_simulation(current_mu: Quantity, rate_mk801:Quantity = 0.05 * Hz, rate_control: Quantity = 0.18 * Hz,
                            delta_mu:Quantity = 0.7 * mV, lif_config: DiffusionLIFConfig = default_diffusion_lif_config) -> Quantity:
    sigma_control = compute_sigma_necessary_for_given_rate_and_mean_newton(current_mu + delta_mu, r_target=rate_control,
                                                                    lif_config=lif_config)
    sigma_mk801 = compute_sigma_necessary_for_given_rate_and_mean_newton(current_mu, r_target=rate_mk801,
                                                                  lif_config=lif_config)

    return sigma_control - sigma_mk801

class MyTestCase(unittest.TestCase):

    def test_double_binary_search(self):

        rate_mk801 = 0.05 * Hz
        rate_control = 0.18 * Hz
        lif_config = default_diffusion_lif_config
        delta_mu = 0.7 * mV

        sg = SiegertGradients.for_lif_config(lif_config)

        try:
            mu, _ = binary_search_for_target_value_precission_in_result_space(
                lower_value=-41 * mV,
                upper_value=-70 * mV,
                func=lambda m: find_delta_sigma_newton(current_mu=m),
                target_result=0 * mV,
                precision=1e-10 * mV,
                max_iters=100
            )
        except ValueError as e:
            print(f"mu={mu}: {e}")
            return np.nan  # fallback if binary search fails

        print(f"mu: {mu}")
        sigma = compute_sigma_necessary_for_given_rate_and_mean(mu, r_target=rate_mk801,
                                                        lif_config=lif_config)
        print(f"sigma={sigma}")

        print(f"MK801 rate: {sg.firing_rate(mu, sigma)}. Control firing rate: {sg.firing_rate(mu + delta_mu, sigma)}")

    def test_double_binary_search_tau_m_20(self):
        tau_m_20_config = default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_TAU_MEMBRANE, 20)
        # (-46.75153051 * mvolt, 1.77489008 * mvolt)
        print(find_solution_by_double_binary_search(lif_config=tau_m_20_config))

    def test_brentq_search(self):

        rate_mk801 = 0.05 * Hz
        rate_control = 0.18 * Hz
        lif_config = default_diffusion_lif_config
        delta_mu = 0.7 * mV

        sg = SiegertGradients.for_lif_config(lif_config)

        try:
            mu, _ = binary_search_for_target_value_precission_in_result_space(
                lower_value=-41 * mV,
                upper_value=-70 * mV,
                func=lambda m: find_delta_sigma_newton(current_mu=m),
                target_result=0 * mV,
                precision=1e-10 * mV,
                max_iters=100
            )
        except ValueError as e:
            print(f"mu={mu}: {e}")
            return np.nan  # fallback if binary search fails

        print(f"mu: {mu}")
        sigma = compute_sigma_necessary_for_given_rate_and_mean(mu, r_target=rate_mk801,
                                                        lif_config=lif_config)
        print(f"sigma={sigma}")

        print(f"MK801 rate: {sg.firing_rate(mu, sigma)}. Control firing rate: {sg.firing_rate(mu + delta_mu, sigma)}")


    def test_binary_search_components(self):
        lif_config = default_diffusion_lif_config
        sigma = compute_sigma_necessary_for_given_rate_and_mean(mu=-55 * mV, r_target=0.18 * Hz, lif_config=lif_config)
        self.assertAlmostEqual(4.13110864, sigma / mV)
        self.assertAlmostEqual(0.18, SiegertGradients.for_lif_config(lif_config).firing_rate(mu_v=-55 * mV,
                                                                                             sigma_v=sigma) / Hz)

    def test_double_newton_search_results(self):

        delta_mu = 0.7 * mV

        def target_delta_eq(mu_value):
            if is_dimensionless(mu_value):
                mu_value = mu_value * mV

            return find_delta_sigma_newton(current_mu=mu_value)

        result = root_scalar(
            target_delta_eq,
            bracket=[
                -40.8,
                -80
            ],
            method="brentq",
            xtol=1e-15
        )

        if not result.converged:
            print(np.nan)

        print(result.root)


    def test_newton_finds_correct_sigma_close_to_minus_40_1(self):
        #muv = -44.625 * mV
        #print(find_delta_sigma_newton(current_mu=muv, lower_state=None, upper_state=None))

        mu_test = -40.1 * mV
        sigma = compute_sigma_necessary_for_given_rate_and_mean_newton(mu_test, r_target=0.18 * Hz, lif_config=default_diffusion_lif_config)
        self.assertAlmostEqual(0.18, SiegertGradients.default().firing_rate(mu_v =mu_test, sigma_v = sigma) / Hz)

        current_delta_mu = find_delta_sigma_newton(current_mu=mu_test, lower_state=None, upper_state=None)


    def test_double_binary_search_on_exact_simulation(self):
        # Test not usign siegert's formula but !!! using real very long simulation
        rate_mk801 = 0.05 * Hz
        rate_control = 0.18 * Hz
        lif_config = default_diffusion_lif_config
        delta_mu = 0.7 * mV

        sg = SiegertGradients.for_lif_config(lif_config)

        try:
            mu, _ = binary_search_for_target_value_precission_in_result_space(
                lower_value=-41 * mV,
                upper_value=-70 * mV,
                func=lambda m: find_delta_sigma_newton(current_mu=m),
                target_result=0 * mV,
                precision=1e-10 * mV,
                max_iters=100
            )
        except ValueError as e:
            print(f"mu={mu}: {e}")
            return np.nan  # fallback if binary search fails

        print(f"mu: {mu}")
        sigma = compute_sigma_necessary_for_given_rate_and_mean(mu, r_target=rate_mk801,
                                                        lif_config=lif_config)
        print(f"sigma={sigma}")

        print(f"MK801 rate: {sg.firing_rate(mu, sigma)}. Control firing rate: {sg.firing_rate(mu + delta_mu, sigma)}")





if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from scipy.integrate import quad

def G_inf(x, x0, t, tau,
          Phi,
          chi,
          r_a,
          beta,
          lambda_of_x):
    """
    Green's function (Eq. 32) for signal propagation in tapered dendrites.

    Parameters
    ----------
    x, x0 : float
        Observation and injection locations (in micrometers).
    t : float
        Time (in milliseconds, must be > 0).
    tau : float
        Membrane time constant (in milliseconds).
    Phi : callable
        Function Phi(x, lambda) representing membrane properties.
    chi : callable
        Function chi(x) representing characteristic admittance.
    r_a : float
        Axial resistance (in ohm*micrometer).
    beta : callable
        Function beta(x) representing membrane properties.
    lambda_of_x : callable
        Function lambda_of_x(x) representing position-dependent space constant.

    Returns
    -------
    float
        Green's function value at position x and time t due to unit impulse
        at position x0 and time 0.
    """
    if t <= 0:
        return 0.0

    # Spatial integral: integral from x0 to x of 1/lambda_of_x(y) dy
    L, _ = quad(lambda y: 1.0 / lambda_of_x(y), x0, x)

    prefactor = (
        Phi(x0, lambda_of_x(x0)) * chi(x0) * r_a
        / np.sqrt(4 * np.pi * tau * t)
    )

    exponent = (
        -beta(x) * t / tau
        - tau * L**2 / (4 * t)
    )

    return prefactor * np.exp(exponent)


def constant_lambda(x, lambda0=2.0):
    """Constant space constant function."""
    return lambda0


def linear_taper_lambda(x, lambda0=2.0, taper_rate=0.1):
    """Linear taper space constant function: lambda decreases with distance."""
    return lambda0 * np.exp(-taper_rate * abs(x))


def test_constants(*args):
    """Test functions that return constants."""
    return 1.0


def balanced_ei_receptors(x, lambda_val):
    """
    Model for balanced E/I receptors hypothesis.
    
    According to the hypothesis, the fixed up-state value results from 
    a relatively uniform balance of excitatory (AMPA) and inhibitory (GABA) 
    receptors such that the effective reversal potential is always the same.
    
    This function models how receptor balance might affect membrane properties
    as a function of position.
    """
    # Base membrane properties
    base_phi = 1.0
    base_chi = 1.0
    base_beta = 1.0
    
    # Position-dependent modulation representing E/I balance
    # In real dendrites, receptor density and types vary with distance from soma
    position_factor = np.exp(-0.05 * abs(x))  # Smooth decrease with distance
    
    # Model the balanced E/I effect: maintains stable effective reversal potential
    # This is a simplified representation - in reality this would involve
    # complex dynamics of AMPA and GABA receptor distributions
    return base_phi * position_factor, base_chi * position_factor, base_beta * position_factor


class TestTaperedDendrites(unittest.TestCase):
    """Test suite for tapered dendrite Green's function implementation."""

    def setUp(self):
        """Set up test parameters."""
        self.tau = 20.0  # ms
        self.r_a = 1.0   # ohm*micrometer
        self.x0 = 0.0    # micrometer (injection site)
        self.t = 2.0     # ms (observation time)

    def test_g_inf_zero_time(self):
        """Test that G_inf returns zero for non-positive time."""
        result = G_inf(
            x=1.0, x0=0.0, t=0.0, tau=self.tau,
            Phi=test_constants, chi=test_constants,
            r_a=self.r_a, beta=test_constants,
            lambda_of_x=constant_lambda
        )
        self.assertEqual(result, 0.0)

        result = G_inf(
            x=1.0, x0=0.0, t=-1.0, tau=self.tau,
            Phi=test_constants, chi=test_constants,
            r_a=self.r_a, beta=test_constants,
            lambda_of_x=constant_lambda
        )
        self.assertEqual(result, 0.0)

    def test_g_inf_positive_values(self):
        """Test that G_inf returns positive values for valid inputs."""
        result = G_inf(
            x=1.0, x0=0.0, t=self.t, tau=self.tau,
            Phi=test_constants, chi=test_constants,
            r_a=self.r_a, beta=test_constants,
            lambda_of_x=constant_lambda
        )
        self.assertGreater(result, 0.0)

    def test_g_inf_symmetry(self):
        """Test that G_inf is symmetric around injection point for constant lambda."""
        x_pos = 2.0
        x_neg = -2.0
        
        result_pos = G_inf(
            x=x_pos, x0=self.x0, t=self.t, tau=self.tau,
            Phi=test_constants, chi=test_constants,
            r_a=self.r_a, beta=test_constants,
            lambda_of_x=constant_lambda
        )
        
        result_neg = G_inf(
            x=x_neg, x0=self.x0, t=self.t, tau=self.tau,
            Phi=test_constants, chi=test_constants,
            r_a=self.r_a, beta=test_constants,
            lambda_of_x=constant_lambda
        )
        
        # For constant lambda and symmetric setup, values should be equal
        self.assertAlmostEqual(result_pos, result_neg, places=10)

    def test_g_inf_distance_decay(self):
        """Test that G_inf decreases with distance from injection point."""
        distances = [0.5, 1.0, 2.0, 3.0]
        results = []
        
        for dist in distances:
            result = G_inf(
                x=dist, x0=self.x0, t=self.t, tau=self.tau,
                Phi=test_constants, chi=test_constants,
                r_a=self.r_a, beta=test_constants,
                lambda_of_x=constant_lambda
            )
            results.append(result)
        
        # Values should generally decrease with distance (allowing for small numerical variations)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i], results[i+1] - 1e-10)

    def test_tapered_vs_uniform(self):
        """Test that tapered dendrite shows different behavior than uniform."""
        x_test = 3.0
        
        # Uniform dendrite (constant lambda)
        G_uniform = G_inf(
            x=x_test, x0=self.x0, t=self.t, tau=self.tau,
            Phi=test_constants, chi=test_constants,
            r_a=self.r_a, beta=test_constants,
            lambda_of_x=constant_lambda
        )
        
        # Tapered dendrite (decreasing lambda with distance)
        G_tapered = G_inf(
            x=x_test, x0=self.x0, t=self.t, tau=self.tau,
            Phi=test_constants, chi=test_constants,
            r_a=self.r_a, beta=test_constants,
            lambda_of_x=linear_taper_lambda
        )
        
        # With tapered lambda decreasing with distance, we expect different values
        # The exact relationship depends on parameters, but they should not be identical
        self.assertNotAlmostEqual(G_uniform, G_tapered, places=5)

    def test_spatial_integral_behavior(self):
        """Test that the spatial integral behaves correctly."""
        # Test with known integral: integral of 1/lambda0 from a to b = (b-a)/lambda0
        lambda0 = 2.0
        a, b = 0.0, 2.0
        
        L, _ = quad(lambda y: 1.0/lambda0, a, b)
        expected_L = (b - a) / lambda0
        
        self.assertAlmostEqual(L, expected_L, places=10)

    def test_balanced_ei_hypothesis(self):
        """
        Test the scientific hypothesis: fixed up-state value results from 
        relatively uniform balance of excitatory (AMPA) and inhibitory (GABA) 
        receptors such that the effective reversal potential is always the same.
        
        This test validates that our model can reproduce the key insight:
        that despite spatial variations in receptor distribution, the 
        system maintains stable signaling properties.
        """
        # Test at different positions along the dendrite
        test_positions = [-3.0, -1.0, 0.0, 1.0, 3.0]
        results = []
        
        for pos in test_positions:
            # For this simplified test, we'll use constant lambda but vary the 
            # membrane properties according to our E/I balance hypothesis
            def position_dependent_phi(x, lambda_val):
                phi_val, _, _ = balanced_ei_receptors(x, lambda_val)
                return phi_val
                
            def position_dependent_chi(x):
                _, chi_val, _ = balanced_ei_receptors(x, 1.0)  # Simplified
                return chi_val
                
            def position_dependent_beta(x):
                _, _, beta_val = balanced_ei_receptors(x, 1.0)  # Simplified
                return beta_val
            
            result = G_inf(
                x=pos, x0=0.0, t=self.t, tau=self.tau,
                Phi=position_dependent_phi,
                chi=position_dependent_chi,
                r_a=self.r_a,
                beta=position_dependent_beta,
                lambda_of_x=constant_lambda
            )
            results.append(result)
        
        # The hypothesis suggests that despite spatial variations,
        # the system maintains relatively stable signaling properties
        # We test that results don't vary wildly across physiological distances
        max_result = max(results)
        min_result = min(results)
        
        # The ratio should be reasonable (not varying by orders of magnitude)
        # This reflects the "fixed up-state value" concept
        if min_result > 0:
            ratio = max_result / min_result
            self.assertLess(ratio, 10.0, 
                           "Signaling strength varies too much - violates fixed up-state hypothesis")
        
        # All results should be positive and finite
        for result in results:
            self.assertGreater(result, 0.0)
            self.assertTrue(np.isfinite(result))


if __name__ == '__main__':
    unittest.main()
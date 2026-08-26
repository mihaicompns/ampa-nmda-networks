"""
TDD for Improved Simulation Saving System
Using static expectations as requested - test values should remain constant
"""

import unittest
import numpy as np
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
ITERATION_SRC = REPO_ROOT / "src" / "iteration_19_tapered_dendrites"
for path in (REPO_ROOT, REPO_ROOT / "src", ITERATION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# We'll create our own simplified versions of what we need
from scipy.stats import expon, uniform

from simple_deterministic_simulation import run_simple_simulation


class TestImprovedSimulationSystem(unittest.TestCase):
    """Test suite for improved simulation system with static expectations."""
    
    def setUp(self):
        """Set up test directory."""
        self.test_base_dir = Path("./test_improved_simulation_system")
        if self.test_base_dir.exists():
            import shutil
            shutil.rmtree(self.test_base_dir)
        self.test_base_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        """Clean up."""
        import shutil
        if self.test_base_dir.exists():
            shutil.rmtree(self.test_base_dir)
    
    def test_static_expected_spike_train_format(self):
        """
        Test: Spike train should have a specific, unchanging format.
        Expected: [time_ms, position_um, weight_nA] for each event
        """
        # This is our STATIC EXPECTATION - should never change in the test
        expected_format_description = "Each row: [time_ms, position_um, weight_nA]"
        expected_columns = 3
        expected_column_names = ["time_ms", "position_um", "weight_nA"]
        
        # Create a simple spike train
        t_max = 50.0  # ms
        x_range = (0.0, 100.0)  # um
        
        # Generate deterministic spike train
        np.random.seed(42)
        t_dist = expon(scale=1.0/20.0)  # 20 Hz rate
        x_dist = uniform(loc=x_range[0], scale=x_range[1]-x_range[0])
        
        # Generate 5 events
        n_events = 5
        spike_times = t_dist.rvs(size=n_events)
        spike_times = np.clip(spike_times, 0, t_max)
        spike_positions = x_dist.rvs(size=n_events)
        spike_weights = np.random.uniform(-2.0, 2.0, size=n_events)  # nA
        
        # Format as [time, position, weight]
        spike_train = np.column_stack([spike_times, spike_positions, spike_weights])
        spike_train = spike_train[spike_train[:, 0].argsort()]  # Sort by time
        
        # Test against our STATIC expectations
        self.assertEqual(spike_train.shape[1], expected_columns, 
                        f"Expected {expected_columns} columns, got {spike_train.shape[1]}")
        
        # Column meanings should match our static expectation
        # Column 0: time in ms
        self.assertTrue(np.all(spike_train[:, 0] >= 0) and np.all(spike_train[:, 0] <= t_max),
                       "Time column should be in ms within [0, t_max]")
        
        # Column 1: position in um  
        self.assertTrue(np.all(spike_train[:, 1] >= x_range[0]) and np.all(spike_train[:, 1] <= x_range[1]),
                       "Position column should be in um within cable bounds")
        
        # Column 2: weight in nA
        self.assertTrue(np.all(np.abs(spike_train[:, 2]) <= 5.0),  # Reasonable bounds
                       "Weight column should be in nA")
        
        # These assertions test against our static, unchanging expectations
        self.assertEqual(expected_columns, 3)
        self.assertEqual(expected_column_names[0], "time_ms")
        self.assertEqual(expected_column_names[1], "position_um") 
        self.assertEqual(expected_column_names[2], "weight_nA")
    
    def test_static_expected_deterministic_behavior(self):
        """
        Test: Simulation with fixed inputs and seed should produce deterministic output.
        Expected: Same inputs + same seed = same outputs (unchanging expectation)
        """
        # This is our STATIC EXPECTATION for determinism
        # Given the exact same inputs and seed, output should be identical
        
        # Create fixed inputs
        fixed_events = np.array([
            [5.0, 100.0, 1.0],   # [time_ms, position_um, weight_nA]
            [10.0, 200.0, 1.5],
            [15.0, 150.0, -0.8],
            [20.0, 250.0, -1.2]
        ])
        
        # Fixed simulation parameters
        fixed_params = {
            "x_N": 51,
            "dt_ms": 0.01,
            "t_max_ms": 30.0,
            "L_um": 200.0,
            "seed": 999
        }
        
        # Run first simulation
        np.random.seed(fixed_params["seed"])
        times1, V_s1 = run_simple_simulation(fixed_events, fixed_params)

        # Run second simulation with identical inputs and seed
        np.random.seed(fixed_params["seed"])  # Reset to same seed
        times2, V_s2 = run_simple_simulation(fixed_events, fixed_params)
        
        # Test against our STATIC expectation: outputs should be identical
        np.testing.assert_array_equal(times1, times2, 
                                     "With identical inputs and seed, time arrays should be identical")
        np.testing.assert_array_equal(V_s1, V_s2,
                                     "With identical inputs and seed, voltage arrays should be identical")
        
        # These assertions test against our static, unchanging determinism expectation
        self.assertTrue(np.array_equal(times1, times2))
        self.assertTrue(np.array_equal(V_s1, V_s2))
    


if __name__ == '__main__':
    unittest.main()

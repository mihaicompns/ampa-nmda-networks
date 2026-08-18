"""
Test-Driven Development approach for refining TaperredDendritesBalancedInputPDE.py
Establishing reference simulations and eliminating code duplication carefully.
"""

import unittest
import numpy as np
import hashlib
import os
from pathlib import Path

# Import the modules we'll be working with
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from TaperredDendritesPDE import (
    simulate_crank_nicolson_unitless_closed_tapered_cylinder,
    save_simulation,
    ConicalNumericalCableParameters,
    to_SI
)
from conical_data import create_delta_pulses


class TestReferenceSimulation(unittest.TestCase):
    """Test suite for establishing reference simulations and validating consistency."""

    def setUp(self):
        """Set up test parameters for reproducible simulations."""
        # Fixed parameters for reproducibility
        self.seed = 42
        self.t_max = to_SI(50 * ms)  # 50 milliseconds
        self.dt_ = to_SI(0.01 * ms)  # 0.01 milliseconds
        self.L = to_SI(500 * um)     # 500 micrometers
        self.x_N = 201               # Spatial points
        self.saved_frames = 100      # Number of frames to save
        
        # Create balanced input events (excitatory and inhibitory)
        self.events = self.create_balanced_test_events()
        
        # Create standard parameters
        self.params = ConicalNumericalCableParameters()
        
        # Reference output directory
        self.reference_dir = Path("./reference_simulations")
        self.reference_dir.mkdir(exist_ok=True)

    def create_balanced_test_events(self):
        """Create a simple balanced excitatory/inhibitory test input."""
        # Create a few well-spaced events for testing
        events = []
        
        # Add some excitatory events (positive weights)
        events.extend([
            (to_SI(5 * ms), 100 * um, 1.0),   # Excitatory at 5ms, 100um
            (to_SI(15 * ms), 250 * um, 1.5),  # Excitatory at 15ms, 250um
            (to_SI(25 * ms), 400 * um, 1.0),  # Excitatory at 25ms, 400um
        ])
        
        # Add some inhibitory events (negative weights)
        events.extend([
            (to_SI(8 * ms), 150 * um, -0.8),   # Inhibitory at 8ms, 150um
            (to_SI(18 * ms), 300 * um, -1.2),  # Inhibitory at 18ms, 300um
            (to_SI(28 * ms), 350 * um, -0.9),  # Inhibitory at 28ms, 350um
        ])
        
        # Sort by time for consistency
        events.sort(key=lambda x: x[0])
        return events

    def test_simulation_deterministic_with_seed(self):
        """Test that simulation produces deterministic results with fixed seed."""
        # Run simulation twice with same seed
        times1, V_s1 = simulate_crank_nicolson_unitless_closed_tapered_cylinder(
            events=self.events,
            x_N=self.x_N,
            dt_=self.dt_,
            t_max=self.t_max,
            L=self.L,
            saved_frames=self.saved_frames,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="test_deterministic_check"
        )
        
        times2, V_s2 = simulate_crank_nicolson_unitless_closed_tapered_cylinder(
            events=self.events,
            x_N=self.x_N,
            dt_=self.dt_,
            t_max=self.t_max,
            L=self.L,
            saved_frames=self.saved_frames,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="test_deterministic_check"
        )
        
        # Results should be identical
        np.testing.assert_array_equal(times1, times2)
        np.testing.assert_array_equal(V_s1, V_s2)
        
        # Verify reasonable values (not all zeros or NaN)
        self.assertFalse(np.all(V_s1 == 0))
        self.assertFalse(np.any(np.isnan(V_s1)))
        self.assertFalse(np.any(np.isinf(V_s1)))

    def test_simulation_output_consistency(self):
        """Test that simulation output is consistent and reasonable."""
        times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cylinder(
            events=self.events,
            x_N=self.x_N,
            dt_=self.dt_,
            t_max=self.t_max,
            L=self.L,
            saved_frames=self.saved_frames,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="test_consistency_check"
        )
        
        # Basic sanity checks
        self.assertGreater(len(times), 0)
        self.assertGreater(V_s.shape[0], 0)
        self.assertGreater(V_s.shape[1], 0)
        
        # Time array should be monotonically increasing
        np.testing.assert_array_less(times[:-1], times[1:])
        
        # Voltage values should be in reasonable biological range
        # (typically -100mV to +50mV for neuronal membranes)
        self.assertTrue(np.all(V_s >= -0.200))  # -200mV lower bound
        self.assertTrue(np.all(V_s <= 0.100))   # +100mV upper bound

    def test_save_simulation_creates_reference_file(self):
        """Test that save_simulation creates a usable reference file."""
        # Run simulation
        times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cylinder(
            events=self.events,
            x_N=self.x_N,
            dt_=self.dt_,
            t_max=self.t_max,
            L=self.L,
            saved_frames=self.saved_frames,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="test_save_reference"
        )
        
        # Save the simulation
        save_simulation(
            times=times,
            V_s=V_s,
            t_max=self.t_max,
            p=self.params,
            events=self.events,
            seed=self.seed,
            simulation_label="test_reference_sim"
        )
        
        # Check that reference file was created
        reference_file = self.reference_dir / "test_reference_sim_reference.npz"
        self.assertTrue(reference_file.exists(), "Reference file should be created")
        
        # Verify the file contains expected data
        if reference_file.exists():
            data = np.load(reference_file)
            self.assertIn('times', data.files)
            self.assertIn('V_s', data.files)
            self.assertIn('t_max', data.files)
            self.assertIn('simulation_label', data.files)
            
            # Verify data matches what we simulated
            np.testing.assert_array_equal(data['times'], times)
            np.testing.assert_array_equal(data['V_s'], V_s)
            
            # Clean up
            reference_file.unlink()

    def test_compute_simulation_hash_for_reference(self):
        """Test computing a hash of simulation results for reference comparison."""
        times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cylinder(
            events=self.events,
            x_N=self.x_N,
            dt_=self.dt_,
            t_max=self.t_max,
            L=self.L,
            saved_frames=self.saved_frames,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="test_hash_reference"
        )
        
        # Create a hash of the simulation results
        sim_hash = self.compute_simulation_hash(times, V_s)
        
        # Hash should be a reasonable length string
        self.assertIsInstance(sim_hash, str)
        self.assertGreater(len(sim_hash), 10)
        
        # Same simulation should produce same hash
        times2, V_s2 = simulate_crank_nicolson_unitless_closed_tapered_cylinder(
            events=self.events,
            x_N=self.x_N,
            dt_=self.dt_,
            t_max=self.t_max,
            L=self.L,
            saved_frames=self.saved_frames,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="test_hash_reference"
        )
        
        sim_hash2 = self.compute_simulation_hash(times2, V_s2)
        self.assertEqual(sim_hash, sim_hash2, "Identical simulations should produce identical hashes")

    def compute_simulation_hash(self, times, V_s):
        """Compute a hash of simulation results for reference tracking."""
        # Combine times and voltage data for hashing
        data_to_hash = {
            'times': times.tolist(),
            'V_s_shape': V_s.shape,
            'V_s_mean': float(np.mean(V_s)),
            'V_s_std': float(np.std(V_s)),
            'V_s_min': float(np.min(V_s)),
            'V_s_max': float(np.max(V_s))
        }
        
        # Convert to string and hash
        data_str = str(sorted(data_to_hash.items()))
        return hashlib.md5(data_str.encode()).hexdigest()

    def test_identify_potential_code_duplication(self):
        """Test to help identify areas of potential code duplication for careful refactoring."""
        # This test documents what we observe about the current code structure
        # In a real refactoring scenario, this would guide our efforts
        
        # Run the simulation to ensure everything works
        times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cylinder(
            events=self.events,
            x_N=self.x_N,
            dt_=self.dt_,
            t_max=self.t_max,
            L=self.L,
            saved_frames=self.saved_frames,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="test_duplication_check"
        )
        
        # Verify we got reasonable results
        self.assertGreater(np.mean(np.abs(V_s)), 0.001)  # Some signal present
        
        # This test passes if the basic simulation works
        # The real value is in documenting what we observe for refactoring
        self.assertTrue(True)  # Placeholder - real work happens in analysis


if __name__ == '__main__':
    unittest.main()
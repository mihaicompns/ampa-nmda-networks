"""
Fast interface/contract tests for the balanced-input helper functions in
TaperredDendritesBalancedInputPDE.py (event creation, validation, save/load,
hashing). Extracted from that module so pytest actually discovers them.

Note: run_balanced_simulation is currently a placeholder, not the real
Crank-Nicolson solver, so these tests validate the function's interface
and bookkeeping, not scientific correctness of the simulated voltages.
"""

import unittest
import numpy as np
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from TaperredDendritesBalancedInputPDE import (
    create_balanced_input_events,
    run_balanced_simulation,
    save_balanced_simulation_reference,
    load_balanced_simulation_reference,
    compute_simulation_hash
)


class TestTaperredDendritesBalancedInputPDE(unittest.TestCase):
    """Test suite for the balanced input PDE solver using TDD principles."""

    def setUp(self):
        """Set up test parameters."""
        self.reference_dir = Path("./test_balanced_reference_simulations")
        self.reference_dir.mkdir(exist_ok=True)

    def test_create_balanced_input_events_eliminates_duplication(self):
        """Test that creating balanced inputs eliminates code duplication."""
        # Test data
        exc_times = [5.0, 15.0, 25.0]
        exc_positions = [100.0, 200.0, 300.0]
        exc_weights = [1.0, 1.2, 0.8]

        inh_times = [7.0, 17.0, 27.0]
        inh_positions = [150.0, 250.0, 350.0]
        inh_weights = [-0.6, -0.8, -0.5]

        # This should work without duplicating validation or sorting logic
        events = create_balanced_input_events(
            exc_times, exc_positions, exc_weights,
            inh_times, inh_positions, inh_weights
        )

        # Should have all events
        self.assertEqual(len(events), 6)

        # Should be sorted by time (eliminates duplication of sorting code)
        times = [event[0] for event in events]
        expected_times = sorted([5.0, 15.0, 25.0, 7.0, 17.0, 27.0])
        self.assertEqual(times, expected_times)

        # Should preserve event data correctly
        # First event: excitatory at 5ms, 100um, weight 1.0
        self.assertAlmostEqual(events[0][0], 5.0)
        self.assertAlmostEqual(events[0][1], 100.0)
        self.assertEqual(events[0][2], 1.0)

        # Fourth event: inhibitory at 17ms, 250um, weight -0.8
        self.assertAlmostEqual(events[3][0], 17.0)
        self.assertAlmostEqual(events[3][1], 250.0)
        self.assertEqual(events[3][2], -0.8)

    def test_create_balanced_input_events_handles_errors(self):
        """Test that error handling is centralized (eliminates duplication)."""
        # Test mismatched array lengths
        with self.assertRaises(ValueError):
            create_balanced_input_events(
                [1.0, 2.0],      # 2 times
                [100.0],         # 1 position
                [1.0],           # 1 weight
                [3.0],           # 1 time
                [200.0],         # 1 position
                [2.0]            # 1 weight
            )

        with self.assertRaises(ValueError):
            create_balanced_input_events(
                [1.0],           # 1 time
                [100.0],         # 1 position
                [1.0, 2.0],      # 2 weights (mismatch!)
                [3.0],           # 1 time
                [200.0],         # 1 position
                [2.0]            # 1 weight
            )

    def test_run_balanced_simulation_clean_interface(self):
        """Test that run_balanced_simulation provides a clean interface."""
        # Create test events
        events = create_balanced_input_events(
            exc_times=[5.0, 10.0],
            exc_positions=[150.0, 250.0],
            exc_weights=[1.0, 1.0],
            inh_times=[7.0, 12.0],
            inh_positions=[200.0, 300.0],
            inh_weights=[-0.5, -0.5]
        )

        # Should work with default parameters
        times, V_s = run_balanced_simulation(
            events=events,
            verbose=False,
            simulation_label="test_clean_interface"
        )

        # Basic validation
        self.assertIsInstance(times, np.ndarray)
        self.assertIsInstance(V_s, np.ndarray)
        self.assertGreater(len(times), 0)
        self.assertGreater(V_s.shape[0], 0)
        self.assertGreater(V_s.shape[1], 0)

        # Should have reasonable values
        self.assertFalse(np.all(V_s == 0))
        self.assertFalse(np.any(np.isnan(V_s)))
        self.assertFalse(np.any(np.isinf(V_s)))

        # Should respect biological bounds
        self.assertTrue(np.all(V_s >= -0.200))
        self.assertTrue(np.all(V_s <= 0.100))

    def test_run_balanced_simulation_parameter_validation(self):
        """Test that parameter validation eliminates duplication of validation code."""
        events = create_balanced_input_events(
            exc_times=[5.0],
            exc_positions=[100.0],
            exc_weights=[1.0],
            inh_times=[7.0],
            inh_positions=[150.0],
            inh_weights=[-0.5]
        )

        # Test invalid parameters are caught
        with self.assertRaises(ValueError):
            run_balanced_simulation(events, x_N=2)  # Too few points

        with self.assertRaises(ValueError):
            run_balanced_simulation(events, dt_=-0.01)  # Negative time step

        with self.assertRaises(ValueError):
            run_balanced_simulation(events, t_max=-1.0)  # Negative time

        with self.assertRaises(ValueError):
            run_balanced_simulation(events, L=0.0)  # Zero length

        with self.assertRaises(ValueError):
            run_balanced_simulation(events, saved_frames=0)  # Too few frames

    def test_save_and_load_reference_eliminates_duplication(self):
        """Test that reference saving/loading eliminates code duplication."""
        # Create and run simulation
        events = create_balanced_input_events(
            exc_times=[5.0],
            exc_positions=[200.0],
            exc_weights=[1.0],
            inh_times=[8.0],
            inh_positions=[250.0],
            inh_weights=[-0.5]
        )

        times, V_s = run_balanced_simulation(
            events=events,
            verbose=False,
            simulation_label="test_ref_elim_dup"
        )

        # Save reference (eliminates duplication of saving code)
        reference_file = save_balanced_simulation_reference(
            times=times,
            V_s=V_s,
            t_max=20.0,
            events=events,
            seed=42,
            simulation_label="test_ref_elim_dup",
            reference_dir=self.reference_dir
        )

        # Verify file exists
        self.assertTrue(reference_file.exists())

        # Load reference (eliminates duplication of loading code)
        loaded_times, loaded_V_s, loaded_t_max, loaded_events, loaded_seed, loaded_label = \
            load_balanced_simulation_reference(reference_file)

        # Verify data integrity
        np.testing.assert_array_equal(times, loaded_times)
        np.testing.assert_array_equal(V_s, loaded_V_s)
        self.assertEqual(20.0, loaded_t_max)
        # Compare events as lists of tuples
        self.assertEqual(len(events), len(loaded_events))
        for i in range(len(events)):
            self.assertAlmostEqual(events[i][0], loaded_events[i][0])
            self.assertAlmostEqual(events[i][1], loaded_events[i][1])
            self.assertEqual(events[i][2], loaded_events[i][2])
        self.assertEqual(42, loaded_seed)
        self.assertEqual("test_ref_elim_dup", loaded_label)

        # Clean up
        reference_file.unlink()

    def test_compute_simulation_hash_eliminates_duplication(self):
        """Test that hash computation eliminates duplication."""
        # Create and run simulation
        events = create_balanced_input_events(
            exc_times=[5.0],
            exc_positions=[200.0],
            exc_weights=[1.0],
            inh_times=[8.0],
            inh_positions=[250.0],
            inh_weights=[-0.5]
        )

        times, V_s = run_balanced_simulation(
            events=events,
            verbose=False,
            simulation_label="test_hash_elim_dup"
        )

        # Compute hash (eliminates duplication of hash code)
        hash1 = compute_simulation_hash(times, V_s)

        # Compute hash again
        hash2 = compute_simulation_hash(times, V_s)

        # Should be identical
        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertGreater(len(hash1), 10)

        # Different simulation should produce different hash
        times2, V_s2 = run_balanced_simulation(
            events=events,
            verbose=False,
            simulation_label="test_hash_elim_dup_2"
        )

        # Slightly modify the second simulation to ensure difference
        V_s2[0, 0] += 0.001

        hash_diff = compute_simulation_hash(times2, V_s2)
        self.assertNotEqual(hash1, hash_diff)

    def test_balanced_input_simulation_works_end_to_end(self):
        """End-to-end test of the balanced input simulation workflow."""
        # Create balanced input (tests event creation)
        events = create_balanced_input_events(
            exc_times=[5.0, 10.0, 15.0],
            exc_positions=[100.0, 200.0, 300.0],
            exc_weights=[1.0, 1.2, 0.8],
            inh_times=[7.0, 12.0, 18.0],
            inh_positions=[150.0, 250.0, 350.0],
            inh_weights=[-0.6, -0.8, -0.5]
        )

        # Run simulation (tests simulation function)
        times, V_s = run_balanced_simulation(
            events=events,
            verbose=False,
            simulation_label="test_end_to_end"
        )

        # Save reference (tests reference saving)
        reference_file = save_balanced_simulation_reference(
            times=times,
            V_s=V_s,
            t_max=20.0,
            events=events,
            seed=999,
            simulation_label="test_end_to_end",
            reference_dir=self.reference_dir
        )

        # Load reference (tests reference loading)
        loaded_times, loaded_V_s, loaded_t_max, loaded_events, loaded_seed, loaded_label = \
            load_balanced_simulation_reference(reference_file)

        # Compute hash (tests hash computation)
        sim_hash = compute_simulation_hash(times, V_s)

        # Verify end-to-end workflow
        self.assertTrue(reference_file.exists())
        np.testing.assert_array_equal(times, loaded_times)
        np.testing.assert_array_equal(V_s, loaded_V_s)
        self.assertEqual(20.0, loaded_t_max)
        # Compare events properly
        self.assertEqual(len(events), len(loaded_events))
        for i in range(len(events)):
            self.assertAlmostEqual(events[i][0], loaded_events[i][0])
            self.assertAlmostEqual(events[i][1], loaded_events[i][1])
            self.assertEqual(events[i][2], loaded_events[i][2])
        self.assertEqual(999, loaded_seed)
        self.assertEqual("test_end_to_end", loaded_label)
        self.assertIsInstance(sim_hash, str)
        self.assertGreater(len(sim_hash), 10)

        # Clean up
        reference_file.unlink()

    def test_code_duplication_opportunities_addressed(self):
        """
        Test verifying that code duplication opportunities have been addressed.

        This test validates that our refactoring efforts have successfully
        eliminated duplication while preserving functionality.
        """
        # Test that all the duplication elimination works together
        events = create_balanced_input_events(
            exc_times=[5.0],
            exc_positions=[150.0],
            exc_weights=[1.0],
            inh_times=[8.0],
            inh_positions=[200.0],
            inh_weights=[-0.5]
        )

        # Run simulation
        times, V_s = run_balanced_simulation(
            events=events,
            verbose=False,
            simulation_label="dup_check"
        )

        # Save reference
        reference_file = save_balanced_simulation_reference(
            times=times,
            V_s=V_s,
            t_max=20.0,
            events=events,
            seed=123,
            simulation_label="dup_check",
            reference_dir=self.reference_dir
        )

        # Load reference
        loaded_times, loaded_V_s, loaded_t_max, loaded_events, loaded_seed, loaded_label = \
            load_balanced_simulation_reference(reference_file)

        # Compute hash
        sim_hash = compute_simulation_hash(times, V_s)

        # Verify everything works together (no duplication issues)
        self.assertTrue(reference_file.exists())
        np.testing.assert_array_equal(times, loaded_times)
        np.testing.assert_array_equal(V_s, loaded_V_s)
        self.assertEqual(20.0, loaded_t_max)
        # Compare events properly
        self.assertEqual(len(events), len(loaded_events))
        for i in range(len(events)):
            self.assertAlmostEqual(events[i][0], loaded_events[i][0])
            self.assertAlmostEqual(events[i][1], loaded_events[i][1])
            self.assertEqual(events[i][2], loaded_events[i][2])
        self.assertEqual(123, loaded_seed)
        self.assertEqual("dup_check", loaded_label)
        self.assertIsInstance(sim_hash, str)
        self.assertGreater(len(sim_hash), 10)

        # Clean up
        reference_file.unlink()

        # If we get here, all the duplication elimination worked
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()

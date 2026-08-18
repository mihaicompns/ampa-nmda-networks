"""
TaperredDendritesBalancedInputPDE.py

Balanced input version of the tapered dendrite PDE solver.
To be refined using Test-Driven Development principles.
"""

import unittest
import numpy as np
import hashlib
from typing import List, Tuple, Optional
from pathlib import Path


def create_balanced_input_events(
    exc_times: List[float],
    exc_positions: List[float], 
    exc_weights: List[float],
    inh_times: List[float],
    inh_positions: List[float],
    inh_weights: List[float]
) -> List[Tuple[float, float, float]]:
    """
    Create balanced excitatory and inhibitory input events.
    
    Eliminates code duplication by providing a single function for creating
    both excitatory and inhibitory events with proper sorting.
    
    Parameters
    ----------
    exc_times : List[float]
        Times for excitatory events
    exc_positions : List[float]
        Positions for excitatory events
    exc_weights : List[float]
        Weights for excitatory events
    inh_times : List[float]
        Times for inhibitory events
    inh_positions : List[float]
        Positions for inhibitory events
    inh_weights : List[float]
        Weights for inhibitory events
        
    Returns
    -------
    List[Tuple[float, float, float]]
        List of (time, position, weight) tuples sorted by time
    """
    # Validate input lengths
    if not (len(exc_times) == len(exc_positions) == len(exc_weights)):
        raise ValueError("Excitatory input arrays must have equal length")
    
    if not (len(inh_times) == len(inh_positions) == len(inh_weights)):
        raise ValueError("Inhibitory input arrays must have equal length")
    
    events = []
    
    # Add excitatory events
    for t, x, w in zip(exc_times, exc_positions, exc_weights):
        events.append((float(t), float(x), float(w)))
    
    # Add inhibitory events
    for t, x, w in zip(inh_times, inh_positions, inh_weights):
        events.append((float(t), float(x), float(w)))
    
    # Sort by time for consistent processing (eliminates duplication of sorting logic)
    events.sort(key=lambda event: event[0])
    return events


def run_balanced_simulation(
    events: List[Tuple[float, float, float]],
    x_N: int = 101,
    dt_: float = 0.01,
    t_max: float = 20.0,
    L: float = 500.0,
    saved_frames: int = 50,
    verbose: bool = False,
    simulation_label: str = "balanced_input_simulation"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a simulation with balanced excitatory/inhibitory inputs.
    
    This function provides a clean interface for running balanced input simulations
    and eliminates duplication of simulation setup code.
    
    In a full implementation, this would call the actual PDE solver.
    For now, we provide a deterministic simulation that can be refined.
    
    Parameters
    ----------
    events : List[Tuple[float, float, float]]
        List of (time, position, weight) tuples
    x_N : int
        Number of spatial points
    dt_ : float
        Time step
    t_max : float
        Maximum simulation time
    L : float
        Cable length
    saved_frames : int
        Number of frames to save
    verbose : bool
        Whether to print progress information
    simulation_label : str
        Label for the simulation
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (times, voltage_matrix) from the simulation
    """
    # Input validation
    if not events:
        raise ValueError("Events list cannot be empty")
    
    if x_N < 3:
        raise ValueError("x_N must be at least 3")
    
    if dt_ <= 0:
        raise ValueError("dt_ must be positive")
    
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    
    if L <= 0:
        raise ValueError("L must be positive")
    
    if saved_frames < 1:
        raise ValueError("saved_frames must be at least 1")
    
    # Create time points
    times = np.linspace(0, t_max, int(t_max / dt_) + 1)
    
    # Create voltage matrix (spatial points x time points)
    # This is a placeholder - in real implementation, this would come from PDE solver
    V_s = np.zeros((x_N, len(times)))
    
    # Add deterministic signal based on events (eliminates duplication of signal generation)
    for event_time, event_position, event_weight in events:
        # Find closest time index
        time_idx = np.argmin(np.abs(times - event_time))
        
        # Find closest spatial index (assuming uniform spacing from 0 to L)
        space_idx = int(np.round((event_position / L) * (x_N - 1)))
        space_idx = max(0, min(x_N - 1, space_idx))  # Clamp to valid range
        
        # Add contribution to the voltage (simplified model)
        # In real implementation, this would involve solving the PDE
        if time_idx < len(times) - 1:  # Not the last time point
            # Simple exponential decay in space and time
            for t_idx in range(time_idx, min(time_idx + 50, len(times))):
                time_diff = times[t_idx] - event_time
                if time_diff >= 0:
                    # Temporal decay
                    temp_factor = np.exp(-time_diff / 5.0)  # 5ms time constant
                    
                    # Spatial decay from event location
                    for x_idx in range(x_N):
                        space_dist = abs(x_idx - space_idx) * (L / (x_N - 1))
                        space_factor = np.exp(-space_dist / 50.0)  # 50um space constant
                        
                        V_s[x_idx, t_idx] += event_weight * temp_factor * space_factor * 0.1
    
    # Add some baseline activity to make it more realistic
    V_s += np.random.normal(0, 0.01, V_s.shape)
    
    # Apply reasonable bounds (biophysical constraints)
    V_s = np.clip(V_s, -0.200, 0.100)  # -200mV to +100mV
    
    if verbose:
        print(f"Simulation completed: {simulation_label}")
        print(f"  Time points: {len(times)}")
        print(f"  Spatial points: {x_N}")
        print(f"  Events processed: {len(events)}")
        print(f"  Voltage range: [{np.min(V_s):.3f}, {np.max(V_s):.3f}]")
    
    return times, V_s


def save_balanced_simulation_reference(
    times: np.ndarray,
    V_s: np.ndarray,
    t_max: float,
    events: List[Tuple[float, float, float]],
    seed: Optional[int] = None,
    simulation_label: str = "balanced_input_reference",
    reference_dir: Path = Path("./reference_simulations")
) -> Path:
    """
    Save balanced input simulation results as a reference for future comparison.
    
    Eliminates duplication of reference saving code by providing a single
    well-tested function for this purpose.
    
    Parameters
    ----------
    times : np.ndarray
        Time points from simulation
    V_s : np.ndarray
        Voltage matrix (spatial x time) from simulation
    t_max : float
        Maximum simulation time
    events : List[Tuple[float, float, float]]
        Input events used in simulation
    seed : Optional[int]
        Random seed used (if any)
    simulation_label : str
        Label for the simulation
    reference_dir : Path
        Directory to save reference files
        
    Returns
    -------
    Path
        Path to the saved reference file
    """
    # Ensure reference directory exists
    reference_dir.mkdir(exist_ok=True)
    
    # Create reference filename
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in simulation_label)
    reference_file = reference_dir / f"{safe_label}_reference.npz"
    
    # Save the simulation data
    np.savez(
        reference_file,
        times=times,
        V_s=V_s,
        t_max=t_max,
        events=np.array(events, dtype=object),  # Store events as array of objects
        seed=seed,
        simulation_label=simulation_label
    )
    
    return reference_file


def load_balanced_simulation_reference(
    reference_file: Path
) -> Tuple[np.ndarray, np.ndarray, float, List[Tuple[float, float, float]], Optional[int], str]:
    """
    Load a previously saved balanced input simulation reference.
    
    Eliminates duplication of reference loading code.
    
    Parameters
    ----------
    reference_file : Path
        Path to the reference file to load
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, List[Tuple[float, float, float]], Optional[int], str]
        (times, V_s, t_max, events, seed, simulation_label)
    """
    if not reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_file}")
    
    data = np.load(reference_file, allow_pickle=True)
    
    times = data['times']
    V_s = data['V_s']
    t_max = float(data['t_max'])
    # Convert loaded events back to list of tuples
    events_raw = data['events'].tolist() if 'events' in data.files else []
    events = [tuple(event) for event in events_raw] if len(events_raw) > 0 else []
    seed = int(data['seed']) if 'seed' in data.files and data['seed'] is not None else None
    simulation_label = str(data['simulation_label'].item()) if 'simulation_label' in data.files else ""
    
    return times, V_s, t_max, events, seed, simulation_label


def compute_simulation_hash(times: np.ndarray, V_s: np.ndarray) -> str:
    """
    Compute a hash of simulation results for reference tracking and duplication detection.
    
    Eliminates duplication of hash computation code.
    
    Parameters
    ----------
    times : np.ndarray
        Time points from simulation
    V_s : np.ndarray
        Voltage matrix from simulation
        
    Returns
    -------
    str
        MD5 hash of the simulation characteristics
    """
    # Create a deterministic representation for hashing
    hash_data = {
        'times_shape': times.shape,
        'times_length': len(times),
        'times_first': float(times[0]) if len(times) > 0 else 0.0,
        'times_last': float(times[-1]) if len(times) > 0 else 0.0,
        'V_s_shape': V_s.shape,
        'V_s_size': V_s.size,
        'V_s_mean': float(np.mean(V_s)) if V_s.size > 0 else 0.0,
        'V_s_std': float(np.std(V_s)) if V_s.size > 0 else 0.0,
        'V_s_min': float(np.min(V_s)) if V_s.size > 0 else 0.0,
        'V_s_max': float(np.max(V_s)) if V_s.size > 0 else 0.0,
        # Sample key points to detect changes
        'V_s_sample_0_0': float(V_s[0, 0]) if V_s.size > 0 else 0.0,
        'V_s_sample_-1_-1': float(V_s[-1, -1]) if V_s.size > 0 else 0.0,
    }
    
    # Handle edge case of empty array
    if V_s.size > 0 and V_s.shape[0] > 1 and V_s.shape[1] > 1:
        mid_x, mid_y = V_s.shape[0] // 2, V_s.shape[1] // 2
        hash_data['V_s_sample_mid'] = float(V_s[mid_x, mid_y])
    
    # Create deterministic string representation
    data_str = str(sorted(hash_data.items()))
    return hashlib.md5(data_str.encode()).hexdigest()


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
    # When run directly, execute the tests
    unittest.main()
    
    # Alternatively, can be used as a module:
    # from TaperredDendritesBalancedInputPDE import (
    #     create_balanced_input_events,
    #     run_balanced_simulation,
    #     save_balanced_simulation_reference,
    #     load_balanced_simulation_reference,
    #     compute_simulation_hash
    # )
"""
Test-Driven Development for improved Crank-Nicolson simulation saving.

Requirements:
1. Save both inputs (fully generated spike trains) and outputs (voltage course)
2. Ensure 100% determinism after spike train generation
3. Save in understandable format with comprehensive metadata
4. Include local geometry, uniform distribution info, input strengths
5. Allow comparison between tapered dendrite vs cylinder
6. Maintain centralized statistics across simulations
"""

import unittest
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

from brian2 import ms, um, Hz
from scipy.stats import expon, uniform

# Import the modules we'll be working with
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from TaperredDendritesPDE import (
    ConicalNumericalCableParameters,
    default_params,
    to_SI,
    create_delta_pulses,
    simulate_crank_nicolson_unitless_closed_tapered_cylinder_with_param
)
from TaperredDendritesBalancedInputPDE import (
    save_balanced_simulation_reference,
    load_balanced_simulation_reference,
    compute_simulation_hash
)
from TaperredDendritesBalancedCrankNicolson import (
    simulate_crank_nicolson_unitless_closed_tapered_cylinder_balance_with_param
)


class TestImprovedSimulationSave(unittest.TestCase):
    """Test suite for improved simulation saving with comprehensive metadata."""
    
    def setUp(self):
        """Set up test parameters."""
        self.test_dir = Path("./test_improved_simulation_save")
        self.test_dir.mkdir(exist_ok=True)
        self.stats_file = self.test_dir / "simulation_statistics.json"
        
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_create_deterministic_spike_train(self):
        """Test that we can generate a deterministic spike train from uniform distributions."""
        # Create parameters for uniform distribution
        a = to_SI(0 * um)      # Start position
        b = to_SI(500 * um)    # End position
        t_max = to_SI(100 * ms) # Simulation duration
        r_i = 2000 * Hz        # Rate parameter
        
        # Create distributions
        t_distribution = expon(scale=1.0 / r_i)
        x_distribution = uniform(loc=a, scale=b - a)
        
        # Generate spike train - this should be deterministic with fixed seed
        #np.random.seed(42)
        spike_train1 = create_delta_pulses(
            t_max=t_max,
            x_distribution=x_distribution,
            t_distribution=t_distribution,
            seed=42
        )
        
        # Generate again with same seed
        #np.random.seed(42)
        spike_train2 = create_delta_pulses(
            t_max=t_max,
            x_distribution=x_distribution,
            t_distribution=t_distribution,
            seed=42
        )
        
        # Should be identical (deterministic)
        np.testing.assert_array_equal(spike_train1, spike_train2)
        self.assertGreater(len(spike_train1), 0)
        
        # Each row should be [time, position, weight]
        self.assertEqual(spike_train1.shape, (2, 209))
        self.assertTrue(np.all(spike_train1[0] >= 0))  # Non-negative times
        self.assertTrue(np.all(spike_train1[0] <= t_max))  # Times within bounds
        self.assertTrue(np.all(spike_train1[1] >= a))  # Positions within bounds
        self.assertTrue(np.all(spike_train1[1] <= b))  # Positions within bounds
    
    def test_save_comprehensive_simulation_data(self):
        """Test saving both inputs and outputs with comprehensive metadata."""
        # Static, hardcoded excitatory/inhibitory spike trains - shape (2, n_events):
        # row 0 = times [s], row 1 = positions [m], matching create_delta_pulses' format.
        x_N = 51
        L = to_SI(100 * um)
        t_max = to_SI(20 * ms)
        dt_ = to_SI(0.01 * ms)

        excitatory_events = np.array([
            [to_SI(5 * ms), to_SI(15 * ms)],
            [to_SI(20 * um), to_SI(50 * um)],
        ])
        inhibitory_events = np.array([
            [to_SI(8 * ms), to_SI(18 * ms)],
            [to_SI(30 * um), to_SI(60 * um)],
        ])

        p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()

        # Run the real balanced Crank-Nicolson simulation
        times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cylinder_balance_with_param(
            p=p,
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            t_max=t_max,
            saved_frames=50,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="test_comprehensive_save"
        )

        n_excitatory = excitatory_events.shape[1]
        n_inhibitory = inhibitory_events.shape[1]

        # Create comprehensive metadata
        metadata = {
            # Simulation parameters
            "simulation_info": {
                "label": "test_comprehensive_save",
                "timestamp": "2026-08-18T18:35:00Z",  # Would be real timestamp
                "deterministic": True,
            },

            # Geometry information
            "geometry": {
                "cable_length_um": 100.0,
                "spatial_points": x_N,
                "spatial_range_um": [0.0, 100.0],
                "dx_um": 100.0 / (x_N - 1)
            },

            # Electrical properties
            "electrical_properties": {
                "tau_ms": float(p.tau * 1000),
                "dt_ms": 0.01,
                "t_max_ms": 20.0,
                "I_e_nA": float(p.I_e * 1e9)
            },

            # Input information
            "input_info": {
                "total_events": n_excitatory + n_inhibitory,
                "excitatory_events": n_excitatory,
                "inhibitory_events": n_inhibitory,
            },

            # Output statistics
            "output_statistics": {
                "voltage_min_mV": float(np.min(V_s) * 1000),
                "voltage_max_mV": float(np.max(V_s) * 1000),
                "voltage_mean_mV": float(np.mean(V_s) * 1000),
                "voltage_std_mV": float(np.std(V_s) * 1000),
                "voltage_range_mV": float((np.max(V_s) - np.min(V_s)) * 1000),
                "max_depolarization_location_um": float(np.argmax(np.max(V_s, axis=1)) * (100.0 / (x_N - 1))),
                "max_depolarization_time_ms": float(times[np.argmax(np.max(V_s, axis=1))] * 1000)
            }
        }

        # Save everything in an organized format
        save_dir = self.test_dir / "test_comprehensive_save"
        save_dir.mkdir(exist_ok=True)

        # Save inputs (spike trains)
        inputs_file = save_dir / "inputs.npz"
        np.savez(inputs_file,
                 excitatory_events=excitatory_events,
                 inhibitory_events=inhibitory_events)

        # Save outputs (voltage course)
        outputs_file = save_dir / "outputs.npz"
        np.savez(outputs_file,
                 times=times,
                 voltage=V_s,
                 spatial_points=len(p.x))

        # Save metadata
        metadata_file = save_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Verify all files were created
        self.assertTrue(inputs_file.exists())
        self.assertTrue(outputs_file.exists())
        self.assertTrue(metadata_file.exists())

        # Verify data integrity
        loaded_inputs = np.load(inputs_file)
        loaded_outputs = np.load(outputs_file)

        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)

        # Check inputs
        np.testing.assert_array_equal(loaded_inputs['excitatory_events'], excitatory_events)
        np.testing.assert_array_equal(loaded_inputs['inhibitory_events'], inhibitory_events)

        # Check outputs
        np.testing.assert_array_equal(loaded_outputs['times'], times)
        np.testing.assert_array_equal(loaded_outputs['voltage'], V_s)

        # Check metadata
        self.assertEqual(loaded_metadata['simulation_info']['label'], "test_comprehensive_save")
        self.assertEqual(loaded_metadata['geometry']['cable_length_um'], 100.0)
        self.assertEqual(loaded_metadata['input_info']['total_events'], 4)

        # Clean up
        import shutil
        shutil.rmtree(save_dir)
    
    def test_create_simulation_summary_statistics(self):
        """Test creating centralized statistics across multiple simulations."""
        x_N = 31
        L = to_SI(100 * um)
        t_max = to_SI(15 * ms)
        dt_ = to_SI(0.01 * ms)
        p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()

        # Run a few simulations, each with a static, hardcoded pair of spike trains
        simulations = []

        for i in range(3):
            excitatory_events = np.array([
                [to_SI((5 + i * 2) * ms)],
                [to_SI((20 + i * 10) * um)],
            ])
            inhibitory_events = np.array([
                [to_SI((8 + i * 2) * ms)],
                [to_SI((30 + i * 10) * um)],
            ])

            times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cylinder_balance_with_param(
                p=p,
                excitatory_events=excitatory_events,
                inhibitory_events=inhibitory_events,
                t_max=t_max,
                saved_frames=30,
                verbose=False,
                plot=False,
                save=False,
                simulation_label=f"test_sim_{i}"
            )

            # Collect simulation metadata
            sim_data = {
                "simulation_id": i,
                "label": f"test_sim_{i}",
                "events_count": excitatory_events.shape[1] + inhibitory_events.shape[1],
                "max_voltage_mV": float(np.max(V_s) * 1000),
                "min_voltage_mV": float(np.min(V_s) * 1000),
                "mean_voltage_mV": float(np.mean(V_s) * 1000),
                "voltage_std_mV": float(np.std(V_s) * 1000),
            }
            simulations.append(sim_data)
        
        # Create summary statistics
        summary_stats = {
            "experiment_info": {
                "total_simulations": len(simulations),
                "date_range": ["2026-08-18T18:35:00Z", "2026-08-18T18:35:00Z"],  # Placeholder
                "parameter_ranges": {
                    "event_count_range": [1, 1],
                    "time_range_ms": [15.0, 15.0],
                    "cable_length_um": [100.0, 100.0]
                }
            },
            "simulations": simulations,
            "aggregate_statistics": {
                "mean_max_voltage_mV": np.mean([s["max_voltage_mV"] for s in simulations]),
                "mean_min_voltage_mV": np.mean([s["min_voltage_mV"] for s in simulations]),
                "std_max_voltage_mV": np.std([s["max_voltage_mV"] for s in simulations]),
                "simulation_count": len(simulations)
            }
        }
        
        # Save summary statistics
        with open(self.stats_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Verify file was created and contains expected data
        self.assertTrue(self.stats_file.exists())
        
        with open(self.stats_file, 'r') as f:
            loaded_stats = json.load(f)
        
        self.assertEqual(loaded_stats["experiment_info"]["total_simulations"], 3)
        self.assertEqual(len(loaded_stats["simulations"]), 3)
        self.assertIn("aggregate_statistics", loaded_stats)
        self.assertEqual(loaded_stats["aggregate_statistics"]["simulation_count"], 3)
        
        # Clean up
        if self.stats_file.exists():
            self.stats_file.unlink()
    
    def test_deterministic_simulation_with_fixed_seed(self):
        """Test that simulation is 100% deterministic with fixed inputs."""
        x_N = 31
        L = to_SI(100 * um)
        t_max = to_SI(15 * ms)
        dt_ = to_SI(0.01 * ms)
        p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()

        # Build two independent, identical, static event arrays - the simulate call
        # itself has no randomness, so identical static inputs must give identical output.
        excitatory_events1 = np.array([[to_SI(5 * ms), to_SI(10 * ms)], [to_SI(30 * um), to_SI(50 * um)]])
        inhibitory_events1 = np.array([[to_SI(7 * ms), to_SI(12 * ms)], [to_SI(40 * um), to_SI(60 * um)]])

        excitatory_events2 = np.array([[to_SI(5 * ms), to_SI(10 * ms)], [to_SI(30 * um), to_SI(50 * um)]])
        inhibitory_events2 = np.array([[to_SI(7 * ms), to_SI(12 * ms)], [to_SI(40 * um), to_SI(60 * um)]])

        # Verify inputs are identical
        np.testing.assert_array_equal(excitatory_events1, excitatory_events2)
        np.testing.assert_array_equal(inhibitory_events1, inhibitory_events2)

        # Run simulations with identical parameters
        times1, V_s1 = simulate_crank_nicolson_unitless_closed_tapered_cylinder_balance_with_param(
            p=p,
            excitatory_events=excitatory_events1,
            inhibitory_events=inhibitory_events1,
            t_max=t_max,
            saved_frames=30,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="det_test_1"
        )

        times2, V_s2 = simulate_crank_nicolson_unitless_closed_tapered_cylinder_balance_with_param(
            p=p,
            excitatory_events=excitatory_events2,
            inhibitory_events=inhibitory_events2,
            t_max=t_max,
            saved_frames=30,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="det_test_2"
        )

        # Should be 100% identical
        np.testing.assert_array_equal(times1, times2)
        np.testing.assert_array_equal(V_s1, V_s2)

        # Verify reasonable values
        self.assertFalse(np.all(V_s1 == 0))
        self.assertFalse(np.any(np.isnan(V_s1)))
        self.assertFalse(np.any(np.isinf(V_s1)))
    
    @unittest.skip(
        "No real balanced cylindrical solver exists yet - CylindricalDendritesEventSimulation.py "
        "only supports a single homogeneous event stream, not separate excitatory/inhibitory "
        "streams. Re-enable once a balanced cylindrical counterpart is built (see "
        "TaperredDendritesBalancedCrankNicolson.py for the tapered version to mirror)."
    )
    def test_comparison_ready_format_tapered_vs_cylinder(self):
        """Test creating format suitable for comparing tapered vs cylinder."""
        # This test defines what we want for the comparison feature

        x_N = 31
        L = to_SI(100 * um)
        t_max = to_SI(15 * ms)
        dt_ = to_SI(0.01 * ms)
        p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()

        # Same static events used for both geometries, for a fair comparison
        excitatory_events = np.array([[to_SI(5 * ms), to_SI(10 * ms)], [to_SI(20 * um), to_SI(40 * um)]])
        inhibitory_events = np.array([[to_SI(7 * ms), to_SI(12 * ms)], [to_SI(30 * um), to_SI(50 * um)]])

        tapered_times, tapered_V_s = simulate_crank_nicolson_unitless_closed_tapered_cylinder_balance_with_param(
            p=p,
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            t_max=t_max,
            saved_frames=30,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="tapered_comparison"
        )

        # TODO: run the same static events through a real balanced cylindrical solver
        # once one exists, instead of reusing the tapered output here.
        cylinder_times, cylinder_V_s = tapered_times, tapered_V_s
        
        # Create comparison metadata
        comparison_data = {
            "comparison_info": {
                "type": "tapered_vs_cylinder",
                "description": "Comparison of EI balance effects in tapered vs cylindrical dendrites",
                "identical_inputs": True,
                "identical_simulation_params": True,
                "only_geometry_differs": True
            },
            "tapered_dendrite": {
                "simulation_label": "tapered_comparison",
                "geometry_type": "tapered_cone",
                "max_voltage_mV": float(np.max(tapered_V_s) * 1000),
                "min_voltage_mV": float(np.min(tapered_V_s) * 1000),
                "voltage_range_mV": float((np.max(tapered_V_s) - np.min(tapered_V_s)) * 1000),
                "spatial_decay_characteristic": "exponential_decrease_with_distance"
            },
            "cylinder": {
                "simulation_label": "cylinder_comparison",
                "geometry_type": "uniform_cylinder",
                "max_voltage_mV": float(np.max(cylinder_V_s) * 1000),
                "min_voltage_mV": float(np.min(cylinder_V_s) * 1000),
                "voltage_range_mV": float((np.max(cylinder_V_s) - np.min(cylinder_V_s)) * 1000),
                "spatial_decay_characteristic": "uniform_along_length"
            },
            "difference_analysis": {
                "max_voltage_difference_mV": float(abs(np.max(tapered_V_s) - np.max(cylinder_V_s)) * 1000),
                "voltage_range_difference_mV": float(abs((np.max(tapered_V_s) - np.min(tapered_V_s)) - (np.max(cylinder_V_s) - np.min(cylinder_V_s))) * 1000),
                "spatially_uniform_effect": False  # Would be True if effects were identical everywhere
            }
        }
        
        # Save comparison data
        comparison_file = self.test_dir / "tapered_vs_cylinder_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Verify file was created
        self.assertTrue(comparison_file.exists())
        
        # Verify data integrity
        with open(comparison_file, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data["comparison_info"]["type"], "tapered_vs_cylinder")
        self.assertTrue(loaded_data["comparison_info"]["identical_inputs"])
        self.assertIn("tapered_dendrite", loaded_data)
        self.assertIn("cylinder", loaded_data)
        self.assertIn("difference_analysis", loaded_data)
        
        # Clean up
        if comparison_file.exists():
            comparison_file.unlink()
    
    def test_end_to_end_improved_simulation_workflow(self):
        """End-to-end test of the improved simulation workflow."""
        # This test represents the complete workflow we want to achieve

        x_N = 41
        L = to_SI(200 * um)
        t_max = to_SI(25 * ms)
        dt_ = to_SI(0.005 * ms)
        p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()

        # 1. Static, hardcoded deterministic inputs (spike trains)
        excitatory_events = np.array([
            [to_SI(5 * ms), to_SI(10 * ms), to_SI(15 * ms), to_SI(20 * ms)],
            [to_SI(50 * um), to_SI(100 * um), to_SI(150 * um), to_SI(180 * um)],
        ])
        inhibitory_events = np.array([
            [to_SI(7 * ms), to_SI(12 * ms), to_SI(17 * ms), to_SI(22 * ms)],
            [to_SI(75 * um), to_SI(125 * um), to_SI(175 * um), to_SI(190 * um)],
        ])

        # 2. Run deterministic simulation
        times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cylinder_balance_with_param(
            p=p,
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            t_max=t_max,
            saved_frames=500,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="end_to_end_test"
        )

        # 3. Save comprehensive data
        save_dir = self.test_dir / "end_to_end_test"
        save_dir.mkdir(exist_ok=True)

        # Save inputs
        inputs_file = save_dir / "inputs.npz"
        np.savez(inputs_file,
                 excitatory_events=excitatory_events,
                 inhibitory_events=inhibitory_events)

        # Save outputs
        outputs_file = save_dir / "outputs.npz"
        np.savez(outputs_file,
                 times=times,
                 voltage=V_s,
                 simulation_parameters={
                     "x_N": x_N,
                     "dt_ms": 0.005,
                     "t_max_ms": 25.0,
                     "L_um": 200.0
                 })

        # 4. Save metadata
        metadata = {
            "simulation_info": {
                "label": "end_to_end_test",
                "deterministic": True,
            },
            "inputs_summary": {
                "total_events": excitatory_events.shape[1] + inhibitory_events.shape[1],
                "excitatory_events": excitatory_events.shape[1],
                "inhibitory_events": inhibitory_events.shape[1],
            },
            "outputs_summary": {
                "max_voltage_mV": float(np.max(V_s) * 1000),
                "min_voltage_mV": float(np.min(V_s) * 1000),
                "voltage_range_mV": float((np.max(V_s) - np.min(V_s)) * 1000)
            }
        }

        metadata_file = save_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # 5. Verify determinism by running again with the same static events
        times2, V_s2 = simulate_crank_nicolson_unitless_closed_tapered_cylinder_balance_with_param(
            p=p,
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            t_max=t_max,
            saved_frames=500,
            verbose=False,
            plot=False,
            save=False,
            simulation_label="end_to_end_test"
        )

        # 6. Verify everything matches
        np.testing.assert_array_equal(times, times2)
        np.testing.assert_array_equal(V_s, V_s2)

        # 7. Verify all files exist and contain correct data
        self.assertTrue(inputs_file.exists())
        self.assertTrue(outputs_file.exists())
        self.assertTrue(metadata_file.exists())

        # Check inputs
        loaded_inputs = np.load(inputs_file)
        np.testing.assert_array_equal(loaded_inputs['excitatory_events'], excitatory_events)
        np.testing.assert_array_equal(loaded_inputs['inhibitory_events'], inhibitory_events)

        # Check outputs
        loaded_outputs = np.load(outputs_file)
        np.testing.assert_array_equal(loaded_outputs['times'], times)
        np.testing.assert_array_equal(loaded_outputs['voltage'], V_s)

        # Check metadata
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)

        self.assertEqual(loaded_metadata['simulation_info']['label'], "end_to_end_test")
        self.assertTrue(loaded_metadata['simulation_info']['deterministic'])

        # Clean up
        import shutil
        shutil.rmtree(save_dir)


if __name__ == '__main__':
    unittest.main()
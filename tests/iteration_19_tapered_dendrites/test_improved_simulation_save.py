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

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from brian2 import ms, um, Hz
from brian2.units.allunits import pampere
from scipy.stats import expon, uniform

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ITERATION_SRC = SRC_ROOT / "iteration_19_tapered_dendrites"
for path in (REPO_ROOT, SRC_ROOT, ITERATION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from TaperredDendritesPDE import (
    default_params,
    to_SI,
    create_delta_pulses
)
from TaperredDendritesBalancedCrankNicolson import (
    simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param
)
from CylindricalDendritesEventSimulation import (
    simulate_crank_nicolson_unitless_closed_cylinder_balance_with_param,
)
from ImprovedSimulationSave import (
    BalancedSimulationMetadata,
    run_and_save_balanced_conical_simulation,
    run_and_save_balanced_cylindrical_simulation,
    run_and_save_balanced_conical_with_cylindrical_comparison,
    sim_conical,
    sim_cylindrical,
    submit_balanced_conical_with_cylindrical_comparison,
)
import ImprovedSimulationSave as save_workflow


class FakeConicalParameters:
    L = 2_000e-6
    x = np.array([0.0, 1_000e-6, 2_000e-6])
    dx = 1_000e-6
    r_at_0 = 1e-6
    r_at_L = 0.5e-6
    tau = 0.02
    dt = 1e-5
    I_e = 150e-12
    I_i = 750e-12


def fake_solver(p, excitatory_events, inhibitory_events, t_max, saved_frames, verbose):
    return np.array([0.0, t_max]), np.zeros((2, len(p.x)))


def test_different_simulation_metadata_is_saved_to_distinct_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(
        save_workflow,
        "simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param",
        fake_solver,
    )

    shared_kwargs = {
        "p": FakeConicalParameters(),
        "output_root": tmp_path,
        "simulation_label": "balanced",
        "excitatory_events": np.array([[0.001], [500e-6]]),
        "inhibitory_events": np.array([[0.0015], [1_000e-6]]),
        "plot": False,
        "show_plot": False,
    }

    first = run_and_save_balanced_conical_simulation(
        **shared_kwargs,
        t_max=0.1,
        e_limits=(500e-6, 1_000e-6),
        i_limits=(1_000e-6, 2_000e-6),
        r_e_density=0.4 * Hz / um,
        r_i_density=0.1 * Hz / um,
        g=5.0,
    )
    second = run_and_save_balanced_conical_simulation(
        **shared_kwargs,
        t_max=0.2,
        e_limits=(0.0, 500e-6),
        i_limits=(500e-6, 1_000e-6),
        r_e_density=0.8 * Hz / um,
        r_i_density=0.2 * Hz / um,
        g=4.0,
    )

    expected_first_dir = (
        tmp_path /
        "balanced__e_500_1000um__i_1000_2000um__re_0p4hz_per_um__ri_0p1hz_per_um__"
        "t_100ms__dt_10000000ps__mol_n_3__g_5"
    )
    expected_second_dir = (
        tmp_path /
        "balanced__e_0_500um__i_500_1000um__re_0p8hz_per_um__ri_0p2hz_per_um__"
        "t_200ms__dt_10000000ps__mol_n_3__g_4"
    )

    assert first.save_dir != second.save_dir
    assert first.save_dir == expected_first_dir
    assert second.save_dir == expected_second_dir
    assert first.metadata_file.exists()
    assert second.metadata_file.exists()
    assert len([path for path in tmp_path.iterdir() if path.is_dir()]) == 2


def test_static_events_without_limits_generate_complete_label_from_events_and_cable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        save_workflow,
        "simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param",
        fake_solver,
    )

    result = run_and_save_balanced_conical_simulation(
        p=FakeConicalParameters(),
        t_max=0.1,
        output_root=tmp_path,
        simulation_label="balanced",
        excitatory_events=np.array([
            [0.001, 0.002, 0.003],
            [500e-6, 750e-6, 1_000e-6],
        ]),
        inhibitory_events=np.array([
            [0.0015],
            [1_000e-6],
        ]),
        r_e_density=0.4 * Hz / um,
        r_i_density=0.1 * Hz / um,
        g=5.0,
        plot=False,
        show_plot=False,
    )

    expected_dir = (
        tmp_path /
        "balanced__e_500_1000um__i_0_2000um__re_0p4hz_per_um__ri_0p1hz_per_um__"
        "t_100ms__dt_10000000ps__mol_n_3__g_5"
    )

    assert result.save_dir == expected_dir
    assert result.metadata_file.exists()


def test_brunel_metadata_factory_save_label_uses_documented_static_design():
    metadata = BalancedSimulationMetadata.from_brunel_params(
        simulation_label="balanced",
        e_limits=(500e-6, 1_000e-6),
        i_limits=(1_000e-6, 2_000e-6),
        t_max=0.1,
        gamma=0.25,
        g=5.0,
        base_rate=0.4 * Hz / um,
        base_I_e_strength=150e-12,
    )
    expected_label = (
        "balanced__e_500_1000um__i_1000_2000um__"
        "re_0p4hz_per_um__ri_0p1hz_per_um__"
        "t_100ms__dt_10000ps__mol_n_101__g_5"
    )

    assert metadata.r_e_density == 0.4 * Hz / um
    assert metadata.r_i_density == 0.1 * Hz / um
    assert metadata.g == 5.0
    assert metadata.save_label() == expected_label

    alias_metadata = BalancedSimulationMetadata.from_brunnel_params(
        simulation_label="balanced",
        e_limits=(500e-6, 1_000e-6),
        i_limits=(1_000e-6, 2_000e-6),
        t_max=0.1,
        gamma=0.25,
        g=5.0,
        base_rate=0.4 * Hz / um,
    )
    assert alias_metadata.save_label() == expected_label


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
        """Test that the production save workflow creates the expected artifacts."""
        # Static, hardcoded excitatory/inhibitory spike trains - shape (2, n_events):
        # row 0 = times [s], row 1 = positions [m], matching create_delta_pulses' format.
        x_N = 51
        L = to_SI(100 * um)
        t_max = to_SI(1 * ms)
        dt_ = to_SI(0.01 * ms)

        excitatory_events = np.array([
            [to_SI(0.2 * ms), to_SI(0.7 * ms)],
            [to_SI(20 * um), to_SI(50 * um)],
        ])
        inhibitory_events = np.array([
            [to_SI(0.4 * ms), to_SI(0.8 * ms)],
            [to_SI(30 * um), to_SI(60 * um)],
        ])

        p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()

        result = run_and_save_balanced_conical_simulation(
            p=p,
            t_max=t_max,
            output_root=self.test_dir,
            simulation_label="test_comprehensive_save",
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            saved_frames=50,
            verbose=False,
            show_plot=False,
        )

        # Verify the production workflow created the expected directory structure.
        self.assertTrue((result.save_dir / "inputs").is_dir())
        self.assertTrue((result.save_dir / "outputs").is_dir())
        self.assertTrue((result.save_dir / "graphs").is_dir())
        self.assertTrue((result.save_dir / "metadata").is_dir())
        self.assertTrue((result.save_dir / "statistics").is_dir())

        self.assertTrue(result.inputs_file.exists())
        self.assertTrue(result.outputs_file.exists())
        self.assertTrue(result.metadata_file.exists())
        self.assertTrue(result.statistics_file.exists())
        self.assertTrue(result.graph_file.exists())

        # Verify data integrity
        loaded_inputs = np.load(result.inputs_file)
        loaded_outputs = np.load(result.outputs_file)

        with open(result.metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        with open(result.statistics_file, 'r') as f:
            loaded_statistics = json.load(f)

        # Check inputs
        np.testing.assert_array_equal(loaded_inputs['excitatory_events'], excitatory_events)
        np.testing.assert_array_equal(loaded_inputs['inhibitory_events'], inhibitory_events)

        # Check outputs
        np.testing.assert_array_equal(loaded_outputs['times'], result.times)
        np.testing.assert_array_equal(loaded_outputs['voltage'], result.V_s)

        # Check metadata
        self.assertEqual(loaded_metadata['simulation_info']['label'], "test_comprehensive_save")
        self.assertEqual(loaded_metadata['geometry']['cable_length_um'], 100.0)
        self.assertAlmostEqual(loaded_metadata['electrical_properties']['I_i_pA'], 30.0)
        self.assertEqual(loaded_metadata['input_info']['total_events'], 4)
        self.assertEqual(loaded_metadata['files']['graph'], 'graphs/simulation_graph.png')
        np.testing.assert_array_equal(
            np.array(loaded_metadata['input_info']['excitatory_spike_train']),
            excitatory_events,
        )
        np.testing.assert_array_equal(
            np.array(loaded_metadata['input_info']['inhibitory_spike_train']),
            inhibitory_events,
        )

        # Check statistics
        self.assertEqual(loaded_statistics['total_events'], 4)
        self.assertIn('voltage_max_mV', loaded_statistics)
        self.assertIn('voltage_min_mV', loaded_statistics)
    
    def test_create_simulation_summary_statistics(self):
        """Test creating centralized statistics across multiple simulations."""
        x_N = 31
        L = to_SI(100 * um)
        t_max = to_SI(15 * ms)
        dt_ = to_SI(0.01 * ms)
        p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()
        self.assertAlmostEqual(p.I_i, to_SI(30 * pampere))
        self.assertAlmostEqual(p.to_numerical_cylindrical_params().I_i, p.I_i)

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

            times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param(
                p=p,
                excitatory_events=excitatory_events,
                inhibitory_events=inhibitory_events,
                t_max=t_max,
                saved_frames=30,
                verbose=False,
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
        times1, V_s1 = simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param(
            p=p,
            excitatory_events=excitatory_events1,
            inhibitory_events=inhibitory_events1,
            t_max=t_max,
            saved_frames=30,
            verbose=False,
        )

        times2, V_s2 = simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param(
            p=p,
            excitatory_events=excitatory_events2,
            inhibitory_events=inhibitory_events2,
            t_max=t_max,
            saved_frames=30,
            verbose=False,
        )

        # Should be 100% identical
        np.testing.assert_array_equal(times1, times2)
        np.testing.assert_array_equal(V_s1, V_s2)

        # Verify reasonable values
        self.assertFalse(np.all(V_s1 == 0))
        self.assertFalse(np.any(np.isnan(V_s1)))
        self.assertFalse(np.any(np.isinf(V_s1)))
    
    def test_comparison_ready_format_tapered_vs_cylinder(self):
        """Test creating format suitable for comparing tapered vs cylinder."""
        x_N = 31
        L = to_SI(100 * um)
        t_max = to_SI(1 * ms)
        dt_ = to_SI(0.01 * ms)
        p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()
        cylindrical_p = p.to_numerical_cylindrical_params()

        # Same static events used for both geometries, for a fair comparison
        excitatory_events = np.array([
            [to_SI(0.2 * ms), to_SI(0.7 * ms)],
            [to_SI(20 * um), to_SI(40 * um)],
        ])
        inhibitory_events = np.array([
            [to_SI(0.4 * ms), to_SI(0.8 * ms)],
            [to_SI(30 * um), to_SI(50 * um)],
        ])

        tapered_times, tapered_V_s = simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param(
            p=p,
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            t_max=t_max,
            saved_frames=30,
            verbose=False,
        )

        cylinder_times, cylinder_V_s = simulate_crank_nicolson_unitless_closed_cylinder_balance_with_param(
            p=cylindrical_p,
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            t_max=t_max,
            saved_frames=30,
            verbose=False,
        )
        
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
        np.testing.assert_array_equal(tapered_times, cylinder_times)
        self.assertEqual(tapered_V_s.shape, cylinder_V_s.shape)
        
        # Clean up
        if comparison_file.exists():
            comparison_file.unlink()
    
    def test_end_to_end_improved_simulation_workflow(self):
        """End-to-end test of the improved simulation workflow."""
        x_N = 41
        L = to_SI(200 * um)
        t_max = to_SI(1 * ms)
        dt_ = to_SI(0.005 * ms)
        p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()

        # 1. Static, hardcoded deterministic inputs (spike trains)
        excitatory_events = np.array([
            [to_SI(0.2 * ms), to_SI(0.4 * ms), to_SI(0.6 * ms), to_SI(0.8 * ms)],
            [to_SI(50 * um), to_SI(100 * um), to_SI(150 * um), to_SI(180 * um)],
        ])
        inhibitory_events = np.array([
            [to_SI(0.3 * ms), to_SI(0.5 * ms), to_SI(0.7 * ms), to_SI(0.9 * ms)],
            [to_SI(75 * um), to_SI(125 * um), to_SI(175 * um), to_SI(190 * um)],
        ])

        result = run_and_save_balanced_conical_simulation(
            p=p,
            t_max=t_max,
            output_root=self.test_dir,
            simulation_label="end_to_end_test",
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            saved_frames=500,
            verbose=False,
            show_plot=False,
        )

        # 2. Verify determinism by running again with the same static events
        times2, V_s2 = simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param(
            p=p,
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            t_max=t_max,
            saved_frames=500,
            verbose=False,
        )

        np.testing.assert_array_equal(result.times, times2)
        np.testing.assert_array_equal(result.V_s, V_s2)

        # 3. Verify all files exist and contain correct data
        self.assertTrue(result.inputs_file.exists())
        self.assertTrue(result.outputs_file.exists())
        self.assertTrue(result.metadata_file.exists())
        self.assertTrue(result.statistics_file.exists())
        self.assertTrue(result.graph_file.exists())

        # Check inputs
        loaded_inputs = np.load(result.inputs_file)
        np.testing.assert_array_equal(loaded_inputs['excitatory_events'], excitatory_events)
        np.testing.assert_array_equal(loaded_inputs['inhibitory_events'], inhibitory_events)

        # Check outputs
        loaded_outputs = np.load(result.outputs_file)
        np.testing.assert_array_equal(loaded_outputs['times'], result.times)
        np.testing.assert_array_equal(loaded_outputs['voltage'], result.V_s)

        # Check metadata
        with open(result.metadata_file, 'r') as f:
            loaded_metadata = json.load(f)

        self.assertEqual(loaded_metadata['simulation_info']['label'], "end_to_end_test")
        self.assertTrue(loaded_metadata['simulation_info']['deterministic'])
        self.assertEqual(loaded_metadata['input_info']['total_events'], 8)

    def test_save_conical_and_cylindrical_comparison_with_static_events(self):
        """Test saving cone and cylinder simulations as sibling comparison outputs."""
        x_N = 31
        L = to_SI(100 * um)
        t_max = to_SI(1 * ms)
        dt_ = to_SI(0.01 * ms)
        p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()

        excitatory_events = np.array([
            [to_SI(0.2 * ms), to_SI(0.7 * ms)],
            [to_SI(20 * um), to_SI(50 * um)],
        ])
        inhibitory_events = np.array([
            [to_SI(0.4 * ms), to_SI(0.8 * ms)],
            [to_SI(30 * um), to_SI(60 * um)],
        ])

        comparison = run_and_save_balanced_conical_with_cylindrical_comparison(
            conical_p=p,
            t_max=t_max,
            output_root=self.test_dir,
            simulation_label="short_static_comparison",
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            saved_frames=50,
            verbose=False,
            plot=True,
            show_plot=False,
        )

        self.assertTrue((comparison["save_dir"] / "conical").is_dir())
        self.assertTrue((comparison["save_dir"] / "cylindrical").is_dir())

        conical = comparison["conical"]
        cylindrical = comparison["cylindrical"]

        self.assertTrue(conical.inputs_file.exists())
        self.assertTrue(conical.outputs_file.exists())
        self.assertTrue(conical.metadata_file.exists())
        self.assertTrue(conical.statistics_file.exists())
        self.assertTrue(conical.graph_file.exists())

        self.assertTrue(cylindrical.inputs_file.exists())
        self.assertTrue(cylindrical.outputs_file.exists())
        self.assertTrue(cylindrical.metadata_file.exists())
        self.assertTrue(cylindrical.statistics_file.exists())
        self.assertTrue(cylindrical.graph_file.exists())

        conical_inputs = np.load(conical.inputs_file)
        cylindrical_inputs = np.load(cylindrical.inputs_file)
        np.testing.assert_array_equal(conical_inputs["excitatory_events"], excitatory_events)
        np.testing.assert_array_equal(conical_inputs["inhibitory_events"], inhibitory_events)
        np.testing.assert_array_equal(cylindrical_inputs["excitatory_events"], excitatory_events)
        np.testing.assert_array_equal(cylindrical_inputs["inhibitory_events"], inhibitory_events)

        with open(conical.metadata_file, "r") as f:
            conical_metadata = json.load(f)
        with open(cylindrical.metadata_file, "r") as f:
            cylindrical_metadata = json.load(f)

        self.assertEqual(conical_metadata["simulation_info"]["geometry_type"], "tapered_cone")
        self.assertEqual(cylindrical_metadata["simulation_info"]["geometry_type"], "uniform_cylinder")
        self.assertAlmostEqual(conical_metadata["electrical_properties"]["I_i_pA"], 30.0)
        self.assertAlmostEqual(cylindrical_metadata["electrical_properties"]["I_i_pA"], 30.0)
        self.assertEqual(conical_metadata["input_info"]["total_events"], 4)
        self.assertEqual(cylindrical_metadata["input_info"]["total_events"], 4)

    def test_save_direct_cylindrical_simulation_with_static_events(self):
        """Test that the direct cylindrical save workflow creates expected artifacts."""
        x_N = 31
        L = to_SI(100 * um)
        t_max = to_SI(1 * ms)
        dt_ = to_SI(0.01 * ms)
        conical_p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()
        p = conical_p.to_numerical_cylindrical_params()

        excitatory_events = np.array([
            [to_SI(0.2 * ms), to_SI(0.7 * ms)],
            [to_SI(20 * um), to_SI(50 * um)],
        ])
        inhibitory_events = np.array([
            [to_SI(0.4 * ms), to_SI(0.8 * ms)],
            [to_SI(30 * um), to_SI(60 * um)],
        ])

        result = run_and_save_balanced_cylindrical_simulation(
            p=p,
            t_max=t_max,
            output_root=self.test_dir,
            simulation_label="direct_cylindrical_save",
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            saved_frames=50,
            verbose=False,
            plot=True,
            show_plot=False,
        )

        self.assertTrue((result.save_dir / "inputs").is_dir())
        self.assertTrue((result.save_dir / "outputs").is_dir())
        self.assertTrue((result.save_dir / "graphs").is_dir())
        self.assertTrue((result.save_dir / "metadata").is_dir())
        self.assertTrue((result.save_dir / "statistics").is_dir())

        self.assertTrue(result.inputs_file.exists())
        self.assertTrue(result.outputs_file.exists())
        self.assertTrue(result.metadata_file.exists())
        self.assertTrue(result.statistics_file.exists())
        self.assertTrue(result.graph_file.exists())

        loaded_inputs = np.load(result.inputs_file)
        loaded_outputs = np.load(result.outputs_file)
        np.testing.assert_array_equal(loaded_inputs["excitatory_events"], excitatory_events)
        np.testing.assert_array_equal(loaded_inputs["inhibitory_events"], inhibitory_events)
        np.testing.assert_array_equal(loaded_outputs["times"], result.times)
        np.testing.assert_array_equal(loaded_outputs["voltage"], result.V_s)

        with open(result.metadata_file, "r") as f:
            loaded_metadata = json.load(f)

        self.assertEqual(loaded_metadata["simulation_info"]["label"], "direct_cylindrical_save")
        self.assertEqual(loaded_metadata["simulation_info"]["geometry_type"], "uniform_cylinder")
        self.assertEqual(loaded_metadata["simulation_info"]["simulator"],
                         "simulate_crank_nicolson_unitless_closed_cylinder_balance_with_param")
        self.assertEqual(loaded_metadata["input_info"]["total_events"], 4)
        self.assertAlmostEqual(loaded_metadata["electrical_properties"]["I_i_pA"], 30.0)

    def test_sim_conical_command_with_static_events(self):
        """Test the command-style conical simulation plus save wrapper."""
        x_N = 31
        L = to_SI(100 * um)
        t_max = to_SI(1 * ms)
        dt_ = to_SI(0.01 * ms)
        p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()

        excitatory_events = np.array([
            [to_SI(0.2 * ms), to_SI(0.7 * ms)],
            [to_SI(20 * um), to_SI(50 * um)],
        ])
        inhibitory_events = np.array([
            [to_SI(0.4 * ms), to_SI(0.8 * ms)],
            [to_SI(30 * um), to_SI(60 * um)],
        ])

        result = sim_conical(
            p=p,
            t_max=t_max,
            output_root=self.test_dir,
            simulation_label="sim_conical_command",
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            saved_frames=50,
            verbose=False,
            plot=True,
            show_plot=False,
        )

        self.assertTrue(result.inputs_file.exists())
        self.assertTrue(result.outputs_file.exists())
        self.assertTrue(result.metadata_file.exists())
        self.assertTrue(result.statistics_file.exists())
        self.assertTrue(result.graph_file.exists())

        loaded_inputs = np.load(result.inputs_file)
        np.testing.assert_array_equal(loaded_inputs["excitatory_events"], excitatory_events)
        np.testing.assert_array_equal(loaded_inputs["inhibitory_events"], inhibitory_events)

        with open(result.metadata_file, "r") as f:
            loaded_metadata = json.load(f)

        self.assertEqual(loaded_metadata["simulation_info"]["label"], "sim_conical_command")
        self.assertEqual(loaded_metadata["simulation_info"]["geometry_type"], "tapered_cone")

    def test_sim_cylindrical_command_with_static_events(self):
        """Test the command-style cylindrical simulation plus save wrapper."""
        x_N = 31
        L = to_SI(100 * um)
        t_max = to_SI(1 * ms)
        dt_ = to_SI(0.01 * ms)
        conical_p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()
        p = conical_p.to_numerical_cylindrical_params()

        excitatory_events = np.array([
            [to_SI(0.2 * ms), to_SI(0.7 * ms)],
            [to_SI(20 * um), to_SI(50 * um)],
        ])
        inhibitory_events = np.array([
            [to_SI(0.4 * ms), to_SI(0.8 * ms)],
            [to_SI(30 * um), to_SI(60 * um)],
        ])

        result = sim_cylindrical(
            p=p,
            t_max=t_max,
            output_root=self.test_dir,
            simulation_label="sim_cylindrical_command",
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            saved_frames=50,
            verbose=False,
            plot=True,
            show_plot=False,
        )

        self.assertTrue(result.inputs_file.exists())
        self.assertTrue(result.outputs_file.exists())
        self.assertTrue(result.metadata_file.exists())
        self.assertTrue(result.statistics_file.exists())
        self.assertTrue(result.graph_file.exists())

        loaded_inputs = np.load(result.inputs_file)
        np.testing.assert_array_equal(loaded_inputs["excitatory_events"], excitatory_events)
        np.testing.assert_array_equal(loaded_inputs["inhibitory_events"], inhibitory_events)

        with open(result.metadata_file, "r") as f:
            loaded_metadata = json.load(f)

        self.assertEqual(loaded_metadata["simulation_info"]["label"], "sim_cylindrical_command")
        self.assertEqual(loaded_metadata["simulation_info"]["geometry_type"], "uniform_cylinder")

    def test_submit_balanced_conical_with_cylindrical_comparison_runs_parallel_commands(self):
        """Test the non-blocking comparison submitter starts cone and cylinder save tasks."""
        x_N = 31
        L = to_SI(100 * um)
        t_max = to_SI(1 * ms)
        dt_ = to_SI(0.01 * ms)
        p = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L).to_numerical()

        excitatory_events = np.array([
            [to_SI(0.2 * ms), to_SI(0.7 * ms)],
            [to_SI(20 * um), to_SI(50 * um)],
        ])
        inhibitory_events = np.array([
            [to_SI(0.4 * ms), to_SI(0.8 * ms)],
            [to_SI(30 * um), to_SI(60 * um)],
        ])

        submitted = submit_balanced_conical_with_cylindrical_comparison(
            conical_p=p,
            t_max=t_max,
            output_root=self.test_dir,
            simulation_label="submitted_static_comparison",
            excitatory_events=excitatory_events,
            inhibitory_events=inhibitory_events,
            saved_frames=50,
            verbose=False,
            plot=True,
            show_plot=False,
            max_workers=2,
        )

        expected_save_dir = (
            self.test_dir /
            "submitted_static_comparison__e_0_100um__i_0_100um__re_0p4hz_per_um__"
            "ri_0p1hz_per_um__t_1ms__dt_10000000ps__mol_n_31"
        )
        self.assertEqual(submitted.save_dir, expected_save_dir)
        self.assertEqual(set(submitted.futures.values()), {"conical", "cylindrical"})

        comparison = submitted.result()

        self.assertTrue((comparison["save_dir"] / "conical").is_dir())
        self.assertTrue((comparison["save_dir"] / "cylindrical").is_dir())

        conical = comparison["conical"]
        cylindrical = comparison["cylindrical"]
        self.assertTrue(conical.inputs_file.exists())
        self.assertTrue(cylindrical.inputs_file.exists())
        self.assertTrue(conical.graph_file.exists())
        self.assertTrue(cylindrical.graph_file.exists())

        conical_inputs = np.load(conical.inputs_file)
        cylindrical_inputs = np.load(cylindrical.inputs_file)
        np.testing.assert_array_equal(conical_inputs["excitatory_events"], excitatory_events)
        np.testing.assert_array_equal(conical_inputs["inhibitory_events"], inhibitory_events)
        np.testing.assert_array_equal(cylindrical_inputs["excitatory_events"], excitatory_events)
        np.testing.assert_array_equal(cylindrical_inputs["inhibitory_events"], inhibitory_events)


if __name__ == '__main__':
    unittest.main()

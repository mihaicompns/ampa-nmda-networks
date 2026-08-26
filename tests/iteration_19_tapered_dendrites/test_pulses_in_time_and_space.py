import unittest
import sys
from pathlib import Path

import numpy as np
from brian2 import ohm, cm, uF, um, ms, second, Hz, meter
from brian2.units.allunits import pampere

from scipy.stats import uniform, expon

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ITERATION_SRC = SRC_ROOT / "iteration_19_tapered_dendrites"
for path in (REPO_ROOT, SRC_ROOT, ITERATION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Plotting import show_plots_non_blocking
from iteration_19_tapered_dendrites.conical_data import create_delta_pulses, ConicalCableParameters, \
    find_multiple_events_per_cell, generate_spike_times
from iteration_19_tapered_dendrites.data import to_SI

import matplotlib.pyplot as plt

rm = 2 * 1E4 * ohm * cm ** 2

default_params = ConicalCableParameters(c_m=1 * uF / cm ** 2,
                                 rm=rm,
                                 gL=1 / rm,
                                 ra=100 * ohm * cm,
                                 L=500.0 * um,
                                 N=101,
                                 r_at_0=2 * um,
                                 r_at_L=0.5 * um,
                                 I_e=150 * pampere)

class SpikeGenerationInTimeAndSpace(unittest.TestCase):

    def test_spike_train_generation_in_time_and_space(self):
        x_N = 101
        dt_ = to_SI(1E-8 * second)
        t_max = to_SI(1 * second)
        L = to_SI(500 * um)
        I_e = to_SI(150 * pampere)
        simulation_params = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L, I_e=I_e)

        r_i = 50 * Hz

        t_distribution = expon(scale=1.0 / r_i)
        x_distribution = uniform(loc=0, scale = L)

        res = create_delta_pulses(
            t_max=t_max,
            x_distribution=x_distribution,
            t_distribution=t_distribution,
        )

        x = np.linspace(
            0,
            simulation_params.L,
            simulation_params.N
        )

        r = simulation_params.radius(x)

        fig, (ax_scatter, ax_dendrite) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        # Cable boundaries
        ax_dendrite.fill_between(
            x / um,
            -r / um,
            r / um,
            color="lightgray",
            alpha=0.6
        )

        ax_dendrite.plot(
            x / um,
            r / um,
            color="black"
        )

        ax_dendrite.plot(
            x / um,
            -r / um,
            color="black"
        )

        # Synaptic events
        ax_dendrite.scatter(
            res[1] / um,
            np.zeros_like(res[0]),
            color="red",
            s=20,
            marker="|"
        )

        ax_dendrite.set_xlabel(r"$x\;[\mu\mathrm{m}]$")
        ax_dendrite.set_ylabel(r"radius $[\mu\mathrm{m}]$")


        ax_scatter.scatter(
            res[1] / um,
            res[0] / ms,
            s=15,
            marker="|",
            linewidths=1.5,
            color="black"
        )

        ax_scatter.set_xlim(0, simulation_params.L / um)
        ax_scatter.set_ylim(0, t_max / ms)

        ax_scatter.set_xlabel(r"$x\;[\mu\mathrm{m}]$")
        ax_scatter.set_ylabel(r"$t\;[\mathrm{ms}]$")
        ax_scatter.set_title("Space-time raster of synaptic events")


        fig.tight_layout()
        show_plots_non_blocking()

        # how to check the process is orderly?

    def test_detects_double_occupied_space_time_cell(self):
        dt = 1.0
        dx = 1.0

        # Two events deliberately placed in the same (dt, dx) cell.
        events = np.array([[ 0.2, 0.8], [ 0.3, 0.7]])

        collision = find_multiple_events_per_cell(
            events,
            dt=dt,
            dx=dx,
        )

        self.assertIsNotNone(collision)

        self.assertEqual(collision["event_index"], 1)
        self.assertEqual(collision["time_index"], 0)
        self.assertEqual(collision["space_index"], 0)

        self.assertAlmostEqual(collision["time"], 0.8)
        self.assertAlmostEqual(collision["position"], 0.7)

    def test_does_not_detect_events_in_different_cells(self):
        dt = 1.0
        dx = 1.0

        events = np.array([[ 0.2, 1.2, 0.2], [ 0.3, 0.3, 1.3]])

        collision = find_multiple_events_per_cell(
            events,
            dt,
            dx,
        )

        self.assertIsNone(collision)

    def test_create_delta_pulses_reproducible(self):
        t_distribution = expon(scale=1.0 / 100.0)
        x_distribution = uniform(
            loc=100e-6,
            scale=300e-6,
        )

        pulses_1 = create_delta_pulses(
            t_max=1.0,
            x_distribution=x_distribution,
            t_distribution=t_distribution,
            seed=12345,
        )

        pulses_2 = create_delta_pulses(
            t_max=1.0,
            x_distribution=x_distribution,
            t_distribution=t_distribution,
            seed=12345,
        )

        np.testing.assert_array_equal(
            pulses_1,
            pulses_2,
        )

    def test_create_delta_pulses_known(self):
        t_distribution = expon(scale=1.0 / 100.0)
        x_distribution = uniform(
            loc=100e-6,
            scale=300e-6,
        )

        pulses = create_delta_pulses(
            t_max=0.01,
            x_distribution=x_distribution,
            t_distribution=t_distribution,
            seed=0,
        )

        self.assertEqual((2, 1), pulses.shape)

        np.testing.assert_array_equal(
            pulses,
            [[0.006799319039689096], [0.00017697181077412833]],
        )

    def test_create_delta_pulses_different_seed(self):
        t_distribution = expon(scale=1.0 / 100.0)
        x_distribution = uniform(
            loc=100e-6,
            scale=300e-6,
        )

        pulses_1 = create_delta_pulses(
            t_max=1.0,
            x_distribution=x_distribution,
            t_distribution=t_distribution,
            seed=12345,
        )

        pulses_2 = create_delta_pulses(
            t_max=1.0,
            x_distribution=x_distribution,
            t_distribution=t_distribution,
            seed=54321,
        )

        self.assertFalse(
            np.array_equal(pulses_1, pulses_2)
        )

if __name__ == '__main__':
    unittest.main()

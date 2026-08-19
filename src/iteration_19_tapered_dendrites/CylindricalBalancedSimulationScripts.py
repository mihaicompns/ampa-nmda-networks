"""
Manual script for the balanced cylindrical Crank-Nicolson simulation.

This is intentionally a script, not a pytest test. It demonstrates the direct
use of simulate_crank_nicolson_unitless_closed_cylinder_balance_with_param with
separate excitatory and inhibitory spike trains.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from brian2 import second, um
from brian2.units import ms
from brian2.units.allunits import pampere


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.iteration_19_tapered_dendrites.data import to_SI
from src.iteration_19_tapered_dendrites.TaperredDendritesPDE import default_params
from src.iteration_19_tapered_dendrites.TaperredDendritesBalancedCrankNicolson import (
    plot_crank_nicolson_unitless_closed_tapered_cone_balance,
)
from src.iteration_19_tapered_dendrites.CylindricalDendritesEventSimulation import (
    simulate_crank_nicolson_unitless_closed_cylinder_balance_with_param,
)
from src.iteration_19_tapered_dendrites.ImprovedSimulationSave import (
    run_and_save_balanced_cylindrical_simulation,
)


def _static_events():
    excitatory_events = np.array([
        [to_SI(0.2 * ms), to_SI(0.7 * ms)],
        [to_SI(20 * um), to_SI(50 * um)],
    ])
    inhibitory_events = np.array([
        [to_SI(0.4 * ms), to_SI(0.8 * ms)],
        [to_SI(30 * um), to_SI(60 * um)],
    ])
    return excitatory_events, inhibitory_events


def _cylindrical_params(L=to_SI(1000 * um), t_max=to_SI(1 * ms)):
    conical_p = default_params.with_SI_properties(
        dt=to_SI(0.01 * ms),
        N=51,
        L=L,
        t=t_max,
        I_e=to_SI(15 * pampere),
    ).to_numerical()
    return conical_p.to_numerical_cylindrical_params()


def test_run_static_balanced_cylinder_example():
    t_max = to_SI(1 * ms)
    p = _cylindrical_params(t_max=t_max)
    excitatory_events, inhibitory_events = _static_events()

    times, V_s = simulate_crank_nicolson_unitless_closed_cylinder_balance_with_param(
        p=p,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        t_max=t_max,
        saved_frames=50,
        verbose=True,
    )

    plot_crank_nicolson_unitless_closed_tapered_cone_balance(
        times=times,
        V_s=V_s,
        p=p,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        simulation_label="balanced_cylinder_static_example",
        desired_positions=[20, 50, 80],
        graph_out_dir=Path("saved_simulations") / "balanced_cylinder_static_example" / "graphs",
        graph_save_name="simulation_graph",
        show_plot=True,
        verbose=True,
    )

    return times, V_s


def test_run_and_save_static_balanced_cylinder_example():
    t_max = to_SI(1 * ms)
    p = _cylindrical_params(t_max=t_max)
    excitatory_events, inhibitory_events = _static_events()

    return run_and_save_balanced_cylindrical_simulation(
        p=p,
        t_max=t_max,
        output_root=Path("saved_simulations") / "balanced_cylindrical",
        simulation_label="balanced_cylinder_static_example",
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        saved_frames=50,
        verbose=True,
        plot=True,
        show_plot=True,
        desired_positions=[20, 50, 80],
    )
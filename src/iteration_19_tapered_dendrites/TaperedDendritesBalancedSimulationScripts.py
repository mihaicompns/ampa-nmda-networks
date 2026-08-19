"""
Manual long-running simulation scripts for balanced tapered-dendrite inputs.

These are intentionally scripts, not pytest tests. They restore the runnable
experiments that used to live in TaperredDendritesBalancedInputPDE.py in the old
repository, while keeping them outside automated test discovery.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from brian2 import Hz, second, um
from brian2.units import ms
from brian2.units.allunits import pampere
from joblib import Parallel, delayed
from scipy.stats import expon, uniform


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.iteration_19_tapered_dendrites.conical_data import create_delta_pulses
from src.iteration_19_tapered_dendrites.data import to_SI
from src.iteration_19_tapered_dendrites.TaperredDendritesPDE import (
    default_params,
    simulate_crank_nicolson_unitless_closed_tapered_cylinder,
)
from src.iteration_19_tapered_dendrites.TaperredDendritesBalancedCrankNicolson import (
    plot_crank_nicolson_unitless_closed_tapered_cone_balance,
    simulate_balanced_input_with_uniform,
    simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param,
)
from src.iteration_19_tapered_dendrites.ImprovedSimulationSave import (
    run_and_save_balanced_conical_with_cylindrical_comparison,
    run_and_save_balanced_conical_simulation,
)


def run_one_spike_position_sweep():
    x0_values = [
        to_SI(100 * um),
        to_SI(150 * um),
        to_SI(250 * um),
        to_SI(300 * um),
        to_SI(400 * um),
        to_SI(450 * um),
    ]
    events = [np.array([[to_SI(0.1 * ms)], [x0]]) for x0 in x0_values]

    return Parallel(n_jobs=-3, verbose=10)(
        delayed(simulate_crank_nicolson_unitless_closed_tapered_cylinder)(
            events=event,
            x_N=101,
            dt_=to_SI(1e-8 * second),
            t_max=to_SI(0.3 * ms),
            verbose=True,
            L=to_SI(500 * um),
            plot=True,
            save=True,
            saved_frames=3 * 10 ** 4,
        )
        for event in events
    )


def run_uniform_excitatory_spike_train():
    x_N = 101
    t_max = to_SI(3 * second)
    dt = to_SI(1e-7 * second)
    L = to_SI(2000 * um)
    r_i = 2000 * Hz

    spike_train = create_delta_pulses(
        t_max=t_max,
        x_distribution=uniform(loc=0, scale=L),
        t_distribution=expon(scale=1.0 / r_i),
    )

    print("Simulating spike train:", spike_train.shape)

    return simulate_crank_nicolson_unitless_closed_tapered_cylinder(
        x_N=x_N,
        dt_=dt,
        t_max=t_max,
        events=spike_train,
        verbose=True,
        L=L,
        plot=True,
        save=True,
        saved_frames=3 * 10 ** 4,
    )


def run_balanced_static_example():
    L = to_SI(500 * um)
    p = default_params.with_SI_properties(
        dt=to_SI(1e-8 * second),
        N=101,
        L=L,
        I_e=to_SI(15 * pampere),
    ).to_numerical()

    excitatory_events = np.array([
        [to_SI(0.10 * ms), to_SI(0.20 * ms), to_SI(0.30 * ms)],
        [to_SI(100 * um), to_SI(250 * um), to_SI(450 * um)],
    ])
    inhibitory_events = np.array([
        [to_SI(0.15 * ms), to_SI(0.25 * ms)],
        [to_SI(150 * um), to_SI(350 * um)],
    ])

    times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param(
        p=p,
        t_max=to_SI(0.5 * ms),
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        verbose=True,
        saved_frames=3 * 10 ** 4,
    )
    plot_crank_nicolson_unitless_closed_tapered_cone_balance(
        times=times,
        V_s=V_s,
        p=p,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        simulation_label="balanced_static_example",
        show_plot=True,
        verbose=True,
    )
    return times, V_s


def run_balanced_500um_example(t_max=to_SI(1 * second)):
    L = to_SI(500 * um)
    p = default_params.with_SI_properties(
        dt=to_SI(1e-8 * second),
        N=101,
        L=L,
        I_e=to_SI(15 * pampere),
    ).to_numerical()

    return run_and_save_balanced_conical_with_cylindrical_comparison(
        conical_p=p,
        t_max=t_max,
        output_root=Path("saved_simulations") / "balanced_comparison",
        simulation_label="balanced_500um_example",
        e_limits=(0.25 * L, L),
        i_limits=(0.0, 0.5 * L),
        saved_frames=3 * 10 ** 4,
        verbose=True,
        plot=True,
        show_plot=True,
    )


def test_run_balanced_2000um_example(t_max=to_SI(1 * second)):
    L = to_SI(2000 * um)
    p = default_params.with_SI_properties(
        dt=to_SI(1e-8 * second),
        N=201,
        L=L,
        I_e=to_SI(15 * pampere),
    ).to_numerical()

    return run_and_save_balanced_conical_with_cylindrical_comparison(
        conical_p=p,
        t_max=t_max,
        output_root=Path("saved_simulations") / "balanced_comparison",
        simulation_label="balanced_2000um_example",
        e_limits=(0.75 * L, L),
        i_limits=(0.0, 0.5 * L),
        saved_frames=3 * 10 ** 4,
        verbose=True,
        plot=True,
        show_plot=True,
    )


def _fractional_limit_sets(L):
    return [
        (0.0, L),
        (0.25 * L, L),
        (0.5 * L, L),
        (0.75 * L, L),
        (0.25 * L, 0.5 * L),
        (0.25 * L, 0.75 * L),
        (0.5 * L, 0.75 * L),
    ]


def run_balanced_limit_sweep_500um(t_max=to_SI(10 * second)):
    L = to_SI(500 * um)
    p = default_params.with_SI_properties(
        dt=to_SI(1e-8 * second),
        N=101,
        L=L,
        I_e=to_SI(15 * pampere),
    ).to_numerical()

    return Parallel(n_jobs=-3, verbose=10)(
        delayed(simulate_balanced_input_with_uniform)(
            p=p,
            t_max=t_max,
            e_limits=(a, b),
            i_limits=(a, b),
        )
        for a, b in _fractional_limit_sets(L)
    )


def run_balanced_limit_sweep_1000um(t_max=to_SI(10 * second)):
    L = to_SI(1000 * um)
    p = default_params.with_SI_properties(
        dt=to_SI(1e-8 * second),
        N=101,
        L=L,
        I_e=to_SI(15 * pampere),
    ).to_numerical()

    return Parallel(n_jobs=-3, verbose=10)(
        delayed(simulate_balanced_input_with_uniform)(
            p=p,
            t_max=t_max,
            e_limits=(a, b),
            i_limits=(a, b),
        )
        for a, b in _fractional_limit_sets(L)
    )

@staticmethod
def test_run_balanced_limit_sweep_1500um(t_max=to_SI(0.001 * second)):
    L = to_SI(1500 * um)
    limits = [
        (0.0, L),
        (0.3 * L, L),
        (0.6 * L, L),
        (2.0 / 3.0 * L, L),
        (to_SI(1250 * um), L),
        (0.0, 0.3 * L),
        (0.0, 0.6 * L),
        (0.0, to_SI(1000 * um)),
        (0.0, to_SI(1250 * um)),
        (to_SI(500 * um), L),
        (to_SI(750 * um), L),
        (to_SI(1000 * um), L),
        (to_SI(500 * um), to_SI(1000 * um)),
    ]
    p = default_params.with_SI_properties(
        dt=to_SI(1e-8 * second),
        N=101,
        L=L,
        I_e=to_SI(15 * pampere),
    ).to_numerical()

    return Parallel(n_jobs=-3, verbose=10)(
        delayed(simulate_balanced_input_with_uniform)(
            p=p,
            t_max=t_max,
            e_limits=(a, b),
            i_limits=(a, b),
        )
        for a, b in limits
    )


SCENARIOS = {
    "one-spike-sweep": run_one_spike_position_sweep,
    "uniform-excitatory": run_uniform_excitatory_spike_train,
    "balanced-static": run_balanced_static_example,
    "balanced-500um": run_balanced_500um_example,
    "balanced-2000um": test_run_balanced_2000um_example,
    "balanced-sweep-500um": run_balanced_limit_sweep_500um,
    "balanced-sweep-1000um": run_balanced_limit_sweep_1000um,
    "balanced-sweep-1500um": test_run_balanced_limit_sweep_1500um,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    args = parser.parse_args()
    SCENARIOS[args.scenario]()


if __name__ == "__main__":
    main()

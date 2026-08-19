"""
Cylindrical counterpart to simulate_crank_nicolson_unitless_closed_tapered_cylinder_with_param,
built to accept the same multi-event spike train interface (events shaped (2, n_events):
row 0 = times, row 1 = positions) so a cone and a cylinder simulation can be run against
identical input for comparison.

Kept as its own module/function rather than folding cylindrical support into
ConicalNumericalCableParameters or the existing single-event cylindrical functions in
CylindricalDendritesPDE.py: the conical and cylindrical PDE coefficients (a, b) are
different enough algebraically (a=0 and b=const for a cylinder, vs. both position-dependent
for a cone) that a shared cable-parameters class would be messy. Both geometries build their
own A matrix externally and pass it into the same crank_nicolson() stepper from
TaperredDendritesPDE.py.
"""

import numpy as np
from brian2 import ms
from scipy.sparse import diags

from data import NumericalCableParameters, to_SI
from TaperredDendritesPDE import crank_nicolson, save_simulation, plot_difussion_unitless_spike_train
from TaperredDendritesBalancedCrankNicolson import crank_nicolson_balanced


def build_cylindrical_A(p: NumericalCableParameters):
    """
    Build the tridiagonal Crank-Nicolson operator A for a cylinder: constant radius
    means no drift term (a=0) and a constant diffusion coefficient b(x) = p.b.
    """
    dx = p.dx
    tau = p.tau

    a = 0.0
    b = np.full(len(p.x), p.b)

    lower = b[1:] / dx ** 2 + a / (2 * dx)
    main = -1 / tau - 2 * b / dx ** 2
    upper = b[:-1] / dx ** 2 - a / (2 * dx)

    return diags(
        diagonals=[lower, main, upper],
        offsets=[-1, 0, 1],
        format="lil")


def simulate_crank_nicolson_unitless_closed_cylinder_with_param(
        events, p: NumericalCableParameters, t_max=to_SI(30 * ms),
        saved_frames=1200, verbose=True, plot=False, save=False,
        simulation_label="cylinder"):
    if verbose:
        print(f"Simulating Crank-Nicolson unitless cylinder with dt = {p.dt: .5e}")

    A = build_cylindrical_A(p)
    u0 = np.zeros(len(p.x))

    times, V_s = crank_nicolson(t_span=(0, t_max), V0=u0, A=A, saved_frames=saved_frames, p=p, events=events)

    if save:
        save_simulation(times=times, V_s=V_s, p=p, events=events, t_max=t_max, simulation_label=simulation_label)

    if plot:
        plot_difussion_unitless_spike_train(times=times, V_s=V_s, p=p, events=events,
                                             sim_type=f"Crank-Nicolson unitless {simulation_label}", save=True)

    return times, V_s


def simulate_crank_nicolson_unitless_closed_cylinder_balance_with_param(
        p: NumericalCableParameters, excitatory_events, inhibitory_events,
        t_max=to_SI(30 * ms), saved_frames=1200, verbose=True):
    """
    Balanced cylindrical counterpart to the balanced tapered-cone simulation.

    The cylinder has a constant-radius operator A, but uses the same balanced
    Crank-Nicolson stepper and the same separate excitatory/inhibitory event
    train format as the cone.
    """
    if verbose:
        print(f"Simulating balanced Crank-Nicolson unitless cylinder with dt = {p.dt: .5e}")

    A = build_cylindrical_A(p)
    u0 = np.zeros(len(p.x))

    return crank_nicolson_balanced(
        t_span=(0, t_max),
        V0=u0,
        A=A,
        saved_frames=saved_frames,
        p=p,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        verbose=verbose,
    )

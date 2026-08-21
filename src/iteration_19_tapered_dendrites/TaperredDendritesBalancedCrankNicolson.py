"""
Real Crank-Nicolson balanced excitatory/inhibitory input simulation for the tapered
dendrite: two separate event streams (excitatory_events, inhibitory_events), each
generated with create_delta_pulses, run through the same tapered-cable Crank-Nicolson
machinery used by simulate_crank_nicolson_unitless_closed_tapered_cylinder_with_param.

Kept as new functions in a new file rather than modifying TaperredDendritesPDE.py's
existing crank_nicolson()/synaptic_spike_train_input(): those only support a single
homogeneous event stream injected at one fixed amplitude (p.I_e), with no sign. Balanced
input needs excitatory events injected as +p.I_e and inhibitory events as -p.I_i, which
needs its own stepping function rather than a change to the existing one.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from brian2 import second, ms, Hz, um, meter, have_same_dimensions
from brian2.units.allunits import pampere
from scipy.stats import expon, uniform
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized

from conical_data import ConicalNumericalCableParameters, create_delta_pulses
from data import to_SI
from CylindricalDendritesPDE import dirac_delta_unitless
from TaperredDendritesPDE import plot_difussion_unitless_balanced_spike_trains, default_params


def experiment_label_for_uniform_balance(e_limits, i_limits):
    e_left, e_right = e_limits
    i_left, i_right = i_limits
    return (
        f"Balanced E~[{int(e_left / 1e-6)} um - {int(e_right / 1e-6)} um] "
        f"I~[{int(i_left / 1e-6)} um - {int(i_right / 1e-6)} um]"
    )


def _consume_event_stream(t, dt, events, next_idx, p, current_amplitude, sign):
    """
    Consume all events in `events` that occur in [t, t+dt), returning their
    combined contribution (with the given sign) and the updated next_idx.
    """
    if events.size == 0 or next_idx >= events.shape[1]:
        return None, next_idx

    event_times, event_positions = events
    n_events = len(event_times)
    t_end = t + dt
    if event_times[next_idx] >= t_end:
        return None, next_idx

    contribution = np.zeros_like(p.x)
    while next_idx < n_events:
        spike_t = event_times[next_idx]

        if spike_t >= t_end:
            break

        if spike_t < t:
            raise ValueError(
                f"Spike at t={spike_t} is before current time t={t}. next_idx={next_idx}."
            )

        spike_x = event_positions[next_idx]

        contribution += sign * dirac_delta_unitless(
            x0=spike_x,
            t0=spike_t,
            I_e=current_amplitude,
            x=p.x,
            t=t,
            dx=p.dx,
            dt=dt,
            tau_m=p.tau,
            r_of_x=p.radius(spike_x),
        )

        next_idx += 1

    return contribution, next_idx


def synaptic_spike_train_input_balanced(t, dt, excitatory_events, inhibitory_events,
                                         next_exc_idx, next_inh_idx, p: ConicalNumericalCableParameters):
    """
    Merge two separate event streams: excitatory events inject +p.I_e, inhibitory
    events inject -p.I_i.
    """
    exc_contribution, next_exc_idx = _consume_event_stream(
        t, dt, excitatory_events, next_exc_idx, p, current_amplitude=p.I_e, sign=+1.0)
    inh_contribution, next_inh_idx = _consume_event_stream(
        t, dt, inhibitory_events, next_inh_idx, p, current_amplitude=p.I_i, sign=-1.0)

    if exc_contribution is None:
        return inh_contribution, next_exc_idx, next_inh_idx
    if inh_contribution is None:
        return exc_contribution, next_exc_idx, next_inh_idx
    return exc_contribution + inh_contribution, next_exc_idx, next_inh_idx


def crank_nicolson_balanced(t_span, V0, A, p: ConicalNumericalCableParameters,
                             excitatory_events, inhibitory_events, saved_frames=1, verbose=False):
    """
    Same Crank-Nicolson stepping as TaperredDendritesPDE.crank_nicolson, but injects
    from two separate signed event streams via synaptic_spike_train_input_balanced.
    """
    t0, tf = t_span

    num_steps = int(np.ceil((tf - t0) / p.dt))
    save_every = int(np.ceil(num_steps / saved_frames))
    num_save = num_steps // save_every + 1

    times = np.zeros(num_save) * t0
    sol = np.zeros((num_save, len(V0)))

    I = eye(A.shape[0], format="csc")
    L = (I - 0.5 * p.dt * A).tocsc()
    R = (I + 0.5 * p.dt * A).tocsc()
    solve = factorized(L)

    t = t0
    V = V0.copy()
    times[0] = t
    sol[0] = V

    next_exc_idx = 0
    next_inh_idx = 0
    for step in range(1, num_steps + 1):
        dt_step = min(p.dt, tf - t)

        synaptic_input_at_t, next_exc_idx, next_inh_idx = synaptic_spike_train_input_balanced(
            t=t, dt=p.dt, excitatory_events=excitatory_events, inhibitory_events=inhibitory_events,
            next_exc_idx=next_exc_idx, next_inh_idx=next_inh_idx, p=p)
        rhs = R @ V
        if synaptic_input_at_t is not None:
            rhs = rhs + p.dt * 1 / p.c_m * synaptic_input_at_t
        V = solve(rhs)

        t += dt_step

        if step % save_every == 0:
            iteration = step // save_every
            if verbose:
                print(f"[CN-balanced {iteration}/{num_save}] step {step}/{num_steps}")
            times[iteration] = t
            sol[iteration] = V

    return times, sol


def simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param(
        p: ConicalNumericalCableParameters, excitatory_events, inhibitory_events,
        t_max=to_SI(30 * second) / 1000, saved_frames=1200, verbose=True):
    if verbose:
        print(f"Simulating balanced Crank-Nicolson unitless cone with dt = {p.dt: .5e}")

    x = p.x
    dx = p.dx
    tau = p.tau

    a = p.a()
    b = p.b(x)

    lower = b[1:] / dx ** 2 + a / (2 * dx)
    main = -1 / tau - 2 * b / dx ** 2
    upper = b[:-1] / dx ** 2 - a / (2 * dx)

    A = diags(diagonals=[lower, main, upper], offsets=[-1, 0, 1], format="lil")

    u0 = np.zeros(len(x))

    times, V_s = crank_nicolson_balanced(
        t_span=(0, t_max), V0=u0, A=A, saved_frames=saved_frames, p=p,
        excitatory_events=excitatory_events, inhibitory_events=inhibitory_events, verbose=verbose)

    return times, V_s


def plot_crank_nicolson_unitless_closed_tapered_cone_balance(
        times, V_s, p: ConicalNumericalCableParameters, excitatory_events, inhibitory_events,
        simulation_label="balanced", desired_positions=[100, 250, 500],
        save=True, graph_save_name=None, graph_out_dir=None, show_plot=True,
        verbose=True):
    plot_difussion_unitless_balanced_spike_trains(
        times=times,
        V_s=V_s,
        p=p,
        excitatory_events=excitatory_events,
        inhibitory_events=inhibitory_events,
        desired_positions=desired_positions,
        verbose=verbose,
        sim_type=f"Crank-Nicolson unitless balanced {simulation_label}",
        save=save,
        save_name=graph_save_name,
        out_dir=graph_out_dir,
        show=show_plot,
    )


def simulate_balanced_input_with_uniform(p: ConicalNumericalCableParameters, t_max=to_SI(1 * second),
                                          e_limits: tuple[float, float] = (0.0, to_SI(500 * um)),
                                          i_limits: tuple[float, float] = (0.0, to_SI(500 * um))):
    if have_same_dimensions(t_max, second):
        t_max = to_SI(t_max)

    r_e_density = 0.02 * Hz / um
    r_i_density = 0.01 * Hz / um

    e_left, e_right = e_limits
    i_left, i_right = i_limits

    r_e = r_e_density * (e_right - e_left) * meter
    r_i = r_i_density * (i_right - i_left) * meter
    exc_x_distribution = uniform(loc=e_left, scale=e_right - e_left)
    inh_x_distribution = uniform(loc=i_left, scale=i_right - i_left)

    spike_train_excitatory = create_delta_pulses(
        t_max=t_max,
        x_distribution=exc_x_distribution,
        t_distribution=expon(scale=1.0 / r_e),
    )

    spike_train_inhibitory = create_delta_pulses(
        t_max=t_max,
        x_distribution=inh_x_distribution,
        t_distribution=expon(scale=1.0 / r_i),
    )

    times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param(
        p=p,
        t_max=t_max,
        excitatory_events=spike_train_excitatory,
        inhibitory_events=spike_train_inhibitory,
        verbose=False,
        saved_frames=3 * 10 ** 4)
    plot_crank_nicolson_unitless_closed_tapered_cone_balance(
        times=times,
        V_s=V_s,
        p=p,
        excitatory_events=spike_train_excitatory,
        inhibitory_events=spike_train_inhibitory,
        simulation_label=experiment_label_for_uniform_balance(e_limits, i_limits),
        show_plot=True,
        verbose=True,
    )
    return times, V_s


@dataclass(frozen=True)
class BalancedUniformSimulationMetadata:
    """
    Everything needed to fully specify and deterministically reproduce a balanced
    excitatory/inhibitory tapered-dendrite simulation with uniformly-distributed
    spike train positions.
    """
    t_max: float  # seconds, SI
    e_limits: Tuple[float, float]  # meters, SI
    i_limits: Tuple[float, float]  # meters, SI
    seed: int
    x_N: int = 101
    dt: float = to_SI(0.001 * ms)  # seconds, SI
    L: float = to_SI(500 * um)  # meters, SI
    I_e: float = to_SI(150 * pampere)  # amperes, SI
    I_i: float = to_SI(30 * pampere)  # amperes, SI
    saved_frames: int = 1200
    simulation_label: Optional[str] = None


def run_balanced_uniform_tapered_simulation(metadata: BalancedUniformSimulationMetadata,
                                             verbose=True, plot=False, save=False):
    """
    Generate deterministic excitatory/inhibitory spike trains from metadata, build the
    tapered cable geometry it describes, and run the real balanced Crank-Nicolson
    simulation - given the same metadata, the result is fully reproducible.

    create_delta_pulses uses its own numpy Generator internally and ignores the
    ambient np.random.seed(), so metadata.seed is passed to it explicitly (with a
    +1 offset for the inhibitory stream so the two event trains aren't identical).
    """
    e_left, e_right = metadata.e_limits
    i_left, i_right = metadata.i_limits

    r_e_density = 0.02 * Hz / um
    r_i_density = 0.01 * Hz / um
    r_e = r_e_density * (e_right - e_left) * meter
    r_i = r_i_density * (i_right - i_left) * meter

    exc_x_distribution = uniform(loc=e_left, scale=e_right - e_left)
    inh_x_distribution = uniform(loc=i_left, scale=i_right - i_left)

    spike_train_excitatory = create_delta_pulses(
        t_max=metadata.t_max, x_distribution=exc_x_distribution,
        t_distribution=expon(scale=1.0 / r_e), seed=metadata.seed)
    spike_train_inhibitory = create_delta_pulses(
        t_max=metadata.t_max, x_distribution=inh_x_distribution,
        t_distribution=expon(scale=1.0 / r_i), seed=metadata.seed + 1)

    p = default_params.with_SI_properties(
        t=metadata.t_max, N=metadata.x_N, dt=metadata.dt, L=metadata.L,
        I_e=metadata.I_e, I_i=metadata.I_i
    ).to_numerical()

    label = metadata.simulation_label or experiment_label_for_uniform_balance(
        metadata.e_limits, metadata.i_limits)

    times, V_s = simulate_crank_nicolson_unitless_closed_tapered_cone_balance_with_param(
        p=p,
        excitatory_events=spike_train_excitatory,
        inhibitory_events=spike_train_inhibitory,
        t_max=metadata.t_max,
        saved_frames=metadata.saved_frames,
        verbose=verbose)
    if plot:
        plot_crank_nicolson_unitless_closed_tapered_cone_balance(
            times=times,
            V_s=V_s,
            p=p,
            excitatory_events=spike_train_excitatory,
            inhibitory_events=spike_train_inhibitory,
            simulation_label=label,
            show_plot=True,
            verbose=verbose,
        )
    return times, V_s

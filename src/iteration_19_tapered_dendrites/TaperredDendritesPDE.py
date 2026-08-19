import unittest
from typing import Callable

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from brian2 import cm, uF, ohm, um, Quantity, is_dimensionless, get_dimensions, volt, have_same_dimensions, mV, uA, \
    uamp, meter, psecond, msecond, Hz
from brian2.units import second, ms
from brian2.units.allunits import ampere, mampere, pampere
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized
from scipy.stats import expon, uniform

import re

from src.Plotting import show_plots_non_blocking
from src.iteration_19_tapered_dendrites.CylindricalDendritesPDE import \
    plot_tuckwell_solution_closed_cable_unitless_separation_of_variables, \
    plot_tuckwell_solution_closed_cable_unitless_method_of_images, dirac_delta_unitless
from src.iteration_19_tapered_dendrites.conical_data import ConicalNumericalCableParameters, ConicalCableParameters, \
    create_delta_pulses
from src.iteration_19_tapered_dendrites.data import to_SI


# TODO: Normalize the way Dayan has it. ie = Ie τm δ(x)δ(t)/ 2πa. A is the radius! We have a slightly different formulation. But Still
def dirac_delta(x0:Quantity, t0:Quantity, w:Quantity, x: np.ndarray[Quantity], r_of_x: np.ndarray[Quantity], t:Quantity, dx: Quantity, dt:Quantity) -> np.ndarray[Quantity]:

    if is_dimensionless(t0):
        t0 = t0 * ms
    if is_dimensionless(x0):
        x0 = x0 * um
    if t0 - t < 0 or t0 - t >= dt:
        return np.zeros(len(x)) * mampere / cm ** 2

    result = np.zeros(len(x)) * mampere / cm ** 2

    if x0 <= x[0]:
        result[0] = w / dx

    elif x0 >= x[-1]:
        result[-1] = w / dx

    else:
        i = np.searchsorted(x, x0) - 1
        alpha = (x0 - x[i]) / dx

        result[i] = w * (1.0 - alpha) / dx
        result[i + 1] = w * alpha / dx

    assert have_same_dimensions(result[0], 1 * mampere / cm ** 2)
    return result

def plot_difussion_solution(times, x, r_of_x, V_s, verbose=True):

    if verbose:
        assert have_same_dimensions(x[0], 1*um)
        assert have_same_dimensions(r_of_x[0], 1*um)
        assert have_same_dimensions(V_s[0][0], 1*mV)
        assert have_same_dimensions(times[0], 1*ms)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(8, 9),
        gridspec_kw={'height_ratios': [3, 2, 1]}
    )

    # ============================================
    # Top: space-time voltage map
    # ============================================
    x = x / um
    r_of_x = r_of_x / um
    times = times / ms
    V_s = V_s / mV

    print("x:", x[0], x[-1], len(x))
    print("times:", times[0], times[-1], len(times))
    print("V_s:", V_s.shape)

    im = ax1.imshow(
        V_s ,
        aspect='auto',
        origin='lower',
        extent=[x[0], x[-1], times[0], times[-1]],
        cmap='cividis_r'
        #vmax=0.05
    )

    fig.colorbar(im, ax=ax1, label="Voltage (mV)")

    ax1.set_title(
        f"Cable equation. Max V = {np.max(V_s):.4f} mV"
    )
    ax1.set_xlabel("x [μm]")
    ax1.set_ylabel("t [ms]")

    indices = [0, 3, 20, 80, 99]
    for i in indices:
        ax2.plot(
            times,
            V_s[:, i],
            label=f"x = {x[i]:.0f} μm"
        )

    ax2.set_xlabel("t [ms]")
    ax2.set_ylabel("V [mV]")
    ax2.set_title("Voltage at selected positions")
    ax2.legend()

    # ============================================
    # Bottom: cone geometry
    # ============================================

    r = r_of_x

    # Cone walls
    ax3.plot(x, r, 'k', linewidth=2)
    ax3.plot(x, -r, 'k', linewidth=2)

    # Fill cone
    ax3.fill_between(
        x,
        -r,
        r,
        color='gray',
        alpha=0.3
    )

    # Center axis y=0
    ax3.axhline(
        0,
        color='gray',
        linestyle=':',
        linewidth=1.5
    )

    # Vertical line at x=0
    ax3.axvline(
        0,
        color='gray',
        linestyle='--',
        linewidth=1.5
    )

    ax3.set_xlabel("x")
    ax3.set_ylabel("radius")
    ax3.set_title("Cable geometry")

    #ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

def analyse_eigenvalues_generalized_locally_toeplitz_matrix(A):
    lam = np.linalg.eigvals(A.toarray())
    print("Eigenvalues")
    # print(lam)
    lam_pos = np.abs(lam)
    stiffness = np.max(lam_pos) / np.min(lam_pos)
    idx = np.argsort(np.abs(lam))
    # yplt.plot(np.arange(len(lam.real)), np.abs(lam.real))
    plt.semilogy(np.abs(lam[idx]))
    plt.xlabel("index")
    plt.ylabel("$\lambda$")
    plt.yscale("log")
    plt.title(f"Eigenvalues of Jacobian matrix. Stiffness = {stiffness: .5f}")
    print(f"Stiffness = {stiffness}. Max eig={np.max(lam_pos)}, min eig={np.min(lam_pos)}")
    idx = np.argsort(np.abs(lam))
    np.set_printoptions(suppress=True, precision=10)
    print("Smallest |lambda|:")
    print(lam[idx[:10]])
    print("\nLargest |lambda|:")
    print(lam[idx[-10:]])
    plt.show()

def plot_difussion_unitless(times, V_s, p: ConicalNumericalCableParameters, x0 =250 * um, t0=0.1 * ms, desired_positions = [100, 250, 500], verbose=True, sim_type="forward Euler", save=True):

    show_difussion_simulation_as_image(times=times, V_s = V_s, p=p, x0=x0, t0=t0, verbose=verbose, sim_type=sim_type)
    #plot_tuckwell_solution_infinite_cable(V_s=V_s, desired_positions=desired_positions, p=p, t0=t0, times=times, x0=x0, sim_type=sim_type)

    plot_tuckwell_solution_closed_cable_unitless_separation_of_variables(V_s=V_s, desired_positions=desired_positions, p=p.to_numerical_cable_params_at(x0), t0=t0, times=times, x0=x0, sim_type=sim_type)
    plot_tuckwell_solution_closed_cable_unitless_method_of_images(V_s=V_s, desired_positions=desired_positions, p=p.to_numerical_cable_params_at(x0), t0=t0, times=times, x0=x0, sim_type=sim_type)

def plot_difussion_unitless_spike_train(times, V_s, p: ConicalNumericalCableParameters, events, desired_positions = [100, 250, 500], verbose=True, sim_type="forward Euler", save=True):

    _, n_events = events.shape

    if n_events == 1:
        t0, x0 = events[0][0], events[1][0]
        plot_difussion_unitless(times=times, V_s=V_s, p=p, x0 = x0, t0=t0, desired_positions = desired_positions, verbose=verbose, sim_type=sim_type, save=save)
        plot_tuckwell_solution_closed_cable_unitless_separation_of_variables(V_s=V_s,
                                                                             desired_positions=desired_positions,
                                                                             p=p.to_numerical_cable_params_at(x0),
                                                                             t0=t0, times=times, x0=x0,
                                                                             sim_type=sim_type)
        plot_tuckwell_solution_closed_cable_unitless_method_of_images(V_s=V_s, desired_positions=desired_positions,
                                                                      p=p.to_numerical_cable_params_at(x0), t0=t0,
                                                                      times=times, x0=x0, sim_type=sim_type)

    show_difussion_simulation_as_image_spike_train(times=times, V_s = V_s, p=p, events=events, desired_positions=desired_positions, verbose=verbose, sim_type=sim_type)
        #plot_tuckwell_solution_infinite_cable(V_s=V_s, desired_positions=desired_positions, p=p, t0=t0, times=times, x0=x0, sim_type=sim_type)


def show_difussion_simulation_as_image(times, V_s, p: ConicalNumericalCableParameters, x0 =250 * um, t0=0.1 * ms, verbose=True, sim_type="forward Euler"):
    if verbose:
        assert is_dimensionless(V_s[0][0])
        assert is_dimensionless(times[0])

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1,
        figsize=(11, 12),
        gridspec_kw={'height_ratios': [2, 2, 2, 1]}
    )

    # ============================================
    # Top: space-time voltage map
    # ============================================
    x = p.x * meter / um
    x0 = x0 / um
    t0 = t0 / ms
    r_of_x = p.radius(p.x) * meter / um
    times = times * second / ms
    V_s = V_s * volt / mV

    print("x:", x[0], x[-1], len(x))
    print("times:", times[0], times[-1], len(times))
    print("V_s:", V_s.shape)

    im = ax1.imshow(
        V_s,
        aspect='auto',
        origin='lower',
        extent=[x[0], x[-1], times[0], times[-1]],
        cmap='cividis_r'
        # vmax=0.05
    )

    fig.colorbar(im, ax=ax1, label="Voltage (mV)")

    ax1.set_title(
        f"{sim_type.capitalize()} simulation for cable equation in cylinder model \n"
        f"x = [{x[0]} - {x[-1]:.2f}] "r"$\mu$"f"m, split in {len(x)} nodes. \n"
        f"Max V = {np.max(V_s):.4f} mV. dx={p.dx * meter / um: .5f} "r"$\mu$"f"m, dt={p.dt: .3e} s \n"
        f""
    )
    ax1.set_xlabel(r"x [$\mu$m]")
    ax1.set_ylabel("t [ms]")


    print("x: ", x)

    desired_positions = [100, x0, 500]
    for desired_distance in desired_positions:
        i =  np.searchsorted(x, desired_distance)
        ax2.plot(
            times,
            V_s[:, i],
            label=f"x = {x[i]:.0f} "r"$\mu$ m",
            alpha=0.6,
            lw=2
        )

    ax2.set_xlabel("t [ms]")
    ax2.set_ylabel("V [mV]")
    ax2.set_title("Voltage at selected positions")
    ax2.legend()

    # ============================================
    # Second to last: Gaussian around injection point!?
    # ============================================
    index_x0 = np.searchsorted(x, x0)


    # which is time of injection?
    v_tot = V_s.sum(axis=1)
    v_t_diff = np.diff(v_tot)
    t_index = np.argwhere(v_t_diff > 0)
    t_index = t_index[0][0] if len(t_index) > 0 else 0

    taum_ms = p.tau * second / ms

    for t_i in range(t_index, t_index+6):

        if len(x) < 200:
            ax3.plot(
                x,
                V_s[t_i, :],
                label=r"t/$\tau$" f"= {times[t_i] / taum_ms :.4f}",
                alpha=0.6,
                lw=2
            )
        else:
            offs = 250 if len(x) > 2000 else 100
            ax3.plot(
                x[index_x0-offs: index_x0+offs],
                V_s[t_i, index_x0-offs: index_x0+offs],
                label=r"t/$\tau$" f"= {times[t_i] / taum_ms :.4f}",
                alpha=0.6,
                lw=2
            )


    ax3.set_xlabel(r"x [$\mu$m]")
    ax3.set_ylabel("V [mV]")
    ax3.legend()
    ax3.set_title(r"Voltage at time steps around $\delta$-pulse")


    # ============================================
    # Bottom: cone geometry
    # ============================================

    r = r_of_x

    # Cone walls
    ax4.plot(x, r, 'k', linewidth=2)
    ax4.plot(x, -r, 'k', linewidth=2)

    # Fill cone
    ax4.fill_between(
        x,
        -r,
        r,
        color='gray',
        alpha=0.3
    )

    # Center axis y=0
    ax4.axhline(
        0,
        color='gray',
        linestyle=':',
        linewidth=1.5
    )

    # Vertical line at x=0
    ax4.axvline(
        0,
        color='gray',
        linestyle='--',
        linewidth=1.5
    )

    ax4.set_xlabel("x")
    ax4.set_ylabel("radius")
    ax4.set_title("Cable geometry")

    # ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    show_plots_non_blocking()

def show_difussion_simulation_as_image_spike_train(times, V_s, p: ConicalNumericalCableParameters, events: np.ndarray, desired_positions: list, verbose=True, sim_type="forward Euler"):
    if verbose:
        assert is_dimensionless(V_s[0][0])
        assert is_dimensionless(times[0])

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

    fig = plt.figure(figsize=(13, 12))

    gs = fig.add_gridspec(
        5, 1,
        height_ratios=[2, 2, 2, 2, 1]
    )

    voltage_diffussion_image = fig.add_subplot(gs[0, 0])
    voltage_at_selected_spots = fig.add_subplot(gs[1, 0])

    plot_voltage_means = fig.add_subplot(
        gs[2, 0],
        sharex=voltage_diffussion_image
    )
    raster_plot = fig.add_subplot(
        gs[3, 0],
        sharex=voltage_diffussion_image
    )
    dendrite_geometry = fig.add_subplot(
        gs[4, 0],
        sharex=voltage_diffussion_image
    )

    # ============================================
    # Top: space-time voltage map
    # ============================================
    x = p.x * meter / um
    r_of_x = p.radius(p.x) * meter / um
    times = times * second / ms
    V_s = V_s * volt / mV

    im = voltage_diffussion_image.imshow(
        V_s,
        aspect='auto',
        origin='lower',
        extent=[x[0], x[-1], times[0], times[-1]],
        cmap='cividis_r',
        vmax=1000
    )

    fig.colorbar(im, ax=voltage_diffussion_image, label="Voltage (mV)")

    voltage_diffussion_image.set_title(
        f"{sim_type.capitalize()} simulation for cable equation in cylinder model \n"
        f"x = [{x[0]} - {x[-1]:.2f}] "r"$\mu$"f"m, split in {len(x)} nodes. \n"
        f"Max V = {np.max(V_s):.4f} mV. dx={p.dx * meter / um: .5f} "r"$\mu$"f"m, dt={p.dt: .3e} s \n"
        f""
    )
    voltage_diffussion_image.set_xlabel(r"x [$\mu$m]")
    voltage_diffussion_image.set_ylabel("t [ms]")

    for desired_distance in desired_positions:
        i =  np.searchsorted(x, desired_distance)
        voltage_at_selected_spots.plot(
            times,
            V_s[:, i],
            label=f"x = {x[i]:.0f} "r"$\mu$ m",
            alpha=0.6,
            lw=2
        )

    voltage_at_selected_spots.set_ylim(-1, 1000)
    voltage_at_selected_spots.set_xlabel("t [ms]")
    voltage_at_selected_spots.set_ylabel("V [mV]")
    voltage_at_selected_spots.set_title("Voltage at selected positions")
    voltage_at_selected_spots.legend()


    number_of_voltages, x_discretizations = V_s.shape

    half_time =  int(number_of_voltages // 2)
    voltage_means = np.average(V_s[half_time:, :], axis=0)

    plot_voltage_means.plot(x, voltage_means)
    plot_voltage_means.set_title("Voltage means")

    raster_plot.scatter(
        events[1] / um,
        events[0] / ms,
        s=15,
        marker="|",
        linewidths=1.5,
        color="black"
    )

    raster_plot.set_xlim(0, p.L / um)

    raster_plot.set_xlabel(r"$x\;[\mu\mathrm{m}]$")
    raster_plot.set_ylabel(r"$t\;[\mathrm{ms}]$")
    raster_plot.set_title("Space-time raster of synaptic events")


    # ============================================
    # Bottom: cone geometry
    # ============================================

    r = r_of_x

    # Cone walls
    dendrite_geometry.plot(x, r, 'k', linewidth=2)
    dendrite_geometry.plot(x, -r, 'k', linewidth=2)

    # Fill cone
    dendrite_geometry.fill_between(
        x,
        -r,
        r,
        color='gray',
        alpha=0.3
    )

    # Center axis y=0
    dendrite_geometry.axhline(
        0,
        color='gray',
        linestyle=':',
        linewidth=1.5
    )

    # Vertical line at x=0
    dendrite_geometry.axvline(
        0,
        color='gray',
        linestyle='--',
        linewidth=1.5
    )

    dendrite_geometry.set_xlabel("x")
    dendrite_geometry.set_ylabel("radius")
    dendrite_geometry.set_title("Cable geometry")

    # ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    show_plots_non_blocking()

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

def synaptic_input_profile(t, x0, dt, p: ConicalNumericalCableParameters):
    return dirac_delta_unitless(x0=x0, t0=to_SI(0.1 * ms), I_e=p.I_e, x=p.x, t=t, dx=p.dx, dt=dt, tau_m=p.tau, r_of_x=p.radius(x0))

def crank_nicolson(t_span, V0, A, p: ConicalNumericalCableParameters, events, saved_frames=1, verbose=False):
    """
    Solve dV/dt = A V using Crank-Nicolson.

    (I - dt/2 A) V[n+1] = (I + dt/2 A) V[n]
    """

    t0, tf = t_span

    # Number of time steps
    num_steps = int(np.ceil((tf - t0) / p.dt))

    # Saving
    save_every = int(np.ceil(num_steps / saved_frames))
    num_save = num_steps // save_every + 1

    times = np.zeros(num_save) * t0
    sol = np.zeros((num_save, len(V0)))

    # Identity matrix
    I = eye(A.shape[0], format="csc")

    # Crank-Nicolson matrices
    L = (I - 0.5 * p.dt * A).tocsc()
    R = (I + 0.5 * p.dt * A).tocsc()

    # Factorize once
    solve = factorized(L)

    # Initial condition
    t = t0
    V = V0.copy()

    times[0] = t
    sol[0] = V

    next_spike_index = 0
    for step in range(1, num_steps + 1):

        dt_step = min(p.dt, tf - t)

        synaptic_input_at_t, next_spike_index = synaptic_spike_train_input(t=t, dt=p.dt, p=p, next_spike_idx=next_spike_index, events=events)
        input_t = p.dt * 1 / p.c_m * synaptic_input_at_t

        V_old = None

        # Before update
        if np.any(input_t != 0):
            V_old = V.copy()

        rhs = R @ V + input_t

        # Solve:
        # (I - dt/2 A) V_new = rhs
        V = solve(rhs)

        if V_old is not None:
            np.set_printoptions(threshold=21)

        t += dt_step

        if step % save_every == 0:
            iteration = step // save_every

            if verbose:
                print(
                    f"[CN {iteration}/{num_save}] "
                    f"step {step}/{num_steps}"
                )

            times[iteration] = t
            sol[iteration] = V

    return times, sol

def save_simulation(times, V_s, t_max, p, events, seed=None, simulation_label="conical_cable_with_spike_train"):
    dt_ns = int(round(p.dt * second / psecond))  # dt in nanoseconds, avoids ugly floats
    t_max =t_max * second / msecond

    filename = (
        f"conical_cable_with_spike_train_{uniform_label_to_filename(simulation_label)}"
        f"_N{len(p.x)}"
        f"_L{round(p.L * meter / um)}"
        f"_t_max{t_max:.3f}".replace(".", "_") + "ms"
        f"_dt{dt_ns}ps"
        f"_n_events{len(events)}"
        f"_seed{seed}"                                    
        ".npz"
    )

    path = Path("saved_simulations") / filename
    path.parent.mkdir(exist_ok=True)

    np.savez_compressed(
        path,
        times=times,
        V_s=V_s,
        dt=p.dt,
        x=p.x,
        events=events,
        tau=p.tau,
        L=p.L,
        N=len(p.x),
        I_e=p.I_e,
    )

    print(f"Saved {path}")
    return path


def synaptic_spike_train_input(
    t,
    dt,
    events,
    next_spike_idx,
    p: ConicalNumericalCableParameters):
    """
    Return the synaptic input generated by all spikes occurring in
    the time interval [t, t + dt), together with the index of the
    next unconsumed spike.

    Parameters
    ----------
    t : float
        Start of the current time step [s].

    dt : float
        Width of the current time step [s].

    event_times : np.ndarray
        Sorted spike times [s].

    event_positions : np.ndarray
        Position corresponding to each spike [m].

    next_spike_idx : int
        Index of the first spike not yet consumed.

    p : ConicalNumericalCableParameters
        Cable parameters.

    Returns
    -------
    input_t : np.ndarray
        Total synaptic input profile for this time step.

    next_spike_idx : int
        Index of the first spike not consumed.
    """

    input_t = np.zeros_like(p.x)

    if next_spike_idx >= len(events[0]):
        return input_t, len(events[0])

    t_end = t + dt

    event_times, event_positions = events
    n_events = len(event_times)

    while next_spike_idx < n_events:

        spike_t = event_times[next_spike_idx]

        # This spike belongs to a future time step.
        if spike_t >= t_end:
            break

        # This should normally never happen if the index is maintained
        # correctly, but protects against stale spikes.
        if spike_t < t:
            raise ValueError(
                f"Spike at t={spike_t} is before current time "
                f"t={t}. next_spike_idx={next_spike_idx}."
            )

        spike_x = event_positions[next_spike_idx]

        input_t += dirac_delta_unitless(
            x0=spike_x,
            t0=spike_t,
            I_e=p.I_e,
            x=p.x,
            t=t,
            dx=p.dx,
            dt=dt,
            tau_m=p.tau,
            r_of_x=p.radius(spike_x),
        )

        next_spike_idx += 1

    return input_t, next_spike_idx

def simulate_crank_nicolson_unitless_closed_tapered_cylinder(events, x_N=301, dt_=to_SI(0.001 * ms), t_max=to_SI(30 * ms), L=to_SI(500 * um),saved_frames = 1200, verbose=True, plot=False, save=False, simulation_label="uniform distribution"):
    simulation_params = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L, I_e = to_SI(150 * pampere))
    p = simulation_params.to_numerical()

    # 1. Parameters
    return simulate_crank_nicolson_unitless_closed_tapered_cylinder_with_param(events=events, p=p, t_max=t_max, saved_frames=saved_frames, verbose=verbose, plot=plot, simulation_label=simulation_label, save=save)

def file_name_prefix_for_uniform(a: float, b: float):
    label = experiment_label_for_uniform(a, b)
    return uniform_label_to_filename(label)

def uniform_label_to_filename(label: str):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', label.replace(" - ", "_").lower())

def experiment_label_for_uniform(a: Quantity, b: Quantity):
    if is_dimensionless(a):
        a = a * meter

    if is_dimensionless(b):
        b = b * meter


    return f"Uniform x ~ [{int(a / um)} um - {int(b / um)} um]"

def simulate_crank_nicolson_unitless_closed_tapered_cylinder_with_param(events, p: ConicalNumericalCableParameters, t_max=to_SI(30 * ms),
                                                                        saved_frames = 1200, verbose=True, plot=False, save=False,
                                                                        simulation_label="uniform distribution"):
    print(f"Simulating Crank-Nicolson unitless cone with dt = {p.dt: .5e}")

    # Spatial domain and initial condition
    x = p.x
    dx = p.dx
    tau = p.tau

    a = p.a()
    b = p.b(x)

    lower = b[1:] / dx ** 2 + a / (2 * dx)
    main = -1 / tau - 2 * b / dx ** 2
    upper = b[:-1] / dx ** 2 - a / (2 * dx)

    # Sparse tridiagonal matrix
    A = diags(
        diagonals=[lower, main, upper],
        offsets=[-1, 0, 1],
        format="lil")

    # empirically, this does not work! Stiffness computed when boundary conditions are applied: 306 030  = 3*1E6 vs 7E4 when conditions are not applied

    def solve(plot=True, t_max=to_SI(300 * ms)):

        print(f"tau = {tau}")
        print(f"dt = {p.dt}")

        u0 = np.zeros(len(x))

        print(f"{u0[0]}. Dimension {get_dimensions(u0[0])}")

        if verbose:
            assert is_dimensionless(u0)

        # 3. Solve the ODE via crank nicolson (x0, t_span, V0, dt =0.01 * ms, saved_frames=1, plot=True, verbose=False):
        times, V_s = crank_nicolson(t_span=(0, t_max), V0=u0, A=A, saved_frames=saved_frames, p=p, events=events)

        if verbose:
            assert is_dimensionless(times[0])
            assert is_dimensionless(V_s[0][0])
            assert is_dimensionless(V_s[0])

        if save:
            save_simulation(times=times, V_s=V_s, p=p, events=events, t_max=t_max, simulation_label="uniform distribution")

        if plot:
            plot_difussion_unitless_spike_train(times=times, V_s=V_s, p=p, events=events,
                                                sim_type=f"Crank-Nicolson unitless {simulation_label}", save=True)

        return np.max(V_s), np.argmax(V_s[1]), p.dt, events

    return solve(t_max=t_max, plot=plot)

def simulate_with_uniform(p: ConicalNumericalCableParameters, t_max=to_SI(1 * second), a = 0, b = to_SI(500 * um)):

    if have_same_dimensions(t_max, second):
        t_max = to_SI(t_max)

    r_i = 2000 * Hz

    t_distribution = expon(scale=1.0 / r_i)
    x_distribution = uniform(loc=a, scale=b)

    spike_train = create_delta_pulses(
        t_max=t_max,
        x_distribution=x_distribution,
        t_distribution=t_distribution,
    )

    print("Simulating spike train: ", spike_train.shape)
    simulate_crank_nicolson_unitless_closed_tapered_cylinder_with_param(
        p=p,
        t_max=t_max,
        events=spike_train,
        verbose=True,
        plot=True,
        save=True,
        saved_frames=3 * 10 ** 4,
        simulation_label=experiment_label_for_uniform(a, b))


class TaperedDendritesPDECase(unittest.TestCase):
    def test_heat_difussion(self):

        # 1. Parameters
        alpha = 0.01  # Thermal diffusivity
        L = 10.0  # Length of the rod
        N = 50  # Number of spatial grid points
        dx = L / (N - 1)  # Spatial step size

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)
        u0 = np.exp(-100 * (x - 3.5) ** 2)  # Gaussian peak in the center

        # 2. Define the PDE as a system of ODEs
        def heat_equation(t, u):
            # Initialize derivative array
            du_dt = np.zeros_like(u)

            # Interior points using central difference
            du_dt[1:-1] = alpha * (u[:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2

            # Boundary conditions (Dirichlet: u = 0 at x=0 and x=L)
            du_dt[0] = 0
            du_dt[-1] = 0

            return du_dt

        # 3. Solve the ODE system using SciPy's solve_ivp
        t_span = (0, 10.0)
        t_eval = np.linspace(0, 10.0, 6)  # Save output at 6 specific times
        sol = solve_ivp(heat_equation, t_span, u0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)

        # 4. Visualize the results
        plt.figure(figsize=(8, 5))
        for i in range(len(sol.t)):
            plt.plot(sol.y[:, i], label=f't = {sol.t[i]:.1f}')

        plt.title("1D Heat Equation via Method of Lines")
        plt.xlabel("Spatial grid (x)")
        plt.ylabel("Temperature (u)")
        plt.legend()
        plt.show()

    def test_heat_difussion_on_cone(self):

        # 1. Parameters
        L = 10.0  # Length of the rod
        N = 50  # Number of spatial grid points
        dx = L / (N - 1)  # Spatial step size

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)
        k = 0.05
        r_0 = 1
        r_of_x = r_0 * (1 - k * x)
        rho_of_x = r_of_x * np.sqrt(1 + k**2 * r_0**2)

        for x_2 in [0.5, 1.5, 2.5, 4.5, 5.5, 6.5, 6.5, 8.5, 9.5, 9.9]:

            #u0_1 = np.exp(-100 * (x - 3.5) ** 2)
            u0_1 = np.zeros_like(x)
            u0_2 = np.exp(-100 * (x - x_2) ** 2)

            u0 = u0_1 + u0_2

            def synaptic_input_profile(t, t0=3.5, x0=6.5, I0=1.0, sigma=0.05):
                """
                Models a spatio-temporal Dirac delta impulse input.
                """
                # 1. Temporal component: Smooth Gaussian regularized delta
                temporal_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))

                # 2. Spatial component: Step indicator normalized by mesh size
                spatial_delta = np.zeros_like(x)
                closest_node_index = np.argmin(np.abs(x - x0))
                spatial_delta[closest_node_index] = 1.0 / dx

                # Combined current injection vector
                return I0 * temporal_delta * spatial_delta

            cm = 10
            gL = 0.1
            ra = 10
            I_syn_func = None
            #def linear_taper_cable_equation(t, V, x, dx, cm, gL, ra, rho_func, r_func, m, I_syn_func):
            def linear_taper_cable_equation(t, V):
                """
                Evaluates dV/dt for a cable with a perfectly linear radius profile r(x) = m*x + b.
                Uses standard analytical expansion and central differences.
                """
                dV_dt = np.zeros_like(V)

                # 1. Compute profiles at standard grid points
                r = r_of_x
                rho = rho_of_x
                I_syn = synaptic_input_profile(t=t)

                # 2. Slice variables for interior nodes (index 1 to N-2)
                V_mid = V[1:-1]
                V_left = V[:-2]
                V_right = V[2:]

                r_mid = r[1:-1]
                rho_mid = rho[1:-1]

                # 3. Compute central differences for the derivatives
                dV2_dx2 = (V_right - 2 * V_mid + V_left) / (dx ** 2)
                dV_dx = (V_right - V_left) / (2 * dx)

                # 4. Reconstruct the expanded diffusion term
                # r^2 * d2V/dx2 + 2 * m * r * dV/dx
                diffusion_term = (r_mid ** 2 * dV2_dx2) + (2 * k * r_mid * dV_dx)

                # 5. Assemble full right-hand side equation
                prefactor = 1.0 / (2.0 * ra * rho_mid)
                dV_dt[1:-1] = (-gL * V_mid + prefactor * diffusion_term + I_syn[1:-1]) / cm

                # 6. Apply Boundary Conditions (Example: Sealed / insulated ends)
                # Simple zero-flux boundary condition approximation
                dV_dt[-1] = 0

                print(f"t = {t: .3f}, V[14, 18]={V[14: 18]}")
                print(f"t = {t: .3f}, dV/dt[14, 18]={dV_dt[15: 19]}")
                print()
                return dV_dt

            # 3. Solve the ODE system using SciPy's solve_ivp
            t_span = (0, 10.0)
            t_eval = np.linspace(0, 10.0, 6)  # Save output at 6 specific times
            sol = solve_ivp(linear_taper_cable_equation, t_span, u0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)

            # 4. Visualize the results
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(8, 6),
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=False
            )

            # Top plot: solution over time
            for i in range(len(sol.t)):
                ax1.plot(sol.y[:, i], label=f't = {sol.t[i]:.1f}')

            ax1.set_title("Cable eq")
            ax1.set_xlabel("Spatial grid (x)")
            ax1.set_ylabel("Voltage (mV)")
            ax1.set_ylim((0, 1.1))
            ax1.legend()

            # Bottom plot: x vs r_of_x
            ax2.plot(x, r_of_x, 'k-')
            ax2.set_xlabel("x")
            ax2.set_ylabel("r(x)")
            ax2.set_title("Radius profile")

            plt.tight_layout()
            plt.show()

    # there is a formula for stability of the simulation delta t <= 0.5 (delta x)^2 / alpha
    # alpha is the prefactor of the second order deriv
    def test_solve_heat_eq_from_youtube_tutorial(self):
        a = 110 # diffusivity coefficient
        length = 50 # mm
        time = 4 # seconds
        nodes = 10
        dx = length / nodes
        dt = 0.5 * dx**2 / a

        u = np.zeros(nodes) + 20 # 20 is initial condition
        u[0] = 100 # Degrees. Initial condition.
        u[-1] = 100

        import matplotlib
        matplotlib.use("TkAgg")
        plt.ion()
        fig, axis = plt.subplots()
        pcm = axis.pcolormesh([u], cmap=plt.cm.jet, vmin=0, vmax=100)
        plt.colorbar(pcm, ax=axis)
        axis.set_ylim((-2, 3))
        #plt.show(block=False)

        counter = 0
        # simulation
        while counter < time:
            w = u.copy()

            for i in range(1, nodes - 1):
                u[i] = dt * a * (w[i-1] - 2 * w[i] + w[i+1]) / dx**2 + w[i]

            counter += dt
            print(f"t: {counter: .3f} [s], Average temperature: {np.mean(u): .2f} C")
            pcm.set_array([u])
            axis.set_title(f"Distribution at t: {counter: .3f}")
            #plt.pause(0.005)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

        plt.pause(1)
        plt.ioff()

    def test_solve_heat_eq_from_youtube_tutorial_improved(self):
        a = 110 # diffusivity coefficient
        length = 50 # mm
        time = 4 # seconds
        nodes = 10
        dx = length / nodes
        dt = 0.5 * dx**2 / a

        u = np.zeros(nodes) + 20 # 20 is initial condition
        u[0] = 100 # Degrees. Initial condition.
        u[-1] = 100

        import matplotlib
        matplotlib.use("TkAgg")
        plt.ion()
        fig, axis = plt.subplots()
        pcm = axis.pcolormesh([u], cmap=plt.cm.jet, vmin=0, vmax=100)
        plt.colorbar(pcm, ax=axis)
        axis.set_ylim((-2, 3))
        #plt.show(block=False)

        counter = 0
        # simulation
        while counter < time:
            w = u.copy()

            u[1:-1] = dt * a * (w[0:-2] - 2 * w[1:-1] + w[2:]) / dx**2 + w[1:-1]

            counter += dt
            print(f"t: {counter: .3f} [s], Average temperature: {np.mean(u): .2f} C")
            pcm.set_array([u])
            axis.set_title(f"Distribution at t: {counter: .3f}")
            #plt.pause(0.005)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

        plt.pause(1)
        plt.ioff()

    def test_difussion_on_cone_colormap(self):
        # 1. Parameters
        L = 10.0  # Length of the rod
        N = 50  # Number of spatial grid points
        dx = L / N  # Spatial step size

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)
        k = 0.05
        r_0 = 1
        r_of_x = r_0 * (1 - k * x)
        rho_of_x = r_of_x * np.sqrt(1 + k ** 2 * r_0 ** 2)

        x_2 = 6.5

        u0_1 = np.exp(-100 * (x - 3.5) ** 2)
        u0_2 = np.exp(-100 * (x - x_2) ** 2)

        u0 = u0_1 + u0_2

        import matplotlib
        matplotlib.use("TkAgg")
        plt.ion()
        fig, axis = plt.subplots()
        pcm = axis.pcolormesh([np.zeros_like(x)], cmap=plt.cm.jet, vmin=0, vmax=0.001)
        plt.colorbar(pcm, ax=axis)
        axis.set_ylim((-2, 3))

        def synaptic_input_profile(t, t0=3.5, x0=6.5, I0=1.0, sigma=0.05):
            """
            Models a spatio-temporal Dirac delta impulse input.
            """
            # 1. Temporal component: Smooth Gaussian regularized delta
            temporal_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))

            # 2. Spatial component: Step indicator normalized by mesh size
            spatial_delta = np.zeros_like(x)
            closest_node_index = np.argmin(np.abs(x - x0))
            spatial_delta[closest_node_index] = 1.0 / dx

            # Combined current injection vector
            return I0 * temporal_delta * spatial_delta

        cm = 10
        gL = 0.1
        ra = 10
        I_syn_func = None

        # def linear_taper_cable_equation(t, V, x, dx, cm, gL, ra, rho_func, r_func, m, I_syn_func):
        def linear_taper_cable_equation(t, V):
            """
            Evaluates dV/dt for a cable with a perfectly linear radius profile r(x) = m*x + b.
            Uses standard analytical expansion and central differences.
            """
            dV_dt = np.zeros_like(V)

            # 1. Compute profiles at standard grid points
            r = r_of_x
            rho = rho_of_x
            I_syn = synaptic_input_profile(t=t)

            # 2. Slice variables for interior nodes (index 1 to N-2)
            V_mid = V[1:-1]
            V_left = V[:-2]
            V_right = V[2:]

            r_mid = r[1:-1]
            rho_mid = rho[1:-1]

            # 3. Compute central differences for the derivatives
            dV2_dx2 = (V_right - 2 * V_mid + V_left) / (dx ** 2)
            dV_dx = (V_right - V_left) / (2 * dx)

            # 4. Reconstruct the expanded diffusion term
            # r^2 * d2V/dx2 + 2 * k * r * dV/dx
            diffusion_term = (r_mid ** 2 * dV2_dx2) - (2 * k * r_mid * dV_dx)

            # 5. Assemble full right-hand side equation
            prefactor = 1.0 / (2.0 * ra * rho_mid)
            dV_dt[1:-1] = (-gL * V_mid + prefactor * diffusion_term + I_syn[1:-1]) / cm

            # 6. Apply Boundary Conditions (Example: Sealed / insulated ends)
            # Simple zero-flux boundary condition approximation
            dV_dt[-1] = 0

            print(f"t = {t: .3f}, V[14, 18]={V[14: 18]}")
            print(f"t = {t: .3f}, dV/dt[14, 18]={dV_dt[15: 19]}")
            print()

            pcm.set_array([V])
            # plt.pause(0.005)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

            return dV_dt

        # 3. Solve the ODE system using SciPy's solve_ivp
        t_span = (0, 10.0)
        t_eval = np.linspace(0, 10.0, 6)  # Save output at 6 specific times
        sol = solve_ivp(linear_taper_cable_equation, t_span, u0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)

        plt.pause(1)
        plt.ioff()

    # this shows that this is a current based model. We definitely need conductance based model for this!
    def test_heat_difussion_on_cone_runge_kutta(self):

        cm = 2
        gL = 0.1
        ra = 1

        # 1. Parameters
        L = 100.0  # Length of the rod
        N = 500  # Number of spatial grid points
        dx = L / (N - 1)  # Spatial step size

        tau = cm / gL

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)
        k = 0.0098
        r_0 = 1
        r_of_x = r_0 * (1 - k * x)
        rho_of_x = r_of_x * np.sqrt(1 + k**2 * r_0**2)
        prefactor = 1 / (cm * ra * np.sqrt(1 + r_0 ** 2 * k ** 2))

        min_prefactor_difussion = rho_of_x[-1] / (2 * prefactor)
        max_prefactor_difussion = rho_of_x[0] / (2 * prefactor)
        # stability requirements
        # second order PDE induces a fourier grid number. Defined as 0.5 delta_x^2 / (max diffussion coeff)
        grid_fourier_number = 0.5 * dx**2 / max_prefactor_difussion
        # first order PDE constraint. Courant-Friedrics-Lewy number. delta x / (1st order diffusion coef)
        cfl_constraint = dx / (k * r_0 / prefactor)

        #dt = min(grid_fourier_number, cfl_constraint)

        dt = 0.25 * dx ** 2 * r_0 * min(min_prefactor_difussion, 1)

        print(f"tau = {tau}")
        print(f"dt = {dt}")

        def solve(x_2, plot=True, t_max=10):
            #u0_1 = np.exp(-100 * (x - 3.5) ** 2) / cm
            u0_1 = np.zeros_like(x)
            u0_2 = np.exp(-100 * (x - x_2) ** 2) / cm

            u0 = u0_1 + u0_2

            def synaptic_input_profile(t, t0=3.5, x0=6.5, I0=1.0, sigma=0.05):
                """
                Models a spatio-temporal Dirac delta impulse input.
                """
                # 1. Temporal component: Smooth Gaussian regularized delta
                temporal_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))

                # 2. Spatial component: Step indicator normalized by mesh size
                spatial_delta = np.zeros_like(x)
                closest_node_index = np.argmin(np.abs(x - x0))
                spatial_delta[closest_node_index] = 1.0 / dx

                # Combined current injection vector
                return I0 * temporal_delta * spatial_delta

            def linear_taper_cable_equation(t, V):
                """
                Evaluates dV/dt for a cable with a perfectly linear radius profile r(x) = m*x + b.
                Uses standard analytical expansion and central differences.
                """
                dV_dt = np.zeros_like(V)

                # 1. Compute profiles at standard grid points
                r = r_of_x
                rho = rho_of_x
                I_syn = synaptic_input_profile(t=-1) / cm

                # 2. Slice variables for interior nodes (index 1 to N-2)
                V_mid = V[1:-1]
                V_left = V[:-2]
                V_right = V[2:]

                r_mid = r[1:-1]


                # 3. Compute central differences for the derivatives
                dV2_dx2 = (V_right - 2 * V_mid + V_left) / (dx ** 2)
                dV_dx = (V_right - V_left) / (2 * dx)

                # 4. Reconstruct the expanded diffusion term

                # 5. Assemble full right-hand side equation
                dV_dt[1:-1] = ( -V_mid / tau  - k * r_0 / prefactor * dV_dx + r_mid / (2 * prefactor) * dV2_dx2 + I_syn[1:-1])

                # 6. Apply Boundary Conditions (Example: Sealed / insulated ends)
                # Simple zero-flux boundary condition approximation
                dV_dt[-1] = 0

                #print(f"t = {t: .3f}, V[14, 18]={V[14: 18]}")
                #print(f"t = {t: .3f}, dV/dt[14, 18]={dV_dt[15: 19]}")
                #print()
                return dV_dt

            def rk4(f, t_span, y0, dt, save_times=None):
                t0, tf = t_span

                t = t0
                y = y0.copy()

                if save_times is None:
                    save_times = np.arange(t0, tf + dt, dt)

                save_times = np.asarray(save_times)

                sol = np.empty((len(save_times), len(y0)))
                sol[0] = y

                save_idx = 1

                while t < tf:
                    delta_t = min(dt, tf - t)

                    k1 = f(t, y)
                    k2 = f(t + delta_t / 2, y + delta_t * k1 / 2)
                    k3 = f(t + delta_t / 2, y + delta_t * k2 / 2)
                    k4 = f(t + delta_t, y + delta_t * k3)

                    y += delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    t += delta_t

                    while save_idx < len(save_times) and t >= save_times[save_idx]:
                        sol[save_idx] = y
                        save_idx += 1

                return save_times, sol

            # 3. Solve the ODE system using manual runge kutta
            t_eval = np.linspace(0, 0.1, 5)  # Save output at 6 specific times
            t_eval = np.arange(0, 5) * 5 * dt
            saved_sols, sol = rk4(linear_taper_cable_equation, t_span=(0, t_max), y0=u0, dt=dt, save_times=t_eval)

            if plot:
                # 4. Visualize the results
                fig, (ax1, ax2) = plt.subplots(
                    2, 1,
                    figsize=(8, 6),
                    gridspec_kw={'height_ratios': [3, 1]},
                    sharex=False
                )

                # Top plot: solution over time
                for i in range(len(saved_sols)):
                    ax1.plot(x, sol[i], label=f't = {saved_sols[i]:.1f}')

                ax1.set_title(f"Cable eq. Second input at {x_2: .1f}. Max V = {np.max(sol[0]): .4f} mV locally at index {np.argmax(sol[0]) * dx : .2f}")
                ax1.set_xlabel("Spatial grid (x)")
                ax1.set_ylabel("Voltage (mV)")
                #ax1.set_ylim((0, 0.5))
                ax1.legend()

                # Bottom plot: x vs r_of_x
                ax2.plot(x, r_of_x, 'k-')
                ax2.set_xlabel("x")
                ax2.set_ylabel("r(x)")
                ax2.set_title("Radius profile")

                plt.tight_layout()
                fig.show()

            return np.max(sol[0]), np.argmax(sol[0])

        #for splits in [50, 100, 200, 250, 500, 750, 1000, 1250, 1500]:
        for splits in [10]:
            x2_values = np.linspace(0.1, L - 0.1, splits)
            results = Parallel(
                n_jobs=-3 if len(x2_values) > 2 else 1,  # use all CPU cores
                backend="loky",  # process-based (default)
                verbose=10
            )(
                delayed(solve)(x2, t_max=0.3, plot=True) for x2 in x2_values
            )

            max_vals, argmax_vals = zip(*results)
            max_vals = np.array(max_vals)
            argmax_vals = np.array(argmax_vals)
            argmax_x = np.asarray(argmax_vals) * dx

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # x2 vs maximum value
            axes[0].plot(x2_values, max_vals, lw=2)
            axes[0].set_xlabel(r"$x_2$")
            axes[0].set_ylabel(r"$\max(V)$")
            axes[0].set_title("Peak voltage")

            # x2 vs argmax
            axes[1].plot(x2_values, argmax_x, lw=2)
            axes[1].set_xlabel(r"$x_2$")
            axes[1].set_ylabel(r"$\arg\max(V)$")
            axes[1].set_title("Peak location")

            # argmax vs max
            axes[2].scatter(argmax_x, max_vals, s=8, alpha=0.6)
            axes[2].set_xlabel(r"$\arg\max(V)$")
            axes[2].set_ylabel(r"$\max(V)$")
            axes[2].set_title("Peak location vs peak voltage")

            fig.suptitle(f"{splits} splits")

            plt.tight_layout()
            plt.show()


    def test_heat_difussion_on_cone_forward_euler(self):

        cm = 2
        gL = 0.1
        ra = 1

        # 1. Parameters
        L = 100.0  # Length of the rod
        N = 500  # Number of spatial grid points
        dx = L / (N - 1)  # Spatial step size

        tau = cm / gL

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)
        k = 0.0098
        r_0 = 1
        r_of_x = r_0 * (1 - k * x)
        rho_of_x = r_of_x * np.sqrt(1 + k**2 * r_0**2)
        prefactor = 1 / (cm * ra * np.sqrt(1 + r_0 ** 2 * k ** 2))

        min_prefactor_difussion = rho_of_x[-1] / (2 * prefactor)
        dt = 0.25 * dx ** 2 * r_0 * min(min_prefactor_difussion, 1)

        print(f"tau = {tau}")
        print(f"dt = {dt}")

        def solve(x_2, plot=True, t_max=10):
            #u0_1 = np.exp(-100 * (x - 3.5) ** 2) / cm
            u0_1 = np.zeros_like(x)
            u0_2 = np.exp(-100 * (x - x_2) ** 2) / cm

            u0 = u0_1 + u0_2

            def synaptic_input_profile(t, t0=3.5, x0=6.5, I0=1.0, sigma=0.05):
                """
                Models a spatio-temporal Dirac delta impulse input.
                """
                # 1. Temporal component: Smooth Gaussian regularized delta
                temporal_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))

                # 2. Spatial component: Step indicator normalized by mesh size
                spatial_delta = np.zeros_like(x)
                closest_node_index = np.argmin(np.abs(x - x0))
                spatial_delta[closest_node_index] = 1.0 / dx

                # Combined current injection vector
                return I0 * temporal_delta * spatial_delta

            def linear_taper_cable_equation(t, V):
                """
                Evaluates dV/dt for a cable with a perfectly linear radius profile r(x) = m*x + b.
                Uses standard analytical expansion and central differences.
                """
                dV_dt = np.zeros_like(V)

                # 1. Compute profiles at standard grid points
                r = r_of_x
                rho = rho_of_x
                I_syn = synaptic_input_profile(t=-1) / cm

                # 2. Slice variables for interior nodes (index 1 to N-2)
                V_mid = V[1:-1]
                V_left = V[:-2]
                V_right = V[2:]

                r_mid = r[1:-1]

                # 3. Compute central differences for the derivatives
                dV2_dx2 = (V_right - 2 * V_mid + V_left) / (dx ** 2)
                dV_dx = (V_right - V_left) / (2 * dx)

                # 4. Reconstruct the expanded diffusion term

                # 5. Assemble full right-hand side equation
                dV_dt[1:-1] = ( -V_mid / tau  - k * r_0 / prefactor * dV_dx + r_mid / (2 * prefactor) * dV2_dx2 + I_syn[1:-1])

                # 6. Apply Boundary Conditions (Example: Sealed / insulated ends)
                # Simple zero-flux boundary condition approximation
                dV_dt[-1] = 0

                #print(f"t = {t: .3f}, V[14, 18]={V[14: 18]}")
                #print(f"t = {t: .3f}, dV/dt[14, 18]={dV_dt[15: 19]}")
                #print()
                return dV_dt

            def rk4(f, t_span, y0, dt, save_times=None):
                t0, tf = t_span

                t = t0
                y = y0.copy()

                if save_times is None:
                    save_times = np.arange(t0, tf + dt, dt)

                save_times = np.asarray(save_times)

                sol = np.empty((len(save_times), len(y0)))
                sol[0] = y

                save_idx = 1

                while t < tf:
                    delta_t = min(dt, tf - t)

                    k1 = f(t, y)
                    k2 = f(t + delta_t / 2, y + delta_t * k1 / 2)
                    k3 = f(t + delta_t / 2, y + delta_t * k2 / 2)
                    k4 = f(t + delta_t, y + delta_t * k3)

                    y += delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    t += delta_t

                    while save_idx < len(save_times) and t >= save_times[save_idx]:
                        sol[save_idx] = y
                        save_idx += 1

                return save_times, sol

            # 3. Solve the ODE system using manual runge kutta
            t_eval = np.linspace(0, 0.1, 5)  # Save output at 6 specific times
            t_eval = np.arange(0, 5) * 5 * dt
            saved_sols, sol = rk4(linear_taper_cable_equation, t_span=(0, t_max), y0=u0, dt=dt, save_times=t_eval)

            if plot:
                # 4. Visualize the results
                fig, (ax1, ax2) = plt.subplots(
                    2, 1,
                    figsize=(8, 6),
                    gridspec_kw={'height_ratios': [3, 1]},
                    sharex=False
                )

                # Top plot: solution over time
                for i in range(len(saved_sols)):
                    ax1.plot(x, sol[i], label=f't = {saved_sols[i]:.1f}')

                ax1.set_title(f"Cable eq. Second input at {x_2: .1f}. Max V = {np.max(sol[0]): .4f} mV locally at index {np.argmax(sol[0]) * dx : .2f}")
                ax1.set_xlabel("Spatial grid (x)")
                ax1.set_ylabel("Voltage (mV)")
                #ax1.set_ylim((0, 0.5))
                ax1.legend()

                # Bottom plot: x vs r_of_x
                ax2.plot(x, r_of_x, 'k-')
                ax2.set_xlabel("x")
                ax2.set_ylabel("r(x)")
                ax2.set_title("Radius profile")

                plt.tight_layout()
                fig.show()

            return np.max(sol[0]), np.argmax(sol[0])

        #for splits in [50, 100, 200, 250, 500, 750, 1000, 1250, 1500]:
        for splits in [10]:
            x2_values = np.linspace(0.1, L - 0.1, splits)
            results = Parallel(
                n_jobs=-3 if len(x2_values) > 2 else 1,  # use all CPU cores
                backend="loky",  # process-based (default)
                verbose=10
            )(
                delayed(solve)(x2, t_max=0.3, plot=True) for x2 in x2_values
            )

            max_vals, argmax_vals = zip(*results)
            max_vals = np.array(max_vals)
            argmax_vals = np.array(argmax_vals)
            argmax_x = np.asarray(argmax_vals) * dx

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # x2 vs maximum value
            axes[0].plot(x2_values, max_vals, lw=2)
            axes[0].set_xlabel(r"$x_2$")
            axes[0].set_ylabel(r"$\max(V)$")
            axes[0].set_title("Peak voltage")

            # x2 vs argmax
            axes[1].plot(x2_values, argmax_x, lw=2)
            axes[1].set_xlabel(r"$x_2$")
            axes[1].set_ylabel(r"$\arg\max(V)$")
            axes[1].set_title("Peak location")

            # argmax vs max
            axes[2].scatter(argmax_x, max_vals, s=8, alpha=0.6)
            axes[2].set_xlabel(r"$\arg\max(V)$")
            axes[2].set_ylabel(r"$\max(V)$")
            axes[2].set_title("Peak location vs peak voltage")

            fig.suptitle(f"{splits} splits")

            plt.tight_layout()
            plt.show()

    def test_pde_triagonal_matrix(self, verbose=False, x_N=100, t_max = 50 * ms):

        c_m = 1 * uF / cm **2
        Rm = 2 * 1E4 * ohm * cm**2
        gL = 1/Rm
        ra = 100 * ohm * cm

        # 1. Parameters
        L = 500.0 * um
        N = x_N
        dx = L / (N - 1) # um

        tau = c_m / gL # ms

        assert have_same_dimensions(tau, 1*second)

        print("tau=", tau)

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)
        r_0 = 2 * um
        r_L = 0.5 * um

        k = (1 - r_L / r_0) / L
        r_of_x = r_0 * (1 - k * x)

        a = k * r_0 / (2 * c_m * ra * np.sqrt(1 + r_0 ** 2 * k ** 2))
        b = r_0 * (1 - k * x) / (2 * c_m * ra * np.sqrt(1 + r_0 ** 2 * k ** 2))

        # TODO: find proper dt
        # CFL condition: delta t <= 1/2 (delta x) ^2 / alpha. Alpha is the prefactor of alpha dV^2 / d^2x
        dt = 0.5 * dx ** 2 / b[0]

        lower = b[1:] / dx ** 2 + a / (2 * dx)
        main = -1 / tau - 2 * b / dx ** 2
        upper = b[:-1] / dx ** 2 - a / (2 * dx)

        # Sparse tridiagonal matrix
        A = diags(
            diagonals=[lower, main, upper],
            offsets=[-1, 0, 1],
            format="lil"
        )

        # Sparse tridiagonal matrix
        A_only_tau_decay = diags(
            diagonals=[np.ones(len(x)) * (-1/tau)],
            offsets=[0],
            format="lil"
        )

        A_no_neumann_conditions = diags(
            diagonals=[lower, main, upper],
            offsets=[-1, 0, 1],
            format="csr"
        )

        # ensure boundary conditions automatically in A matrix
        A[0, 0] = -1 / tau - 2 * b[0] / dx ** 2
        A[0, 1] = 2 * b[0] / dx ** 2
        A[-1, -2] = 2 * b[-1] / dx ** 2
        A[-1, -1] = -1 / tau - 2 * b[-1] / dx ** 2
        # empirically, this does not work! Stiffness computed when boundary conditions are applied: 306 030  = 3*1E6 vs 7E4 when conditions are not applied

        #analyse_eigenvalues_generalized_locally_toeplitz_matrix(A)

        A = A.tocsr().toarray() * (1 / second)

        print(f"tau = {tau}")
        print(f"dt = {dt}")

        def solve(x_2, plot=True, t_max=300 * ms):

            if is_dimensionless(t_max):
                t_max = t_max * ms

            # u0_1 = np.exp(-100 * (x - 3.5) ** 2) / c_m
            u0_1 = np.zeros(x.shape) * mV
            sigma_v = 10 * um
            u0_2 = 100 * np.exp(- 0.5 * (((x - x_2) / sigma_v) ** 2))  * mV

            u0 = u0_1 + u0_2

            print(f"{u0[0] / volt}. Dimension {get_dimensions(u0[0] / volt)}")

            if verbose:
                assert have_same_dimensions(1 * volt / second, 1 * ampere / uF)
                assert have_same_dimensions(u0, 1 * volt)
                assert have_same_dimensions(u0, 1 * volt)

            def synaptic_input_profile(t, t0=3.5, x0=6.5*um):

               return dirac_delta(x0=x0, t0=t0, w=1 * uamp / cm**2, x=x, t=t, dx=dx, dt=dt)

            def linear_taper_cable_equation(t, V):
                """
                Computes dV/dt = A @ V + I_syn/c_m
                """
                I_syn = synaptic_input_profile(t)

                if verbose:
                    print(f"A: {get_dimensions(A)}")
                    print(f"V: {get_dimensions(V)}")
                    print(f"I Syn: {get_dimensions(I_syn)}")

                    print(f"A @ V: {get_dimensions(A @ V)}")

                    assert have_same_dimensions(A[0, 0], 1 / second)
                    assert have_same_dimensions(V[0], 1*volt)
                    assert have_same_dimensions(I_syn[0] / c_m, 1*volt/second)
                result = A @ V + I_syn / c_m

                if verbose:
                    assert have_same_dimensions(result[0], 1 * volt / second)

                return result

            def forward_euler(f: Callable[[float, np.ndarray], np.ndarray], t_span, V0: np.ndarray, dt: Quantity, saved_frames=1):
                """
                Forward Euler solver.

                Parameters
                ----------
                f : callable
                    RHS function f(t, V)
                t_span : tuple
                    (t0, tf)
                V0 : array
                    Initial condition
                dt : float
                    Time step
                save_every : int
                    Save every N iterations

                Returns
                -------
                times : array
                    Saved times
                sol : array
                    Saved solutions, shape = (time, space)
                """

                t0, tf = t_span

                # Number of Euler steps
                num_steps = int(np.ceil((tf - t0) / dt))

                # Number of saved states
                save_every = int(np.ceil(num_steps / saved_frames))
                num_save = num_steps//save_every + 1

                # Preallocate
                times = np.zeros(num_save) * ms
                sol = np.zeros((num_save, len(V0))) * mV
                t = t0
                V = V0.copy()

                times[0] = t0
                sol[0] = V

                if verbose:
                    assert have_same_dimensions(V0[0], 1*mV)
                    assert have_same_dimensions(V, 1*mV)
                    assert have_same_dimensions(sol[0][0], 1*mV)

                for step in range(1, num_steps + 1):
                        dt_step = min(dt, tf - t)

                        # Forward Euler step
                        V = V + dt_step * f(t, V)

                        t += dt_step

                        # If another save_every bunch
                        if step % save_every == 0:
                            iteration = step // save_every
                            print(f"[f {iteration}/{num_save}]: Reached step {step} from {num_steps} ({100 * step / num_steps:.2f}%)")
                            times[iteration] = t
                            sol[iteration] = V

                if verbose:
                    assert have_same_dimensions(sol[0], 1 * mV)
                    assert have_same_dimensions(sol[0][0], 1 * mV)
                    assert have_same_dimensions(V0[0], 1 * mV)
                    assert have_same_dimensions(V, 1 * mV)

                return times, sol

            # 3. Solve the ODE system using manual runge kutta
            times, V_s = forward_euler(linear_taper_cable_equation, t_span=(0 * ms, t_max), V0=u0, dt=dt, saved_frames=400)

            if verbose:
                assert have_same_dimensions(times[0], 1 * ms)
                assert have_same_dimensions(V_s[0][0], 1 * mV)
                assert have_same_dimensions(V_s[0], 1 * mV)

            if plot:
                plot_difussion_solution(times=times, x=x, r_of_x=r_of_x, V_s=V_s)

            return np.max(V_s[1]), np.argmax(V_s[1])

        # for splits in [50, 100, 200, 250, 500, 750, 1000, 1250, 1500]:
        for splits in [10]:
            x2_values = np.linspace(0.1 * um, L - 0.1 * um, splits)
            x2_values = [6.5 * um, 490*um]
            results = Parallel(
                n_jobs=-3 if len(x2_values) > 2 else 1,  # use all CPU cores
                backend="loky",  # process-based (default)
                verbose=10
            )(
                delayed(solve)(x2, t_max=t_max, plot=True) for x2 in x2_values
            )

            max_vals, argmax_vals = zip(*results)
            max_vals = np.array(max_vals)
            argmax_vals = np.array(argmax_vals)
            argmax_x = np.asarray(argmax_vals) * dx

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # x2 vs maximum value
            axes[0].plot(x2_values, max_vals, lw=2)
            axes[0].set_xlabel(r"$x_2$")
            axes[0].set_ylabel(r"$\max(V)$")
            axes[0].set_title("Peak voltage")

            # x2 vs argmax
            axes[1].plot(x2_values, argmax_x, lw=2)
            axes[1].set_xlabel(r"$x_2$")
            axes[1].set_ylabel(r"$\arg\max(V)$")
            axes[1].set_title("Peak location")

            # argmax vs max
            axes[2].scatter(argmax_x, max_vals, s=8, alpha=0.6)
            axes[2].set_xlabel(r"$\arg\max(V)$")
            axes[2].set_ylabel(r"$\max(V)$")
            axes[2].set_title("Peak location vs peak voltage")

            fig.suptitle(f"{splits} splits")

            plt.tight_layout()
            plt.show()

    def test_simulate_diffusion_on_closed_conical_dendrite_one_spike(self):

        x0_values = [to_SI(100 * um), to_SI(150 * um), to_SI(250 * um), to_SI(300 * um), to_SI(400 * um), to_SI(450 * um)]

        events = [np.array([[to_SI(0.1 * ms)], [x0]]) for x0 in x0_values]

        is_debug = False
        results = Parallel(n_jobs=1 if is_debug else  -3, verbose=10)(
            delayed(simulate_crank_nicolson_unitless_closed_tapered_cylinder)(
                events=event,
                x_N=101,
                dt_=to_SI(1E-8 * second),
                #t_max=to_SI(0.3 * second),
                t_max=to_SI(0.3 * ms),
                verbose=True,
                L=to_SI(500 * um),
                plot=True,
                save=True,
                saved_frames=3 * 10 ** 4,
            )
            for event in events
        )

    def test_simulate_diffusion_on_closed_conical_dendrite_spike_train(self):
        x_N = 101
        t_max = to_SI(3 * second)

        dt = to_SI(1E-7 * second)
        L = to_SI(2000 * um)
        r_i = 2000 * Hz

        t_distribution = expon(scale=1.0 / r_i)
        x_distribution = uniform(loc=0, scale=L)

        spike_train = create_delta_pulses(
            t_max=t_max,
            x_distribution=x_distribution,
            t_distribution=t_distribution,
        )

        print("Simulating spike train: ", spike_train.shape)

        simulate_crank_nicolson_unitless_closed_tapered_cylinder(
                x_N=x_N,
                dt_=dt,
                t_max=t_max,
                events=spike_train,
                verbose=True,
                L=L,
                plot=True,
                save=True,
                saved_frames=3 * 10 ** 4)

    def test_simulate_diffusion_on_closed_conical_dendrite_spike_trainfirst_half(self):

        limits = [(0, to_SI(2000 * um)), (to_SI(250 * um), to_SI(500 * um)), (to_SI(400 * um), to_SI(500 * um)),
                  (to_SI(150 * um), to_SI(350 * um)), (to_SI(450 * um), to_SI(500 * um))]
        t_max = to_SI(10 * second)
        p = default_params.with_SI_properties(dt=to_SI(1E-8 * second), N = 101, L=to_SI(500 * um)).to_numerical()


        simulate_with_uniform(p=p, t_max=t_max, a = 0, b = to_SI(500 * um))
        is_debug=False
        results = Parallel(n_jobs=1 if is_debug else -3, verbose=10)(
            delayed(simulate_with_uniform)(p=p, t_max=t_max, a = a, b=b) for a, b in limits
        )

    def test_simulate_diffusion_on_closed_conical_dendrite_spike_small_L(self):

        limits = [(0, to_SI(2000 * um)), (to_SI(250 * um), to_SI(2000 * um)), (to_SI(1000 * um), to_SI(2000 * um)),
                  (to_SI(250 * um), to_SI(750 * um)), (to_SI(500 * um), to_SI(750 * um)) ,(to_SI(1500 * um), to_SI(2000 * um))]
        t_max = to_SI(10 * second)
        p = default_params.with_SI_properties(dt=to_SI(1E-8 * second), N = 101, L=to_SI(500 * um)).to_numerical()


        simulate_with_uniform(p=p, t_max=t_max, a = 0, b = to_SI(500 * um))
        is_debug=False
        results = Parallel(n_jobs=1 if is_debug else -3, verbose=10)(
            delayed(simulate_with_uniform)(p=p, t_max=t_max, a = a, b=b) for a, b in limits
        )

    def test_simulate_diffusion_on_closed_conical_dendrite_spike_trainfirst_half(self):

        limits = [(0, to_SI(2000 * um)), (to_SI(250 * um), to_SI(2000 * um)), (to_SI(1000 * um), to_SI(2000 * um)),
                  (to_SI(250 * um), to_SI(750 * um)), (to_SI(500 * um), to_SI(750 * um)) ,(to_SI(1500 * um), to_SI(2000 * um))]
        t_max = to_SI(10 * second)
        p = default_params.with_SI_properties(dt=to_SI(5E-8 * second), N = 101, L=to_SI(1500 * um)).to_numerical()


        simulate_with_uniform(p=p, t_max=t_max, a = 0, b = to_SI(1500 * um))
        is_debug=False
        results = Parallel(n_jobs=1 if is_debug else -3, verbose=10)(
            delayed(simulate_with_uniform)(p=p, t_max=t_max, a = a, b=b) for a, b in limits
        )

    def test_difussions(self):
        self.simulate_500_um(t_max=to_SI(10 * ms))


    def params_500_um(self):
        L = to_SI(500 * um)
        limits = [(0, L), (0.25 * L, L), (0.5 * L, L), (0.75 * L, L),
                  (0.25 * L, 0.5 * L), (0.25 * L, 0.75 * L), (0.5 * L, 0.75 * L)]
        p = default_params.with_SI_properties(dt=to_SI(1E-8 * second), N=101, L=L,
                                              I_e=to_SI(15 * pampere)).to_numerical()

        return [(a, b, p) for a, b in limits]

    def params_1000_um(self):
        L = to_SI(1000 * um)
        limits = [(0, L), (0.25 * L, L), (0.5 * L, L), (0.75 * L, L),
                  (0.25 * L, 0.5 * L), (0.25 * L, 0.75 * L), (0.5 * L, 0.75 * L)]
        p = default_params.with_SI_properties(dt=to_SI(1E-8 * second), N=101, L=L,
                                              I_e=to_SI(15 * pampere)).to_numerical()

        return [(a, b, p) for a, b in limits]

    def params_1500_um(self):
        L = to_SI(1500 * um)
        limits = [(0, L), (0.3 * L, L), (0.6 * L, L), (0.6666666666666666 * L, L) , (to_SI(1250 * um), L),
                  (0, 0.3 * L), (0, 0.6 * L), (0, to_SI(1250 * um)) , (0, to_SI(1000 * um)), (0, to_SI(1250 * um)),
                  (to_SI(500 * um), L), (to_SI(750 * um), L), (to_SI(1000 * um), L),
                  (to_SI(500 * um), to_SI(1000)), (to_SI(1250 * um), L)]

        p = default_params.with_SI_properties(dt=to_SI(1E-8 * second), N=101, L=L,
                                              I_e=to_SI(15 * pampere)).to_numerical()

        return [(a, b, p) for a, b in limits]

    def test_sim_all(self, t_max=to_SI(10 * second)):
        is_debug=False
        almost_all_sim = self.params_500_um() + self.params_1000_um() + self.params_1500_um()
        print(almost_all_sim)
        results = Parallel(n_jobs=1 if is_debug else -2, verbose=10)(
            delayed(simulate_with_uniform)(p=p, t_max=t_max, a=a, b=b) for a, b, p in almost_all_sim)



if __name__ == '__main__':
    unittest.main()

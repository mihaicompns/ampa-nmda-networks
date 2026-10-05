"""
Manual long-running simulation scripts for balanced tapered-dendrite inputs.

These are intentionally scripts, not pytest tests. They restore the runnable
experiments that used to live in TaperredDendritesBalancedInputPDE.py in the old
repository, while keeping them outside automated test discovery.
"""

import argparse
import csv
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from brian2 import Hz, second, um
from brian2.units import ms, meter
from brian2.units.allunits import pampere
from joblib import Parallel, delayed
from scipy.sparse import diags
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
    crank_nicolson,
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
    run_balanced_comparison_sweep,
    submit_balanced_conical_with_cylindrical_comparison,
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


def _build_conical_crank_nicolson_matrix(p):
    x = p.x
    dx = p.dx
    tau = p.tau
    a = p.a()
    b = p.b(x)

    lower = b[1:] / dx ** 2 + a / (2 * dx)
    main = -1 / tau - 2 * b / dx ** 2
    upper = b[:-1] / dx ** 2 - a / (2 * dx)

    return diags(
        diagonals=[lower, main, upper],
        offsets=[-1, 0, 1],
        format="lil",
    )


def _build_cylindrical_crank_nicolson_matrix(p):
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
        format="lil",
    )


def simulate_single_epsp_trace(p, input_x, input_t, t_max, saved_frames):
    events = np.array([[input_t], [input_x]])
    V0 = np.zeros(len(p.x))
    A = _build_conical_crank_nicolson_matrix(p)
    times, V_s = crank_nicolson(
        t_span=(0, t_max),
        V0=V0,
        A=A,
        saved_frames=saved_frames,
        p=p,
        events=events,
        verbose=False,
    )
    return times, V_s, events


def simulate_single_cylindrical_epsp_trace(p, input_x, input_t, t_max, saved_frames):
    events = np.array([[input_t], [input_x]])
    V0 = np.zeros(len(p.x))
    A = _build_cylindrical_crank_nicolson_matrix(p)
    times, V_s = crank_nicolson(
        t_span=(0, t_max),
        V0=V0,
        A=A,
        saved_frames=saved_frames,
        p=p,
        events=events,
        verbose=False,
    )
    return times, V_s, events


def run_single_input_local_epsp_summary(
        output_root=Path("saved_simulations") / "one_spike_position_epsp",
        L=to_SI(500 * um),
        N=101,
        dt=to_SI(1e-8 * second),
        t_max=to_SI(0.3 * ms),
        input_t=to_SI(0.1 * ms),
        input_positions_um=(50, 100, 150, 250, 350, 450),
        I_e=to_SI(15 * pampere),
        saved_frames=3 * 10 ** 4,
        output_stem="single_input_local_epsp_summary"):
    """
    Run one excitatory input at several cone positions and summarize the local EPSP.

    The key readout is the peak EPSP at the input node compared with fixed
    proximal and distal readout sites. This makes the local amplification visible
    in one reproducible figure instead of several independent simulation plots.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    p = default_params.with_SI_properties(
        dt=dt,
        N=N,
        L=L,
        I_e=I_e,
    ).to_numerical()

    x_um = p.x * meter / um
    times_by_position = {}
    voltages_by_position = {}
    summary_rows = []

    soma_index = 0
    distal_index = len(p.x) - 1

    for input_position_um in input_positions_um:
        input_x = to_SI(input_position_um * um)
        input_index = int(np.argmin(np.abs(p.x - input_x)))
        times, V_s, events = simulate_single_epsp_trace(
            p=p,
            input_x=input_x,
            input_t=input_t,
            t_max=t_max,
            saved_frames=saved_frames,
        )

        voltage_mV = V_s * 1000.0
        local_peak_mV = float(np.max(voltage_mV[:, input_index]))
        soma_peak_mV = float(np.max(voltage_mV[:, soma_index]))
        distal_peak_mV = float(np.max(voltage_mV[:, distal_index]))
        global_peak_mV = float(np.max(voltage_mV))
        global_peak_index = int(np.argmax(np.max(voltage_mV, axis=0)))
        strongest_remote_mV = max(soma_peak_mV, distal_peak_mV)

        times_by_position[str(input_position_um)] = times
        voltages_by_position[str(input_position_um)] = voltage_mV
        summary_rows.append({
            "input_position_um": float(input_position_um),
            "nearest_grid_position_um": float(x_um[input_index]),
            "local_peak_mV": local_peak_mV,
            "soma_peak_mV": soma_peak_mV,
            "distal_peak_mV": distal_peak_mV,
            "global_peak_mV": global_peak_mV,
            "global_peak_position_um": float(x_um[global_peak_index]),
            "local_over_strongest_endpoint": (
                local_peak_mV / strongest_remote_mV
                if strongest_remote_mV != 0.0 else np.inf
            ),
        })

    csv_path = output_root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    npz_path = output_root / f"{output_stem}.npz"
    np.savez_compressed(
        npz_path,
        x_um=x_um,
        input_positions_um=np.asarray(input_positions_um, dtype=float),
        summary=np.asarray([[row[key] for key in summary_rows[0]] for row in summary_rows], dtype=float),
        summary_columns=np.asarray(list(summary_rows[0].keys())),
        **{f"times_{key}um": value for key, value in times_by_position.items()},
        **{f"voltage_mV_{key}um": value for key, value in voltages_by_position.items()},
    )

    figure_path = output_root / f"{output_stem}.png"
    _plot_single_input_local_epsp_summary(
        figure_path=figure_path,
        p=p,
        x_um=x_um,
        summary_rows=summary_rows,
        times_by_position=times_by_position,
        voltages_by_position=voltages_by_position,
        geometry_label="conical",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {figure_path}")
    return summary_rows


def run_single_input_local_epsp_cylinder_summary(
        output_root=Path("saved_simulations") / "one_spike_position_epsp_cylinder",
        L=to_SI(500 * um),
        N=101,
        dt=to_SI(1e-8 * second),
        t_max=to_SI(0.3 * ms),
        input_t=to_SI(0.1 * ms),
        input_positions_um=(50, 100, 150, 250, 350, 450),
        I_e=to_SI(15 * pampere),
        saved_frames=3 * 10 ** 4,
        output_stem="single_input_local_epsp_summary"):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    conical_parameters = default_params.with_SI_properties(
        dt=dt,
        N=N,
        L=L,
        I_e=I_e,
    )
    p = conical_parameters.to_numerical().to_numerical_cylindrical_params(x0=0.0)

    x_um = p.x * meter / um
    times_by_position = {}
    voltages_by_position = {}
    summary_rows = []

    soma_index = 0
    distal_index = len(p.x) - 1

    for input_position_um in input_positions_um:
        input_x = to_SI(input_position_um * um)
        input_index = int(np.argmin(np.abs(p.x - input_x)))
        times, V_s, events = simulate_single_cylindrical_epsp_trace(
            p=p,
            input_x=input_x,
            input_t=input_t,
            t_max=t_max,
            saved_frames=saved_frames,
        )

        voltage_mV = V_s * 1000.0
        local_peak_mV = float(np.max(voltage_mV[:, input_index]))
        soma_peak_mV = float(np.max(voltage_mV[:, soma_index]))
        distal_peak_mV = float(np.max(voltage_mV[:, distal_index]))
        global_peak_mV = float(np.max(voltage_mV))
        global_peak_index = int(np.argmax(np.max(voltage_mV, axis=0)))
        strongest_remote_mV = max(soma_peak_mV, distal_peak_mV)

        times_by_position[str(input_position_um)] = times
        voltages_by_position[str(input_position_um)] = voltage_mV
        summary_rows.append({
            "input_position_um": float(input_position_um),
            "nearest_grid_position_um": float(x_um[input_index]),
            "local_peak_mV": local_peak_mV,
            "soma_peak_mV": soma_peak_mV,
            "distal_peak_mV": distal_peak_mV,
            "global_peak_mV": global_peak_mV,
            "global_peak_position_um": float(x_um[global_peak_index]),
            "local_over_strongest_endpoint": (
                local_peak_mV / strongest_remote_mV
                if strongest_remote_mV != 0.0 else np.inf
            ),
        })

    csv_path = output_root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    npz_path = output_root / f"{output_stem}.npz"
    np.savez_compressed(
        npz_path,
        x_um=x_um,
        input_positions_um=np.asarray(input_positions_um, dtype=float),
        summary=np.asarray([[row[key] for key in summary_rows[0]] for row in summary_rows], dtype=float),
        summary_columns=np.asarray(list(summary_rows[0].keys())),
        **{f"times_{key}um": value for key, value in times_by_position.items()},
        **{f"voltage_mV_{key}um": value for key, value in voltages_by_position.items()},
    )

    figure_path = output_root / f"{output_stem}.png"
    _plot_single_input_local_epsp_summary(
        figure_path=figure_path,
        p=p,
        x_um=x_um,
        summary_rows=summary_rows,
        times_by_position=times_by_position,
        voltages_by_position=voltages_by_position,
        geometry_label="cylindrical",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {figure_path}")
    return summary_rows


def run_single_input_local_epsp_2000um_summary():
    return run_single_input_local_epsp_summary(
        output_root=Path("saved_simulations") / "one_spike_position_epsp_2000um_conical",
        L=to_SI(2000 * um),
        N=201,
        input_positions_um=(200, 400, 600, 1000, 1400, 1800, 1900, 1950),
        output_stem="single_input_local_epsp_conical_2000um",
    )


def run_single_input_local_epsp_cylinder_2000um_summary():
    return run_single_input_local_epsp_cylinder_summary(
        output_root=Path("saved_simulations") / "one_spike_position_epsp_2000um_cylinder",
        L=to_SI(2000 * um),
        N=201,
        input_positions_um=(200, 400, 600, 1000, 1400, 1800, 1900, 1950),
        output_stem="single_input_local_epsp_cylinder_2000um",
    )


def _plot_single_input_local_epsp_summary(
        figure_path, p, x_um, summary_rows, times_by_position, voltages_by_position,
        geometry_label):
    positions = [row["input_position_um"] for row in summary_rows]
    local_peaks = [row["local_peak_mV"] for row in summary_rows]
    soma_peaks = [row["soma_peak_mV"] for row in summary_rows]
    distal_peaks = [row["distal_peak_mV"] for row in summary_rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)

    trace_axis = axes[0]
    for row in summary_rows:
        key = str(int(row["input_position_um"]) if row["input_position_um"].is_integer() else row["input_position_um"])
        times_ms = times_by_position[key] * second / ms
        voltage_mV = voltages_by_position[key]
        local_index = int(np.argmin(np.abs(x_um - row["nearest_grid_position_um"])))
        trace_axis.plot(
            times_ms,
            voltage_mV[:, local_index],
            label=f"{row['nearest_grid_position_um']:.0f} um",
        )
    trace_axis.set_title(f"Local EPSP trace for one input at different {geometry_label} positions")
    trace_axis.set_xlabel("t [ms]")
    trace_axis.set_ylabel("V at input [mV]")
    trace_axis.legend(title="input x", ncols=3)

    peak_axis = axes[1]
    peak_axis.plot(positions, local_peaks, "o-", label="local at input")
    peak_axis.plot(positions, soma_peaks, "s--", label="proximal endpoint")
    peak_axis.plot(positions, distal_peaks, "^--", label="distal endpoint")
    peak_axis.set_title("Peak EPSP depends on where the same input lands")
    peak_axis.set_xlabel("input position [um]")
    peak_axis.set_ylabel("peak V [mV]")
    peak_axis.legend()

    geometry_axis = axes[2]
    geometry_axis.plot(positions, local_peaks, "o-", color="tab:red", label="local peak at input")
    geometry_axis.set_title("Non-normalized local EPSP peak")
    geometry_axis.set_xlabel("x [um]")
    geometry_axis.set_ylabel("local peak V [mV]")
    geometry_axis.legend(loc="upper right")

    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


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
        I_i=to_SI(30 * pampere),
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


def test_sim_one_example():
    L = to_SI(2000 * um)

    p = default_params.with_SI_properties(
        dt=to_SI(1e-8 * second),
        N=201,
        L=L,
        I_e=to_SI(15 * pampere),
        I_i=to_SI(30 * pampere),
    ).to_numerical()

    return simulate_balanced_input_with_uniform(p=p, t_max=to_SI(1 * second), e_limits=(0, L), i_limits=(0.0, L))

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


def _um_label(value):
    return int(round(value / to_SI(1 * um)))


DEFAULT_LIMIT_SETS = object()


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
            i_limits=(c, d),
        )
        for a, b, c, d in itertools.product(_fractional_limit_sets(L), _fractional_limit_sets(L))
    )


def test_run_balanced_limit_sweep_1000um(
        L = to_SI(1000 * um),
        t_max=to_SI(2 * second),
        limit_sets=DEFAULT_LIMIT_SETS,
        output_root=None,
        p=None,
        dt=to_SI(1e-8 * second),
        N=101,
        I_e=to_SI(10 * pampere),
        I_i=to_SI(40 * pampere),
        r_e_density=0.4 * Hz / um,
        r_i_density=0.1 * Hz / um,
        g=None,
        saved_frames=3 * 10 ** 4,
        verbose=True,
        plot=True,
        show_plot=True,
        max_workers=12):
    if p is None:
        p = default_params.with_SI_properties(
            dt=dt,
            N=N,
            L=L,
            I_e=I_e,
            I_i=I_i,
        ).to_numerical()

    if limit_sets is DEFAULT_LIMIT_SETS:
        limit_sets = _fractional_limit_sets(L)
    if output_root is None:
        output_root = Path("saved_simulations") / f"{int(L * meter / um)}um"
    else:
        output_root = Path(output_root)
    if g is None:
        g = float(to_SI(I_i) / to_SI(I_e))

    comparison_kwargs = [
        {
            "conical_p": p,
            "t_max": t_max,
            "output_root": output_root,
            "simulation_label": "exponential_uniform",
            "e_limits": (a, b),
            "i_limits": (c, d),
            "r_e_density": r_e_density,
            "r_i_density": r_i_density,
            "g": g,
            "saved_frames": saved_frames,
            "verbose": verbose,
            "plot": plot,
            "show_plot": show_plot,
        }
        for (a, b), (c, d) in itertools.product(limit_sets, limit_sets)
    ]

    return run_balanced_comparison_sweep(
        comparison_kwargs,
        max_workers=max_workers,
    )

def test_run_balanced_limit_sweep_2000um():
    test_run_balanced_limit_sweep_1000um(L=to_SI(2000 * um), t_max=to_SI(2 * second))

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
    "single-input-local-epsp": run_single_input_local_epsp_summary,
    "single-input-local-epsp-cylinder": run_single_input_local_epsp_cylinder_summary,
    "single-input-local-epsp-2000um": run_single_input_local_epsp_2000um_summary,
    "single-input-local-epsp-cylinder-2000um": run_single_input_local_epsp_cylinder_2000um_summary,
    "one-spike-sweep": run_one_spike_position_sweep,
    "uniform-excitatory": run_uniform_excitatory_spike_train,
    "balanced-static": run_balanced_static_example,
    "balanced-500um": run_balanced_500um_example,
    "balanced-2000um": test_run_balanced_2000um_example,
    "balanced-sweep-500um": run_balanced_limit_sweep_500um,
    "balanced-sweep-1000um": test_run_balanced_limit_sweep_1000um,
    "balanced-sweep-1500um": test_run_balanced_limit_sweep_1500um,
    "sim-one-example": test_sim_one_example,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    args = parser.parse_args()
    SCENARIOS[args.scenario]()


if __name__ == "__main__":
    main()

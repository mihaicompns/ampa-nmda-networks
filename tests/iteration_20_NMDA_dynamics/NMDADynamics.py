import unittest
from dataclasses import dataclass, field, replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from brian2 import kHz, mmole, ms, mV, nA, nF, nS
from joblib import Parallel, cpu_count, delayed
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import FormatStrFormatter
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from Plotting import show_plots_non_blocking


@dataclass(frozen=True)
class NMDADynamicsParameters:
    # Values follow the Brian2 Wang_2002 example for excitatory neurons.
    dt: object = field(default_factory=lambda: 0.01 * ms)
    t_stop: object = field(default_factory=lambda: 15.0 * ms)
    capacitance: object = field(default_factory=lambda: 0.5 * nF)
    g_leak: object = field(default_factory=lambda:1 * nS)
    e_leak: object = field(default_factory=lambda: -70.0 * mV)
    e_excitatory: object = field(default_factory=lambda: 0.0 * mV)
    g_ampa: object = field(default_factory=lambda: 0.05 * nS)
    g_nmda: object = field(default_factory=lambda: 0.165 * nS)
    magnesium: object = field(default_factory=lambda: 1.0 * mmole)
    tau_ampa: object = field(default_factory=lambda: 2.0 * ms)
    tau_nmda_rise: object = field(default_factory=lambda: 2.0 * ms)
    tau_nmda_decay: object = field(default_factory=lambda: 100.0 * ms)
    alpha: object = field(default_factory=lambda: 0.5 * kHz)
    initial_v: object = field(default_factory=lambda: -70.0 * mV)


def nmda_voltage_block(v, magnesium=1.0 * mmole):
    return 1.0 / (1.0 + 0.25 * np.exp(-0.075 * v / mV))


def simulate_nmda_only_forward_euler(
        params=NMDADynamicsParameters(),
        compartment_matrix=((1.0,),),
        nmda_current_weights=None,
        ampa_current_weights=None,
        spike_train=(5.0 * ms,),
        presynaptic_spike_weight=1.0,
        include_ampa=False,
):
    compartment_matrix = np.asarray(compartment_matrix, dtype=float)
    if compartment_matrix.ndim != 2 or 0 in compartment_matrix.shape:
        raise ValueError("compartment_matrix must have shape K x N_presynaptic")

    n_compartments, n_presynaptic = compartment_matrix.shape
    if nmda_current_weights is None:
        nmda_current_weights = np.ones(n_compartments)
    nmda_current_weights = np.asarray(nmda_current_weights, dtype=float)
    if nmda_current_weights.shape != (n_compartments,):
        raise ValueError("nmda_current_weights must have one value per compartment")
    if ampa_current_weights is None:
        ampa_current_weights = np.ones(n_presynaptic)
    ampa_current_weights = np.asarray(ampa_current_weights, dtype=float)
    if ampa_current_weights.shape != (n_presynaptic,):
        raise ValueError("ampa_current_weights must have one value per presynaptic neuron")

    n_steps = int(round(float(params.t_stop / params.dt))) + 1
    times = np.arange(n_steps) * params.dt
    x_ampa = np.zeros((n_steps, n_presynaptic))
    x = np.zeros((n_steps, n_compartments))
    s = np.zeros((n_steps, n_compartments))
    v = np.zeros(n_steps) * mV
    i_ampa = np.zeros(n_steps) * nA
    i_nmda = np.zeros(n_steps) * nA
    v[0] = params.initial_v

    presynaptic_input_by_step = _presynaptic_input_by_step(
        spike_train=spike_train,
        dt=params.dt,
        n_steps=n_steps,
        n_presynaptic=n_presynaptic,
        presynaptic_spike_weight=presynaptic_spike_weight,
    )

    for step in range(n_steps - 1):
        x_ampa_current = x_ampa[step].copy()
        if include_ampa:
            x_ampa_current += presynaptic_input_by_step[step]

        x_current = x[step].copy()
        x_current += compartment_matrix @ presynaptic_input_by_step[step]

        s_current = s[step]
        weighted_ampa = np.dot(ampa_current_weights, x_ampa_current)
        weighted_s = np.dot(nmda_current_weights, s_current)
        block = nmda_voltage_block(v[step], params.magnesium)
        i_ampa[step] = params.g_ampa * (v[step] - params.e_excitatory) * weighted_ampa
        i_nmda[step] = params.g_nmda * block * (v[step] - params.e_excitatory) * weighted_s

        dx_ampa = params.dt * (-x_ampa_current / params.tau_ampa)
        dx = params.dt * (-x_current / params.tau_nmda_rise)
        ds = params.dt * (
                -s_current / params.tau_nmda_decay
                + params.alpha * x_current * (1.0 - s_current)
        )
        dv = params.dt * (
                (
                        -params.g_leak * (v[step] - params.e_leak)
                        - i_ampa[step]
                        - i_nmda[step]
                ) / params.capacitance
        )

        x_ampa[step + 1] = x_ampa_current + np.asarray(dx_ampa, dtype=float)
        x[step + 1] = x_current + np.asarray(dx, dtype=float)
        s[step + 1] = s_current + np.asarray(ds, dtype=float)
        v[step + 1] = v[step] + dv

    final_step = n_steps - 1
    final_block = nmda_voltage_block(v[final_step], params.magnesium)
    i_ampa[final_step] = (
            params.g_ampa
            * (v[final_step] - params.e_excitatory)
            * np.dot(ampa_current_weights, x_ampa[final_step])
    )
    i_nmda[final_step] = (
            params.g_nmda
            * final_block
            * (v[final_step] - params.e_excitatory)
            * np.dot(nmda_current_weights, s[final_step])
    )

    return {
        "t_ms": times / ms,
        "presynaptic_input": presynaptic_input_by_step,
        "x_AMPA": x_ampa,
        "x": x,
        "s": s,
        "V_mV": v / mV,
        "I_AMPA_nA": i_ampa / nA,
        "I_NMDA_nA": i_nmda / nA,
    }


def _presynaptic_input_by_step(
        spike_train,
        dt,
        n_steps,
        n_presynaptic,
        presynaptic_spike_weight,
):
    normalized_spike_train = _normalize_spike_train(spike_train, n_presynaptic)
    presynaptic_input_by_step = np.zeros((n_steps, n_presynaptic))
    for presynaptic_index, spike_times in enumerate(normalized_spike_train):
        for spike_time in spike_times:
            step = int(round(float(spike_time / dt)))
            time_grid_error_ms = float((step * dt - spike_time) / ms)
            if not np.isclose(time_grid_error_ms, 0.0):
                raise ValueError("spike times must fall on the Euler time grid")
            if step < 0 or step >= n_steps:
                raise ValueError(f"spike time {spike_time} is outside the simulation interval")
            presynaptic_input_by_step[step, presynaptic_index] += presynaptic_spike_weight
    return presynaptic_input_by_step


def _normalize_spike_train(spike_train, n_presynaptic):
    if _is_flat_spike_time_list(spike_train):
        if n_presynaptic != 1:
            raise ValueError("a flat spike train can only drive one presynaptic neuron")
        return [spike_train]

    if len(spike_train) != n_presynaptic:
        raise ValueError("nested spike_train must have one spike-time list per presynaptic neuron")
    return spike_train


def _is_flat_spike_time_list(spike_train):
    return all(np.isscalar(spike_time) or hasattr(spike_time, "dim") for spike_time in spike_train)


def simulate_single_spike_at_5_ms(include_ampa=False, g_ampa=None):
    params = NMDADynamicsParameters(t_stop=100.0 * ms)
    if g_ampa is not None:
        params = replace(params, g_ampa=g_ampa)
    return simulate_nmda_only_forward_euler(
        params=params,
        spike_train=(5.0 * ms,),
        include_ampa=include_ampa,
    )


def simulate_two_spikes_at_5_and_25_ms(g_nmda=6.0 * nS, g_ampa=None, include_ampa=False, params_template=None):
    if params_template is None:
        params_template = NMDADynamicsParameters(t_stop=100.0 * ms)
    params = replace(params_template, g_nmda=g_nmda)
    if g_ampa is not None:
        params = replace(params, g_ampa=g_ampa)
    return simulate_nmda_only_forward_euler(
        params=params,
        spike_train=(5.0 * ms, 25.0 * ms),
        include_ampa=include_ampa,
    )


def simulate_three_spikes_at_5_25_and_45_ms(g_nmda=6.0 * nS, g_ampa=None, include_ampa=False, params_template=None):
    if params_template is None:
        params_template = NMDADynamicsParameters(t_stop=150.0 * ms)
    params = replace(params_template, g_nmda=g_nmda)
    if g_ampa is not None:
        params = replace(params, g_ampa=g_ampa)
    return simulate_nmda_only_forward_euler(
        params=params,
        spike_train=(5.0 * ms, 25.0 * ms, 45.0 * ms),
        include_ampa=include_ampa,
    )


def simulate_spike_train_with_two_compartments():
    params = NMDADynamicsParameters(t_stop=120.0 * ms)
    spike_train = (
        (5.0 * ms, 40.0 * ms),
        (15.0 * ms, 25.0 * ms, 60.0 * ms),
    )
    compartment_matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    return simulate_nmda_only_forward_euler(
        params=params,
        compartment_matrix=compartment_matrix,
        nmda_current_weights=(1.0, 0.5),
        spike_train=spike_train,
    )


def scan_g_nmda_max_vs_delta_v_max(
        g_nmda_values=None,
        params_template=None,
        spike_train=(5.0 * ms,),
        include_ampa=False,
        g_ampa=None,
        n_jobs=None,
):
    if g_nmda_values is None:
        g_nmda_values = np.linspace(0, 100, 40) * nS
    if params_template is None:
        params_template = NMDADynamicsParameters(t_stop=100.0 * ms)

    results = Parallel(n_jobs=_safe_n_jobs(n_jobs))(
        delayed(_simulate_delta_v_max_for_g_nmda)(g_nmda, params_template, spike_train, include_ampa, g_ampa)
        for g_nmda in g_nmda_values
    )
    g_nmda_nS, delta_v_max_mV = zip(*results)
    return {
        "g_nmda_nS": np.array(g_nmda_nS),
        "delta_v_max_mV": np.array(delta_v_max_mV),
    }


def _safe_n_jobs(n_jobs):
    if n_jobs is not None:
        return n_jobs
    return min(4, cpu_count())


def _simulate_delta_v_max_for_g_nmda(g_nmda, params_template, spike_train, include_ampa, g_ampa):
    params = replace(params_template, g_nmda=g_nmda)
    if g_ampa is not None:
        params = replace(params, g_ampa=g_ampa)
    result = simulate_nmda_only_forward_euler(
        params=params,
        spike_train=spike_train,
        include_ampa=include_ampa,
    )
    delta_v_max = np.max(result["V_mV"] - result["V_mV"][0])
    return float(g_nmda / nS), float(delta_v_max)


def scan_g_nmda_max_vs_delta_v_max_for_two_spikes(
        g_nmda_values=None,
        params_template=None,
        include_ampa=False,
        g_ampa=None,
        n_jobs=None,
):
    return scan_g_nmda_max_vs_delta_v_max(
        g_nmda_values=g_nmda_values,
        params_template=params_template,
        spike_train=(5.0 * ms, 25.0 * ms),
        include_ampa=include_ampa,
        g_ampa=g_ampa,
        n_jobs=n_jobs,
    )


def scan_g_nmda_max_vs_delta_v_max_for_three_spikes(
        g_nmda_values=None,
        params_template=None,
        include_ampa=False,
        g_ampa=None,
        n_jobs=None,
):
    return scan_g_nmda_max_vs_delta_v_max(
        g_nmda_values=g_nmda_values,
        params_template=params_template,
        spike_train=(5.0 * ms, 25.0 * ms, 45.0 * ms),
        include_ampa=include_ampa,
        g_ampa=g_ampa,
        n_jobs=n_jobs,
    )


def scan_g_nmda_max_vs_delta_v_max_for_one_spike(
        g_nmda_values=None,
        params_template=None,
        include_ampa=False,
        g_ampa=None,
        n_jobs=None,
):
    return scan_g_nmda_max_vs_delta_v_max(
        g_nmda_values=g_nmda_values,
        params_template=params_template,
        spike_train=(5.0 * ms,),
        include_ampa=include_ampa,
        g_ampa=g_ampa,
        n_jobs=n_jobs,
    )

def plot_g_nmda_max_vs_delta_v_max(scan_result, title="NMDA conductance scan"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scan_result["g_nmda_nS"], scan_result["delta_v_max_mV"], marker="o")
    ax.set_title(title)
    ax.set_xlabel("g_NMDA max [nS]")
    ax.set_ylabel(r"max $\Delta$ V [mV]")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fig.tight_layout()
    return fig, ax


def _g_leak_label(params):
    return rf"$g_L$ = {float(params.g_leak / nS):.3g} nS"


def plot_nmda_dynamics(result):
    t_ms = result["t_ms"]
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(8, 10))
    fig.suptitle("AMPA + NMDA response to presynaptic spike train")
    axes[0].plot(t_ms, result["x_AMPA"])
    axes[0].set_ylabel("x_AMPA")
    axes[1].plot(t_ms, result["x"])
    axes[1].set_ylabel("x_NMDA")
    axes[2].plot(t_ms, result["s"])
    axes[2].set_ylabel("s_NMDA")
    axes[3].plot(t_ms, result["I_AMPA_nA"])
    axes[3].set_ylabel("I_AMPA [nA]")
    axes[4].plot(t_ms, result["I_NMDA_nA"])
    axes[4].set_ylabel("I_NMDA [nA]")
    axes[5].plot(t_ms, result["V_mV"])
    axes[5].set_ylabel("V [mV]")
    axes[5].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[5].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[5].set_xlabel("t [ms]")
    fig.tight_layout()
    return fig, axes


def simulate_voltage_nullcline_for_single_spike_state(
        g_nmda=250.0 * nS,
        g_ampa=0.05 * nS,
        selected_t=7.0 * ms,
        second_spike_delta=20.0 * ms,
        third_spike_delta=40.0 * ms,
        include_ampa=True,
        params_template=None,
):
    if params_template is None:
        params_template = NMDADynamicsParameters(t_stop=100.0 * ms)
    params = replace(params_template, g_nmda=g_nmda, g_ampa=g_ampa)
    simulation = simulate_nmda_only_forward_euler(
        params=params,
        spike_train=(5.0 * ms,),
        include_ampa=include_ampa,
    )
    two_spike_simulation = simulate_nmda_only_forward_euler(
        params=params,
        spike_train=(5.0 * ms, 5.0 * ms + second_spike_delta),
        include_ampa=include_ampa,
    )
    three_spike_simulation = simulate_nmda_only_forward_euler(
        params=params,
        spike_train=(5.0 * ms, 5.0 * ms + second_spike_delta, 5.0 * ms + third_spike_delta),
        include_ampa=include_ampa,
    )
    s_at_selected_t = _s_value_at_time(simulation, selected_t)
    ampa_at_selected_t = _ampa_value_at_time(simulation, selected_t)
    v_values_mV = np.linspace(-90.0, 20.0, 500)
    f_nA = compute_voltage_nullcline_function(
        v_values_mV=v_values_mV,
        g_nmda=g_nmda,
        s_value=s_at_selected_t,
        g_ampa=g_ampa,
        ampa_value=ampa_at_selected_t,
        params=params,
    )

    first_spike_max_v_mV = float(np.max(simulation["V_mV"]))
    second_spike_max_v_mV = float(np.max(two_spike_simulation["V_mV"]))
    delta_v_after_first_spike_mV = first_spike_max_v_mV - float(params.initial_v / mV)
    delta_v_after_second_spike_mV = second_spike_max_v_mV - first_spike_max_v_mV

    return {
        "simulation": simulation,
        "two_spike_simulation": two_spike_simulation,
        "three_spike_simulation": three_spike_simulation,
        "s_at_selected_t": s_at_selected_t,
        "ampa_at_selected_t": ampa_at_selected_t,
        "v_values_mV": v_values_mV,
        "f_nA": f_nA,
        "g_nmda_nS": float(g_nmda / nS),
        "g_ampa_nS": float(g_ampa / nS),
        "g_leak_nS": float(params.g_leak / nS),
        "selected_t_ms": float(selected_t / ms),
        "second_spike_delta_ms": float(second_spike_delta / ms),
        "third_spike_delta_ms": float(third_spike_delta / ms),
        "include_ampa": bool(include_ampa),
        "initial_v_mV": float(params.initial_v / mV),
        "first_spike_max_v_mV": first_spike_max_v_mV,
        "second_spike_max_v_mV": second_spike_max_v_mV,
        "delta_v_after_first_spike_mV": float(delta_v_after_first_spike_mV),
        "delta_v_after_second_spike_mV": float(delta_v_after_second_spike_mV),
    }


def plot_voltage_nullcline_from_single_spike_state_data(plot_data):
    simulation = plot_data["simulation"]
    two_spike_simulation = plot_data["two_spike_simulation"]
    three_spike_simulation = plot_data["three_spike_simulation"]
    s_at_selected_t = plot_data["s_at_selected_t"]
    ampa_at_selected_t = plot_data["ampa_at_selected_t"]
    v_values_mV = plot_data["v_values_mV"]
    f_nA = plot_data["f_nA"]
    g_nmda_nS = plot_data["g_nmda_nS"]
    g_leak_nS = plot_data["g_leak_nS"]
    selected_t_ms = plot_data["selected_t_ms"]
    second_spike_delta_ms = plot_data["second_spike_delta_ms"]
    third_spike_delta_ms = plot_data["third_spike_delta_ms"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    nullcline_axis = axes[0, 0]
    voltage_axis = axes[0, 1]
    conductance_axis = axes[1, 0]
    empty_axis = axes[1, 1]
    fig.suptitle(
        rf"$g_{{NMDA}}$ = {g_nmda_nS:.1f} nS, $g_L$ = {g_leak_nS:.3g} nS"
    )
    nullcline_axis.plot(v_values_mV, f_nA)
    nullcline_axis.axhline(0.0, color="black", linewidth=1.0)
    nullcline_axis.set_title(
        f"f(v), t = {selected_t_ms:.1f} ms, s = {s_at_selected_t:.3f}, AMPA = {ampa_at_selected_t:.3f}"
    )
    nullcline_axis.set_xlabel("v [mV]")
    nullcline_axis.set_ylabel("f(v) [nA]")
    nullcline_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    nullcline_axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    voltage_axis.plot(simulation["t_ms"], simulation["V_mV"], label="one spike")
    voltage_axis.plot(
        two_spike_simulation["t_ms"],
        two_spike_simulation["V_mV"],
        color="tab:red",
        linestyle="--",
        label=f"second spike after {second_spike_delta_ms:.1f} ms",
    )
    voltage_axis.plot(
        three_spike_simulation["t_ms"],
        three_spike_simulation["V_mV"],
        color="tab:green",
        linestyle=":",
        label=f"third spike after {third_spike_delta_ms:.1f} ms",
    )
    voltage_axis.axvline(selected_t_ms, color="black", linewidth=1.0, alpha=0.35)
    _annotate_voltage_deltas(voltage_axis, plot_data)
    voltage_axis.set_title(f"V(t), g_NMDA = {g_nmda_nS:.1f} nS")
    voltage_axis.set_xlabel("t [ms]")
    voltage_axis.set_ylabel("V [mV]")
    voltage_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    voltage_axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    voltage_axis.legend()

    conductance_axis.plot(
        simulation["t_ms"],
        g_nmda_nS * simulation["s"][:, 0],
        label="one spike",
    )
    conductance_axis.plot(
        two_spike_simulation["t_ms"],
        g_nmda_nS * two_spike_simulation["s"][:, 0],
        color="tab:red",
        linestyle="--",
        label=f"second spike after {second_spike_delta_ms:.1f} ms",
    )
    conductance_axis.plot(
        three_spike_simulation["t_ms"],
        g_nmda_nS * three_spike_simulation["s"][:, 0],
        color="tab:green",
        linestyle=":",
        label=f"third spike after {third_spike_delta_ms:.1f} ms",
    )
    conductance_axis.axvline(selected_t_ms, color="black", linewidth=1.0, alpha=0.35)
    conductance_axis.set_title(r"$g_{NMDA} max \cdot s(t)$")
    conductance_axis.set_xlabel("t [ms]")
    conductance_axis.set_ylabel(r"$g_{NMDA} max \cdot s$ [nS]")
    conductance_axis.set_ylim(bottom=0.0)
    conductance_axis.legend()

    empty_axis.axis("off")
    fig.tight_layout()
    return fig, axes, plot_data


def _annotate_voltage_deltas(ax, plot_data):
    first_peak_index = int(np.argmax(plot_data["simulation"]["V_mV"]))
    second_peak_index = int(np.argmax(plot_data["two_spike_simulation"]["V_mV"]))
    first_peak_t_ms = float(plot_data["simulation"]["t_ms"][first_peak_index])
    second_peak_t_ms = float(plot_data["two_spike_simulation"]["t_ms"][second_peak_index])
    first_peak_v_mV = float(plot_data["first_spike_max_v_mV"])
    second_peak_v_mV = float(plot_data["second_spike_max_v_mV"])
    baseline_v_mV = float(plot_data["initial_v_mV"])

    ax.annotate(
        rf"$\Delta V_1$ = {plot_data['delta_v_after_first_spike_mV']:.3f} mV",
        xy=(first_peak_t_ms, first_peak_v_mV),
        xytext=(first_peak_t_ms + 0.08 * (ax.get_xlim()[1] - ax.get_xlim()[0]), first_peak_v_mV),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "tab:blue"},
        color="tab:blue",
        fontsize=9,
    )
    ax.annotate(
        "",
        xy=(first_peak_t_ms, first_peak_v_mV),
        xytext=(first_peak_t_ms, baseline_v_mV),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.0, "color": "tab:blue"},
    )
    ax.annotate(
        rf"$\Delta V_2$ = {plot_data['delta_v_after_second_spike_mV']:.3f} mV",
        xy=(second_peak_t_ms, second_peak_v_mV),
        xytext=(second_peak_t_ms + 0.04 * (ax.get_xlim()[1] - ax.get_xlim()[0]), second_peak_v_mV),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "tab:red"},
        color="tab:red",
        fontsize=9,
    )
    ax.annotate(
        "",
        xy=(second_peak_t_ms, second_peak_v_mV),
        xytext=(second_peak_t_ms, first_peak_v_mV),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.0, "color": "tab:red"},
    )


def plot_voltage_nullcline_for_single_spike_state(*args, **kwargs):
    plot_data = simulate_voltage_nullcline_for_single_spike_state(*args, **kwargs)
    return plot_voltage_nullcline_from_single_spike_state_data(plot_data)


def _s_value_at_time(simulation, selected_t, compartment_index=0):
    selected_t_ms = float(selected_t / ms)
    index = int(np.argmin(np.abs(simulation["t_ms"] - selected_t_ms)))
    return float(simulation["s"][index, compartment_index])


def _ampa_value_at_time(simulation, selected_t):
    selected_t_ms = float(selected_t / ms)
    index = int(np.argmin(np.abs(simulation["t_ms"] - selected_t_ms)))
    return float(np.dot(np.ones(simulation["x_AMPA"].shape[1]), simulation["x_AMPA"][index]))


def compute_voltage_nullcline_function(
        v_values_mV,
        g_nmda,
        s_value,
        g_ampa=0.0 * nS,
        ampa_value=0.0,
        params=NMDADynamicsParameters(),
):
    v_values_mV = np.asarray(v_values_mV, dtype=float)
    v = v_values_mV * mV
    f = (
            -params.g_leak * (v - params.e_leak)
            - g_ampa * ampa_value * (v - params.e_excitatory)
            - g_nmda * s_value * nmda_voltage_block(v, params.magnesium) * (v - params.e_excitatory)
    )
    return f / nA


def compute_voltage_nullcline(
        g_nmda,
        s_values=None,
        params=NMDADynamicsParameters(),
        v_range_mV=(-90.0, 20.0),
):
    if s_values is None:
        s_values = np.linspace(0.0, 1.0, 400)
    s_values = np.asarray(s_values, dtype=float)
    return _bisect_voltage_nullcline_for_s_values(s_values, g_nmda, params, v_range_mV)


def _find_nullcline_voltage_for_s(s_value, g_nmda, params, v_range_mV):
    return _bisect_voltage_nullcline_for_s_values(
        np.array([s_value], dtype=float),
        g_nmda,
        params,
        v_range_mV,
    )[0]


def _bisect_voltage_nullcline_for_s_values(s_values, g_nmda, params, v_range_mV):
    lower_v_mV = np.full_like(s_values, v_range_mV[0], dtype=float)
    upper_v_mV = np.full_like(s_values, v_range_mV[1], dtype=float)
    lower_residual = _voltage_nullcline_residual(lower_v_mV, s_values, g_nmda, params)
    upper_residual = _voltage_nullcline_residual(upper_v_mV, s_values, g_nmda, params)
    has_root = np.signbit(lower_residual) != np.signbit(upper_residual)

    for _ in range(80):
        middle_v_mV = 0.5 * (lower_v_mV + upper_v_mV)
        middle_residual = _voltage_nullcline_residual(middle_v_mV, s_values, g_nmda, params)
        lower_matches_middle = np.signbit(lower_residual) == np.signbit(middle_residual)
        lower_v_mV = np.where(lower_matches_middle, middle_v_mV, lower_v_mV)
        lower_residual = np.where(lower_matches_middle, middle_residual, lower_residual)
        upper_v_mV = np.where(lower_matches_middle, upper_v_mV, middle_v_mV)

    roots = 0.5 * (lower_v_mV + upper_v_mV)
    exact_lower = np.isclose(lower_residual, 0.0, atol=1e-12)
    exact_upper = np.isclose(upper_residual, 0.0, atol=1e-12)
    roots = np.where(exact_lower, lower_v_mV, roots)
    roots = np.where(exact_upper, upper_v_mV, roots)
    return np.where(has_root | exact_lower | exact_upper, roots, np.nan)


def _voltage_nullcline_residual(v_mV, s_value, g_nmda, params):
    v_mV = np.asarray(v_mV, dtype=float)
    s_value = np.asarray(s_value, dtype=float)
    sigma_v = 1.0 / (1.0 + 0.25 * np.exp(-0.075 * v_mV))
    g_leak_nS = float(params.g_leak / nS)
    g_nmda_nS = float(g_nmda / nS)
    e_leak_mV = float(params.e_leak / mV)
    e_excitatory_mV = float(params.e_excitatory / mV)
    v_star_rhs = (
            g_leak_nS * e_leak_mV
            + g_nmda_nS * s_value * sigma_v * e_excitatory_mV
    ) / (
            g_leak_nS
            + g_nmda_nS * s_value * sigma_v
    )
    residual = v_star_rhs - v_mV
    if residual.ndim == 0:
        return float(residual)
    return residual


class NMDANullclineExplorer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NMDA nullcline explorer")
        self.params = NMDADynamicsParameters(t_stop=100.0 * ms)
        self.v_values_mV = np.linspace(-90.0, 20.0, 500)
        self.single_spike = None
        self.single_spike_g_nmda_nS = None
        self.two_spikes = None
        self.two_spike_g_nmda_nS = None
        self.two_spike_delta_ms = None

        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvas(self.figure)
        self.g_slider, self.g_label = self._make_slider(1, 4000, 60)
        self.t_slider, self.t_label = self._make_slider(0, 1000, 70)
        self.second_spike_delta_slider, self.second_spike_delta_label = self._make_slider(1, 190, 40)

        sliders = QGridLayout()
        sliders.addWidget(self.g_label, 0, 0)
        sliders.addWidget(self.g_slider, 0, 1)
        sliders.addWidget(self.t_label, 1, 0)
        sliders.addWidget(self.t_slider, 1, 1)
        sliders.addWidget(self.second_spike_delta_label, 2, 0)
        sliders.addWidget(self.second_spike_delta_slider, 2, 1)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addLayout(sliders)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        for slider in [self.g_slider, self.t_slider, self.second_spike_delta_slider]:
            slider.valueChanged.connect(self.update_plot)

        self.update_plot()

    def _make_slider(self, minimum, maximum, value):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        slider.setTracking(True)
        return slider, QLabel()

    def update_plot(self):
        g_nmda = self.g_slider.value() / 10.0 * nS
        selected_t = self.t_slider.value() / 10.0 * ms
        second_spike_delta = self.second_spike_delta_slider.value() / 2.0 * ms

        self.g_label.setText(f"g_NMDA max: {g_nmda / nS:.1f} nS")
        self.t_label.setText(f"t: {selected_t / ms:.1f} ms")
        self.second_spike_delta_label.setText(f"second spike delay: {second_spike_delta / ms:.1f} ms")

        single_spike = self._single_spike_simulation(g_nmda)
        two_spikes = self._two_spike_simulation(second_spike_delta)

        single_s = self._s_at_time(single_spike, selected_t)
        single_ampa = _ampa_value_at_time(single_spike, selected_t)
        f_nA = compute_voltage_nullcline_function(
            self.v_values_mV,
            g_nmda,
            single_s,
            g_ampa=self.params.g_ampa,
            ampa_value=single_ampa,
            params=self.params,
        )

        self._plot_nullcline_axis(
            self.axes[0],
            single_s,
            single_ampa,
            f_nA,
            selected_t,
        )
        self._plot_time_course_axis(
            self.axes[1],
            single_spike,
            two_spikes,
            selected_t,
            second_spike_delta,
        )
        self.figure.tight_layout()
        self.canvas.draw()

    def _single_spike_simulation(self, g_nmda):
        g_nmda_nS = float(g_nmda / nS)
        if self.single_spike is None or not np.isclose(self.single_spike_g_nmda_nS, g_nmda_nS):
            self.single_spike = simulate_nmda_only_forward_euler(
                params=replace(self.params, g_nmda=g_nmda),
                spike_train=(5.0 * ms,),
                include_ampa=True,
            )
            self.single_spike_g_nmda_nS = g_nmda_nS
        return self.single_spike

    def _two_spike_simulation(self, second_spike_delta):
        second_spike_delta_ms = float(second_spike_delta / ms)
        g_nmda_nS = self.g_slider.value() / 10.0
        needs_update = (
                self.two_spikes is None
                or not np.isclose(self.two_spike_delta_ms, second_spike_delta_ms)
                or not np.isclose(self.two_spike_g_nmda_nS, g_nmda_nS)
        )
        if needs_update:
            self.two_spikes = simulate_nmda_only_forward_euler(
                params=replace(self.params, g_nmda=g_nmda_nS * nS),
                spike_train=(5.0 * ms, 5.0 * ms + second_spike_delta),
                include_ampa=True,
            )
            self.two_spike_g_nmda_nS = g_nmda_nS
            self.two_spike_delta_ms = second_spike_delta_ms
        return self.two_spikes

    def _plot_nullcline_axis(self, ax, selected_s, selected_ampa, f_nA, selected_t):
        ax.clear()
        ax.plot(self.v_values_mV, f_nA)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(
            f"f(v), t = {selected_t / ms:.1f} ms, s = {selected_s:.3f}, AMPA = {selected_ampa:.3f}"
        )
        ax.set_xlabel("v [mV]")
        ax.set_ylabel("f(v) [nA]")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    def _plot_time_course_axis(self, ax, single_spike, two_spikes, selected_t, second_spike_delta):
        ax.clear()
        ax.plot(single_spike["t_ms"], single_spike["V_mV"], label="one spike")
        ax.plot(
            two_spikes["t_ms"],
            two_spikes["V_mV"],
            color="tab:red",
            linestyle="--",
            label=f"second spike after {second_spike_delta / ms:.1f} ms",
        )
        ax.axvline(selected_t / ms, color="black", linewidth=1.0, alpha=0.35)
        ax.set_title("V(t)")
        ax.set_xlabel("t [ms]")
        ax.set_ylabel("V [mV]")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.legend()

    def _s_at_time(self, simulation, selected_t):
        return _s_value_at_time(simulation, selected_t)


def launch_nmda_nullcline_explorer():
    app = QApplication.instance()
    created_app = app is None
    if created_app:
        app = QApplication([])

    window = NMDANullclineExplorer()
    window.show()

    if created_app:
        app.exec()
    return window


class NMDASimulationScripts(unittest.TestCase):

    def test_simulate_one_presynaptic_input(self):
        simulation = simulate_single_spike_at_5_ms(include_ampa=True)
        plot_nmda_dynamics(simulation)
        show_plots_non_blocking()

    def test_scan_g_nmda_max_vs_delta_v_max(self):
        scan_result = scan_g_nmda_max_vs_delta_v_max(include_ampa=True)
        plot_g_nmda_max_vs_delta_v_max(scan_result)
        show_plots_non_blocking()

    def test_scan_g_nmda_max_vs_delta_v_max_for_one_spike(self):
        scan_result = scan_g_nmda_max_vs_delta_v_max_for_one_spike(include_ampa=True, g_nmda_values=np.linspace(0, 200, 40) * nS)
        plot_g_nmda_max_vs_delta_v_max(
            scan_result,
            title="NMDA conductance scan, spike at 5 ms",
        )
        show_plots_non_blocking()

    def test_scan_g_nmda_max_vs_delta_v_max_for_two_spikes(self):
        scan_result = scan_g_nmda_max_vs_delta_v_max_for_two_spikes(include_ampa=True, g_nmda_values=np.linspace(0, 100, 40) * nS)
        plot_g_nmda_max_vs_delta_v_max(
            scan_result,
            title="NMDA conductance scan, spikes at 5 ms and 25 ms",
        )
        show_plots_non_blocking()

    def test_plot_one_two_and_three_spike_g_nmda_scan(self):
        from PlotOneVsTwoSpikeNMDAScan import run_and_plot_one_two_and_three_spike_g_nmda_scan

        g_nmda_values = np.linspace(0, 200, 40) * nS
        run_and_plot_one_two_and_three_spike_g_nmda_scan(
            g_nmda_values=g_nmda_values,
            params_template=NMDADynamicsParameters(t_stop=150.0 * ms),
            include_ampa=True,
        )
        show_plots_non_blocking()


    def test_plot_voltage_nullcline_for_selected_g_nmda_values(self):
        params_template = NMDADynamicsParameters(t_stop=150.0 * ms)
        for g_nmda in [25.0 * nS, 75.0 * nS, 100.0 * nS]:
            plot_voltage_nullcline_for_single_spike_state(
                g_nmda=g_nmda,
                g_ampa=25.0 * nS,
                params_template=params_template,
                include_ampa=True,
            )
        show_plots_non_blocking()

    def test_simulate_two_presynaptic_inputs_with_configurable_g_nmda(self):
        simulation = simulate_two_spikes_at_5_and_25_ms(
            g_nmda=25.0 * nS,
            include_ampa=True,
        )
        plot_nmda_dynamics(simulation)
        show_plots_non_blocking()

    def test_plot_v_nullcline_2_ms_after_initial_pulse(self):
        plot_voltage_nullcline_for_single_spike_state(
            g_nmda=250.0 * nS,
            g_ampa=25 * nS,
            selected_t=7.0 * ms,
            include_ampa=True,
        )
        show_plots_non_blocking()



if __name__ == '__main__':
    plot_voltage_nullcline_for_single_spike_state()
    show_plots_non_blocking()

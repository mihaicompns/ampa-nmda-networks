from dataclasses import dataclass, field, replace

import matplotlib.pyplot as plt
import numpy as np
from brian2 import (
    BrianLogger,
    Network,
    NeuronGroup,
    prefs,
    SpikeGeneratorGroup,
    StateMonitor,
    Synapses,
    defaultclock,
    exp,
    mmole,
    ms,
    mV,
    nA,
    nF,
    nS,
)

BrianLogger.log_level_error()
BrianLogger.suppress_name("brian2.codegen")
prefs.codegen.target = "numpy"


@dataclass(frozen=True)
class NMDASpikeParameters:
    dt: object = field(default_factory=lambda: 0.01 * ms)
    t_stop: object = field(default_factory=lambda: 100.0 * ms)
    capacitance: object = field(default_factory=lambda: 0.5 * nF)
    g_leak: object = field(default_factory=lambda: 0.0 * nS)
    e_leak: object = field(default_factory=lambda: -70.0 * mV)
    e_excitatory: object = field(default_factory=lambda: 0.0 * mV)
    g_nmda_max: object = field(default_factory=lambda: 65.0 * nS)
    magnesium: object = field(default_factory=lambda: 1.0 * mmole)
    tau_rise: object = field(default_factory=lambda: 2.0 * ms)
    tau_decay: object = field(default_factory=lambda: 100.0 * ms)
    alpha: object = field(default_factory=lambda: 0.5 / ms)
    presynaptic_spike_weight: float = 1.0
    initial_v: object = field(default_factory=lambda: -70.0 * mV)


def nmda_voltage_block(v, magnesium=1.0 * mmole):
    return 1.0 / (1.0 + 0.25 * np.exp(-0.075 * v / mV))


def simulate_nmda_spike_plain(spike_times, params=NMDASpikeParameters()):
    spike_steps = _spike_steps_for_grid(spike_times, params.dt, params.t_stop)
    n_steps = int(round(float(params.t_stop / params.dt))) + 1
    times = np.arange(n_steps) * params.dt

    x = np.zeros(n_steps)
    s = np.zeros(n_steps)
    v = np.zeros(n_steps) * mV
    i_nmda = np.zeros(n_steps) * nA
    presynaptic_input = np.zeros(n_steps)

    x_current = 0.0
    s_current = 0.0
    v_current = params.initial_v

    for step in range(n_steps):
        if step in spike_steps:
            pulse_weight = spike_steps[step] * params.presynaptic_spike_weight
            x_current += pulse_weight
            presynaptic_input[step] = pulse_weight

        block = nmda_voltage_block(v_current, params.magnesium)
        i_current = (
            params.g_nmda_max
            * block
            * s_current
            * (v_current - params.e_excitatory)
        )

        x[step] = x_current
        s[step] = s_current
        v[step] = v_current
        i_nmda[step] = i_current

        if step == n_steps - 1:
            break

        dx = params.dt * (-x_current / params.tau_rise)
        ds = params.dt * (-s_current / params.tau_decay + params.alpha * x_current)
        dv = params.dt * (
            (
                -params.g_leak * (v_current - params.e_leak)
                - i_current
            )
            / params.capacitance
        )
        x_current += float(dx)
        s_current += float(ds)
        v_current += dv

    return {
        "t_ms": times / ms,
        "presynaptic_input": presynaptic_input,
        "x": x,
        "s": s,
        "V_mV": v / mV,
        "I_NMDA_nA": i_nmda / nA,
    }


def simulate_nmda_spike_brian2(spike_times, params=NMDASpikeParameters()):
    normalized_spike_times = _normalize_spike_times(spike_times)
    previous_dt = defaultclock.dt
    defaultclock.dt = params.dt
    try:
        source = SpikeGeneratorGroup(
            1,
            indices=np.zeros(len(normalized_spike_times), dtype=int),
            times=normalized_spike_times,
        )
        neuron = NeuronGroup(
            1,
            """
            dx/dt = -x / tau_rise : 1
            ds/dt = -s / tau_decay + alpha * x : 1
            block = 1 / (1 + 0.25 * exp(-0.075 * V / mV)) : 1
            I_NMDA = g_nmda_max * block * s * (V - e_excitatory) : amp
            dV/dt = (-g_leak * (V - e_leak) - I_NMDA) / capacitance : volt
            tau_rise : second
            tau_decay : second
            alpha : Hz
            magnesium : mole
            g_nmda_max : siemens
            g_leak : siemens
            e_leak : volt
            e_excitatory : volt
            capacitance : farad
            """,
            method="euler",
            dt=params.dt,
        )
        neuron.x = 0.0
        neuron.s = 0.0
        neuron.V = params.initial_v
        neuron.tau_rise = params.tau_rise
        neuron.tau_decay = params.tau_decay
        neuron.alpha = params.alpha
        neuron.magnesium = params.magnesium
        neuron.g_nmda_max = params.g_nmda_max
        neuron.g_leak = params.g_leak
        neuron.e_leak = params.e_leak
        neuron.e_excitatory = params.e_excitatory
        neuron.capacitance = params.capacitance

        synapse = Synapses(
            source,
            neuron,
            model="w_x : 1",
            on_pre="x_post += w_x",
            dt=params.dt,
        )
        synapse.connect()
        synapse.w_x = params.presynaptic_spike_weight

        monitor = StateMonitor(
            neuron,
            ["x", "s", "V", "I_NMDA"],
            record=True,
            dt=params.dt,
            when="end",
        )
        net = Network(source, neuron, synapse, monitor)
        net.run(params.t_stop + params.dt)

        expected_n_steps = int(round(float(params.t_stop / params.dt))) + 1
        return {
            "t_ms": np.asarray(monitor.t[:expected_n_steps] / ms),
            "x": np.asarray(monitor.x[0][:expected_n_steps]),
            "s": np.asarray(monitor.s[0][:expected_n_steps]),
            "V_mV": np.asarray(monitor.V[0][:expected_n_steps] / mV),
            "I_NMDA_nA": np.asarray(monitor.I_NMDA[0][:expected_n_steps] / nA),
        }
    finally:
        defaultclock.dt = previous_dt


def simulate_presynaptic_spike_train(
    spike_times,
    params=NMDASpikeParameters(),
    backend="plain",
):
    simulate = _backend_simulator(backend)
    normalized_spike_times = _normalize_spike_times(spike_times)
    return {
        spike_count: simulate(normalized_spike_times[:spike_count], params=params)
        for spike_count in [1, 2, 3]
    }


def incremental_peak_depolarizations(simulations, spike_times=None):
    if spike_times is None:
        one_spike_delta_v = _delta_v_max(simulations[1])
        two_spike_delta_v = _delta_v_max(simulations[2])
        three_spike_delta_v = _delta_v_max(simulations[3])
    else:
        normalized_spike_times = _normalize_spike_times(spike_times)
        if len(normalized_spike_times) < 3:
            raise ValueError("spike_times must contain at least three spike times")

        first_peak_v = _window_peak_v_mV(
            simulations[1],
            start_ms=float(normalized_spike_times[0] / ms),
            stop_ms=float(normalized_spike_times[1] / ms),
        )
        second_peak_v = _window_peak_v_mV(
            simulations[2],
            start_ms=float(normalized_spike_times[1] / ms),
            stop_ms=float(normalized_spike_times[2] / ms),
        )
        third_peak_v = _window_peak_v_mV(
            simulations[3],
            start_ms=float(normalized_spike_times[2] / ms),
            stop_ms=None,
        )

        one_spike_delta_v = first_peak_v - float(simulations[1]["V_mV"][0])
        two_spike_delta_v = second_peak_v - float(simulations[2]["V_mV"][0])
        three_spike_delta_v = third_peak_v - float(simulations[3]["V_mV"][0])

    return {
        "one_spike_delta_v_max_mV": one_spike_delta_v,
        "two_spike_delta_v_max_mV": two_spike_delta_v,
        "three_spike_delta_v_max_mV": three_spike_delta_v,
        "second_spike_increment_mV": two_spike_delta_v - one_spike_delta_v,
        "third_spike_increment_mV": three_spike_delta_v - two_spike_delta_v,
    }


def plot_one_two_three_spike_voltage_overlay(
    simulations,
    params=NMDASpikeParameters(),
    spike_times=None,
    title=None,
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4),
        sharex=True,
    )
    voltage_ax, gating_ax = axes
    line_specs = {
        1: ("-", "1 spike"),
        2: ("--", "2 spikes"),
        3: (":", "3 spikes"),
    }
    windowed_increments = None
    if spike_times is not None:
        windowed_increments = incremental_peak_depolarizations(
            simulations=simulations,
            spike_times=spike_times,
        )
    increment_key_by_spike_count = {
        1: "one_spike_delta_v_max_mV",
        2: "second_spike_increment_mV",
        3: "third_spike_increment_mV",
    }
    for spike_count, simulation in simulations.items():
        if windowed_increments is None:
            delta_v = _delta_v_max(simulation)
        else:
            delta_v = windowed_increments[increment_key_by_spike_count[spike_count]]
        linestyle, label = line_specs[spike_count]
        voltage_ax.plot(
            simulation["t_ms"],
            simulation["V_mV"],
            linestyle=linestyle,
            linewidth=2,
            label=f"{label}; dV{spike_count}={delta_v:.3g} mV",
        )
        gating_ax.plot(
            simulation["t_ms"],
            simulation["s"],
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )

    if title is None:
        title = f"NMDA-only local model, g_NMDA,max={float(params.g_nmda_max / nS):.3g} nS"
    fig.suptitle(title)

    voltage_ax.set_title("Voltage response")
    voltage_ax.set_xlabel(r"$t$ [ms]")
    voltage_ax.set_ylabel(r"$V_{\mathrm{local}}$ [mV]")
    voltage_ax.legend()

    gating_ax.set_title(r"NMDA gating variable $s(t)$")
    gating_ax.set_xlabel(r"$t$ [ms]")
    gating_ax.set_ylabel(r"$s(t)$")
    gating_ax.legend()
    fig.tight_layout()
    return fig, axes


def _backend_simulator(backend):
    if backend == "plain":
        return simulate_nmda_spike_plain
    if backend == "brian2":
        return simulate_nmda_spike_brian2
    raise ValueError(f"Unknown backend: {backend}")


def _delta_v_max(simulation):
    return float(np.max(simulation["V_mV"] - simulation["V_mV"][0]))


def _window_peak_v_mV(simulation, start_ms, stop_ms):
    times_ms = np.asarray(simulation["t_ms"])
    voltages_mV = np.asarray(simulation["V_mV"])
    if stop_ms is None:
        window_mask = times_ms >= start_ms
    else:
        window_mask = (times_ms >= start_ms) & (times_ms < stop_ms)
    if not np.any(window_mask):
        raise ValueError(f"no samples in voltage window starting at {start_ms} ms")
    return float(np.max(voltages_mV[window_mask]))


def _spike_steps_for_grid(spike_times, dt, t_stop):
    spike_steps = {}
    for spike_time in _normalize_spike_times(spike_times):
        step = int(round(float(spike_time / dt)))
        time_grid_error_ms = float((step * dt - spike_time) / ms)
        if not np.isclose(time_grid_error_ms, 0.0):
            raise ValueError("spike times must fall on the simulation time grid")
        if spike_time < 0 * ms or spike_time > t_stop:
            raise ValueError(f"spike time {spike_time} is outside the simulation interval")
        spike_steps[step] = spike_steps.get(step, 0) + 1
    return spike_steps


def _normalize_spike_times(spike_times):
    return tuple(spike_times)


def params_with(params, **kwargs):
    return replace(params, **kwargs)

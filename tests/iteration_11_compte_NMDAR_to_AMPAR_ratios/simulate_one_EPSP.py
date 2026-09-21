import numpy as np
from brian2 import plt, mpl, StateMonitor, mV, start_scope, defaultclock, mmole, kHz, NeuronGroup, run, \
    second, devices as brian2devices, seed, Synapses, \
    ms, SpikeGeneratorGroup

from src.iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from src.iteration_7_one_compartment_step_input.one_compartment_with_up_down import SimulationResults

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

epsp_model = '''
v : volt (constant)

I_ampa = g_e * (v - E_ampa): amp
I_gaba = g_i * (v - E_gaba): amp
I_nmda = g_nmda * (v - E_nmda): amp

dg_e/dt = -g_e / tau_ampa : siemens
dg_i/dt = -g_i / tau_gaba  : siemens

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
dx_nmda/dt = - x_nmda / tau_nmda_rise : 1

sigmoid_v = 1/(1 + (MG_C/mmole)/3.57 * exp(-0.062*(v/mvolt))): 1
'''

def simulate_one_epsp(experiment: Experiment):

    start_scope()

    defaultclock.dt = experiment.sim_clock

    C = experiment.neuron_params.C

    g_ampa = experiment.synaptic_params.g_ampa
    g_gaba = experiment.synaptic_params.g_gaba
    g_x_nmda = experiment.synaptic_params.g_x_nmda
    g_nmda_max = experiment.synaptic_params.g_nmda

    E_ampa = experiment.synaptic_params.e_ampa
    E_gaba = experiment.synaptic_params.e_gaba
    E_nmda = experiment.synaptic_params.e_ampa

    MG_C = 1 * mmole  # extracellular magnesium concentration

    tau_ampa = experiment.synaptic_params.tau_ampa
    tau_gaba = experiment.synaptic_params.tau_gaba
    tau_nmda_rise = experiment.synaptic_params.tau_nmda_rise
    tau_nmda_decay = experiment.synaptic_params.tau_nmda_decay

    alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates

    neuron = NeuronGroup(1, epsp_model, method='exponential_euler')
    neuron.v = -65 * mV

    G = SpikeGeneratorGroup(1, np.array([0]), np.array([100]) * ms)

    S_ampa = Synapses(G, neuron, on_pre='g_e += g_ampa',
                               method=experiment.integration_method)
    S_gaba = Synapses(G, neuron, on_pre='g_i += g_gaba',
                               method=experiment.integration_method)
    S_nmda = Synapses(G, neuron, on_pre='x_nmda += g_x_nmda',
                                method=experiment.integration_method)
    S_ampa.connect(p=1)
    S_gaba.connect(p=1)
    S_nmda.connect(p=1)
    currents_monitor = StateMonitor(neuron, experiment.plot_params.recorded_currents, record=True)

    reporting = "text" if experiment.in_testing else None
    run(experiment.sim_time, report=reporting, report_period=1 * second)


    return SimulationResults(experiment, None, None, None, None, None,
                      currents_monitor, mean_field_values=None)


import copy
import enum
import math

import numpy as np
from brian2 import ufarad, siemens, mV, ms, Hz, nS, nsiemens, mmole, kHz, psiemens
from brian2.units.allunits import pampere
from loguru import logger

from src.iteration_8_compute_mean_steady_state.equations import sigmoid_v


class SynapticParams:
    KEY_SYNAPTIC_STRENGTH = "J"
    KEY_SYNAPTIC_DELAY = "D"

    KEY_G_AMPA = "g_ampa"
    KEY_G_GABA = "g_gaba"
    KEY_G_NMDA = "g_nmda"
    KEY_X_NMDA = "x_nmda"

    KEY_TAU_AMPA = "tau_ampa_ms"
    KEY_TAU_GABA = "tau_gaba_ms"

    KEY_TAU_NMDA_RISE = "tau_nmda_rise"
    KEY_TAU_NMDA_DECAY = "tau_nmda_decay"

    KEY_E_AMPA = "E_ampa"
    KEY_E_GABA = "E_gaba"

    KEY_MG_EXTRACELLULAR_CONCENTRATION = "MG_C"
    KEY_ALPHA_NMDA = "alpha_nmda"

    def __init__(self, params: dict, g: float):

        self.J = params.get(SynapticParams.KEY_SYNAPTIC_STRENGTH, 0.5 * mV)
        self.D = params.get(SynapticParams.KEY_SYNAPTIC_DELAY, 1.5 * ms)

        self.g_ampa = params.get(SynapticParams.KEY_G_AMPA, 0) * siemens
        if g > 0:
            self.g_gaba = g * self.g_ampa
            self.g = g
        else:
            self.g_gaba = params.get(SynapticParams.KEY_G_GABA, 0) * siemens
            self.g = self.g_gaba / self.g_ampa

        self.g_x_nmda = params.get(SynapticParams.KEY_X_NMDA, 1)

        self.g_nmda = params.get(SynapticParams.KEY_G_NMDA, 0) * siemens

        self.tau_ampa = params.get(SynapticParams.KEY_TAU_AMPA, 2) * ms
        self.tau_gaba = params.get(SynapticParams.KEY_TAU_GABA, 2) * ms

        self.e_ampa = params.get(SynapticParams.KEY_E_AMPA, 0) * mV
        self.e_gaba = params.get(SynapticParams.KEY_E_AMPA, -80) * mV
        self.tau_nmda_rise = params.get(NeuronModelParams.KEY_TAU_NMDA_RISE, 2) * ms
        self.tau_nmda_decay = params.get(NeuronModelParams.KEY_TAU_NMDA_DECAY, 100) * ms

        # extracellular magnesium concentration
        self.MG_C = params.get(SynapticParams.KEY_MG_EXTRACELLULAR_CONCENTRATION,
                               1) * mmole  # extracellular magnesium concentration
        self.alpha_nmda = params.get(SynapticParams.KEY_ALPHA_NMDA, 0.5) * kHz  # NMDA saturation

    def __str__(self):
        return f"{self.__class__}(J={self.J}, D={self.D})"


'''
Brunel page 185:
The parameter space remains large, even for such
simple model neurons. In the following, using anatom-
ical estimates for neocortex, we choose N E = 0.8N ,
N I = 0.2N (80% of excitatory neurons). This implies
C E = 4C I . We rewrite C I = γ C E —that is, γ = 0.25.

N_I = gamma * N_E
'''


class State:
    KEY_STATE_UP = "up_state"
    KEY_STATE_DOWN = "down_state"

    KEY_N = "N"
    KEY_N_E = "N_E"
    KEY_N_I = "N_I"
    KEY_N_NMDA = "N_nmda"

    KEY_NU = "nu"

    KEY_NU_NMDA = "nu_nmda"

    KEY_GAMMA = "gamma"
    KEY_OMEGA = "omega"

    KEY_X_VAR_MULT = "x_var_mult"

    def __init__(self, params: dict):

        self.params = params.copy()

        self.gamma = params.get(State.KEY_GAMMA, 0.25)
        self.omega = params.get(State.KEY_OMEGA, 0)

        if self.KEY_N in params and self.KEY_N_E in params:
            raise ValueError("Please provide only one of N and N_E. Gamma will set the proper ration")

        if self.KEY_N in params:
            self.N = params.get(State.KEY_N)
            self.N_E = math.ceil(self.N / (1 + self.gamma))
            self.N_I = self.N - self.N_E
        else:
            self.N_E = params.get(State.KEY_N_E, 10_000)
            self.N_I = round(self.gamma * self.N_E)
            self.N = self.N_E + self.N_I

        if self.KEY_N_NMDA in params:
            self.N_NMDA = params.get(State.KEY_N_NMDA, 0)
            self.omega = self.N_NMDA / self.N if self.N > 0 else 0
        else:
            self.N_NMDA = int(self.omega * self.N)

        self.nu = params.get(State.KEY_NU, 0) * Hz
        self.nu_nmda = params.get(State.KEY_NU_NMDA, 0) * Hz

        self.x_var_mult = params.get(State.KEY_X_VAR_MULT, 1)

    def gen_plot_title(self):
        return fr"$N_E={self.N_E}$, $N_I={self.N_I}$, $N_\mathrm{{NMDA}}={self.N_NMDA}$,  $\nu={self.nu}$, $\nu_\mathrm{{NMDA}}={self.nu_nmda}$, $\gamma={self.gamma}$"




class NetworkParams:
    KEY_G = "g"
    KEY_NU_THR = "nu_thr"
    KEY_NU_E_OVER_NU_THR = "nu_ext_over_nu_thr"

    KEY_RECORD_EXTRA = "record_N"

    KEY_C_EXT = "C_ext"

    KEY_EPSILON = "epsilon"

    KEY_UP_STATE = "UP_STATE"
    KEY_DOWN_STATE = "DOWN_STATE"

    def __init__(self, params: dict):
        self.g = params.get(NetworkParams.KEY_G, 0)
        self.synaptic_params = SynapticParams(params, g=self.g)

        self.epsilon = params.get(NetworkParams.KEY_EPSILON, 0.1)

        self.up_state = State(params[State.KEY_STATE_UP]) if State.KEY_STATE_UP in params else None
        self.down_state = State(params[State.KEY_STATE_DOWN]) if State.KEY_STATE_DOWN in params else None

    def __str__(self):
        return f"{self.__class__}({self.KEY_G}={self.g}, {self.KEY_GAMMA}={self.gamma}, {self.KEY_EPSILON}={self.epsilon}, {self.KEY_N_E}={self.N_E}, {self.KEY_N_I}={self.N_I}, \
                N={self.N}, C_E={self.C_E}, {self.KEY_C_EXT}={self.C_ext})"

class CurrentClampParams:

    KEY_I_INJECTED = "I_CLAMP" # pico amperre

    def __init__(self, params: dict):
        self.i_inj = params.get(CurrentClampParams.KEY_I_INJECTED, 0) * pampere

class NeuronModelParams:
    KEY_NEURON_C = "C"
    KEY_NEURON_G_L = "g_L"
    KEY_NEURON_THRESHOLD = "theta"
    KEY_NEURON_V_R = "v_reset"
    KEY_NEURON_E_L = "E_leak"
    KEY_TAU_REF = "tau_ref"

    KEY_TAU_NMDA_RISE = "tau_nmda_rise"
    KEY_TAU_NMDA_DECAY = "tau_nmda_decay"

    def __init__(self, params: dict, network_params: NetworkParams = NetworkParams):
        self.synaptic_params = SynapticParams(params, g=network_params.g)

        self.C = params.get(NeuronModelParams.KEY_NEURON_C, 1) * ufarad
        self.g_L = params.get(NeuronModelParams.KEY_NEURON_G_L, 0.04e-3) * siemens
        self.theta = params.get(NeuronModelParams.KEY_NEURON_THRESHOLD, -40) * mV
        self.V_r = params.get(NeuronModelParams.KEY_NEURON_V_R, -65) * mV
        self.E_leak = params.get(NeuronModelParams.KEY_NEURON_E_L, -65) * mV
        self.tau_rp = params.get(NeuronModelParams.KEY_TAU_REF, 2) * ms

        self.tau = self.C / self.g_L
        if network_params.up_state is not None:
            self.nu_thr = (self.theta - self.E_leak) / (self.synaptic_params.J * network_params.up_state.N_E * self.tau)
        else:
            self.nu_thr = (self.theta - self.E_leak) / (
                    self.synaptic_params.J * network_params.down_state.N_E * self.tau)

        logger.debug("Computed tau membrane = {}, nu threshold = {}", self.tau, self.nu_thr)

    def __str__(self):
        return (
            f"{self.__class__}({NeuronModelParams.KEY_NEURON_C}={self.C}, {NeuronModelParams.KEY_NEURON_G_L}={self.g_L}, \
                {NeuronModelParams.KEY_NEURON_THRESHOLD}={self.theta}, {NeuronModelParams.KEY_NEURON_V_R}={self.V_r}, \
                {NeuronModelParams.KEY_NEURON_E_L}={self.E_leak}, {NeuronModelParams.KEY_TAU_REF}={self.tau_rp}, nu_thr(computed)={self.nu_thr},\
                tau membrane(computed)={self.tau})")


class PlotParams:
    KEY_PANEL = "panel"

    KEY_T_RANGE = "t_range"
    KEY_RATE_RANGE = "rate_range"
    KEY_VOLTAGE_RANGE = "voltage_range"
    KEY_RATE_TICK_STEP = "rate_tick_step"

    KEY_PLOT_SMOOTH_WIDTH = "smoothened_rate_width"
    KEY_PLOT_TURN_OFF_SMOOTH_RATE = "smoothened_rate_width"

    KEY_PLOT_NEURONS_W_HIDDEN_VARIABLES = "neurons_w_hidden_variables_to_plot"

    KEY_WHAT_PLOTS_TO_SHOW = "show_plots"

    class AvailablePlots(enum.Enum):
        RASTER_AND_RATE = 1
        PSD_AND_CVS = 2
        HIDDEN_VARIABLES = 3
        CURRENTS = 4

    class AvailableHiddenVariables(enum.Enum):
        sigmoid_v = "sigmoid_v"
        x_nmda = "x_nmda"
        s_nmda = "s_nmda"
        g_nmda = "g_nmda"
        I_nmda = "I_nmda"
        one_minus_s_nmda = "one_minus_s_nmda"

        g_e = "g_e"
        g_i = "g_i"

        s_drive = "s_drive"
        alpha_x_t = "alpha_x_t"
        g_nmda_max = "g_nmda_max"
        v_minus_e_gaba = "v_minus_e_gaba"

    hidden_variable_plot_details = {
        AvailableHiddenVariables.sigmoid_v.value: {
            "title": "Sigmoid",
            "y_label": "activation \n [unitless]"
        },
        AvailableHiddenVariables.x_nmda.value: {
            "title": "X variable (NMDA upstroke)",
            "y_label": r"$x_\mathrm{NMDA}$"
        },
        AvailableHiddenVariables.g_nmda.value: {
            "title": "Total NMDA conductance",
            "y_label": r'$g_\mathrm{NMDA}$''\n'r'[nS]',
            "scaling": nsiemens
        },
        AvailableHiddenVariables.I_nmda.value: {
            "title": "NMDA current",
            "y_label": r"$I_\mathrm{NMDA}$"
        },
        AvailableHiddenVariables.one_minus_s_nmda.value: {
            "title": "how much free g_nmda exists? (1 - s) variable",
            "y_label": "% unsaturated NMDA [unitless]"
        },

        AvailableHiddenVariables.s_nmda.value: {
            "title": "S NMDA variable",
            "y_label": r'$s_\mathrm{NMDA}$ \\n [unitless]'
        },

        AvailableHiddenVariables.s_drive.value: {
            "title": r'Driving force of $s_\mathrm{NMDA}$ is $\alpha \cdot x_\mathrm{NMDA} \cdot (1-s_\mathrm{NMDA})$',
            "y_label": r"$[s^{-1}]$"
        },

        AvailableHiddenVariables.alpha_x_t.value: {
            "title": r'$\alpha \cdot x_\mathrm{NMDA}$',
            "y_label": r"$[s^{-1}]$"
        },

        AvailableHiddenVariables.g_nmda_max.value: {
            "title": r'G NMDA max',
            "y_label": r"$[\frac{\mathrm{siemens}}{m^2}]$"
        },

        AvailableHiddenVariables.v_minus_e_gaba.value: {
            "title": r'$v-E_\mathrm{GABA}$',
            "y_label": r"$[mV]$",
            "scaling": mV
        },

        AvailableHiddenVariables.g_e.value: {
            "title": r'$g_\mathrm{AMPA}$',
            "y_label": "[nS]",
            "scaling": nsiemens
        },

        AvailableHiddenVariables.g_i.value: {
            "title": r'$g_\mathrm{GABA}$',
            "y_label": "[nS]",
            "scaling": nsiemens
        }
    }

    class AvailableCurrents(enum.Enum):
        I_Leak = "I_L"
        I_ampa = "I_ampa"
        I_gaba = "I_gaba"
        I_nmda = "I_nmda"
        I_Ext_syn = "I_ext_syn"

        standard = ["I_L", "I_ampa", "I_gaba", "I_nmda"]

    def __init__(self, params):
        self.panel = params.get(PlotParams.KEY_PANEL, "")
        self.t_range = params.get(PlotParams.KEY_T_RANGE, [0, 100])
        self.rate_range = params.get(PlotParams.KEY_RATE_RANGE, [0, 150])
        self.voltage_range = params.get(PlotParams.KEY_VOLTAGE_RANGE, None)

        self.rate_tick_step = params.get(PlotParams.KEY_RATE_TICK_STEP, 30)
        self.smoothened_rate_width = params.get(PlotParams.KEY_PLOT_SMOOTH_WIDTH, 0.5) * ms
        self.plot_smoothened_rate = PlotParams.KEY_PLOT_TURN_OFF_SMOOTH_RATE not in params
        self.plot_smoothened_rate = True

        self.neurons_to_plot = params.get(PlotParams.KEY_PLOT_NEURONS_W_HIDDEN_VARIABLES, [])
        self.plots = params.get(PlotParams.KEY_WHAT_PLOTS_TO_SHOW, [PlotParams.AvailablePlots.RASTER_AND_RATE])
        self.recorded_hidden_variables = params.get(Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD,
                                                    ["g_nmda", "I_nmda"])

        for hidden_variable in self.recorded_hidden_variables:
            if hidden_variable not in self.AvailableHiddenVariables:
                raise ValueError(
                    f"{hidden_variable} not found in known hidden variables {self.AvailableHiddenVariables}")

        self.recorded_currents = params.get(Experiment.KEY_CURRENTS_TO_RECORD, [])
        if self.recorded_currents is None:
            self.recorded_currents = []
        self.recorded_g_s = params.get(Experiment.KEY_G_S_TO_RECORD, ["g_nmda", "g_e", "g_i"])

    def show_raster_and_rate(self):
        return self.plots is not None and PlotParams.AvailablePlots.RASTER_AND_RATE in self.plots

    def show_psd_and_cv(self):
        return self.plots is not None and PlotParams.AvailablePlots.PSD_AND_CVS in self.plots

    def show_hidden_variables(self):
        return self.plots is not None and PlotParams.AvailablePlots.HIDDEN_VARIABLES in self.plots

    def create_hidden_variables_plots_grid(self):
        if not self.show_hidden_variables():
            return {}

        result = {}
        for hidden_variable in self.recorded_hidden_variables:
            result[hidden_variable] = {
                "index": len(result.items()),
                "title": PlotParams.hidden_variable_plot_details[hidden_variable]["title"],
                "y_label": PlotParams.hidden_variable_plot_details[hidden_variable]["y_label"],
                "scaling": PlotParams.hidden_variable_plot_details[hidden_variable]["scaling"] if "scaling" in
                                                                                                  PlotParams.hidden_variable_plot_details[
                                                                                                      hidden_variable] else 1
            }
        return result

    def show_currents_plots(self):
        return self.plots is not None and PlotParams.AvailablePlots.CURRENTS in self.plots and len(
            self.recorded_currents) > 0


class NMDAParams:
    KEY_BETA = "beta"

    def __init__(self, params):
        self.beta = params.get(NMDAParams.KEY_BETA, 1)


def check_plot_times_inside_sim_time(plot_params: PlotParams, sim_time):
    if list is type(plot_params.t_range[0]):
        for t_range in plot_params.t_range:
            if t_range[1] > sim_time / ms:
                raise ValueError(f"Sim time was only {sim_time}. We can not plot unit {t_range[1]}.")
    elif plot_params.t_range[0] > sim_time / ms or plot_params.t_range[1] > sim_time / ms:
        raise ValueError(f"Sim time was only {sim_time}. We can not plot unit {plot_params.t_range}.")


class DiffusionProcess:
    KEY_X_VARIANCE_MULTIPLICATION = "x_noise_multiplier"

    def __init__(self, params: dict):
        self.x_variance_mult = params.get(DiffusionProcess.KEY_X_VARIANCE_MULTIPLICATION, 0)


class Experiment:
    KEY_SIM_TIME = "sim_time"
    KEY_SIMULATION_CLOCK = "simulation_clock"

    KEY_SIMULATION_METHOD = "method"

    KEY_SELECTED_MODEL = "model"

    KEY_STEADY_MODEL = "steady_model"
    KEY_DIFFUSION_MODEL = "diffusion_model"

    KEY_HIDDEN_VARIABLES_TO_RECORD = "hidden_variables_to_record"
    KEY_CURRENTS_TO_RECORD = "currents_to_record"
    KEY_G_S_TO_RECORD = "g_s_to_record"

    KEY_IN_TESTING = "in_testing"

    def __init__(self, params: dict):
        self.params = copy.deepcopy(params)

        self.sim_time = self.__extract_simulation_time__(params) * ms
        self.sim_clock = params.get(Experiment.KEY_SIMULATION_CLOCK, 0.05) * ms

        self.integration_method = params.get(Experiment.KEY_SIMULATION_METHOD, "euler")

        self.network_params = NetworkParams(params)
        self.synaptic_params = SynapticParams(params, g=self.network_params.g)
        self.neuron_params = NeuronModelParams(params=params, network_params=self.network_params)
        self.current_clamp_params = CurrentClampParams(params)

        self.plot_params = PlotParams(params)


        check_plot_times_inside_sim_time(self.plot_params, self.sim_time)

        self.nu_ext_over_nu_thr = params.get(NetworkParams.KEY_NU_E_OVER_NU_THR, 1)
        self.nu_thr = self.neuron_params.nu_thr
        self.nu_ext = self.nu_ext_over_nu_thr * self.nu_thr

        # self.mean_excitatory_input = self.synaptic_params.J * self.neuron_params.tau * self.network_params.up_state.N_E * self.nu_ext
        # self.mean_inhibitory_input = - self.network_params.g * self.synaptic_params.J * self.neuron_params.tau * self.network_params.up_state.N_E * self.nu_ext

        self.nmda_params = NMDAParams(params)

        self.model = params.get(Experiment.KEY_SELECTED_MODEL)
        self.steady_state_model = params.get(Experiment.KEY_STEADY_MODEL, None)
        self.diffusion_model = params.get(Experiment.KEY_DIFFUSION_MODEL, None)

        self.recorded_hidden_variables = self.plot_params.recorded_hidden_variables

        self.in_testing = params.get(Experiment.KEY_IN_TESTING, False)

        self.effective_time_constant_up_state = EffectiveTimeConstantEstimation(self, self.network_params.up_state,
                                                                                label="Up") \
            if self.network_params.up_state is not None else None
        self.effective_time_constant_down_state = EffectiveTimeConstantEstimation(self, self.network_params.down_state,
                                                                                  label="Down") \
            if self.network_params.down_state is not None else None

    def with_property(self, key: str, value: object):
        return self.with_properties({key: value})

    def with_label(self, value: str):
        return self.with_property(PlotParams.KEY_PANEL, value)

    def with_properties(self, values: dict[str, object]):
        new_params = copy.deepcopy(self.params)
        for key, value in values.items():
            new_params[key] = value
        return Experiment(new_params)

    def __extract_simulation_time__(self, params):
        if Experiment.KEY_SIM_TIME not in params and PlotParams.KEY_T_RANGE not in params:
            raise ValueError("Either simulation time or time range must be provided")

        if Experiment.KEY_SIM_TIME in params:
            return params[Experiment.KEY_SIM_TIME]

        return np.max(params[PlotParams.KEY_T_RANGE])

    def gen_plot_title(self):
        return fr"""{self.plot_params.panel} Single Compartment Level
    Network: [N={self.network_params.N}, $N_E={self.network_params.N_E}$, $N_I={self.network_params.N_I}$, $\gamma={self.network_params.gamma}$, $\epsilon={self.network_params.epsilon}$]
    Input: [$\nu_T={self.nu_thr}$, $\frac{{\nu_E}}{{\nu_T}}={self.nu_ext_over_nu_thr:.2f}$, $\nu_E={self.nu_ext:.2f}$ Hz]
    Neuron: [$C={self.neuron_params.C}$, $g_L={self.neuron_params.g_L}$, $\theta={self.neuron_params.theta}$, $V_R={self.neuron_params.V_r}$, $E_L={self.neuron_params.E_leak}$, $\tau_M={self.neuron_params.tau}$, $\tau_{{\mathrm{{ref}}}}={self.neuron_params.tau_rp}$]
    Synapse: [$g_{{\mathrm{{AMPA}}}}={self.synaptic_params.g_ampa:.2f}$, $g_{{\mathrm{{GABA}}}}={self.synaptic_params.g_gaba:.2f}$, $g={self.network_params.g}$]"""

    def with_no_current(self):
        return self.with_property(CurrentClampParams.KEY_I_INJECTED, 0)

# Richardson Synaptic Shot Noise and Conductance Fluctuations Affect the Membrane Voltage with Equal Significance, 2005
class EffectiveTimeConstantEstimation:

    def __init__(self, config: Experiment, state: State, label="Up"):
        self.config = config
        self.state = state
        self.__check_is_diffusion_approximation_valid__()

        self.state.effective_timeconstant_estimation = self

        self.__cached_g_nmda = None

        # logger.debug("Effective Reversal {} State with included Poisson Rate {}", label, self.E_0())

    def E_0(self):
        effective_reversal = (self.config.neuron_params.g_L * self.config.neuron_params.E_leak +
                              self.mean_excitatory_conductance() * self.config.synaptic_params.e_ampa +
                              self.mean_inhibitory_conductance() * self.config.synaptic_params.e_gaba) / self.mean_total_conductance()
        return effective_reversal

    # 2.6, 2.12
    def mean_excitatory_conductance(self):
        # According to units computation, this is not in the same unit aas g_ampa. Check section 6.5 Units, Composed
        # return self.config.synaptic_params.tau_ampa * self.state.N_E * self.state.nu * self.config.synaptic_params.g_ampa / 18**3
        return self.config.synaptic_params.tau_ampa * self.state.N_E * self.state.nu * self.config.synaptic_params.g_ampa

    # 2.6, 2.12
    def mean_inhibitory_conductance(self):
        # return self.config.synaptic_params.tau_gaba * self.state.N_I * self.state.nu * self.config.synaptic_params.g_gaba / 18**3
        return self.config.synaptic_params.tau_gaba * self.state.N_I * self.state.nu * self.config.synaptic_params.g_gaba

    def mean_nmda_activation(self):
        return self.config.synaptic_params.tau_nmda_rise * self.state.N_NMDA * self.state.nu_nmda

    # 2.12
    def mean_total_conductance(self):
        return self.config.neuron_params.g_L + self.mean_excitatory_conductance() + self.mean_inhibitory_conductance()

    # 2.6
    def std_excitatory_conductance(self):
        return self.config.synaptic_params.g_ampa * np.sqrt(
            1 / 2 * self.config.synaptic_params.tau_ampa * self.state.N_E * self.state.nu
        )

    # 2.6
    def std_inhibitory_conductance(self):
        return self.config.synaptic_params.g_gaba * np.sqrt(
            1 / 2 * self.config.synaptic_params.tau_gaba * self.state.N_I * self.state.nu
        )

    # 2.19
    def std_voltage(self):
        # 2.13
        g_0 = self.mean_total_conductance()
        tau_0 = self.config.neuron_params.C / g_0
        tau_e = self.config.synaptic_params.tau_ampa
        tau_i = self.config.synaptic_params.tau_gaba

        s_e = self.std_excitatory_conductance()
        s_i = self.std_inhibitory_conductance()

        E_0 = self.E_0()
        E_e = self.config.synaptic_params.e_ampa
        E_i = self.config.synaptic_params.e_gaba

        return np.sqrt(
            (s_e / g_0) ** 2 * (E_e - E_0) ** 2 * (tau_e / (tau_e + tau_0)) + (s_i / g_0) ** 2 * (E_i - E_0) ** 2 * (
                    tau_i / (tau_i + tau_0)))

    def shunt_level(self):
        return (self.mean_excitatory_conductance() + self.mean_inhibitory_conductance()) / self.mean_total_conductance()

    # 2.9 check if diffusion approximation holds
    def __check_is_diffusion_approximation_valid__(self):
        return
        sigma_e_over_g_e_0 = self.std_excitatory_conductance() / self.mean_excitatory_conductance()
        sigma_i_over_g_i_0 = self.std_inhibitory_conductance() / self.mean_inhibitory_conductance()
        logger.debug("Is diffusion approximation valid? sigma_e / g_e0 ={} << 1? {}", sigma_e_over_g_e_0,
                     sigma_e_over_g_e_0 < 0.01)
        logger.debug("Is diffusion approximation valid? sigma_i / g_i0 ={} << 1? {}", sigma_i_over_g_i_0,
                     sigma_i_over_g_i_0 < 0.01)

    def gen_plot_title(self):
        return fr"$V_\mathrm{{eff, rev}}$ = {self.E_0() / mV: .2f} mV, $\sigma_v$ = {self.std_voltage() / mV : .2f} mV, $g_{{\mathrm{{0, AMPA}}}}={self.mean_excitatory_conductance() / nS:.2f}\,n\mathrm{{S}}$, $g_{{\mathrm{{0, GABA}}}}={self.mean_inhibitory_conductance() / nS:.2f}\,n\mathrm{{S}}$, Shunt Level = {self.shunt_level(): .2f}"

    def mean_x_nmda(self):
        return self.config.synaptic_params.tau_nmda_rise * self.config.synaptic_params.g_x_nmda * self.state.N_NMDA * self.state.nu_nmda

    def mean_s_nmda(self):
        mean_x_nmda = self.mean_x_nmda()
        return 1 - 1 / (
                1 + self.config.synaptic_params.alpha_nmda * self.config.synaptic_params.tau_nmda_decay * mean_x_nmda)

    def std_x_nmda(self):
        return self.config.synaptic_params.g_x_nmda * np.sqrt(
            1 / 2 * self.config.synaptic_params.tau_nmda_rise * self.state.N_NMDA * self.state.nu_nmda
        )

    def mean_total_conductance_with_nmda(self):
        g_nmda = self.compute_mean_g_nmda()

        return self.mean_total_conductance() + g_nmda

    def E_0_with_nmda(self, g_nmda=None):

        if self.state.N_NMDA < 1 or self.state.nu_nmda == 0:
            return self.E_0()

        if g_nmda is None:
            g_nmda = self.compute_mean_g_nmda()

        total_conductance = self.mean_total_conductance() + g_nmda

        effective_reversal = (self.config.neuron_params.g_L * self.config.neuron_params.E_leak +
                              self.mean_excitatory_conductance() * self.config.synaptic_params.e_ampa +
                              g_nmda * self.config.synaptic_params.e_ampa +
                              self.mean_inhibitory_conductance() * self.config.synaptic_params.e_gaba) / total_conductance

        return effective_reversal

    def compute_mean_g_nmda(self, v=None, current_g_nmda=0 * nsiemens, err=1E-10):

        if self.__cached_g_nmda is not None:
            return self.__cached_g_nmda

        if v is None:
            v = self.E_0()

        g_nmda = self.config.synaptic_params.g_nmda * sigmoid_v(self.config, v) * self.mean_s_nmda()

        diff = g_nmda - current_g_nmda
        if np.abs(diff) / nsiemens < err:
            self.__cached_g_nmda = g_nmda
            return g_nmda

        current_v = self.E_0_with_nmda(g_nmda)
        return self.compute_mean_g_nmda(v=current_v, current_g_nmda=g_nmda, err=err)

    def std_s_nmda(self):

        tau_s = self.config.synaptic_params.tau_nmda_decay * (1 - self.mean_x_nmda())

        in_sqrt = 1 / 2 * (self.state.N_NMDA * self.state.nu_nmda) / (tau_s + self.config.synaptic_params.tau_nmda_rise)
        return self.config.synaptic_params.alpha_nmda * (
                    1 - self.mean_x_nmda()) * tau_s * self.config.synaptic_params.tau_nmda_rise / tau_s * self.config.synaptic_params.tau_nmda_decay * np.sqrt(
            in_sqrt)

    def std_g_nmda(self):
        return self.config.synaptic_params.g_nmda * sigmoid_v(self.config, self.E_0_with_nmda()) * self.std_s_nmda()

    def std_voltage_with_nmda(self):
        # 2.13
        g_0 = self.mean_total_conductance()
        tau_0 = self.config.neuron_params.C / g_0
        tau_e = self.config.synaptic_params.tau_ampa
        tau_i = self.config.synaptic_params.tau_gaba
        tau_n = self.config.synaptic_params.tau_nmda_decay * (1 - self.mean_s_nmda())

        s_e = self.std_excitatory_conductance()
        s_i = self.std_inhibitory_conductance()
        s_n = self.std_g_nmda()

        E_0 = self.E_0()
        E_e = self.config.synaptic_params.e_ampa
        E_i = self.config.synaptic_params.e_gaba


        return np.sqrt(
            (s_e / g_0) ** 2 * (E_e - E_0) ** 2 * (tau_e / (tau_e + tau_0)) + (s_i / g_0) ** 2 * (E_i - E_0) ** 2 * (
                    tau_i / (tau_i + tau_0)) + (s_n / g_0)**2 * (E_e - E_0)**2 * ( tau_n / (tau_n + tau_0))
        )

    def tau_eff(self):
        return self.config.neuron_params.C / self.mean_total_conductance()

    def tau_eff_with_nmda(self):
        return self.config.neuron_params.C / self.mean_total_conductance_with_nmda()

    def crazy_s_nmda_mean_and_variance(self):

        tau_rise = self.config.synaptic_params.tau_nmda_rise
        tau_decay = self.config.synaptic_params.tau_nmda_decay
        alpha = self.config.synaptic_params.alpha_nmda
        """
        Returns mean and variance of NMDA gating variable s.
        """
        mu_x = self.mean_x_nmda()
        mu_s = (alpha * tau_decay * mu_x) / (1 + alpha * tau_decay * mu_x)

        var_x = self.std_x_nmda()**2

        # Effective rate constant for s fluctuations
        gamma = 1 / tau_decay + alpha * mu_x

        # Simplified variance formula (from solving covariance)
        # This is an approximation valid for small fluctuations
        var_s = (alpha * (1 - mu_s)) ** 2 * var_x * tau_rise / (gamma * tau_rise + 1)

        return mu_s, var_s
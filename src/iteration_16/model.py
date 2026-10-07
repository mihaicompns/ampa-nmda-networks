import math
from loguru import logger

from dataclasses import dataclass, field, replace

import numpy as np
from brian2 import (
    ms,
    mV,
    nS,
    Hz,
    Quantity, Mohm, kHz, nF, is_dimensionless, second,
)

WANG_MODEL = """
dv/dt = 1/C * (-g_L * (v - E_L) -g_ampa * (v - e_ampa) -g_gaba * (v - e_gaba) - g_nmda_max * s_nmda * sigma_of_v * (v - e_nmda)): volt

dg_ampa/dt = -g_ampa / tau_ampa : siemens
dg_gaba/dt = -g_gaba / tau_gaba : siemens

dx_nmda/dt = -x_nmda / tau_nmda_rise : 1
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha_nmda * x_nmda * (1 - s_nmda): 1

sigma_of_v = 1 / (1 + exp(-0.062 * v / mV) * (mg_concentration / 3.57)) : 1
"""

WANG_MODEL_FOR_FULL_NMDA_INPUT = """
dv/dt = 1/C * (-g_L * (v - E_L) -g_ampa * (v - e_ampa) -g_gaba * (v - e_gaba) - g_nmda_max * s_nmda * (v - e_nmda)): volt

dg_ampa/dt = -g_ampa / tau_ampa : siemens
dg_gaba/dt = -g_gaba / tau_gaba : siemens

dx_nmda/dt = -x_nmda / tau_nmda_rise : 1
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha_nmda * x_nmda * (1 - s_nmda): 1
"""


def copy_of(arr: np.ndarray):
    return None if arr is None else arr.copy()


class Results:

    def __init__(self, mu_v: Quantity, sigma_v: Quantity):
        if is_dimensionless(mu_v):
            mu_v = mu_v * mV
        if is_dimensionless(sigma_v):
            sigma_v = sigma_v * mV

        self.mu_v = mu_v
        self.sigma_v = sigma_v


chapter1Results = Results(mu_v=- 47.61595645 * mV, sigma_v=1.9 * mV)
chapter1Result_tau_m_20_ms = Results(mu_v=-46.75153051 * mV, sigma_v=1.77489008 * mV)

from enum import Enum
class MeanfieldScaling(Enum):
    NONE = "none"
    WEAK = "weak"
    STRONG = "strong"

@dataclass(frozen=True)
class ConductanceDiffusionSimulationConfig:
    # simulation
    simulation_time: Quantity = field(default_factory=lambda: 500 * ms)
    dt: Quantity = field(default_factory=lambda: 0.05 * ms)

    model: str = WANG_MODEL

    # membrane
    membrane_resistance: Quantity = field(default_factory=lambda: 50 * Mohm)  # Ohm
    membrane_capacitance: Quantity = field(default_factory=lambda: 0.5 * nF)  # value from Wang is 0.5 * nF

    e_L: Quantity = field(default_factory=lambda: -65 * mV)
    resting_voltage: Quantity = field(default_factory=lambda: -65 * mV)

    # synaptic reversals
    e_ampa: Quantity = field(default_factory=lambda: 0 * mV)
    e_nmda: Quantity = field(default_factory=lambda: 0 * mV)
    e_gaba: Quantity = field(default_factory=lambda: -80 * mV)

    # conductance decay
    tau_ampa: Quantity = field(default_factory=lambda: 2 * ms)
    tau_gaba: Quantity = field(default_factory=lambda: 5 * ms)

    tau_nmda_rise: Quantity = field(default_factory=lambda: 2 * ms)
    tau_nmda_decay: Quantity = field(default_factory=lambda: 100 * ms)

    alpha_nmda: Quantity = field(default_factory=lambda: 0.5 * kHz)

    N: int | None = 1
    # in our manuscript, this is actually k
    k: float = 1

    N_E: int | None = None
    N_I: int | None = None

    k_comp: int = 1

    r_e: Quantity = field(default_factory=lambda: 0 * Hz)
    r_i: Quantity = field(default_factory=lambda: 0 * Hz)
    r_n: Quantity = field(default_factory=lambda: 0 * Hz)

    theta: Quantity = field(default_factory=lambda: -40 * mV)
    v_reset: Quantity = field(default_factory=lambda: -55 * mV)

    ampa_spike_times: np.ndarray | None = None
    gaba_spike_times: np.ndarray | None = None
    nmda_spike_times: np.ndarray | None = None

    g_L: Quantity = field(default_factory=lambda: 20 * nS)
    w_ampa: Quantity = field(default_factory=lambda: 0.5 * nS)
    w_gaba: Quantity = field(default_factory=lambda: 0.5 * nS)
    w_x: Quantity = field(default_factory=lambda: 1)

    g_nmda_max: Quantity = field(default_factory=lambda: 1 * nS)

    magnesium_concentration: float = 1.0

    label: str = ""

    seed: int | None = None

    scaling: MeanfieldScaling = MeanfieldScaling.NONE

    # from script \frac{g_{i, 0}}{g_{e, 0}} =  \gamma \\
    def g(self):
        return self.tau_ampa * self.w_ampa * self.N_E

    def g_e0(self):
        return self.tau_ampa * self.w_ampa * self.r_e * self.N_E

    def g_i0(self):
        return self.tau_gaba * self.w_gaba * self.r_i * self.N_I

    def get_g_nmda_max(self):

        if self.scaling == MeanfieldScaling.NONE or self.k_comp == 0:
            return self.g_nmda_max
        elif self.scaling == MeanfieldScaling.WEAK:
            return self.g_nmda_max / self.k_comp
        elif self.scaling == MeanfieldScaling.STRONG:
            return self.g_nmda_max / (math.sqrt(self.k_comp))

    def tau_m(self):
        return self.membrane_capacitance / self.g_L

    def tau_0(self):
        return self.membrane_capacitance / (self.g_L + self.g_e0() + self.g_i0())

    def with_property(self, **changes):

        if "k" in changes and ("N_E" or "N_I") in changes:
            raise ValueError("Either (N, k) or (N E, N I)")

        if changes.__contains__("N") or changes.__contains__("k"):
            changes["N_E"] = None,
            changes["N_I"] = None

        elif changes.__contains__("N_E"):
            N_E = changes["N_E"]
            changes["N"] = None
            if "N_I" not in changes:
                changes["N_I"] = int(N_E / self.k)

        if "ampa_spike_times" in changes:
            changes["ampa_spike_times"] = copy_of(changes["ampa_spike_times"])
        else:
            changes["ampa_spike_times"] = copy_of(self.ampa_spike_times)

        if "gaba_spike_times" in changes:
            changes["gaba_spike_times"] = copy_of(changes["gaba_spike_times"])
        else:
            changes["gaba_spike_times"] = copy_of(self.gaba_spike_times)

        if "nmda_spike_times" in changes:
            changes["nmda_spike_times"] = copy_of(changes["nmda_spike_times"])
        else:
            changes["nmda_spike_times"] = copy_of(self.nmda_spike_times)

        return replace(
            self,
            **changes,
        )

    def __post_init__(self):

        if self.N_E is not None and self.N_I is not None:
            object.__setattr__(self, "N", self.N_E + self.N_I)
            if self.N_I > 0:
                object.__setattr__(self, "k", self.N_E / self.N_I)
        elif self.N is not None:
            N_I = int(self.N / (1 + self.k))
            N_E = self.N - N_I

            object.__setattr__(self, "N_E", N_E)
            object.__setattr__(self, "N_I", N_I)

        else:
            raise ValueError("Must provide either (N, gamma) or (N_E, N_I)")

    def with_fitted_solution(self, gr: float, gamma: float):
        # if we fix w_e, w_i and also ratio inhibition to excitation, the only thing we can adapt is the ratio of the firing rate
        logger.debug("Is k = {} the same as N E / NI {}? Difference is {}", self.k, self.N_E / self.N_I,
                     self.k - self.N_E / self.N_I)
        p = self.tau_ampa * self.w_ampa / (self.tau_gaba * self.w_gaba)
        q = gamma * self.k * p

        rate_e = gr / (self.g() / nS)
        rate_i = q * rate_e
        return self.with_property(r_e=rate_e, r_i=rate_i)

config_with_tau_m_10 = ConductanceDiffusionSimulationConfig(
    g_L = 50 * nS,
    w_ampa=2.399104714393616 * nS,
    w_gaba=5.077383026480675 * nS,
    g_nmda_max=4.074575871229172 * nS,
    N_E=1,
    N_I=1,
)

config_with_tau_m_20 = ConductanceDiffusionSimulationConfig(
    g_L = 25 * nS,
    w_ampa=2.399104714393616 * nS,
    w_gaba=5.077383026480675 * nS,
    g_nmda_max=4.074575871229172 * nS,
    N_E=1,
    N_I=1,
)

config_with_weak_synapses = ConductanceDiffusionSimulationConfig(
    w_ampa=2.399104714393616 * nS,
    w_gaba=5.077383026480675 * nS,
    g_nmda_max=4.074575871229172 * nS,
    N_E=1,
    N_I=1,
    label="weak synapses"
)

config_with_intermediate_synapses = ConductanceDiffusionSimulationConfig(
    w_ampa=4.819055736064912 * nS,
    w_gaba=10.371847078204151 * nS,
    g_nmda_max=4.074575871229172 * nS, # I have set alpha = alpha / 2
    N_E=1,
    N_I=1,
    label="intermediate synapses"
)

config_with_medium_synapses = ConductanceDiffusionSimulationConfig(
    w_ampa=9.7228533 * nS,
    w_gaba=21.6878134 * nS,
    g_nmda_max=4.074575871229172 * nS,
    N_E=1,
    N_I=1,
    label="moderate synapses"
)

wang_config_recurrent_synapses = ConductanceDiffusionSimulationConfig(
    w_ampa=0.05 * nS,
    w_gaba=1.3 * nS,
    g_nmda_max=0.165 * nS,
    N_E=1600,
    N_I=400,
    label="recurrent ampa"
)

wang_config_external_ampa_synapses = ConductanceDiffusionSimulationConfig(
    w_ampa=2.1 * nS,
    w_gaba=1.3 * nS,
    g_nmda_max=0.165 * nS,
    N_E=1600,
    N_I=400,
    label="wang external ampa"
)

high_shunt_config = config_with_intermediate_synapses.with_property(N_E=1600, N_I=400, simulation_time=0.5 * second,
                                                              alpha_nmda=0.25 * kHz,
                                                              g_nmda_max=4 * 4.074575871229172 * nS).with_fitted_solution(gr= 534.14, gamma=1.45)

''' for pyramidal cells, g ext,AMPA = 2.1, g rec,AMPA = 0.05, gNMDA = 0.165, and g GABA = 1.3 '''
''' What are mean g_s in Wang? g_ampa = g ext,AMPA_0 * 2.4 kHz + g rec,AMPA_0'''


@dataclass
class WangSimulationResult:
    time_ms: np.ndarray

    membrane_voltage_mV: np.ndarray

    g_ampa_nS: np.ndarray
    g_gaba_nS: np.ndarray

    s_nmda: np.ndarray
    x_nmda: np.ndarray

    ampa_presyn_spikes: np.ndarray
    gaba_presyn_spikes: np.ndarray
    nmda_presyn_spikes: np.ndarray

    @classmethod
    def from_monitors(
            cls,
            state_monitor,
            ampa_spike_monitor,
            gaba_spike_monitor,
            nmda_spike_monitor):
        return cls(
            time_ms=state_monitor.t[:] / ms, membrane_voltage_mV=state_monitor.v[0] / mV,
            g_ampa_nS=state_monitor.g_ampa[0] / nS, g_gaba_nS=state_monitor.g_gaba[0] / nS,
            s_nmda=state_monitor.s_nmda[0], x_nmda=state_monitor.x_nmda[0],
            ampa_presyn_spikes=np.asarray(ampa_spike_monitor.t[:] / ms),
            gaba_presyn_spikes=np.asarray(gaba_spike_monitor.t[:] / ms),
            nmda_presyn_spikes=np.asarray(nmda_spike_monitor.t[:] / ms),
        )

    def ampa_spike_delta_v(self, spike_index: int = 0, window_ms: float = 10.0) -> float:
        return self._spike_delta_v(self.ampa_presyn_spikes, spike_index, window_ms)

    def gaba_spike_delta_v(self, spike_index: int = 0, window_ms: float = 30.0) -> float:
        return self._spike_delta_v(self.gaba_presyn_spikes, spike_index, window_ms)

    def nmda_spike_delta_v(self, spike_index: int = 0, window_ms: float = 300.0) -> float:
        return self._spike_delta_v(self.nmda_presyn_spikes, spike_index, window_ms)

    def _spike_delta_v(self, spike_times_ms: np.ndarray, spike_index: int, window_ms: float) -> float:
        """
        Estimate the ΔV produced by a single presynaptic spike.

        Returns:
            V_peak_after_spike - V_before_spike
        """

        if len(spike_times_ms) - 1 < spike_index:
            return

        dt_ms = self.time_ms[1] - self.time_ms[0]
        spike_time_ms = spike_times_ms[spike_index]

        # first sample at or after the spike
        index_of_spike = int(spike_time_ms / dt_ms) - 1

        if index_of_spike < 0:
            raise ValueError("Spike occurs before first voltage sample.")

        v_before = self.membrane_voltage_mV[index_of_spike]

        n_window = max(1, int(round(window_ms / dt_ms)))
        index_end_of_spike = min(len(self.time_ms), index_of_spike + n_window)

        v_segment = self.membrane_voltage_mV[index_of_spike:index_end_of_spike]

        dv_max = np.max(v_segment) - v_before
        dv_min = np.min(v_segment) - v_before

        # choose whichever excursion is larger
        return float(
            dv_max if abs(dv_max) >= abs(dv_min) else dv_min
        )

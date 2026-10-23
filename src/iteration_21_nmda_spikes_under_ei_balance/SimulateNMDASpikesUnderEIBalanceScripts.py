import json
import csv
import math
import sys
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from brian2.units import pfarad
from joblib import Parallel, delayed
from scipy.special import logsumexp, ndtr
from scipy.stats import kurtosis, norm, skew, shapiro
from brian2 import (
    Hz,
    Clock,
    Network,
    NeuronGroup,
    PoissonInput,
    StateMonitor,
    kHz,
    ms,
    mV,
    nF,
    nS,
    seed,
    start_scope,
)

from Plotting import show_plots_non_blocking

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iteration_16.SolveCubicEquation import (  # noqa: E402
    E_0,
    compute_siegert_firing_rate_for_fitted_solution,
    evaluate_sympy_solution,
    polynomial_r,
    sigma_sq,
)
from iteration_16.model import (  # noqa: E402
    ConductanceDiffusionSimulationConfig,
    chapter1Results,
    config_with_intermediate_synapses,
)
from iteration_16.simpy import bind_config_to_sympy_values  # noqa: E402
from iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20NullclineScripts import (  # noqa: E402
    read_fitted_apic20_parameters_from_csv,
)
from iteration_21_nmda_spikes_under_ei_balance.GeneralizedLambert import (  # noqa: E402
    find_fixed_points,
    fixed_point_current_slope,
)

ITERATION_16_SOLUTIONS_PATH = SRC_DIR / "iteration_16" / "solutions" / "solutions.txt"
OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"
DEBUG_DISTRIBUTION_DIR = Path(__file__).resolve().parent / "debug_distributions"

# Keep the stored names identical to the simulation result keys.
DEBUG_DISTRIBUTION_VARIABLES = {
    "effective_g_nmda_nS": "g_NMDA [nS]",
    "s_nmda": "s_NMDA [1]",
    "x_nmda": "x_NMDA [1]",
    "g_ampa_nS": "g_AMPA [nS]",
    "voltage_mV": "voltage [mV]",
}

# Voltage, g_AMPA, g_NMDA, s_NMDA, x_NMDA, in plot row order.
TRACE_DISTRIBUTION_GAUSSIAN_COMPONENTS = (2, 1, 1, 1, 2)

TRACE_MODEL_EQUATIONS = (
    (
        r"$C_m\,\frac{dV}{dt}=-g_L(V-E_L)-g_{AMPA}(V-E_E)-g_{NMDA}(V-E_E)$"
    ),
    (
        r"$\frac{dg_{AMPA}}{dt}=-\frac{g_{AMPA}}{\tau_{AMPA}}+w_{AMPA}\sum_k\delta(t-t_k)$"
    ),
    (
        r"$g_{NMDA}=g_{NMDA,max}s_{NMDA}\,\sigma_{Mg}(V) \sigma_{Mg}(V)=\left(1+e^{-0.08V/\mathrm{mV}}Mg/3.57\right)^{-1}$"
    ),
    r"$\frac{ds_{NMDA}}{dt}=-\frac{s_{NMDA}}{\tau_{decay}}+\alpha_{NMDA}x_{NMDA}$",
    (
        r"$\frac{dx_{NMDA}}{dt}=-\frac{x_{NMDA}}{\tau_{rise}}+w_{NMDA}\sum_k\delta(t-t_k)$"
    ),
)

TRACE_DYNAMIC_MAGNESIUM_MODEL_EQUATIONS = (
    TRACE_MODEL_EQUATIONS[0],
    TRACE_MODEL_EQUATIONS[1],
    (
        r"$g_{NMDA}=g_{NMDA,max}s_{NMDA}\,\sigma_{Mg}(t)$" "\n"
        r"$\tau_{\sigma}\frac{d\sigma_{Mg}}{dt}=\sigma_{\infty}(V)-\sigma_{Mg}$" "\n"
        r"$\sigma_{\infty}(V)=\left(1+e^{-0.08V/\mathrm{mV}}Mg/3.57\right)^{-1}$"
    ),
    TRACE_MODEL_EQUATIONS[3],
    TRACE_MODEL_EQUATIONS[4],
)


@dataclass(frozen=True)
class DebugHistogramSaveConfiguration:
    """Select the compact histogram debug files produced by the long run."""

    enabled: bool = True
    save_g_nmda: bool = False
    save_s_nmda: bool = False
    save_x_nmda: bool = False
    save_g_ampa: bool = True
    save_voltage: bool = False
    save_csv: bool = True
    save_json: bool = False
    bins: int = 200
    fit_sample_size: int = 500_000

    def selected_result_keys(self):
        flags = {
            "effective_g_nmda_nS": self.save_g_nmda,
            "s_nmda": self.save_s_nmda,
            "x_nmda": self.save_x_nmda,
            "g_ampa_nS": self.save_g_ampa,
            "voltage_mV": self.save_voltage,
        }
        return [key for key, selected in flags.items() if selected]


@dataclass(frozen=True)
class AmpaNmdaExperimentConfig:
    """Complete, reusable definition of one AMPA–NMDA ensemble experiment."""

    n_models: int
    n_e: int
    n_nmda: int
    simulation_time: object
    burn_in: object
    dt: object
    random_seed: int
    r_e: object
    C_m: object
    g_L: object
    E_L: object
    E_E: object
    tau_ampa: object
    w_ampa: object
    tau_nmda_rise: object
    tau_nmda_decay: object
    alpha_nmda: object
    g_nmda_max: object
    w_nmda: float
    magnesium: float
    magnesium_block_tau: object = None

    def with_property(self, **changes):
        """Return a copy with only the named configuration fields replaced."""
        return replace(self, **changes)

    @classmethod
    def from_fitted_apic20(cls, *, r_e=20 * Hz, magnesium_block_tau=None):
        fitted = read_fitted_apic20_parameters_from_csv()
        return cls(
            n_models=1_000, n_e=800, n_nmda=10,
            simulation_time=1_000 * ms, burn_in=500 * ms, dt=0.1 * ms,
            random_seed=21, r_e=r_e,
            C_m=fitted.capacitance_nF * nF,
            g_L=fitted.g_leak_nS * nS,
            E_L=fitted.e_leak_mV * mV,
            E_E=fitted.e_excitatory_mV * mV,
            tau_ampa=2 * ms, w_ampa=0.05 * nS,
            tau_nmda_rise=fitted.tau_rise_ms * ms,
            tau_nmda_decay=fitted.tau_decay_ms * ms,
            alpha_nmda=fitted.alpha_per_ms / ms,
            g_nmda_max=fitted.g_nmda_max_nS * nS,
            w_nmda=fitted.spike_weight, magnesium=1.0,
            magnesium_block_tau=magnesium_block_tau,
        )

    @classmethod
    def from_metadata(cls, metadata):
        value = lambda name: float(metadata[name])
        tau_sigma = metadata.get("magnesium_block_tau_ms", "")
        return cls(
            n_models=int(metadata["n_models"]), n_e=int(metadata["n_e"]),
            n_nmda=int(metadata["n_nmda"]),
            simulation_time=value("simulation_time_ms") * ms,
            burn_in=value("burn_in_ms") * ms, dt=value("dt_ms") * ms,
            random_seed=int(metadata["random_seed"]),
            r_e=value("input_rate_per_source_hz") * Hz,
            C_m=value("capacitance_nF") * nF, g_L=value("g_leak_nS") * nS,
            E_L=value("e_leak_mV") * mV, E_E=value("e_excitatory_mV") * mV,
            tau_ampa=value("tau_ampa_ms") * ms, w_ampa=value("w_ampa_nS") * nS,
            tau_nmda_rise=value("tau_nmda_rise_ms") * ms,
            tau_nmda_decay=value("tau_nmda_decay_ms") * ms,
            alpha_nmda=value("alpha_nmda_per_ms") / ms,
            g_nmda_max=value("g_nmda_max_nS") * nS,
            w_nmda=value("w_nmda"), magnesium=value("magnesium"),
            magnesium_block_tau=(float(tau_sigma) * ms if tau_sigma else None),
        )

    def build_metadata(self, result):
        model_name = "dynamic_magnesium" if self.magnesium_block_tau is not None else "instantaneous_magnesium"
        s0_theory = filtered_nmda_shot_noise_cumulants(
            float(self.n_nmda * self.r_e / Hz), float(self.tau_nmda_rise / ms),
            float(self.tau_nmda_decay / ms), float(self.alpha_nmda / kHz), self.w_nmda,
        )[0]
        g_ampa_theory = exponential_shot_noise_cumulants(
            float(self.n_e * self.r_e / Hz), float(self.tau_ampa / ms),
            float(self.w_ampa / nS),
        )[0]
        return {
            "model_name": model_name, "model_equations": result["model_equations"],
            "n_models": self.n_models, "n_e": self.n_e, "n_nmda": self.n_nmda,
            "simulation_time_ms": float(self.simulation_time / ms),
            "burn_in_ms": float(self.burn_in / ms), "dt_ms": float(self.dt / ms),
            "random_seed": self.random_seed,
            "input_rate_per_source_hz": float(self.r_e / Hz),
            "ampa_total_input_rate_hz": float(self.n_e * self.r_e / Hz),
            "nmda_total_input_rate_hz": float(self.n_nmda * self.r_e / Hz),
            "capacitance_nF": float(self.C_m / nF), "g_leak_nS": float(self.g_L / nS),
            "e_leak_mV": float(self.E_L / mV), "e_excitatory_mV": float(self.E_E / mV),
            "tau_ampa_ms": float(self.tau_ampa / ms), "w_ampa_nS": float(self.w_ampa / nS),
            "tau_nmda_rise_ms": float(self.tau_nmda_rise / ms),
            "tau_nmda_decay_ms": float(self.tau_nmda_decay / ms),
            "alpha_nmda_per_ms": float(self.alpha_nmda / kHz),
            "g_nmda_max_nS": float(self.g_nmda_max / nS), "w_nmda": self.w_nmda,
            "magnesium": self.magnesium,
            "magnesium_block_tau_ms": result["magnesium_block_tau_ms"],
            "theoretical_s0_mean": s0_theory,
            "theoretical_g_ampa_0_nS_mean": g_ampa_theory,
            "empirical_ensemble_voltage_mean_mV": result["ensemble_mean_mV"],
            "empirical_ensemble_voltage_variance_mV2": result["ensemble_variance_mV2"],
            "voltage_histogram_sample_size": min(result["voltage_mV"].size, 500_000),
            "voltage_histogram_sample_seed": 0, "histogram_bin_count": 150,
            "fixed_point_normalization": "joint lower+upper probability mass",
        }


@dataclass(frozen=True)
class AmpaNmdaExperimentConfigFactoryBuilder:
    """Immutable builder for explicit experiment variants in tests/runnables."""

    configuration: AmpaNmdaExperimentConfig

    @classmethod
    def fitted_apic20(cls):
        return cls(AmpaNmdaExperimentConfig.from_fitted_apic20())

    def with_property(self, **changes):
        return replace(
            self,
            configuration=self.configuration.with_property(**changes),
        )

    def build(self):
        return self.configuration


# Edit this configuration before running the 1,000-compartment runnable.
# The camelCase name is retained because this is the requested runnable flag.
saveDebugFile = DebugHistogramSaveConfiguration(
    enabled=True,
    save_g_nmda=False,
    save_s_nmda=False,
    save_x_nmda=False,
    save_g_ampa=True,
    save_voltage=True,
    save_csv=True,
    save_json=False,
    bins=200,
)

saveAMPAFile = DebugHistogramSaveConfiguration(
    enabled=True,
    save_g_nmda=False,
    save_s_nmda=False,
    save_x_nmda=False,
    save_g_ampa=True,
    save_voltage=False,
    save_csv=True,
    save_json=False,
    bins=500,
)


# Kept as a named, inspectable Brian2 equation string.  Every dynamic variable
# and subexpression declares its physical unit.
AMPA_NMDA_SINGLE_COMPARTMENT_MODEL = """
dV/dt = (-I_leak - I_AMPA - I_NMDA) / C_m : volt

dg_e/dt = -g_e / tau_AMPA : siemens
dx_NMDA/dt = -x_NMDA / tau_NMDA_rise : 1
ds_NMDA/dt = -s_NMDA / tau_NMDA_decay
             + alpha_NMDA * x_NMDA : 1

sigma_Mg_inf = 1 / (1 + exp(-0.08 * V / mV) * Mg / 3.57) : 1
sigma_Mg = sigma_Mg_inf : 1
g_NMDA_open = g_NMDA_max * s_NMDA : siemens
g_NMDA = g_NMDA_open * sigma_Mg : siemens

I_leak = g_L * (V - E_L) : amp
I_AMPA = g_e * (V - E_E) : amp
I_NMDA = g_NMDA * (V - E_E) : amp
"""


AMPA_NMDA_DYNAMIC_MAGNESIUM_MODEL = """
dV/dt = (-I_leak - I_AMPA - I_NMDA) / C_m : volt

dg_e/dt = -g_e / tau_AMPA : siemens
dx_NMDA/dt = -x_NMDA / tau_NMDA_rise : 1
ds_NMDA/dt = -s_NMDA / tau_NMDA_decay
             + alpha_NMDA * x_NMDA : 1

sigma_Mg_inf = 1 / (1 + exp(-0.08 * V / mV) * Mg / 3.57) : 1
dsigma_Mg/dt = (sigma_Mg_inf - sigma_Mg) / tau_sigma_Mg : 1
g_NMDA_open = g_NMDA_max * s_NMDA : siemens
g_NMDA = g_NMDA_open * sigma_Mg : siemens

I_leak = g_L * (V - E_L) : amp
I_AMPA = g_e * (V - E_E) : amp
I_NMDA = g_NMDA * (V - E_E) : amp
"""


def simulate_ampa_nmda_single_compartment_ensemble(experiment):
    """Simulate independent copies and return voltage samples and statistics.

    ``n_e * r_e`` and ``n_NMDA * r_e`` are superposed Poisson rates for each
    compartment.  Brian2's ``PoissonInput`` gives every compartment an
    independent realization, so all 1,000 copies can be integrated together.
    """
    n_models, n_e, n_NMDA = experiment.n_models, experiment.n_e, experiment.n_nmda
    simulation_time, burn_in, dt = experiment.simulation_time, experiment.burn_in, experiment.dt
    random_seed, r_e = experiment.random_seed, experiment.r_e
    C_m, g_L, E_L, E_E = experiment.C_m, experiment.g_L, experiment.E_L, experiment.E_E
    tau_AMPA, w_AMPA = experiment.tau_ampa, experiment.w_ampa
    tau_NMDA_rise, tau_NMDA_decay = experiment.tau_nmda_rise, experiment.tau_nmda_decay
    alpha_NMDA, g_NMDA_max = experiment.alpha_nmda, experiment.g_nmda_max
    w_NMDA, Mg = experiment.w_nmda, experiment.magnesium
    magnesium_block_tau = experiment.magnesium_block_tau
    start_scope()
    seed(random_seed)
    clock = Clock(dt=dt)
    g_NMDA = g_NMDA_max / n_NMDA
    dynamic_magnesium_block = magnesium_block_tau is not None
    model_equations = (
        AMPA_NMDA_DYNAMIC_MAGNESIUM_MODEL
        if dynamic_magnesium_block
        else AMPA_NMDA_SINGLE_COMPARTMENT_MODEL
    )
    namespace = {
        "C_m": C_m,
        "g_L": g_L,
        "E_L": E_L,
        "E_E": E_E,
        "tau_AMPA": tau_AMPA,
        "tau_NMDA_rise": tau_NMDA_rise,
        "tau_NMDA_decay": tau_NMDA_decay,
        "alpha_NMDA": alpha_NMDA,
        "g_NMDA_max": g_NMDA,
        "Mg": Mg,
    }
    if dynamic_magnesium_block:
        namespace["tau_sigma_Mg"] = magnesium_block_tau
    compartments = NeuronGroup(
        n_models,
        model_equations,
        method="euler",
        namespace=namespace,
        clock=clock,
        name="single_compartments",
    )
    compartments.V = E_L
    compartments.g_e = 0 * nS
    compartments.x_NMDA = 0
    compartments.s_NMDA = 0
    if dynamic_magnesium_block:
        resting_sigma_mg = 1.0 / (
            1.0 + np.exp(-0.08 * float(E_L / mV)) * float(Mg) / 3.57
        )
        compartments.sigma_Mg = resting_sigma_mg

    ampa_input = PoissonInput(
        compartments, "g_e", N=n_e, rate=r_e, weight=w_AMPA
    )
    nmda_input = PoissonInput(
        compartments, "x_NMDA", N=n_NMDA, rate=r_e, weight=w_NMDA
    )
    state = StateMonitor(
        compartments,
        ["V", "g_e", "g_NMDA", "s_NMDA", "x_NMDA", "sigma_Mg", "sigma_Mg_inf"],
        record=True,
        clock=clock,
    )
    Network(compartments, ampa_input, nmda_input, state).run(simulation_time)

    sample_mask = state.t >= burn_in
    voltage_mV = np.asarray(state.V[:, sample_mask] / mV)
    g_ampa_nS = np.asarray(state.g_e[:, sample_mask] / nS)
    effective_g_nmda_nS = np.asarray(state.g_NMDA[:, sample_mask] / nS)
    s_nmda = np.asarray(state.s_NMDA[:, sample_mask])
    x_nmda = np.asarray(state.x_NMDA[:, sample_mask])
    sigma_mg = np.asarray(state.sigma_Mg[:, sample_mask])
    sigma_mg_inf = np.asarray(state.sigma_Mg_inf[:, sample_mask])
    per_model_mean_mV = np.mean(voltage_mV, axis=1)
    per_model_variance_mV2 = np.var(voltage_mV, axis=1)
    return {
        "model_equations": model_equations,
        "trace_model_equations": (
            TRACE_DYNAMIC_MAGNESIUM_MODEL_EQUATIONS
            if dynamic_magnesium_block
            else TRACE_MODEL_EQUATIONS
        ),
        "magnesium_block_tau_ms": (
            float(magnesium_block_tau / ms) if dynamic_magnesium_block else None
        ),
        "time_ms": np.asarray(state.t[sample_mask] / ms),
        "voltage_mV": voltage_mV,
        "g_ampa_nS": g_ampa_nS,
        "effective_g_nmda_nS": effective_g_nmda_nS,
        "s_nmda": s_nmda,
        "x_nmda": x_nmda,
        "sigma_mg": sigma_mg,
        "sigma_mg_inf": sigma_mg_inf,
        "per_model_mean_mV": per_model_mean_mV,
        "per_model_variance_mV2": per_model_variance_mV2,
        "ensemble_mean_mV": float(np.mean(voltage_mV)),
        "ensemble_variance_mV2": float(np.var(voltage_mV)),
    }


def estimate_full_text_json_size(result, sample_size=10_000):
    """Estimate full-precision JSON size without materializing huge lists.

    JSON has no native float array representation, so its size depends on the
    decimal spellings of the values.  We measure those spellings on an evenly
    spaced sample and add one comma per value.  ``binary_bytes`` is exact for
    the five NumPy arrays and is the useful comparison for the debug files.
    """
    number_of_float_values = 0
    binary_bytes = 0
    estimated_json_bytes = 2  # outer braces
    by_variable = {}
    for key, label in DEBUG_DISTRIBUTION_VARIABLES.items():
        values = np.asarray(result[key])
        flat = values.ravel()
        count = int(flat.size)
        number_of_float_values += count
        binary_bytes += int(values.nbytes)
        if count:
            indices = np.linspace(
                0, count - 1, min(count, int(sample_size)), dtype=np.int64
            )
            encoded_lengths = [len(json.dumps(float(value))) for value in flat[indices]]
            average_float_bytes = float(np.mean(encoded_lengths))
        else:
            average_float_bytes = 0.0
        # Includes commas and a modest allowance for row brackets and the key.
        array_bytes = int(round(count * (average_float_bytes + 1.0)))
        array_bytes += int(values.shape[0] * 2 if values.ndim == 2 else 2)
        array_bytes += len(json.dumps(key)) + 2
        estimated_json_bytes += array_bytes
        by_variable[key] = {
            "label": label,
            "shape": list(values.shape),
            "number_of_float_values": count,
            "binary_bytes": int(values.nbytes),
            "estimated_json_bytes": array_bytes,
        }
    return {
        "number_of_float_values": number_of_float_values,
        "binary_bytes": binary_bytes,
        "estimated_json_bytes": estimated_json_bytes,
        "variables": by_variable,
    }


def format_storage_size(number_of_bytes):
    """Return a compact decimal/GiB storage-size description."""
    return f"{number_of_bytes / 1e9:.2f} GB ({number_of_bytes / 2**30:.2f} GiB)"


def save_debug_histograms(result, output_dir, configuration):
    """Save selected histograms and balanced raw samples for later fitting."""
    if not configuration.enabled:
        return {}
    if not configuration.save_csv and not configuration.save_json:
        raise ValueError("Enable save_csv, save_json, or both")
    if configuration.bins < 2:
        raise ValueError("Histogram bins must be at least 2")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    for key in configuration.selected_result_keys():
        label = DEBUG_DISTRIBUTION_VARIABLES[key]
        values = np.asarray(result[key])
        counts, bin_edges = np.histogram(values.ravel(), bins=configuration.bins)
        raw_fit_sample = sample_original_values_by_compartment(
            values,
            sample_size=configuration.fit_sample_size,
        )
        paths = []
        if configuration.save_json:
            json_path = output_dir / f"{key}_histogram.json"
            json_path.write_text(
                json.dumps(
                    {
                        "variable": key,
                        "label": label,
                        "counts": counts.tolist(),
                        "bin_edges": bin_edges.tolist(),
                    },
                    indent=2,
                )
            )
            paths.append(json_path)
            raw_json_path = output_dir / f"{key}_fit_sample.json"
            raw_json_path.write_text(
                json.dumps(
                    {
                        "variable": key,
                        "label": label,
                        "values": raw_fit_sample.tolist(),
                    }
                )
            )
        if configuration.save_csv:
            csv_path = output_dir / f"{key}_histogram.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["bin_left", "bin_right", "count"])
                writer.writerows(zip(bin_edges[:-1], bin_edges[1:], counts))
            paths.append(csv_path)
            raw_csv_path = output_dir / f"{key}_fit_sample.csv"
            np.savetxt(
                raw_csv_path,
                raw_fit_sample,
                delimiter=",",
                header="value",
                comments="",
            )
        saved_paths[key] = paths
    return saved_paths


def sample_original_values_by_compartment(values, sample_size=500_000, random_seed=21):
    """Draw a balanced sample of original values across model compartments."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("Expected a models-by-time array")
    n_models, n_times = values.shape
    sample_size = min(int(sample_size), int(values.size))
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    allocations = np.full(n_models, sample_size // n_models, dtype=int)
    allocations[: sample_size % n_models] += 1
    rng = np.random.default_rng(random_seed)
    samples = []
    for model_index, allocation in enumerate(allocations):
        if allocation:
            time_indices = rng.choice(n_times, size=allocation, replace=False)
            samples.append(values[model_index, time_indices])
    return np.concatenate(samples).astype(float, copy=False)


def load_debug_fit_sample(sample_path, variable_name):
    """Load original, non-binned observations saved for Gaussian fitting."""
    sample_path = Path(sample_path)
    if sample_path.suffix.lower() == ".csv":
        values = np.loadtxt(sample_path, delimiter=",", skiprows=1)
    elif sample_path.suffix.lower() == ".json":
        data = json.loads(sample_path.read_text())
        if data["variable"] != variable_name:
            raise ValueError(f"File contains {data['variable']!r}, not {variable_name!r}")
        values = np.asarray(data["values"], dtype=float)
    else:
        raise ValueError("Fit-sample file must have a .csv or .json suffix")
    return np.asarray(values, dtype=float).ravel()


def fit_saved_debug_sample(
    sample_path,
    variable_name,
    output_path=None,
    number_of_gaussians=2,
):
    """Fit and plot one saved raw sample without histogram reconstruction."""
    values = load_debug_fit_sample(sample_path, variable_name)
    fit = fit_gaussian_mixture(
        values,
        number_of_gaussians=number_of_gaussians,
        max_samples=values.size,
    )
    shapiro_values = values[
        np.linspace(0, values.size - 1, min(values.size, 5_000), dtype=np.int64)
    ]
    shapiro_result = shapiro(shapiro_values)
    fit["shapiro_w"] = float(shapiro_result.statistic)
    fit["shapiro_p_value"] = float(shapiro_result.pvalue)
    fit["shapiro_sample_size"] = int(shapiro_values.size)
    sample_mean = float(np.mean(values))
    sample_standard_deviation = float(np.std(values))
    if sample_standard_deviation <= np.finfo(float).eps:
        raise ValueError("Gaussianity measures require non-constant values")
    standardized_values = (values - sample_mean) / sample_standard_deviation
    fit["skewness"] = float(skew(standardized_values, bias=False))
    fit["excess_kurtosis"] = float(
        kurtosis(standardized_values, fisher=True, bias=False)
    )
    standardized_density, standardized_edges = np.histogram(
        standardized_values,
        bins=200,
        density=True,
    )
    standardized_centers = 0.5 * (
        standardized_edges[:-1] + standardized_edges[1:]
    )
    standard_normal_density = norm.pdf(standardized_centers)
    fit["standardized_histogram_rmse"] = float(
        np.sqrt(np.mean((standardized_density - standard_normal_density) ** 2))
    )
    if output_path is None:
        output_path = Path(sample_path).with_name(f"fit_{variable_name}.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, (distribution_axis, qq_axis) = plt.subplots(1, 2, figsize=(14, 6))
    distribution_axis.hist(
        values, bins=200, density=True, color="0.85", label="raw sample"
    )
    for component_index in range(number_of_gaussians):
        distribution_axis.plot(
            fit["grid"],
            fit["component_density"][:, component_index],
            linestyle=":",
            label=f"Gaussian {component_index + 1}",
        )
    distribution_axis.plot(
        fit["grid"], fit["mixture_density"], linewidth=2, label="mixture"
    )
    interpretation = (
        r"single Gaussian $\bf{plausible}$"
        if fit["shapiro_p_value"] >= 0.05
        else r"single Gaussian $\bf{implausible}$"
    )
    distribution_axis.set_xlabel(DEBUG_DISTRIBUTION_VARIABLES[variable_name])
    distribution_axis.set_ylabel("density")
    distribution_axis.set_title(
        f"Distribution and {number_of_gaussians}-Gaussian mixture"
    )
    distribution_axis.legend()

    qq_sample_size = min(standardized_values.size, 20_000)
    qq_indices = np.linspace(
        0, standardized_values.size - 1, qq_sample_size, dtype=np.int64
    )
    ordered_standardized_values = np.sort(standardized_values)[qq_indices]
    probabilities = (np.arange(qq_sample_size) + 0.5) / qq_sample_size
    theoretical_quantiles = norm.ppf(probabilities)
    qq_axis.scatter(
        theoretical_quantiles,
        ordered_standardized_values,
        s=5,
        alpha=0.35,
        rasterized=True,
    )
    qq_limits = [
        min(theoretical_quantiles[0], ordered_standardized_values[0]),
        max(theoretical_quantiles[-1], ordered_standardized_values[-1]),
    ]
    qq_axis.plot(qq_limits, qq_limits, "k--", linewidth=1, label="normal reference")
    qq_axis.set_xlim(qq_limits)
    qq_axis.set_ylim(qq_limits)
    qq_axis.set_xlabel("theoretical standard-normal quantiles")
    qq_axis.set_ylabel("observed standardized quantiles")
    qq_axis.set_title(f"Normal Q–Q plot (n={qq_sample_size:,})")
    qq_axis.legend()

    figure.suptitle(
        f"{DEBUG_DISTRIBUTION_VARIABLES[variable_name]} Gaussianity\n"
        f"Shapiro–Wilk: W={fit['shapiro_w']:.5f}, p={fit['shapiro_p_value']:.3g}, "
        f"n={fit['shapiro_sample_size']:,} — {interpretation} at α=0.05; \n"
        f"standardized histogram RMSE={fit['standardized_histogram_rmse']:.4g}; "
        f"skewness={fit['skewness']:.4g}; "
        f"excess kurtosis={fit['excess_kurtosis']:.4g}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path, fit


def load_debug_histogram(histogram_path, variable_name):
    """Load one variable's histogram without touching the other distributions."""
    histogram_path = Path(histogram_path)
    if variable_name not in DEBUG_DISTRIBUTION_VARIABLES:
        raise ValueError(
            f"Unknown variable {variable_name!r}; choose from "
            f"{tuple(DEBUG_DISTRIBUTION_VARIABLES)}"
        )
    if histogram_path.suffix.lower() == ".json":
        histogram = json.loads(histogram_path.read_text())
        if histogram["variable"] != variable_name:
            raise ValueError(
                f"File contains {histogram['variable']!r}, not {variable_name!r}"
            )
        counts = np.asarray(histogram["counts"], dtype=np.int64)
        bin_edges = np.asarray(histogram["bin_edges"], dtype=float)
        label = histogram["label"]
    elif histogram_path.suffix.lower() == ".csv":
        table = np.genfromtxt(histogram_path, delimiter=",", names=True)
        table = np.atleast_1d(table)
        counts = np.asarray(table["count"], dtype=np.int64)
        bin_edges = np.r_[table["bin_left"], table["bin_right"][-1]].astype(float)
        label = DEBUG_DISTRIBUTION_VARIABLES[variable_name]
    else:
        raise ValueError("Histogram file must have a .csv or .json suffix")
    return {
        "label": label,
        "counts": counts,
        "bin_edges": bin_edges,
    }


def fit_saved_debug_histogram(
    histogram_path,
    variable_name,
    output_path=None,
    max_fit_samples=500_000,
    number_of_gaussians=2,
):
    """Fit and plot one saved histogram without rerunning the simulation."""
    histogram = load_debug_histogram(histogram_path, variable_name)
    counts = histogram["counts"]
    bin_edges = histogram["bin_edges"]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total_count = int(np.sum(counts))
    if total_count < 5:
        raise ValueError("At least five histogram observations are required")

    # Deterministic proportional reconstruction of the binned empirical
    # distribution using at most 500k bin-centre observations. Integer sample
    # allocations use the largest-remainder apportionment method.
    fit_sample_count = min(total_count, int(max_fit_samples))
    exact_repetitions = counts.astype(float) * fit_sample_count / total_count
    repetitions = np.floor(exact_repetitions).astype(int)
    missing = fit_sample_count - int(np.sum(repetitions))
    if missing:
        residual_order = np.argsort(exact_repetitions - repetitions)[::-1]
        repetitions[residual_order[:missing]] += 1
    fit_values = np.repeat(bin_centers, repetitions)
    fit = fit_gaussian_mixture(
        fit_values,
        max_samples=max_fit_samples,
        number_of_gaussians=number_of_gaussians,
    )
    shapiro_sample_size = min(fit_values.size, 5_000)
    shapiro_indices = np.linspace(
        0,
        fit_values.size - 1,
        shapiro_sample_size,
        dtype=np.int64,
    )
    shapiro_result = shapiro(np.sort(fit_values)[shapiro_indices])
    fit["shapiro_w"] = float(shapiro_result.statistic)
    fit["shapiro_p_value"] = float(shapiro_result.pvalue)
    fit["shapiro_sample_size"] = int(shapiro_sample_size)

    if output_path is None:
        output_path = Path(histogram_path).with_name(f"fit_{variable_name}.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    widths = np.diff(bin_edges)
    density = counts / (total_count * widths)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.stairs(density, bin_edges, fill=True, color="0.85", label="saved histogram")
    for component_index in range(number_of_gaussians):
        axis.plot(
            fit["grid"],
            fit["component_density"][:, component_index],
            linestyle=":",
            label=f"Gaussian {component_index + 1}",
        )
    axis.plot(fit["grid"], fit["mixture_density"], linewidth=2, label="mixture")
    axis.set_xlabel(histogram["label"])
    axis.set_ylabel("density")
    normality_interpretation = (
        "single Gaussian not rejected"
        if fit["shapiro_p_value"] >= 0.05
        else "single Gaussian rejected"
    )
    axis.set_title(
        f"{number_of_gaussians}-Gaussian fit: {histogram['label']}\n"
        f"Shapiro–Wilk: W={fit['shapiro_w']:.5f}, "
        f"p={fit['shapiro_p_value']:.3g}, n={fit['shapiro_sample_size']:,} — "
        f"{normality_interpretation} at α=0.05"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path, fit


def fit_gaussian_mixture(
    values,
    number_of_gaussians=2,
    random_seed=21,
    max_samples=500_000,
):
    """Fit a 1-D Gaussian mixture and score it on held-out samples.

    Several quantile-based initializations are tried because the distributions
    produced here are usually one broad Gaussian with a small lower shoulder.
    A single EM initialization can split the broad peak instead of describing
    that shoulder, even though both solutions are local likelihood maxima.
    """
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not isinstance(number_of_gaussians, (int, np.integer)):
        raise TypeError("number_of_gaussians must be an integer")
    if number_of_gaussians < 1:
        raise ValueError("number_of_gaussians must be at least 1")
    if values.size < max(5, number_of_gaussians + 1):
        raise ValueError(
            f"At least {max(5, number_of_gaussians + 1)} finite samples are required"
        )
    rng = np.random.default_rng(random_seed)
    if values.size > max_samples:
        values = values[rng.choice(values.size, max_samples, replace=False)]
    rng.shuffle(values)
    split = max(2, int(0.8 * values.size))
    train, test = values[:split], values[split:]
    overall_std = max(float(np.std(train)), np.finfo(float).eps)
    minimum_std = max(overall_std * 1e-3, np.finfo(float).eps)
    best_fit = None
    percentile_step = 100.0 / number_of_gaussians
    base_percentiles = np.linspace(
        percentile_step / 2,
        100.0 - percentile_step / 2,
        number_of_gaussians,
    )
    percentile_initializations = (
        np.clip(base_percentiles + shift * percentile_step, 1.0, 99.0)
        for shift in (-0.2, 0.0, 0.2)
    )
    for percentiles in percentile_initializations:
        means = np.percentile(train, percentiles)
        standard_deviations = np.full(number_of_gaussians, overall_std)
        weights = np.full(number_of_gaussians, 1.0 / number_of_gaussians)

        for _ in range(200):
            old_parameters = np.r_[weights, means, standard_deviations]
            log_components = (
                np.log(np.maximum(weights, np.finfo(float).eps))[None, :]
                - np.log(standard_deviations)[None, :]
                - 0.5 * np.log(2 * np.pi)
                - 0.5
                * ((train[:, None] - means[None, :]) / standard_deviations[None, :]) ** 2
            )
            responsibilities = np.exp(
                log_components - logsumexp(log_components, axis=1)[:, None]
            )
            component_mass = np.maximum(
                np.sum(responsibilities, axis=0), np.finfo(float).eps
            )
            weights = component_mass / np.sum(component_mass)
            means = np.sum(responsibilities * train[:, None], axis=0) / component_mass
            variances = (
                np.sum(responsibilities * (train[:, None] - means) ** 2, axis=0)
                / component_mass
            )
            standard_deviations = np.maximum(np.sqrt(variances), minimum_std)
            if np.allclose(
                old_parameters,
                np.r_[weights, means, standard_deviations],
                rtol=1e-7,
                atol=1e-9,
            ):
                break

        train_log_density = logsumexp(
            np.log(np.maximum(weights, np.finfo(float).eps))[None, :]
            - np.log(standard_deviations)[None, :]
            - 0.5 * np.log(2 * np.pi)
            - 0.5
            * ((train[:, None] - means[None, :]) / standard_deviations[None, :]) ** 2,
            axis=1,
        )
        train_log_likelihood = float(np.sum(train_log_density))
        if best_fit is None or train_log_likelihood > best_fit[0]:
            best_fit = (
                train_log_likelihood,
                weights.copy(),
                means.copy(),
                standard_deviations.copy(),
            )

    train_log_likelihood, weights, means, standard_deviations = best_fit
    component_order = np.argsort(means)
    weights = weights[component_order]
    means = means[component_order]
    standard_deviations = standard_deviations[component_order]

    def mixture_log_density(samples):
        log_components = (
            np.log(weights)[None, :]
            - np.log(standard_deviations)[None, :]
            - 0.5 * np.log(2 * np.pi)
            - 0.5
            * ((samples[:, None] - means[None, :]) / standard_deviations[None, :]) ** 2
        )
        return logsumexp(log_components, axis=1)

    held_out_mean_log_likelihood = float(np.mean(mixture_log_density(test)))
    sorted_test = np.sort(test)
    fitted_cdf = np.sum(
        weights[None, :]
        * ndtr(
            (sorted_test[:, None] - means[None, :])
            / standard_deviations[None, :]
        ),
        axis=1,
    )
    empirical_cdf = (np.arange(sorted_test.size) + 0.5) / sorted_test.size
    cdf_max_error = float(np.max(np.abs(empirical_cdf - fitted_cdf)))
    grid = np.linspace(float(np.min(values)), float(np.max(values)), 400)
    component_density = (
        weights[None, :]
        / (standard_deviations[None, :] * np.sqrt(2 * np.pi))
        * np.exp(
            -0.5
            * ((grid[:, None] - means[None, :]) / standard_deviations[None, :]) ** 2
        )
    )
    return {
        "weights": weights,
        "means": means,
        "standard_deviations": standard_deviations,
        "grid": grid,
        "component_density": component_density,
        "mixture_density": np.sum(component_density, axis=1),
        "held_out_mean_log_likelihood": held_out_mean_log_likelihood,
        "bic": -2 * train_log_likelihood
        + (3 * number_of_gaussians - 1) * np.log(train.size),
        "cdf_max_error": cdf_max_error,
    }


def fit_single_exponential(values, random_seed=21, max_samples=20_000):
    """Fit an exponential distribution on non-negative values by MLE."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values) & (values >= 0.0)]
    if values.size < 3:
        raise ValueError("At least three non-negative finite samples are required")

    rng = np.random.default_rng(random_seed)
    if values.size > max_samples:
        values = values[rng.choice(values.size, max_samples, replace=False)]
    rng.shuffle(values)
    split = max(2, int(0.8 * values.size))
    train, test = values[:split], values[split:]
    scale = max(float(np.mean(train)), np.finfo(float).eps)
    rate = 1.0 / scale
    grid = np.linspace(0.0, float(np.max(values)), 500)
    density = rate * np.exp(-rate * grid)
    train_log_likelihood = float(train.size * np.log(rate) - rate * np.sum(train))
    held_out_mean_log_likelihood = float(np.mean(np.log(rate) - rate * test))
    sorted_test = np.sort(test)
    fitted_cdf = 1.0 - np.exp(-rate * sorted_test)
    empirical_cdf = np.arange(1, test.size + 1) / test.size

    return {
        "scale": scale,
        "rate": rate,
        "grid": grid,
        "density": density,
        "held_out_mean_log_likelihood": held_out_mean_log_likelihood,
        "bic": -2 * train_log_likelihood + np.log(train.size),
        "cdf_max_error": float(np.max(np.abs(empirical_cdf - fitted_cdf))),
    }


def exponential_shot_noise_cumulants(rate_hz, tau_ms, amplitude):
    """Return stationary cumulants 1..4 for exponential Poisson shot noise."""
    rate_tau = float(rate_hz) * float(tau_ms) / 1000.0
    return np.asarray(
        [rate_tau * float(amplitude) ** order / order for order in range(1, 5)]
    )


def filtered_nmda_shot_noise_cumulants(
    rate_hz, tau_rise_ms, tau_decay_ms, alpha_per_ms, weight
):
    """Cumulants 1..4 of s for Poisson-driven linear x-to-s NMDA kinetics."""
    tau_rise = float(tau_rise_ms) / 1000.0
    tau_decay = float(tau_decay_ms) / 1000.0
    alpha = float(alpha_per_ms) * 1000.0
    amplitude_scale = (
        alpha * float(weight) * tau_rise * tau_decay / (tau_decay - tau_rise)
    )
    cumulants = []
    for order in range(1, 5):
        kernel_power_integral = amplitude_scale**order * sum(
            (-1) ** index
            * math.comb(order, index)
            / ((order - index) / tau_decay + index / tau_rise)
            for index in range(order + 1)
        )
        cumulants.append(float(rate_hz) * kernel_power_integral)
    return np.asarray(cumulants)


def _estimated_cumulants(values, max_samples=500_000, random_seed=0):
    values = np.asarray(values, dtype=float).ravel()
    if values.size > max_samples:
        indices = np.random.default_rng(random_seed).choice(
            values.size, size=max_samples, replace=False
        )
        values = values[indices]
    centered = values - np.mean(values)
    variance = float(np.mean(centered**2))
    third = float(np.mean(centered**3))
    fourth = float(np.mean(centered**4) - 3.0 * variance**2)
    return np.asarray([float(np.mean(values)), variance, third, fourth])


def _with_standardized_higher_cumulants(theoretical, empirical):
    return {
        "theoretical_cumulants": theoretical,
        "empirical_cumulants": empirical,
        "theoretical_skewness": float(theoretical[2] / theoretical[1] ** 1.5),
        "empirical_skewness": float(empirical[2] / empirical[1] ** 1.5),
        "theoretical_excess_kurtosis": float(theoretical[3] / theoretical[1] ** 2),
        "empirical_excess_kurtosis": float(empirical[3] / empirical[1] ** 2),
    }


def compute_shot_noise_gaussianity_summary(
    result,
    *,
    ampa_input_rate_hz,
    nmda_input_rate_hz,
    ampa_tau_ms,
    ampa_weight_nS,
    nmda_tau_rise_ms,
    nmda_tau_decay_ms,
    nmda_alpha_per_ms,
    nmda_weight,
    nmda_open_conductance_scale_nS,
):
    """Compare theoretical and sampled higher cumulants with Gaussian zeros."""
    x_theory = exponential_shot_noise_cumulants(
        nmda_input_rate_hz, nmda_tau_rise_ms, nmda_weight
    )
    s_theory = filtered_nmda_shot_noise_cumulants(
        nmda_input_rate_hz,
        nmda_tau_rise_ms,
        nmda_tau_decay_ms,
        nmda_alpha_per_ms,
        nmda_weight,
    )
    g_nmda_theory = s_theory * np.asarray(
        [nmda_open_conductance_scale_nS**order for order in range(1, 5)]
    )
    g_ampa_theory = exponential_shot_noise_cumulants(
        ampa_input_rate_hz, ampa_tau_ms, ampa_weight_nS
    )
    pairs = {
        "x": (x_theory, result["x_nmda"]),
        "s": (s_theory, result["s_nmda"]),
        # Theory is for open g_NMDA; data include the voltage-dependent Mg block.
        "g_NMDA": (g_nmda_theory, result["effective_g_nmda_nS"]),
        "g_AMPA": (g_ampa_theory, result["g_ampa_nS"]),
    }
    return {
        name: _with_standardized_higher_cumulants(
            theoretical, _estimated_cumulants(values)
        )
        for name, (theoretical, values) in pairs.items()
    }


def format_shot_noise_gaussianity_rows(summary):
    """Return table-like text aligned with V, g_AMPA, g_NMDA, s, and x rows."""
    rows = ["not evaluated for V"]
    for name in ("g_AMPA", "g_NMDA", "s", "x"):
        values = summary[name]
        theory_label = "open theory" if name == "g_NMDA" else "theory"
        rows.append(
            f"γ₁  {theory_label:<11} {values['theoretical_skewness']:>8.3g}\n"
            f"    data        {values['empirical_skewness']:>8.3g}\n"
            f"γ₂  {theory_label:<11} {values['theoretical_excess_kurtosis']:>8.3g}\n"
            f"    data        {values['empirical_excess_kurtosis']:>8.3g}"
        )
    return rows


def plot_chosen_ampa_nmda_traces(
    result,
    chosen_model_indices=(0, 1, 2),
    output_path=OUTPUT_DIR / "ampa_nmda_three_chosen_traces.png",
    figure_title=None,
    gaussianity_summary=None,
):
    """Plot V, AMPA, and Mg-blocked NMDA conductance for chosen models."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_ms = result["time_ms"]
    fig, axes = plt.subplots(
        5,
        4,
        figsize=(20, 14),
        sharex=False,
        sharey=False,
        gridspec_kw={"width_ratios": [4, 1, 1.35, 3.2]},
    )
    trace_axes = axes[:, 0]
    (
        voltage_histogram_ax,
        ampa_histogram_ax,
        nmda_histogram_ax,
        s_nmda_histogram_ax,
        x_nmda_histogram_ax,
    ) = axes[:, 1]
    fit_axes = axes[:, 2]
    gaussianity_axes = axes[:, 3]
    for model_index in chosen_model_indices:
        label = f"model {model_index}"
        trace_axes[0].plot(time_ms, result["voltage_mV"][model_index], label=label)
        trace_axes[1].plot(time_ms, result["g_ampa_nS"][model_index], label=label)
        trace_axes[2].plot(
            time_ms,
            result["effective_g_nmda_nS"][model_index],
            label=label,
        )
        trace_axes[3].plot(time_ms, result["s_nmda"][model_index], label=label)
        trace_axes[4].plot(time_ms, result["x_nmda"][model_index], label=label)

    # Use all recorded samples from all compartments for the distributions.
    voltage_histogram_ax.hist(
        result["voltage_mV"].ravel(),
        bins=50,
        density=True,
        orientation="horizontal",
        color="tab:green",
        alpha=0.75,
    )
    ampa_histogram_ax.hist(
        result["g_ampa_nS"].ravel(),
        bins=50,
        density=True,
        orientation="horizontal",
        color="tab:blue",
        alpha=0.75,
    )
    nmda_histogram_ax.hist(
        result["effective_g_nmda_nS"].ravel(),
        bins=50,
        density=True,
        orientation="horizontal",
        color="tab:orange",
        alpha=0.75,
    )
    s_nmda_histogram_ax.hist(
        result["s_nmda"].ravel(),
        bins=50,
        density=True,
        orientation="horizontal",
        color="tab:red",
        alpha=0.75,
    )
    x_nmda_histogram_ax.hist(
        result["x_nmda"].ravel(),
        bins=50,
        density=True,
        orientation="horizontal",
        color="tab:purple",
        alpha=0.75,
    )
    distribution_values = (
        result["voltage_mV"],
        result["g_ampa_nS"],
        result["effective_g_nmda_nS"],
        result["s_nmda"],
        result["x_nmda"],
    )
    distribution_colors = ("tab:green", "tab:blue", "tab:orange", "tab:red", "tab:purple")
    for fit_axis, values, color, number_of_gaussians in zip(
        fit_axes,
        distribution_values,
        distribution_colors,
        TRACE_DISTRIBUTION_GAUSSIAN_COMPONENTS,
    ):
        fit_axis.hist(
            np.asarray(values).ravel(),
            bins=50,
            density=True,
            orientation="horizontal",
            color="0.85",
        )
        fit = fit_gaussian_mixture(
            values, number_of_gaussians=number_of_gaussians
        )
        for component_index in range(fit["component_density"].shape[1]):
            fit_axis.plot(
                fit["component_density"][:, component_index],
                fit["grid"],
                linestyle=":",
                color=color,
                alpha=0.7,
            )
        fit_axis.plot(
            fit["mixture_density"], fit["grid"], color=color, linewidth=2
        )
        parameter_text = (
            "single Gaussian"
            if number_of_gaussians == 1
            else f"{number_of_gaussians}-Gaussian mixture"
        )
        fit_axis.set_xlabel("density")
        fit_axis.text(
            0.98,
            0.98,
            f"{parameter_text}\nheld-out LL={fit['held_out_mean_log_likelihood']:.3g}\n"
            f"BIC={fit['bic']:.3g}\nD={fit['cdf_max_error']:.3g}",
            transform=fit_axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
    trace_axes[0].set_ylabel("V [mV]")
    trace_axes[1].set_ylabel(r"$g_{AMPA}$ [nS]")
    trace_axes[2].set_ylabel(r"$g_{NMDA,max}s\sigma_{Mg}(V)$ [nS]")
    trace_axes[3].set_ylabel(r"$s_{NMDA}$ [1]")
    trace_axes[4].set_ylabel(r"$x_{NMDA}$ [1]")
    trace_axes[4].set_xlabel("time [ms]")
    trace_axes[0].set_title("Three single-compartment realizations")
    voltage_histogram_ax.set_title("distribution", fontsize=10)
    fit_axes[0].set_title("distribution fit", fontsize=10)
    gaussianity_axes[0].set_title(
        "model equation and higher cumulants\n"
        "(Gaussian reference: γ₁ = γ₂ = 0)",
        fontsize=10,
    )
    gaussianity_rows = (
        format_shot_noise_gaussianity_rows(gaussianity_summary)
        if gaussianity_summary is not None
        else ["not available"] * 5
    )
    trace_model_equations = result.get(
        "trace_model_equations", TRACE_MODEL_EQUATIONS
    )
    for row_index, (gaussianity_axis, row_text, equation) in enumerate(
        zip(gaussianity_axes, gaussianity_rows, trace_model_equations)
    ):
        gaussianity_axis.axis("off")
        gaussianity_axis.text(
            0.02,
            0.72 if row_index else 0.5,
            equation,
            transform=gaussianity_axis.transAxes,
            ha="left",
            va="center",
            fontsize=9,
        )
        if row_index:
            gaussianity_axis.text(
                0.02,
                0.27,
                row_text,
                transform=gaussianity_axis.transAxes,
                ha="left",
                va="center",
                fontsize=8,
                fontfamily="monospace",
            )
    voltage_histogram_ax.set_xlabel("density")
    ampa_histogram_ax.set_xlabel("density")
    nmda_histogram_ax.set_xlabel("density")
    s_nmda_histogram_ax.set_xlabel("density")
    x_nmda_histogram_ax.set_xlabel("density")
    for axis in trace_axes:
        axis.legend(loc="best")
    if figure_title:
        fig.suptitle(figure_title, fontsize=10, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def compute_per_unit_s0_and_fixed_points(
    result,
    *,
    experiment=None,
    g_nmda_max_nS=None, n_nmda=None, g_leak_nS=None,
    e_leak_mV=None, e_excitatory_mV=None, mg=None,
):
    """Compute post-burn-in s0 and stable Lambert fixed points per unit."""
    s0 = np.mean(np.asarray(result["s_nmda"], dtype=float), axis=1)
    g_ampa_zero_nS = (
        np.mean(np.asarray(result["g_ampa_nS"], dtype=float), axis=1)
        if "g_ampa_nS" in result
        else np.zeros(s0.size)
    )
    return compute_fixed_points_from_per_unit_s0(
        s0,
        experiment=experiment,
        g_ampa_zero_nS=g_ampa_zero_nS,
        g_nmda_max_nS=g_nmda_max_nS,
        n_nmda=n_nmda,
        g_leak_nS=g_leak_nS,
        e_leak_mV=e_leak_mV,
        e_excitatory_mV=e_excitatory_mV,
        mg=mg,
    )


def compute_fixed_points_from_per_unit_s0(
    s0,
    *,
    experiment=None,
    g_ampa_zero_nS=0.0,
    g_nmda_max_nS=None, n_nmda=None, g_leak_nS=None,
    e_leak_mV=None, e_excitatory_mV=None, mg=None,
):
    """Recompute stable fixed points from stored per-unit post-burn-in s0."""
    if experiment is not None:
        g_nmda_max_nS = float(experiment.g_nmda_max / nS)
        n_nmda = experiment.n_nmda
        g_leak_nS = float(experiment.g_L / nS)
        e_leak_mV = float(experiment.E_L / mV)
        e_excitatory_mV = float(experiment.E_E / mV)
        mg = experiment.magnesium
    s0 = np.asarray(s0, dtype=float)
    g_ampa_zero_nS = np.broadcast_to(np.asarray(g_ampa_zero_nS, dtype=float), s0.shape)
    open_g_nmda_nS = (float(g_nmda_max_nS) / int(n_nmda)) * s0
    lower_fixed_point_mV = [None] * s0.size
    upper_fixed_point_mV = [None] * s0.size
    magnesium_coefficient = float(mg) / 3.57
    for unit_index, g_n in enumerate(open_g_nmda_nS):
        g_ampa_zero = g_ampa_zero_nS[unit_index]
        effective_leak_nS = float(g_leak_nS) + g_ampa_zero
        e_0 = (
            float(g_leak_nS) * float(e_leak_mV)
            + g_ampa_zero * float(e_excitatory_mV)
        ) / effective_leak_nS
        voltage_separator_mV = -45 # mV
        roots = find_fixed_points(
            g_n,
            g_l=effective_leak_nS,
            e_l=e_0,
            e_e=float(e_excitatory_mV),
            a=magnesium_coefficient,
            b=0.08,
        )
        stable_roots = [
            voltage
            for voltage in roots
            if fixed_point_current_slope(
                voltage,
                g_n,
                g_l=effective_leak_nS,
                e_e=float(e_excitatory_mV),
                a=magnesium_coefficient,
                b=0.08,
            )
            > 0.0
        ]
        if len(stable_roots) >= 2:
            lower_fixed_point_mV[unit_index] = float(stable_roots[0])
            upper_fixed_point_mV[unit_index] = float(stable_roots[-1])
        elif stable_roots:
            if stable_roots[0] < voltage_separator_mV:
                lower_fixed_point_mV[unit_index] = float(stable_roots[0])
            else:
                upper_fixed_point_mV[unit_index] = float(stable_roots[0])

    return {
        "s0": s0,
        "open_g_nmda_nS": open_g_nmda_nS,
        "lower_fixed_point_mV": lower_fixed_point_mV,
        "upper_fixed_point_mV": upper_fixed_point_mV,
    }


def save_voltage_fixed_point_distribution_data(
    result,
    s0,
    output_dir,
    *,
    metadata,
    voltage_sample_size=500_000,
    random_seed=0,
):
    """Save simulation observables needed to recompute and plot fixed points."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("name", "value"))
        for name, value in metadata.items():
            writer.writerow((name, "" if value is None else value))

    s0 = np.asarray(s0, dtype=float)
    g_ampa_0 = np.mean(np.asarray(result["g_ampa_nS"], dtype=float), axis=1)
    voltage_mean = np.asarray(
        result.get(
            "per_model_mean_mV",
            np.mean(np.asarray(result["voltage_mV"], dtype=float), axis=1),
        ),
        dtype=float,
    )
    per_unit_path = output_dir / "per_unit_post_burn_in_means.csv"
    fieldnames = (
        "unit_index",
        "s0_empirical",
        "g_ampa_0_nS_empirical",
        "voltage_mean_mV",
    )
    with per_unit_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for unit_index in range(s0.size):
            writer.writerow(
                {
                    "unit_index": unit_index,
                    "s0_empirical": s0[unit_index],
                    "g_ampa_0_nS_empirical": g_ampa_0[unit_index],
                    "voltage_mean_mV": voltage_mean[unit_index],
                }
            )

    voltage_mV = np.asarray(result["voltage_mV"], dtype=float).ravel()
    sample_size = min(voltage_mV.size, int(voltage_sample_size))
    sample_indices = np.random.default_rng(random_seed).choice(
        voltage_mV.size, size=sample_size, replace=False
    )
    voltage_sample_path = output_dir / "voltage_histogram_sample.csv"
    with voltage_sample_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("flat_sample_index", "voltage_mV"))
        writer.writerows(zip(sample_indices, voltage_mV[sample_indices]))

    return {
        "metadata": metadata_path,
        "per_unit": per_unit_path,
        "voltage_sample": voltage_sample_path,
    }


def load_voltage_fixed_point_distribution_data(output_dir):
    """Load saved simulation observables; fixed points are intentionally absent."""
    output_dir = Path(output_dir)
    per_unit_path = output_dir / "per_unit_post_burn_in_means.csv"
    if not per_unit_path.exists():
        # Read exports made before fixed points were removed from persisted data.
        per_unit_path = output_dir / "per_unit_steady_state_and_fixed_points.csv"
    voltage_sample_path = output_dir / "voltage_histogram_sample.csv"
    metadata_path = output_dir / "metadata.csv"

    with per_unit_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    per_unit_result = {
        "s0": np.asarray([float(row["s0_empirical"]) for row in rows]),
        "g_ampa_0_nS": np.asarray(
            [float(row["g_ampa_0_nS_empirical"]) for row in rows]
        ),
        "voltage_mean_mV": np.asarray(
            [float(row["voltage_mean_mV"]) for row in rows]
        ),
    }
    with metadata_path.open(newline="") as stream:
        metadata = {row["name"]: row["value"] for row in csv.DictReader(stream)}
    voltage_rows = np.loadtxt(
        voltage_sample_path,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1),
        ndmin=2,
    )
    voltage_mV = voltage_rows[np.argsort(voltage_rows[:, 0]), 1]
    return {"voltage_mV": voltage_mV}, per_unit_result, metadata


def plot_voltage_and_fixed_point_distributions(
    result,
    fixed_point_result,
    output_path,
):
    """Compare simulated voltage with per-unit Lambert fixed-point distributions."""
    voltage_mV = np.asarray(result["voltage_mV"], dtype=float).ravel()
    lower = fixed_point_result["lower_fixed_point_mV"]
    upper = fixed_point_result["upper_fixed_point_mV"]
    finite_lower = np.asarray(
        [value for value in lower if value is not None], dtype=float
    )
    finite_upper = np.asarray(
        [value for value in upper if value is not None], dtype=float
    )
    voltage_sample_size = min(voltage_mV.size, 500_000)
    voltage_sample_indices = np.random.default_rng(0).choice(
        voltage_mV.size, size=voltage_sample_size, replace=False
    )
    sampled_voltage_mV = voltage_mV[voltage_sample_indices]
    total_fixed_point_count = finite_lower.size + finite_upper.size

    def fixed_point_histogram_parameters(values):
        """Return bins/weights whose density area is this branch's joint mass."""
        bin_edges = np.histogram_bin_edges(values, bins=150)
        bin_width = bin_edges[1] - bin_edges[0]
        weights = np.full(
            values.size,
            1.0 / (total_fixed_point_count * bin_width),
            dtype=float,
        )
        return bin_edges, weights

    figure, axis = plt.subplots(figsize=(11, 7))
    axis.hist(
        sampled_voltage_mV,
        bins=150,
        density=True,
        histtype="step",
        linewidth=2,
        color="tab:green",
        label=(
            f"simulated V (sample n={sampled_voltage_mV.size:,}; "
            f"total n={voltage_mV.size:,})"
        ),
    )
    if finite_lower.size:
        lower_bin_edges, lower_weights = fixed_point_histogram_parameters(finite_lower)
        axis.hist(
            finite_lower,
            bins=lower_bin_edges,
            weights=lower_weights,
            density=False,
            histtype="step",
            linewidth=2,
            color="tab:blue",
            label=f"lower stable fixed point (n={finite_lower.size:,})",
        )
    if finite_upper.size:
        upper_bin_edges, upper_weights = fixed_point_histogram_parameters(finite_upper)
        axis.hist(
            finite_upper,
            bins=upper_bin_edges,
            weights=upper_weights,
            density=False,
            histtype="step",
            linewidth=2,
            color="tab:orange",
            label=f"upper stable fixed point (n={finite_upper.size:,})",
        )
    axis.set_xlabel("voltage [mV]")
    axis.set_ylabel("probability density")
    axis.set_title(
        "Simulated membrane-voltage distribution versus per-unit fixed points\n"
        r"$s_0^{(i)}=\langle s_{NMDA}^{(i)}(t)\rangle$ after burn-in"
    )
    axis.legend()
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def fitted_apic20_ei_balance_config(
    base_config: ConductanceDiffusionSimulationConfig = config_with_intermediate_synapses,
):
    fitted_parameters = read_fitted_apic20_parameters_from_csv()
    return base_config.with_property(
        membrane_capacitance=float(fitted_parameters.capacitance_nF) * nF,
        g_L=float(fitted_parameters.g_leak_nS) * nS,
        g_nmda_max=float(fitted_parameters.g_nmda_max_nS) * nS,
        e_L=float(fitted_parameters.e_leak_mV) * mV,
        resting_voltage=float(fitted_parameters.e_leak_mV) * mV,
        e_ampa=float(fitted_parameters.e_excitatory_mV) * mV,
        e_nmda=float(fitted_parameters.e_excitatory_mV) * mV,
        label="APIC20 fitted numbers under EI balance",
    )


def read_sympy_cubic_solution_expressions(
    solutions_file_name=ITERATION_16_SOLUTIONS_PATH,
):
    return sp.sympify(Path(solutions_file_name).read_text())


def assert_solution_symbols_are_bound(solutions, config):
    bound_names = set(bind_config_to_sympy_values(config))
    solution_symbols = {
        str(symbol)
        for solution in solutions
        for symbol in getattr(solution, "free_symbols", set())
    }
    unbound_names = sorted(solution_symbols - bound_names)
    if unbound_names:
        raise ValueError(f"Unbound SymPy solution symbols: {unbound_names}")
    return {
        name: bind_config_to_sympy_values(config)[name]
        for name in sorted(solution_symbols)
    }


def read_solutions(
    config: ConductanceDiffusionSimulationConfig,
    solutions_file_name=ITERATION_16_SOLUTIONS_PATH,
):
    sols = []
    solutions = read_sympy_cubic_solution_expressions(solutions_file_name)
    binding_values = assert_solution_symbols_are_bound(solutions, config)
    for solution in solutions:
        x_sol, y_sol = evaluate_sympy_solution(
            solution,
            config,
            mu_v_target=chapter1Results.mu_v,
            sigma_target=chapter1Results.sigma_v,
        )
        rate_sol = x_sol / (config.g() / nS)
        if rate_sol > 0 and y_sol > 0:
            sols.append((rate_sol, y_sol))
    return sols, binding_values


def plot_fitted_apic20_cubic_solutions(
    config: ConductanceDiffusionSimulationConfig | None = None,
    solutions_file_name=ITERATION_16_SOLUTIONS_PATH,
    output_png_path=OUTPUT_DIR / "fitted_apic20_ei_balance_cubic_solutions.png",
    output_json_path=OUTPUT_DIR / "fitted_apic20_ei_balance_cubic_solutions.json",
    r_max_hz=10_000.0,
    polynomial_full_min_hz=-500.0,
    polynomial_full_max_hz=65_000.0,
    polynomial_zoom_min_hz=-400.0,
    polynomial_zoom_max_hz=100.0,
):
    if config is None:
        config = fitted_apic20_ei_balance_config()
    sols, binding_values = read_solutions(config, solutions_file_name)
    output_png_path = Path(output_png_path)
    output_json_path = Path(output_json_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    r = np.linspace(0.1, float(r_max_hz), 3000) * Hz
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    mean_ax = axes[0, 0]
    variance_ax = axes[0, 1]
    rate_ax = axes[1, 0]
    unused_ax = axes[1, 1]
    polynomial_full_ax = axes[2, 0]
    polynomial_zoom_ax = axes[2, 1]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    polynomial_roots = []
    polynomial_full_rates = np.linspace(
        float(polynomial_full_min_hz),
        float(polynomial_full_max_hz),
        6000,
    ) * Hz
    polynomial_zoom_rates = np.linspace(
        float(polynomial_zoom_min_hz),
        float(polynomial_zoom_max_hz),
        3000,
    ) * Hz
    for index, (rate_sol, gamma) in enumerate(sols):
        color = colors[index % len(colors)]
        mean_vm = E_0(r, config, gamma)
        variance_vm = sigma_sq(r, config, gamma)
        polynomial_full_values, roots = polynomial_r(
            polynomial_full_rates,
            config,
            gamma,
            mu_v=chapter1Results.mu_v,
            sigma_v=chapter1Results.sigma_v,
        )
        polynomial_zoom_values, _zoom_roots = polynomial_r(
            polynomial_zoom_rates,
            config,
            gamma,
            mu_v=chapter1Results.mu_v,
            sigma_v=chapter1Results.sigma_v,
        )
        mean_ax.plot(r / Hz, mean_vm / mV, color=color, label=rf"$\gamma={gamma:.3g}$")
        variance_ax.plot(r / Hz, variance_vm / (mV**2), color=color, label=rf"$\gamma={gamma:.3g}$")
        rate_ax.plot(
            r / Hz,
            compute_siegert_firing_rate_for_fitted_solution(
                config,
                gamma,
                r,
                means_vm=mean_vm,
                vars_vm=variance_vm,
            )
            / Hz,
            color=color,
            label=rf"$\gamma={gamma:.3g}$",
        )
        polynomial_full_scale = float(np.max(np.abs(polynomial_full_values)))
        scaled_polynomial_full_values = (
            polynomial_full_values / polynomial_full_scale
            if polynomial_full_scale > 0.0
            else polynomial_full_values
        )
        polynomial_zoom_scale = float(np.max(np.abs(polynomial_zoom_values)))
        scaled_polynomial_zoom_values = (
            polynomial_zoom_values / polynomial_zoom_scale
            if polynomial_zoom_scale > 0.0
            else polynomial_zoom_values
        )
        polynomial_full_ax.plot(
            polynomial_full_rates[polynomial_full_rates >= 0 * Hz] / Hz,
            scaled_polynomial_full_values[polynomial_full_rates >= 0 * Hz],
            color=color,
            label=rf"$\gamma={gamma:.3g}$",
        )
        polynomial_full_ax.plot(
            polynomial_full_rates[polynomial_full_rates < 0 * Hz] / Hz,
            scaled_polynomial_full_values[polynomial_full_rates < 0 * Hz],
            color="tab:red",
            linestyle=":",
            linewidth=1.4,
            label="negative input rate",
        )
        polynomial_zoom_ax.plot(
            polynomial_zoom_rates[polynomial_zoom_rates >= 0 * Hz] / Hz,
            scaled_polynomial_zoom_values[polynomial_zoom_rates >= 0 * Hz],
            color=color,
            label=rf"$\gamma={gamma:.3g}$",
        )
        polynomial_zoom_ax.plot(
            polynomial_zoom_rates[polynomial_zoom_rates < 0 * Hz] / Hz,
            scaled_polynomial_zoom_values[polynomial_zoom_rates < 0 * Hz],
            color="tab:red",
            linestyle=":",
            linewidth=1.4,
            label="negative input rate",
        )
        root_rates_hz = root_rates_from_conductance_roots(roots, config)
        polynomial_roots.append(
            {
                "gamma": float(gamma),
                "polynomial_full_scale": polynomial_full_scale,
                "polynomial_zoom_scale": polynomial_zoom_scale,
                "root_rates_hz": root_rates_hz,
            }
        )
        full_visible_root_rates_hz = [
            root_rate_hz
            for root_rate_hz in root_rates_hz
            if float(polynomial_full_min_hz)
            <= root_rate_hz
            <= float(polynomial_full_max_hz)
        ]
        zoom_visible_root_rates_hz = [
            root_rate_hz
            for root_rate_hz in root_rates_hz
            if float(polynomial_zoom_min_hz)
            <= root_rate_hz
            <= float(polynomial_zoom_max_hz)
        ]
        for ax, visible_root_rates_hz in [
            (polynomial_full_ax, full_visible_root_rates_hz),
            (polynomial_zoom_ax, zoom_visible_root_rates_hz),
        ]:
            if not visible_root_rates_hz:
                continue
            ax.scatter(
                visible_root_rates_hz,
                np.zeros(len(visible_root_rates_hz)),
                color=color,
                edgecolors="white",
                marker="o",
                s=42,
                zorder=5,
            )
            for root_rate_hz in visible_root_rates_hz:
                ax.annotate(
                    f"{root_rate_hz:.3g} Hz",
                    (root_rate_hz, 0.0),
                    xytext=(5, 8),
                    textcoords="offset points",
                    fontsize=7,
                )

    mean_ax.axhline(chapter1Results.mu_v / mV, color="black", linestyle="--", linewidth=1.0)
    variance_ax.axhline(
        (chapter1Results.sigma_v / mV) ** 2,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    unused_ax.axis("off")
    polynomial_full_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    polynomial_zoom_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    mean_ax.set_ylabel("E0 [mV]")
    variance_ax.set_ylabel("sigma_v^2 [mV^2]")
    rate_ax.set_ylabel("Siegert rate [Hz]")
    polynomial_full_ax.set_ylabel("normalized P(r)")
    polynomial_zoom_ax.set_ylabel("normalized P(r)")
    mean_ax.set_xlabel("input rate [Hz]")
    variance_ax.set_xlabel("input rate [Hz]")
    rate_ax.set_xlabel("input rate [Hz]")
    polynomial_full_ax.set_xlabel("input rate [Hz]")
    polynomial_zoom_ax.set_xlabel("input rate [Hz]")
    mean_ax.set_title("Mean voltage from cubic EI-balance solution")
    variance_ax.set_title("Voltage variance from cubic EI-balance solution")
    rate_ax.set_title("Output firing-rate check")
    polynomial_full_ax.set_title("Cubic polynomial over full root range")
    polynomial_zoom_ax.set_title("Cubic polynomial zoom around local roots")
    polynomial_full_ax.set_xlim(float(polynomial_full_min_hz), float(polynomial_full_max_hz))
    polynomial_zoom_ax.set_xlim(float(polynomial_zoom_min_hz), float(polynomial_zoom_max_hz))
    for ax in [mean_ax, variance_ax, rate_ax, polynomial_full_ax, polynomial_zoom_ax]:
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(config.label)
    fig.tight_layout()
    fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    output_json_path.write_text(
        json.dumps(
            {
                "plot_path": str(output_png_path),
                "solutions_file_name": str(solutions_file_name),
                "binding_values": binding_values,
                "solutions": [
                    {"rate_hz": float(rate / Hz), "gamma": float(gamma)}
                    for rate, gamma in sols
                ],
                "polynomial_roots": polynomial_roots,
                "r_max_hz": float(r_max_hz),
                "polynomial_full_min_hz": float(polynomial_full_min_hz),
                "polynomial_full_max_hz": float(polynomial_full_max_hz),
                "polynomial_zoom_min_hz": float(polynomial_zoom_min_hz),
                "polynomial_zoom_max_hz": float(polynomial_zoom_max_hz),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return output_png_path, output_json_path, sols


def root_rates_from_conductance_roots(roots, config):
    root_rates_hz = []
    for root in roots:
        root_rate = root / config.g()
        root_rate_value = complex(root_rate / Hz)
        if abs(root_rate_value.imag) < 1e-8:
            root_rates_hz.append(float(root_rate_value.real))
    return sorted(root_rates_hz)


class SimulateNMDASpikesUnderEIBalanceScriptTestCases(unittest.TestCase):
    """Runnables for iteration 21: NMDA spikes under EI balance."""

    def test_ampa_nmda_brian2_model_runs_for_100_ms(self):
        """Fast smoke test: Brian2 accepts the units and integrates the model."""
        simulation_time = 100 * ms
        n_models = 4

        experiment = AmpaNmdaExperimentConfigFactoryBuilder.fitted_apic20().with_property(
            n_models=n_models,
            simulation_time=simulation_time,
            burn_in=0 * ms,
            dt=0.1 * ms,
            random_seed=21,
            C_m=0.5 * nF,
            g_L=20 * nS,
            E_L=-65 * mV,
            E_E=0 * mV,
            tau_ampa=2 * ms,
            w_ampa=0.05 * nS,
            r_e=5 * Hz,
            n_e=8,
            tau_nmda_rise=2 * ms,
            tau_nmda_decay=100 * ms,
            alpha_nmda=0.5 * kHz,
            g_nmda_max=4 * nS,
            n_nmda=8,
            w_nmda=1.0,
            magnesium=1.0,
        ).build()
        result = simulate_ampa_nmda_single_compartment_ensemble(experiment)

        self.assertEqual(result["voltage_mV"].shape, (n_models, 1000))
        self.assertTrue(np.all(np.isfinite(result["voltage_mV"])))
        self.assertAlmostEqual(result["time_ms"][0], 0.0)
        self.assertAlmostEqual(result["time_ms"][-1], 99.9)

    def test_read_fitted_apic20_cubic_solutions(self):
        config = fitted_apic20_ei_balance_config()
        sols, binding_values = read_solutions(config)

        self.assertGreater(len(binding_values), 0)
        self.assertIsInstance(sols, list)
        self.assertAlmostEqual(
            float(read_fitted_apic20_parameters_from_csv().capacitance_nF),
            float(config.membrane_capacitance / nF),
        )

    def test_plot_fitted_apic20_cubic_solutions(self):
        plot_path, json_path, _sols = plot_fitted_apic20_cubic_solutions()

        self.assertTrue(plot_path.exists(), str(plot_path))
        self.assertGreater(plot_path.stat().st_size, 0, str(plot_path))
        self.assertTrue(json_path.exists(), str(json_path))
        self.assertGreater(json_path.stat().st_size, 0, str(json_path))

    def test_simulate_1000_ampa_nmda_with_magnesium_tau(self):
        self.test_simulate_1_000_ampa_nmda_single_compartments(tau_mg = 100 * ms)

    def test_simulate_1_000_ampa_nmda_single_compartments(
        self,
        r_e=20 * Hz,
        tau_mg=None,
        show_plot=True
    ):

        # Principal experiment inputs are intentionally plain and local: edit
        # these values, then run this one test from the IDE.
        chosen_model_indices = (0, 1, 2)

        # Parameters fitted in iteration 20.  The individual assignments stay
        # here so it is obvious which fitted values drive this experiment.
        fitted_cm = 13.16560131 * pfarad
        experiment = (
            AmpaNmdaExperimentConfigFactoryBuilder.fitted_apic20()
            .with_property(
                C_m = fitted_cm,
                n_models=1_000, n_e = 800, n_nmda = 10,
                simulation_time = 1_000 * ms, burn_in = 500 * ms, dt = 0.01 * ms, random_seed = 21,
                r_e=r_e, magnesium_block_tau=tau_mg)
            .build()
        )
        result = simulate_ampa_nmda_single_compartment_ensemble(experiment)

        fixed_point_result = compute_per_unit_s0_and_fixed_points(
            result, experiment=experiment
        )
        metadata = experiment.build_metadata(result)
        model_name = metadata["model_name"]
        distribution_data_paths = save_voltage_fixed_point_distribution_data(
            result,
            fixed_point_result["s0"],
            OUTPUT_DIR
            / "voltage_fixed_point_data"
            / f"{float(r_e / Hz):g}_Hz_{model_name}",
            metadata=metadata,
        )
        for data_name, data_path in distribution_data_paths.items():
            print(f"Saved fixed-point analysis {data_name}: {data_path}")
        fixed_point_distribution_path = plot_voltage_and_fixed_point_distributions(
            result,
            fixed_point_result,
            OUTPUT_DIR
            / f"voltage_vs_fixed_point_distributions_{float(r_e / Hz):g}_Hz.png",
        )
        number_of_lower_points = sum(
            value is not None
            for value in fixed_point_result["lower_fixed_point_mV"]
        )
        number_of_upper_points = sum(
            value is not None
            for value in fixed_point_result["upper_fixed_point_mV"]
        )
        print(
            f"Per-unit s0 fixed points: lower={number_of_lower_points:,}, "
            f"upper={number_of_upper_points:,}, units={experiment.n_models:,}"
        )

        debug_output_dir = (
            DEBUG_DISTRIBUTION_DIR / f"{float(r_e / Hz):g}_Hz"
        )
        if saveDebugFile.enabled:
            storage = estimate_full_text_json_size(result)
            print(
                "Estimated storage for all five complete arrays: "
                f"binary={format_storage_size(storage['binary_bytes'])}, "
                f"full-text JSON≈{format_storage_size(storage['estimated_json_bytes'])}"
            )
            saved_debug_paths = save_debug_histograms(
                result,
                debug_output_dir,
                saveDebugFile,
            )
            for variable_name, paths in saved_debug_paths.items():
                for path in paths:
                    print(
                        f"Saved {variable_name} histogram: {path} "
                        f"({format_storage_size(path.stat().st_size)})"
                    )
                for suffix in ("csv", "json"):
                    raw_path = debug_output_dir / f"{variable_name}_fit_sample.{suffix}"
                    if raw_path.exists():
                        print(
                            f"Saved {variable_name} raw fit sample: {raw_path} "
                            f"({format_storage_size(raw_path.stat().st_size)})"
                        )

        print(
            f"V = {result['ensemble_mean_mV']:.3f} mV, "
            f"variance = {result['ensemble_variance_mV2']:.3f} mV^2"
        )
        gaussianity_summary = compute_shot_noise_gaussianity_summary(
            result,
            ampa_input_rate_hz=float(experiment.n_e * experiment.r_e / Hz),
            nmda_input_rate_hz=float(experiment.n_nmda * r_e / Hz),
            ampa_tau_ms=float(experiment.tau_ampa / ms),
            ampa_weight_nS=float(experiment.w_ampa / nS),
            nmda_tau_rise_ms=float(experiment.tau_nmda_rise / ms),
            nmda_tau_decay_ms=float(experiment.tau_nmda_decay / ms),
            nmda_alpha_per_ms=float(experiment.alpha_nmda / kHz),
            nmda_weight=float(experiment.w_nmda),
            nmda_open_conductance_scale_nS=float(experiment.g_nmda_max / nS) / experiment.n_nmda,
        )
        figure_title = (
            f"AMPA–NMDA ensemble: N={experiment.n_models:,}, r_e={float(experiment.r_e / Hz):g} Hz, "
            f"N_e={experiment.n_e}, N_NMDA={experiment.n_nmda}, T={float(experiment.simulation_time / ms):g} ms, "
            f"burn-in={float(experiment.burn_in / ms):g} ms, dt={float(experiment.dt / ms):g} ms, seed={experiment.random_seed}\n"
            f"Membrane: C_m={float(experiment.C_m / nF):.4g} nF, g_L={float(experiment.g_L / nS):.4g} nS, "
            f"E_L={float(experiment.E_L / mV):.4g} mV, E_E={float(experiment.E_E / mV):.4g} mV; "
            f"AMPA: tau={float(experiment.tau_ampa / ms):.4g} ms, w={float(experiment.w_ampa / nS):.4g} nS\n"
            f"NMDA: tau_r={float(experiment.tau_nmda_rise / ms):.4g} ms, "
            f"tau_d={float(experiment.tau_nmda_decay / ms):.4g} ms, "
            f"alpha={float(experiment.alpha_nmda / kHz):.4g} kHz, "
            f"g_max={float(experiment.g_nmda_max / nS):.4g} nS, w={experiment.w_nmda:g}, "
            f"Mg={experiment.magnesium:g}; fits: g_AMPA/g_NMDA/s_NMDA single Gaussian, "
            f"V/x_NMDA two-Gaussian"
        )
        plot_path = plot_chosen_ampa_nmda_traces(
            result,
            chosen_model_indices=chosen_model_indices,
            output_path=(
                OUTPUT_DIR
                / f"ampa_nmda_three_chosen_traces_{float(r_e / Hz):g}_Hz.png"
            ),
            figure_title=figure_title,
            gaussianity_summary=gaussianity_summary,
        )

        self.assertEqual(result["voltage_mV"].shape[0], experiment.n_models)
        self.assertTrue(np.all(np.isfinite(result["voltage_mV"])))
        self.assertGreater(result["ensemble_variance_mV2"], 0.0)
        self.assertTrue(plot_path.exists(), str(plot_path))
        self.assertGreater(plot_path.stat().st_size, 0, str(plot_path))
        self.assertTrue(
            fixed_point_distribution_path.exists(),
            str(fixed_point_distribution_path),
        )
        self.assertEqual(fixed_point_result["s0"].shape, (experiment.n_models,))
        if show_plot:
            show_plots_non_blocking()
            return None
        plt.close("all")
        return {
            "rate_hz": float(r_e / Hz),
            "ensemble_mean_mV": result["ensemble_mean_mV"],
            "ensemble_variance_mV2": result["ensemble_variance_mV2"],
            "plot_path": str(plot_path),
        }

    def _fit_one_saved_debug_distribution(
        self,
        variable_name,
        rate_hz=8.0,
        number_of_gaussians=2,
        file_format="csv",
        show_plot=True,
    ):
        fit_sample_path = (
            DEBUG_DISTRIBUTION_DIR
            / f"{rate_hz:g}_Hz"
            / f"{variable_name}_fit_sample.{file_format}"
        )
        plot_path, fit = fit_saved_debug_sample(
            fit_sample_path,
            variable_name,
            number_of_gaussians=number_of_gaussians,
        )
        self.assertTrue(plot_path.exists(), str(plot_path))
        self.assertTrue(np.all(np.isfinite(fit["means"])))
        if show_plot:
            show_plots_non_blocking()
        else:
            plt.close("all")
        return plot_path, fit

    def test_fit_saved_g_nmda_distribution(self):
        return self._fit_one_saved_debug_distribution("effective_g_nmda_nS")

    def test_fit_saved_s_nmda_distribution(self):
        return self._fit_one_saved_debug_distribution("s_nmda")

    def test_fit_saved_x_nmda_distribution(self):
        return self._fit_one_saved_debug_distribution("x_nmda")

    def test_fit_saved_g_ampa_distribution(self):
        return self._fit_one_saved_debug_distribution("g_ampa_nS", rate_hz=80, number_of_gaussians=1)

    def test_fit_saved_voltage_distribution(self):
        return self._fit_one_saved_debug_distribution("voltage_mV")

    def test_simulate_1_000_ampa_nmda_single_compartments_multiple_r_es(self):
        rates_hz = np.arange(1, 50, dtype=float)
        # Every rate gets its own process and therefore its own independent
        # Brian2 scope.  n_jobs=-1 uses all CPUs available to joblib.
        completed_jobs = Parallel(
            n_jobs=3,
            backend="loky",
            return_as="generator_unordered",
        )(
            delayed(run_ampa_nmda_rate_in_worker)(rate_hz)
            for rate_hz in rates_hz
        )

        # Plotting stays in the parent process (GUI backends are not reliably
        # process-safe), but results are yielded and shown as soon as each
        # Brian2 worker finishes.  Other rates keep simulating in parallel.
        results = []
        for result in completed_jobs:
            results.append(result)
            plot_image = plt.imread(result["plot_path"])
            figure, axis = plt.subplots(figsize=(15, 14))
            axis.imshow(plot_image)
            axis.axis("off")
            axis.set_title(f"Completed Brian2 run: r_e={result['rate_hz']:g} Hz")
            figure.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

        results.sort(key=lambda result: result["rate_hz"])
        self.assertEqual([result["rate_hz"] for result in results], rates_hz.tolist())
        for result in results:
            self.assertTrue(Path(result["plot_path"]).exists(), result["plot_path"])


def run_ampa_nmda_rate_in_worker(rate_hz):
    """Pickle-safe joblib entry point for one independent Brian2 rate run."""
    test_case = SimulateNMDASpikesUnderEIBalanceScriptTestCases(
        methodName="test_simulate_1_000_ampa_nmda_single_compartments"
    )
    return test_case.test_simulate_1_000_ampa_nmda_single_compartments(
        r_e=float(rate_hz) * Hz,
        show_plot=False,
    )

class UnderstandLambertRootsComingFromSimulations(unittest.TestCase):

    def test_roots_distribution_for_simulation(self):
        saved_run_dir = (
            OUTPUT_DIR
            / "voltage_fixed_point_data"
            / "20_Hz_instantaneous_magnesium"
        )
        result, per_unit_result, metadata = load_voltage_fixed_point_distribution_data(
            saved_run_dir
        )
        experiment = AmpaNmdaExperimentConfig.from_metadata(metadata)
        fixed_point_result = compute_fixed_points_from_per_unit_s0(
            per_unit_result["s0"],
            experiment=experiment,
            g_ampa_zero_nS=per_unit_result["g_ampa_0_nS"],
        )
        plot_path = plot_voltage_and_fixed_point_distributions(
            result,
            fixed_point_result,
            saved_run_dir / "voltage_vs_fixed_point_distributions_from_csv.png",
        )

        self.assertTrue(plot_path.exists(), str(plot_path))
        self.assertGreater(plot_path.stat().st_size, 0, str(plot_path))
        show_plots_non_blocking()




if __name__ == "__main__":
    unittest.main()

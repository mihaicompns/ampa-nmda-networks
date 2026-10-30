"""Aggregate-input variant of the iteration-21 AMPA--NMDA experiment.

This companion deliberately leaves ``SimulateNMDASpikesUnderEIBalanceScripts``
unchanged.  Its public analysis and plotting helpers are re-exported here, but
the simulator below replaces the ``n_nmda`` separate shared Poisson sources
per compartment by their superposition: one source at ``n_nmda * r_e``.
That source drives both the shared AMPA conductance and the compartment's one
summed ``x_NMDA`` state; consequently there is also only one ``s_NMDA`` state.
"""

import csv
import unittest
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np
from joblib import Parallel, delayed
from scipy.special import logsumexp
from iteration_21_nmda_spikes_under_ei_balance.GeneralizedLambert import (
    find_fixed_points,
    fixed_point_current_slope,
)
from iteration_21_nmda_spikes_under_ei_balance.CompartmentDistributionPlots import (
    plot_compartment_s_voltage_fixed_point_distributions,
)
from iteration_21_nmda_spikes_under_ei_balance.ShotNoiseCumulants import (
    compute_shot_noise_gaussianity_summary,
    format_shot_noise_gaussianity_rows,
)
from brian2 import (
    Clock,
    Hz,
    Network,
    NeuronGroup,
    PoissonGroup,
    PoissonInput,
    StateMonitor,
    Synapses,
    kHz,
    ms,
    mV,
    nF,
    nS,
    seed,
    start_scope,
)


SUMMED_OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output_summed_nmda_input"

# Edit both values to control the rows in the always-generated compartment plot.
COMPARTMENT_DISTRIBUTION_INDICES = (0, 1, 2)
NUMBER_OF_COMPARTMENT_DISTRIBUTIONS = 3

# Number of post-burn-in (neuron, time) s states used for the aggregate
# fixed-point histogram.  This is independent of the 1,000 per-neuron means
# retained for the lower panel.
FIXED_POINT_STATE_SAMPLE_COUNT = 50_000

# A low FPS makes the moving-nullcline GIFs easier to inspect than the
# standard fitted-APIC20 protocol animation.
SUMMED_NMDA_ANIMATION_FPS = 10
SUMMED_NMDA_ANIMATION_FRAME_COUNT = 1600


@dataclass(frozen=True)
class SummedNmdaAnimationRequest:
    """Select one rate/unit trace and an optional post-burn-in time window."""

    r_e_hz: float
    unit_index: int
    t_min_ms: float = None
    t_max_ms: float = None
    output_format: str = "gif"
    frame_count: int = SUMMED_NMDA_ANIMATION_FRAME_COUNT
    fps: int = SUMMED_NMDA_ANIMATION_FPS


def _number_label(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _nmda_time_constants_title(tau_rise_ms, tau_decay_ms):
    """Return consistently rendered simulated NMDA time constants."""
    return (
        rf"$\tau_r={float(tau_rise_ms):g}\,\mathrm{{ms}}$, "
        rf"$\tau_D={float(tau_decay_ms):g}\,\mathrm{{ms}}$"
    )


def _experiment_nmda_time_constants_title(experiment):
    return _nmda_time_constants_title(
        experiment.tau_nmda_rise / ms,
        experiment.tau_nmda_decay / ms,
    )


def summed_nmda_animation_filename(request, random_seed):
    output_format = str(request.output_format).lower().lstrip(".")
    if output_format not in {"gif", "mp4"}:
        raise ValueError("animation output_format must be 'gif' or 'mp4'")
    if request.t_min_ms is None and request.t_max_ms is None:
        time_label = "all"
    else:
        start = "start" if request.t_min_ms is None else _number_label(request.t_min_ms)
        stop = "end" if request.t_max_ms is None else _number_label(request.t_max_ms)
        time_label = f"{start}_to_{stop}"
    return (
        f"summed_nmda_re_{_number_label(request.r_e_hz)}_Hz_"
        f"unit_{request.unit_index}_t_{time_label}_ms_seed_{random_seed}."
        f"{output_format}"
    )


def summed_nmda_trace_filename(r_e_hz, unit_index, random_seed):
    return (
        f"summed_nmda_re_{_number_label(r_e_hz)}_Hz_unit_{unit_index}_"
        f"seed_{random_seed}_trace.csv"
    )


@dataclass(frozen=True)
class AmpaNmdaExperimentConfig:
    """Inputs for one aggregate AMPA--NMDA ensemble experiment."""

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
    initial_voltage_min: object = field(default_factory=lambda: -80 * mV)
    initial_voltage_max: object = field(default_factory=lambda: 0 * mV)
    magnesium_block_tau: object = None
    with_nullcline: bool = False

    def with_property(self, **changes):
        return replace(self, **changes)

    @classmethod
    def from_fitted_apic20(cls, *, r_e=20 * Hz, magnesium_block_tau=None):
        # Lazy import keeps the summed-input simulator independent of the
        # original iteration-21 script and its legacy iteration-16 imports.
        from iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20NullclineScripts import (
            read_fitted_apic20_parameters_from_csv,
        )

        fitted = read_fitted_apic20_parameters_from_csv()
        return cls(
            n_models=1_000,
            n_e=800,
            n_nmda=10,
            simulation_time=2_000 * ms,
            burn_in=500 * ms,
            dt=0.1 * ms,
            random_seed=22,
            r_e=r_e,
            C_m=fitted.capacitance_nF * nF,
            g_L=fitted.g_leak_nS * nS,
            E_L=fitted.e_leak_mV * mV,
            E_E=fitted.e_excitatory_mV * mV,
            tau_ampa=2 * ms,
            w_ampa=0.05 * nS,
            tau_nmda_rise=fitted.tau_rise_ms * ms,
            tau_nmda_decay=fitted.tau_decay_ms * ms,
            alpha_nmda=fitted.alpha_per_ms / ms,
            g_nmda_max=fitted.g_nmda_max_nS * nS,
            w_nmda=fitted.spike_weight,
            magnesium=1.0,
            initial_voltage_min=-80 * mV,
            initial_voltage_max=0 * mV,
            magnesium_block_tau=magnesium_block_tau,
        )


AMPA_NMDA_SINGLE_COMPARTMENT_MODEL = """
dV/dt = (-I_leak - I_AMPA - I_NMDA) / C_m : volt

dg_e_ampa_only/dt = -g_e_ampa_only / tau_AMPA : siemens
dg_e_shared/dt = -g_e_shared / tau_AMPA : siemens
g_e = g_e_ampa_only + g_e_shared : siemens
dx_NMDA/dt = -x_NMDA / tau_NMDA_rise : 1
ds_NMDA/dt = -s_NMDA / tau_NMDA_decay + alpha_NMDA * x_NMDA : 1

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

dg_e_ampa_only/dt = -g_e_ampa_only / tau_AMPA : siemens
dg_e_shared/dt = -g_e_shared / tau_AMPA : siemens
g_e = g_e_ampa_only + g_e_shared : siemens
dx_NMDA/dt = -x_NMDA / tau_NMDA_rise : 1
ds_NMDA/dt = -s_NMDA / tau_NMDA_decay + alpha_NMDA * x_NMDA : 1

sigma_Mg_inf = 1 / (1 + exp(-0.08 * V / mV) * Mg / 3.57) : 1
dsigma_Mg/dt = (sigma_Mg_inf - sigma_Mg) / tau_sigma_Mg : 1
g_NMDA_open = g_NMDA_max * s_NMDA : siemens
g_NMDA = g_NMDA_open * sigma_Mg : siemens

I_leak = g_L * (V - E_L) : amp
I_AMPA = g_e * (V - E_E) : amp
I_NMDA = g_NMDA * (V - E_E) : amp
"""


TRACE_MODEL_EQUATIONS = (
    r"$C_m dV/dt=-I_L-I_{AMPA}-I_{NMDA}$",
    r"$dg_{AMPA}/dt=-g_{AMPA}/tau_{AMPA}$",
    r"$g_{NMDA}=g_{NMDA,max}s_{NMDA}\sigma_{Mg}(V)$",
    r"$ds_{NMDA}/dt=-s_{NMDA}/tau_{decay}+alpha_{NMDA}x_{NMDA}$",
    r"$dx_{NMDA}/dt=-x_{NMDA}/tau_{rise}+w_{NMDA}\sum_k delta(t-t_k)$",
)
TRACE_DYNAMIC_MAGNESIUM_MODEL_EQUATIONS = TRACE_MODEL_EQUATIONS


def _fit_gaussian_mixture(
    values,
    number_of_components,
    random_seed=21,
    maximum_sample_count=500_000,
):
    """Fit a small one-dimensional Gaussian mixture by deterministic EM."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    maximum_sample_count = int(maximum_sample_count)
    if maximum_sample_count < 1:
        raise ValueError("maximum_sample_count must be at least 1")
    if values.size > maximum_sample_count:
        rng = np.random.default_rng(random_seed)
        values = values[
            rng.choice(values.size, maximum_sample_count, replace=False)
        ]
    if values.size < max(5, number_of_components + 1):
        raise ValueError("Not enough finite samples for Gaussian fitting")
    means = np.percentile(
        values,
        np.linspace(
            50 / number_of_components,
            100 - 50 / number_of_components,
            number_of_components,
        ),
    )
    overall_std = max(float(np.std(values)), np.finfo(float).eps)
    standard_deviations = np.full(number_of_components, overall_std)
    weights = np.full(number_of_components, 1.0 / number_of_components)
    minimum_std = max(overall_std * 1e-3, np.finfo(float).eps)
    for _ in range(200):
        old_parameters = np.r_[weights, means, standard_deviations]
        log_components = (
            np.log(np.maximum(weights, np.finfo(float).eps))[None, :]
            - np.log(standard_deviations)[None, :]
            - 0.5 * np.log(2 * np.pi)
            - 0.5
            * ((values[:, None] - means[None, :]) / standard_deviations[None, :])
            ** 2
        )
        responsibilities = np.exp(
            log_components - logsumexp(log_components, axis=1)[:, None]
        )
        masses = np.maximum(responsibilities.sum(axis=0), np.finfo(float).eps)
        weights = masses / masses.sum()
        means = (responsibilities * values[:, None]).sum(axis=0) / masses
        variances = (
            responsibilities * (values[:, None] - means[None, :]) ** 2
        ).sum(axis=0) / masses
        standard_deviations = np.maximum(np.sqrt(variances), minimum_std)
        if np.allclose(
            old_parameters,
            np.r_[weights, means, standard_deviations],
            rtol=1e-7,
            atol=1e-9,
        ):
            break
    order = np.argsort(means)
    weights = weights[order]
    means = means[order]
    standard_deviations = standard_deviations[order]
    grid = np.linspace(float(values.min()), float(values.max()), 400)
    components = (
        weights[None, :]
        / (standard_deviations[None, :] * np.sqrt(2 * np.pi))
        * np.exp(
            -0.5
            * ((grid[:, None] - means[None, :]) / standard_deviations[None, :])
            ** 2
        )
    )
    return {
        "weights": weights,
        "means": means,
        "standard_deviations": standard_deviations,
        "grid": grid,
        "component_density": components,
        "mixture_density": components.sum(axis=1),
    }


def _summed_shot_noise_summary(result, experiment):
    """Return common theory/data cumulant summaries for every synaptic state."""
    return compute_shot_noise_gaussianity_summary(
        result,
        ampa_input_rate_hz=float(experiment.n_e * experiment.r_e / Hz),
        nmda_input_rate_hz=float(experiment.n_nmda * experiment.r_e / Hz),
        ampa_tau_ms=float(experiment.tau_ampa / ms),
        ampa_weight_nS=float(experiment.w_ampa / nS),
        nmda_tau_rise_ms=float(experiment.tau_nmda_rise / ms),
        nmda_tau_decay_ms=float(experiment.tau_nmda_decay / ms),
        nmda_alpha_per_ms=float(experiment.alpha_nmda / kHz),
        nmda_weight=float(experiment.w_nmda),
    )


# For dx/dt=-x/tau_r+w*sum_k delta(t-t_k) and
# ds/dt=-s/tau_D+alpha*x, stationarity gives
# s_0 = alpha*tau_D*<x> = alpha*w*(N*r_e)*tau_r*tau_D.
def compute_theoretical_s0(experiment):
    """Return the stationary mean summed NMDA s state."""
    summed_rate_per_ms = float(experiment.n_nmda * experiment.r_e / kHz)
    return float(
        (experiment.alpha_nmda / kHz)
        * experiment.w_nmda
        * summed_rate_per_ms
        * (experiment.tau_nmda_rise / ms)
        * (experiment.tau_nmda_decay / ms)
    )


def compute_theoretical_ito_correction(experiment, voltage_mV):
    r"""Return the NMDA multiplicative-noise Itô drift correction [mV/ms].

    With C_x=alpha*w*tau_r and
    D_s=int_0^infinity Cov[s(t),s(t+tau)]d tau
       =N*r_e*C_x^2*tau_D^2/2, the correction is
    D_s*(g_NMDA/(N*C_m))^2*sigma(V)^2*(V-E_E)
    *[1+B*(V-E_E)*(1-sigma(V))].
    """
    voltage_mV = np.asarray(voltage_mV, dtype=float)
    tau_rise_ms = float(experiment.tau_nmda_rise / ms)
    tau_decay_ms = float(experiment.tau_nmda_decay / ms)
    alpha_per_ms = float(experiment.alpha_nmda / kHz)
    total_rate_per_ms = float(experiment.n_nmda * experiment.r_e / kHz)
    c_x = alpha_per_ms * float(experiment.w_nmda) * tau_rise_ms
    d_s_ms = total_rate_per_ms * c_x**2 * tau_decay_ms**2 / 2.0
    nmda_scale_per_ms = (
        float(experiment.g_nmda_max / nS)
        / int(experiment.n_nmda)
        / float(experiment.C_m / nF)
        / 1000.0
    )
    e_excitatory_mV = float(experiment.E_E / mV)
    voltage_from_excitation_mV = voltage_mV - e_excitatory_mV
    magnesium_block = float(experiment.magnesium) / 3.57
    magnesium_slope_per_mV = 0.08
    sigma_mg = 1.0 / (
        1.0
        + magnesium_block
        * np.exp(-magnesium_slope_per_mV * voltage_mV)
    )
    return (
        d_s_ms
        * nmda_scale_per_ms**2
        * sigma_mg**2
        * voltage_from_excitation_mV
        * (
            1.0
            + magnesium_slope_per_mV
            * voltage_from_excitation_mV
            * (1.0 - sigma_mg)
        )
    )


def plot_summed_nmda_three_neuron_diagnostics(
    result,
    experiment,
    *,
    chosen_model_indices=(0, 1, 2),
    output_path=SUMMED_OUTPUT_DIR / "summed_nmda_three_neuron_diagnostics.png",
    show_plots=False,
):
    """Plot three traces, population histograms, fits, and x shot noise."""
    if any(index < 0 or index >= experiment.n_models for index in chosen_model_indices):
        raise IndexError("chosen_model_indices must refer to simulated neurons")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        5,
        4,
        figsize=(20, 14),
        gridspec_kw={"width_ratios": [4.0, 1.1, 1.5, 3.2]},
    )
    names = ("voltage_mV", "g_ampa_nS", "effective_g_nmda_nS", "s_nmda", "x_nmda")
    labels = (
        "V [mV]",
        r"$g_{AMPA}$ [nS]",
        r"effective $g_{NMDA}$ [nS]",
        r"$s_{NMDA}$",
        r"summed $x_{NMDA}$",
    )
    colors = ("tab:green", "tab:blue", "tab:orange", "tab:red", "tab:purple")
    component_counts = (2, 1, 1, 1, None)
    for row, (name, label, color, number_of_components) in enumerate(
        zip(names, labels, colors, component_counts)
    ):
        values = np.asarray(result[name])
        for neuron_index in chosen_model_indices:
            axes[row, 0].plot(
                result["time_ms"],
                values[neuron_index],
                label=f"neuron {neuron_index}",
            )
        axes[row, 0].set_ylabel(label)
        axes[row, 0].legend(loc="best", fontsize=8)
        axes[row, 1].hist(
            values.ravel(), bins=50, density=True, orientation="horizontal",
            color=color, alpha=0.75,
        )
        axes[row, 1].set_xlabel("density")
        axes[row, 2].hist(
            values.ravel(), bins=50, density=True, orientation="horizontal",
            color="0.88",
        )
        if number_of_components is not None:
            fit = _fit_gaussian_mixture(values, number_of_components)
            for component in fit["component_density"].T:
                axes[row, 2].plot(component, fit["grid"], ":", color=color)
            axes[row, 2].plot(
                fit["mixture_density"], fit["grid"], color=color, linewidth=2
            )
        axes[row, 2].set_xlabel("fit density")

    shot_noise = _summed_shot_noise_summary(result, experiment)
    gaussianity_rows = format_shot_noise_gaussianity_rows(shot_noise)
    equations = result.get("trace_model_equations", TRACE_MODEL_EQUATIONS)
    for row, (axis, equation, cumulant_text) in enumerate(
        zip(axes[:, 3], equations, gaussianity_rows)
    ):
        axis.axis("off")
        axis.text(0.02, 0.72 if row else 0.5, equation, va="center", fontsize=9)
        if row:
            axis.text(
                0.02,
                0.27,
                cumulant_text,
                va="center",
                fontsize=8,
                fontfamily="monospace",
            )
    axes[0, 0].set_title("Three chosen neurons")
    axes[0, 1].set_title("population distribution")
    axes[0, 2].set_title("Gaussian fit")
    axes[0, 3].set_title(
        "model equation and standardized cumulants"
    )
    axes[-1, 0].set_xlabel("time [ms]")
    figure.suptitle(
        f"Summed AMPA–NMDA input: r_e={float(experiment.r_e / Hz):g} Hz, "
        f"N_e={experiment.n_e}, N_NMDA={experiment.n_nmda}\n"
        f"{_experiment_nmda_time_constants_title(experiment)}\n"
        f"Initial V: uniform[{float(experiment.initial_voltage_min / mV):g}, "
        f"{float(experiment.initial_voltage_max / mV):g}] mV; "
        f"seed={experiment.random_seed}; realized min/mean/max="
        f"{np.min(result['initial_voltage_mV']):.4g}/"
        f"{np.mean(result['initial_voltage_mV']):.4g}/"
        f"{np.max(result['initial_voltage_mV']):.4g} mV",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plots:
        plt.show(block=False)
    else:
        plt.close(figure)
    return output_path, shot_noise


def simulate_ampa_nmda_single_compartment_ensemble_with_summed_nmda_input(
    experiment,
):
    """Simulate one aggregate shared AMPA--NMDA stream per compartment.

    ``n_e - n_nmda`` independent AMPA-only inputs retain rate ``r_e``.  The
    superposition theorem for independent Poisson processes lets the shared
    inputs be represented by one stream with rate ``n_nmda * r_e``.  Each
    shared event increments both ``g_e_shared`` and the single summed
    ``x_NMDA`` state by their per-synapse weights.
    """
    n_models = experiment.n_models
    n_e = experiment.n_e
    n_nmda = experiment.n_nmda
    if not 0 < n_nmda <= n_e:
        raise ValueError("n_nmda must satisfy 0 < n_nmda <= n_e")

    initial_voltage_min_mV = float(experiment.initial_voltage_min / mV)
    initial_voltage_max_mV = float(experiment.initial_voltage_max / mV)
    if initial_voltage_min_mV > initial_voltage_max_mV:
        raise ValueError("initial_voltage_min must not exceed initial_voltage_max")

    start_scope()
    seed(experiment.random_seed)
    clock = Clock(dt=experiment.dt)
    dynamic_magnesium_block = experiment.magnesium_block_tau is not None
    model_equations = (
        AMPA_NMDA_DYNAMIC_MAGNESIUM_MODEL
        if dynamic_magnesium_block
        else AMPA_NMDA_SINGLE_COMPARTMENT_MODEL
    )
    namespace = {
        "C_m": experiment.C_m,
        "g_L": experiment.g_L,
        "E_L": experiment.E_L,
        "E_E": experiment.E_E,
        "tau_AMPA": experiment.tau_ampa,
        "tau_NMDA_rise": experiment.tau_nmda_rise,
        "tau_NMDA_decay": experiment.tau_nmda_decay,
        "alpha_NMDA": experiment.alpha_nmda,
        # x and s now contain the sum over n_nmda inputs, so retain the
        # original per-input normalization of the open NMDA conductance.
        "g_NMDA_max": experiment.g_nmda_max / n_nmda,
        "Mg": experiment.magnesium,
    }
    if dynamic_magnesium_block:
        namespace["tau_sigma_Mg"] = experiment.magnesium_block_tau

    compartments = NeuronGroup(
        n_models,
        model_equations,
        method="euler",
        namespace=namespace,
        clock=clock,
        name="summed_input_single_compartments",
    )
    initial_voltage_mV = np.random.default_rng(experiment.random_seed).uniform(
        initial_voltage_min_mV,
        initial_voltage_max_mV,
        size=n_models,
    )
    compartments.V = initial_voltage_mV * mV
    compartments.g_e_ampa_only = 0 * nS
    compartments.g_e_shared = 0 * nS
    compartments.x_NMDA = 0
    compartments.s_NMDA = 0
    if dynamic_magnesium_block:
        resting_sigma_mg = 1.0 / (
            1.0
            + np.exp(-0.08 * initial_voltage_mV)
            * float(experiment.magnesium)
            / 3.57
        )
        compartments.sigma_Mg = resting_sigma_mg

    network_objects = [compartments]
    number_of_ampa_only_inputs = n_e - n_nmda
    if number_of_ampa_only_inputs:
        ampa_only_input = PoissonInput(
            compartments,
            "g_e_ampa_only",
            N=number_of_ampa_only_inputs,
            rate=experiment.r_e,
            weight=experiment.w_ampa,
        )
        network_objects.append(ampa_only_input)

    summed_source_rate = n_nmda * experiment.r_e
    summed_sources = PoissonGroup(
        n_models,
        rates=summed_source_rate,
        clock=clock,
        name="summed_ampa_nmda_sources",
    )
    summed_synapses = Synapses(
        summed_sources,
        compartments,
        on_pre="""
        g_e_shared_post += w_AMPA
        x_NMDA_post += w_NMDA
        """,
        namespace={
            "w_AMPA": experiment.w_ampa,
            "w_NMDA": experiment.w_nmda,
        },
        clock=clock,
        name="summed_ampa_nmda_synapses",
    )
    summed_synapses.connect(j="i")
    network_objects.extend([summed_sources, summed_synapses])

    state = StateMonitor(
        compartments,
        [
            "V",
            "g_e",
            "g_e_ampa_only",
            "g_e_shared",
            "g_NMDA",
            "s_NMDA",
            "x_NMDA",
            "sigma_Mg",
            "sigma_Mg_inf",
        ],
        record=True,
        clock=clock,
    )
    network_objects.append(state)
    Network(*network_objects).run(experiment.simulation_time)

    sample_mask = state.t >= experiment.burn_in
    voltage_mV = np.asarray(state.V[:, sample_mask] / mV)
    g_ampa_nS = np.asarray(state.g_e[:, sample_mask] / nS)
    result = {
        "model_equations": model_equations,
        "trace_model_equations": (
            TRACE_DYNAMIC_MAGNESIUM_MODEL_EQUATIONS
            if dynamic_magnesium_block
            else TRACE_MODEL_EQUATIONS
        ),
        "magnesium_block_tau_ms": (
            float(experiment.magnesium_block_tau / ms)
            if dynamic_magnesium_block
            else None
        ),
        "initial_voltage_mV": initial_voltage_mV,
        "number_of_summed_sources": n_models,
        "summed_source_rate_hz": float(summed_source_rate / Hz),
        "time_ms": np.asarray(state.t[sample_mask] / ms),
        "voltage_mV": voltage_mV,
        "g_ampa_nS": g_ampa_nS,
        "g_ampa_only_nS": np.asarray(state.g_e_ampa_only[:, sample_mask] / nS),
        "g_ampa_shared_nS": np.asarray(state.g_e_shared[:, sample_mask] / nS),
        "effective_g_nmda_nS": np.asarray(state.g_NMDA[:, sample_mask] / nS),
        "s_nmda": np.asarray(state.s_NMDA[:, sample_mask]),
        "x_nmda": np.asarray(state.x_NMDA[:, sample_mask]),
        "sigma_mg": np.asarray(state.sigma_Mg[:, sample_mask]),
        "sigma_mg_inf": np.asarray(state.sigma_Mg_inf[:, sample_mask]),
        "per_model_mean_mV": np.mean(voltage_mV, axis=1),
        "per_model_variance_mV2": np.var(voltage_mV, axis=1),
        "ensemble_mean_mV": float(np.mean(voltage_mV)),
        "ensemble_variance_mV2": float(np.var(voltage_mV)),
    }
    return result


SUMMED_NMDA_TRACE_ARRAY_COLUMNS = (
    "time_ms",
    "voltage_mV",
    "effective_g_nmda_nS",
    "s_nmda",
    "g_ampa_nS",
)
SUMMED_NMDA_TRACE_METADATA_COLUMNS = (
    "r_e_hz",
    "unit_index",
    "random_seed",
    "n_nmda",
    "g_nmda_max_nS",
    "g_leak_nS",
    "e_leak_mV",
    "e_excitatory_mV",
    "capacitance_nF",
    "magnesium",
    "tau_nmda_rise_ms",
    "tau_nmda_decay_ms",
    "alpha_nmda_per_ms",
    "w_nmda",
)


def save_summed_nmda_unit_trace_csv(result, experiment, unit_index, output_path):
    """Save one unit's trace plus the constants required for later nullclines."""
    unit_index = int(unit_index)
    if not 0 <= unit_index < experiment.n_models:
        raise IndexError(f"unit_index {unit_index} is outside 0..{experiment.n_models - 1}")
    metadata = {
        "r_e_hz": float(experiment.r_e / Hz),
        "unit_index": unit_index,
        "random_seed": int(experiment.random_seed),
        "n_nmda": int(experiment.n_nmda),
        "g_nmda_max_nS": float(experiment.g_nmda_max / nS),
        "g_leak_nS": float(experiment.g_L / nS),
        "e_leak_mV": float(experiment.E_L / mV),
        "e_excitatory_mV": float(experiment.E_E / mV),
        "capacitance_nF": float(experiment.C_m / nF),
        "magnesium": float(experiment.magnesium),
        "tau_nmda_rise_ms": float(experiment.tau_nmda_rise / ms),
        "tau_nmda_decay_ms": float(experiment.tau_nmda_decay / ms),
        "alpha_nmda_per_ms": float(experiment.alpha_nmda / kHz),
        "w_nmda": float(experiment.w_nmda),
    }
    arrays = {
        "time_ms": np.asarray(result["time_ms"], dtype=float),
        "voltage_mV": np.asarray(result["voltage_mV"][unit_index], dtype=float),
        "effective_g_nmda_nS": np.asarray(
            result["effective_g_nmda_nS"][unit_index], dtype=float
        ),
        "s_nmda": np.asarray(result["s_nmda"][unit_index], dtype=float),
        "g_ampa_nS": np.asarray(result["g_ampa_nS"][unit_index], dtype=float),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (*SUMMED_NMDA_TRACE_METADATA_COLUMNS, *SUMMED_NMDA_TRACE_ARRAY_COLUMNS)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index in range(len(arrays["time_ms"])):
            writer.writerow(
                metadata
                | {name: arrays[name][sample_index] for name in SUMMED_NMDA_TRACE_ARRAY_COLUMNS}
            )
    return output_path


def load_summed_nmda_unit_trace_csv(input_path):
    """Load a trace CSV written by :func:`save_summed_nmda_unit_trace_csv`."""
    with Path(input_path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"saved trace CSV is empty: {input_path}")
    trace = {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in SUMMED_NMDA_TRACE_ARRAY_COLUMNS
    }
    first = rows[0]
    for name in SUMMED_NMDA_TRACE_METADATA_COLUMNS:
        converter = int if name in {"unit_index", "random_seed", "n_nmda"} else float
        if name in first:
            trace[name] = converter(first[name])
    if "alpha_nmda_per_ms" not in trace or "w_nmda" not in trace:
        from iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20NullclineScripts import (
            read_fitted_apic20_parameters_from_csv,
        )

        fitted = read_fitted_apic20_parameters_from_csv()
        trace.setdefault("alpha_nmda_per_ms", float(fitted.alpha_per_ms))
        trace.setdefault("w_nmda", float(fitted.spike_weight))
    return trace


def _saved_trace_nullcline(trace, index, voltage_grid_mV):
    sigma = 1.0 / (
        1.0
        + np.exp(-0.08 * voltage_grid_mV) * float(trace["magnesium"]) / 3.57
    )
    open_g_nmda_nS = (
        float(trace["g_nmda_max_nS"])
        / int(trace["n_nmda"])
        * float(trace["s_nmda"][index])
    )
    total_current_pA = (
        float(trace["g_leak_nS"])
        * (voltage_grid_mV - float(trace["e_leak_mV"]))
        + float(trace["g_ampa_nS"][index])
        * (voltage_grid_mV - float(trace["e_excitatory_mV"]))
        + open_g_nmda_nS
        * sigma
        * (voltage_grid_mV - float(trace["e_excitatory_mV"]))
    )
    return -1e-3 * total_current_pA / float(trace["capacitance_nF"])


def plot_summed_nmda_trace_snapshot(saved_trace, snapshot_time_ms, output_path):
    """Plot one selected saved-trace iteration in the animation's layout."""
    from iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20NullclineScripts import (
        classify_intersection_stability,
        create_fitted_nullcline_figure,
        derivative_ylim,
        estimate_zero_crossings,
    )

    time_ms = np.asarray(saved_trace["time_ms"], dtype=float)
    index = int(np.argmin(np.abs(time_ms - float(snapshot_time_ms))))
    voltage_grid_mV = np.linspace(-90.0, 20.0, 500)
    nullcline = _saved_trace_nullcline(saved_trace, index, voltage_grid_mV)
    axes = create_fitted_nullcline_figure()
    figure = axes["fig"]
    axes["model_ax"].axis("off")
    axes["model_ax"].text(
        0.5,
        0.5,
        r"$C_m dV/dt=-I_L-I_{AMPA}-I_{NMDA}$" "\n"
        r"$g_{NMDA}=g_{NMDA,max}s(t)\sigma_{Mg}(V)$",
        ha="center",
        va="center",
        fontsize=12,
    )
    panels = (
        ("voltage_ax", "voltage_mV", r"$V(t)$", "V [mV]", "tab:blue"),
        (
            "g_nmda_ax",
            "effective_g_nmda_nS",
            r"$g_{NMDA}(t)$",
            "g_NMDA [nS]",
            "tab:green",
        ),
        ("s_ax", "s_nmda", r"$s(t)$", "s", "tab:purple"),
    )
    for axis_name, value_name, title_text, ylabel, color in panels:
        values = np.asarray(saved_trace[value_name], dtype=float)
        axis = axes[axis_name]
        axis.plot(time_ms, values, color=color, linewidth=1.5)
        axis.plot([time_ms[index]], [values[index]], "o", color=color)
        axis.axvline(time_ms[index], color="0.35", linestyle="--", linewidth=1)
        axis.set(title=title_text, xlabel="t [ms]", ylabel=ylabel)
    nullcline_axis = axes["nullcline_ax"]
    nullcline_axis.axhline(0.0, color="black", linewidth=1)
    nullcline_axis.plot(voltage_grid_mV, nullcline, color="tab:orange", linewidth=2)
    current_v_mV = float(saved_trace["voltage_mV"][index])
    current_dvdt = float(np.interp(current_v_mV, voltage_grid_mV, nullcline))
    nullcline_axis.plot([current_v_mV], [current_dvdt], "o", color="tab:orange")
    nullcline_axis.axvline(current_v_mV, color="tab:blue", linewidth=1)
    intersections = estimate_zero_crossings(voltage_grid_mV, nullcline)
    stable, unstable = classify_intersection_stability(intersections)
    nullcline_axis.plot(stable, np.zeros_like(stable), "o", color="black")
    nullcline_axis.plot(
        unstable,
        np.zeros_like(unstable),
        "o",
        color="black",
        markerfacecolor="none",
    )
    nullcline_axis.set(
        xlim=(-90, 20),
        ylim=derivative_ylim(nullcline),
        title=r"$dV/dt(V; s(t), g_{AMPA}(t))$",
        xlabel="V [mV]",
        ylabel="dV/dt [mV/ms]",
    )
    figure.suptitle(
        f"Saved summed-NMDA trace: r_e={saved_trace['r_e_hz']:g} Hz, "
        f"unit={saved_trace['unit_index']}, seed={saved_trace['random_seed']}\n"
        f"{_nmda_time_constants_title(saved_trace['tau_nmda_rise_ms'], saved_trace['tau_nmda_decay_ms'])}\n"
        f"requested t={float(snapshot_time_ms):g} ms, sampled t={time_ms[index]:g} ms, "
        f"V={current_v_mV:.2f} mV"
    )
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.86)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_summed_nmda_moving_nullcline_animation(
    result,
    experiment,
    request,
    output_path,
    *,
    frame_count=None,
    fps=None,
    display_unit_index=None,
):
    """Animate V, effective g_NMDA, s, and the instantaneous V-nullcline."""
    from iteration_20_nonlin_dynamics_NMDA.SimulateFittedApic20NullclineScripts import (
        classify_intersection_stability,
        create_fitted_nullcline_figure,
        derivative_ylim,
        estimate_zero_crossings,
    )

    unit_index = int(request.unit_index)
    if display_unit_index is None:
        display_unit_index = unit_index
    frame_count = request.frame_count if frame_count is None else frame_count
    fps = request.fps if fps is None else fps
    output_format = str(request.output_format).lower().lstrip(".")
    if output_format not in {"gif", "mp4"}:
        raise ValueError("animation output_format must be 'gif' or 'mp4'")
    if int(frame_count) < 1 or int(fps) < 1:
        raise ValueError("animation frame_count and fps must be at least 1")
    if not 0 <= unit_index < experiment.n_models:
        raise IndexError(
            f"animation unit_index {unit_index} is outside 0..{experiment.n_models - 1}"
        )
    time_ms = np.asarray(result["time_ms"], dtype=float)
    t_min_ms = float(time_ms[0] if request.t_min_ms is None else request.t_min_ms)
    t_max_ms = float(time_ms[-1] if request.t_max_ms is None else request.t_max_ms)
    if t_min_ms > t_max_ms:
        raise ValueError("animation t_min_ms must not exceed t_max_ms")
    candidate_indexes = np.flatnonzero((time_ms >= t_min_ms) & (time_ms <= t_max_ms))
    if not len(candidate_indexes):
        raise ValueError(
            f"animation time range {t_min_ms:g}..{t_max_ms:g} ms has no samples"
        )
    frame_indexes = np.unique(
        np.linspace(
            candidate_indexes[0],
            candidate_indexes[-1],
            min(int(frame_count), len(candidate_indexes)),
            dtype=int,
        )
    )
    voltage_mV = np.asarray(result["voltage_mV"][unit_index], dtype=float)
    effective_g_nmda_nS = np.asarray(
        result["effective_g_nmda_nS"][unit_index], dtype=float
    )
    s_nmda = np.asarray(result["s_nmda"][unit_index], dtype=float)
    g_ampa_nS = np.asarray(result["g_ampa_nS"][unit_index], dtype=float)
    voltage_grid_mV = np.linspace(-90.0, 20.0, 500)
    magnesium_coefficient = float(experiment.magnesium) / 3.57
    sigma_grid = 1.0 / (
        1.0 + np.exp(-0.08 * voltage_grid_mV) * magnesium_coefficient
    )
    g_leak_nS = float(experiment.g_L / nS)
    e_leak_mV = float(experiment.E_L / mV)
    e_excitatory_mV = float(experiment.E_E / mV)
    capacitance_nF = float(experiment.C_m / nF)
    g_nmda_max_per_source_nS = float(experiment.g_nmda_max / nS) / experiment.n_nmda

    def nullcline_for(s_value, g_ampa_value_nS):
        open_g_nmda_nS = g_nmda_max_per_source_nS * s_value
        total_current_pA = (
            g_leak_nS * (voltage_grid_mV - e_leak_mV)
            + g_ampa_value_nS * (voltage_grid_mV - e_excitatory_mV)
            + open_g_nmda_nS
            * sigma_grid
            * (voltage_grid_mV - e_excitatory_mV)
        )
        return -1e-3 * total_current_pA / capacitance_nF

    nullclines = [
        nullcline_for(s_nmda[index], g_ampa_nS[index])
        for index in frame_indexes
    ]
    theoretical_s0 = compute_theoretical_s0(experiment)
    theoretical_s0_nullcline = nullcline_for(
        theoretical_s0,
        float(np.mean(g_ampa_nS[candidate_indexes])),
    )
    theoretical_ito_corrected_nullcline = (
        theoretical_s0_nullcline
        + compute_theoretical_ito_correction(experiment, voltage_grid_mV)
    )
    axes = create_fitted_nullcline_figure()
    figure = axes["fig"]
    axes["model_ax"].axis("off")
    axes["model_ax"].text(
        0.5,
        0.5,
        r"$C_m dV/dt=-I_L-I_{AMPA}-I_{NMDA}$" "\n"
        r"$g_{NMDA}=g_{NMDA,max}s(t)\sigma_{Mg}(V)$",
        ha="center",
        va="center",
        fontsize=12,
    )
    panels = (
        ("voltage_ax", voltage_mV, r"$V(t)$", "V [mV]", "tab:blue"),
        ("g_nmda_ax", effective_g_nmda_nS, r"$g_{NMDA}(t)$", "g_NMDA [nS]", "tab:green"),
        ("s_ax", s_nmda, r"$s(t)$", "s", "tab:purple"),
    )
    active_lines = []
    points = []
    for axis_name, values, title_text, ylabel, color in panels:
        axis = axes[axis_name]
        axis.plot(time_ms[candidate_indexes], values[candidate_indexes], color="0.75")
        axis.set(xlim=(t_min_ms, t_max_ms), title=title_text, xlabel="t [ms]", ylabel=ylabel)
        value_min = float(np.min(values[candidate_indexes]))
        value_max = float(np.max(values[candidate_indexes]))
        padding = max((value_max - value_min) * 0.1, 1e-6)
        axis.set_ylim(value_min - padding, value_max + padding)
        active_lines.append(axis.plot([], [], color=color, linewidth=2)[0])
        points.append(axis.plot([], [], "o", color=color)[0])
    nullcline_axis = axes["nullcline_ax"]
    nullcline_axis.axhline(0.0, color="black", linewidth=1)
    # Draw the stationary s_0 reference first, behind the moving nullcline.
    nullcline_axis.plot(
        voltage_grid_mV,
        theoretical_s0_nullcline,
        linestyle="-.",
        color="tab:red",
        linewidth=1.8,
        label=rf"theoretical $s_0={theoretical_s0:.4g}$",
    )
    nullcline_axis.plot(
        voltage_grid_mV,
        theoretical_ito_corrected_nullcline,
        linestyle=":",
        color="tab:purple",
        linewidth=2.0,
        label=r"theoretical $s_0$ nullcline + Itô correction",
    )
    theoretical_intersections = estimate_zero_crossings(
        voltage_grid_mV, theoretical_s0_nullcline
    )
    theoretical_stable, theoretical_unstable = classify_intersection_stability(
        theoretical_intersections
    )
    nullcline_axis.plot(
        theoretical_stable,
        np.zeros_like(theoretical_stable),
        "D",
        color="tab:red",
        markerfacecolor="tab:red",
        markersize=6,
        linestyle="None",
        label=r"theoretical $s_0$ stable fixed point",
    )
    if theoretical_unstable.size:
        nullcline_axis.plot(
            theoretical_unstable,
            np.zeros_like(theoretical_unstable),
            "D",
            color="tab:red",
            markerfacecolor="none",
            markersize=6,
            linestyle="None",
            label=r"theoretical $s_0$ unstable fixed point",
        )
    nullcline_axis.set(
        xlim=(-90, 20),
        ylim=derivative_ylim(
            np.concatenate(
                [
                    *nullclines,
                    theoretical_s0_nullcline,
                    theoretical_ito_corrected_nullcline,
                ]
            )
        ),
        title=r"$dV/dt(V; s(t), g_{AMPA}(t))$",
        xlabel="V [mV]",
        ylabel="dV/dt [mV/ms]",
    )
    nullcline_line = nullcline_axis.plot([], [], color="tab:orange", linewidth=2)[0]
    nullcline_axis.legend(loc="best", fontsize=8)
    nullcline_point = nullcline_axis.plot([], [], "o", color="tab:orange")[0]
    stable_points = nullcline_axis.plot([], [], "o", color="black", linestyle="None")[0]
    unstable_points = nullcline_axis.plot(
        [], [], "o", color="black", markerfacecolor="none", linestyle="None"
    )[0]
    voltage_line = nullcline_axis.axvline(0.0, color="tab:blue", linewidth=1)
    title = figure.suptitle("")

    def update(frame_number):
        index = int(frame_indexes[frame_number])
        visible = np.arange(candidate_indexes[0], index + 1)
        for line, point, (_, values, _, _, _) in zip(active_lines, points, panels):
            line.set_data(time_ms[visible], values[visible])
            point.set_data([time_ms[index]], [values[index]])
        nullcline = nullclines[frame_number]
        nullcline_line.set_data(voltage_grid_mV, nullcline)
        current_v_mV = voltage_mV[index]
        current_dvdt = float(np.interp(current_v_mV, voltage_grid_mV, nullcline))
        nullcline_point.set_data([current_v_mV], [current_dvdt])
        voltage_line.set_xdata([current_v_mV, current_v_mV])
        intersections = estimate_zero_crossings(voltage_grid_mV, nullcline)
        stable, unstable = classify_intersection_stability(intersections)
        stable_points.set_data(stable, np.zeros_like(stable))
        unstable_points.set_data(unstable, np.zeros_like(unstable))
        title.set_text(
            f"Summed NMDA moving nullcline: r_e={request.r_e_hz:g} Hz, "
            f"unit={display_unit_index}, seed={experiment.random_seed}\n"
            f"{_experiment_nmda_time_constants_title(experiment)}\n"
            f"t={time_ms[index]:.1f} ms, V={current_v_mV:.2f} mV, "
            f"g_NMDA={effective_g_nmda_nS[index]:.3g} nS, "
            f"s={s_nmda[index]:.3g}, theoretical s_0={theoretical_s0:.4g}"
        )
        return (*active_lines, *points, nullcline_line, nullcline_point,
                stable_points, unstable_points, voltage_line, title)

    animation = FuncAnimation(
        figure, update, frames=len(frame_indexes), interval=1000 / fps, blit=False
    )
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.86)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = (
        PillowWriter(fps=fps)
        if output_format == "gif"
        else FFMpegWriter(fps=fps, codec="mpeg4", bitrate=2400)
    )
    animation.save(output_path, writer=writer, dpi=150)
    plt.close(figure)
    return output_path


def generate_summed_nmda_moving_nullcline_animation_from_saved_trace(
    saved_trace,
    output_path,
    *,
    t_min_ms=None,
    t_max_ms=None,
    frame_count=1_000,
    fps=20,
    output_format="mp4",
):
    """Regenerate a complete animation from a saved unit trace CSV payload."""
    result = {
        "time_ms": np.asarray(saved_trace["time_ms"], dtype=float),
        "voltage_mV": np.asarray(saved_trace["voltage_mV"], dtype=float)[None, :],
        "effective_g_nmda_nS": np.asarray(
            saved_trace["effective_g_nmda_nS"], dtype=float
        )[None, :],
        "s_nmda": np.asarray(saved_trace["s_nmda"], dtype=float)[None, :],
        "g_ampa_nS": np.asarray(saved_trace["g_ampa_nS"], dtype=float)[None, :],
    }
    experiment = SimpleNamespace(
        n_models=1,
        n_nmda=int(saved_trace["n_nmda"]),
        r_e=float(saved_trace["r_e_hz"]) * Hz,
        random_seed=int(saved_trace["random_seed"]),
        magnesium=float(saved_trace["magnesium"]),
        g_nmda_max=float(saved_trace["g_nmda_max_nS"]) * nS,
        g_L=float(saved_trace["g_leak_nS"]) * nS,
        E_L=float(saved_trace["e_leak_mV"]) * mV,
        E_E=float(saved_trace["e_excitatory_mV"]) * mV,
        C_m=float(saved_trace["capacitance_nF"]) * nF,
        tau_nmda_rise=float(saved_trace["tau_nmda_rise_ms"]) * ms,
        tau_nmda_decay=float(saved_trace["tau_nmda_decay_ms"]) * ms,
        alpha_nmda=float(saved_trace["alpha_nmda_per_ms"]) * kHz,
        w_nmda=float(saved_trace["w_nmda"]),
    )
    request = SummedNmdaAnimationRequest(
        r_e_hz=float(saved_trace["r_e_hz"]),
        unit_index=0,
        t_min_ms=t_min_ms,
        t_max_ms=t_max_ms,
        output_format=output_format,
        frame_count=frame_count,
        fps=fps,
    )
    return generate_summed_nmda_moving_nullcline_animation(
        result,
        experiment,
        request,
        output_path,
        display_unit_index=int(saved_trace["unit_index"]),
    )


def compute_summed_per_unit_s0_and_fixed_points(
    result,
    experiment,
    max_fixed_point_samples=FIXED_POINT_STATE_SAMPLE_COUNT,
):
    """Calculate fixed points from sampled post-burn-in neuron/time states.

    ``s0`` remains the vector of per-neuron mean s values used by the lower
    plot.  Fixed points instead use up to ``max_fixed_point_samples`` actual s
    observations, paired with the corresponding neuron's mean AMPA
    conductance, so the overlaid distribution is not limited to one point per
    neuron.
    """
    s0 = np.mean(np.asarray(result["s_nmda"], dtype=float), axis=1)
    g_ampa_zero_nS = np.mean(
        np.asarray(result["g_ampa_nS"], dtype=float), axis=1
    )
    s_states = np.asarray(result["s_nmda"], dtype=float)
    number_of_states = s_states.size
    sample_count = min(number_of_states, int(max_fixed_point_samples))
    if sample_count < 1:
        raise ValueError("max_fixed_point_samples must be at least 1")
    if sample_count < number_of_states:
        flat_indices = np.random.default_rng(0).choice(
            number_of_states, sample_count, replace=False
        )
    else:
        flat_indices = np.arange(number_of_states)
    unit_indices = flat_indices // s_states.shape[1]
    sampled_s = s_states.ravel()[flat_indices]
    sampled_g_ampa_zero_nS = g_ampa_zero_nS[unit_indices]
    open_g_nmda_nS = (
        float(experiment.g_nmda_max / nS) / experiment.n_nmda * sampled_s
    )
    lower_fixed_point_mV = [None] * sample_count
    upper_fixed_point_mV = [None] * sample_count
    magnesium_coefficient = float(experiment.magnesium) / 3.57
    g_leak_nS = float(experiment.g_L / nS)
    e_leak_mV = float(experiment.E_L / mV)
    e_excitatory_mV = float(experiment.E_E / mV)
    for unit_index, open_g_nmda in enumerate(open_g_nmda_nS):
        effective_leak_nS = g_leak_nS + sampled_g_ampa_zero_nS[unit_index]
        effective_reversal_mV = (
            g_leak_nS * e_leak_mV
            + sampled_g_ampa_zero_nS[unit_index] * e_excitatory_mV
        ) / effective_leak_nS
        roots = find_fixed_points(
            open_g_nmda,
            g_l=effective_leak_nS,
            e_l=effective_reversal_mV,
            e_e=e_excitatory_mV,
            a=magnesium_coefficient,
            b=0.08,
        )
        stable_roots = [
            voltage
            for voltage in roots
            if fixed_point_current_slope(
                voltage,
                open_g_nmda,
                g_l=effective_leak_nS,
                e_e=e_excitatory_mV,
                a=magnesium_coefficient,
                b=0.08,
            )
            > 0.0
        ]
        if len(stable_roots) >= 2:
            lower_fixed_point_mV[unit_index] = float(stable_roots[0])
            upper_fixed_point_mV[unit_index] = float(stable_roots[-1])
        elif stable_roots:
            target = (
                lower_fixed_point_mV
                if stable_roots[0] < -45.0
                else upper_fixed_point_mV
            )
            target[unit_index] = float(stable_roots[0])
    return {
        "s0": s0,
        "fixed_point_s_samples": sampled_s,
        "open_g_nmda_nS": open_g_nmda_nS,
        "lower_fixed_point_mV": lower_fixed_point_mV,
        "upper_fixed_point_mV": upper_fixed_point_mV,
    }


def plot_voltage_and_fixed_point_distributions(
    result,
    fixed_point_result,
    output_path,
    *,
    show_plots=False,
    experiment=None,
):
    """Plot voltage/fixed-point distributions and per-neuron mean s below."""
    voltage_mV = np.asarray(result["voltage_mV"], dtype=float).ravel()
    if voltage_mV.size > 500_000:
        indices = np.random.default_rng(0).choice(
            voltage_mV.size, 500_000, replace=False
        )
        voltage_mV = voltage_mV[indices]
    finite_lower = np.asarray(
        [
            value
            for value in fixed_point_result["lower_fixed_point_mV"]
            if value is not None
        ],
        dtype=float,
    )
    finite_upper = np.asarray(
        [
            value
            for value in fixed_point_result["upper_fixed_point_mV"]
            if value is not None
        ],
        dtype=float,
    )
    s0 = np.asarray(fixed_point_result["s0"], dtype=float)
    figure, (voltage_axis, s0_axis) = plt.subplots(
        2, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [3, 1]}
    )
    voltage_axis.hist(
        voltage_mV,
        bins=150,
        density=True,
        histtype="step",
        linewidth=2,
        color="tab:green",
        label=f"simulated V (sample n={voltage_mV.size:,})",
    )
    total_fixed_points = finite_lower.size + finite_upper.size
    for values, color, label in (
        (finite_lower, "tab:blue", "lower stable fixed point"),
        (finite_upper, "tab:orange", "upper stable fixed point"),
    ):
        if values.size:
            bin_edges = np.histogram_bin_edges(values, bins=150)
            bin_width = bin_edges[1] - bin_edges[0]
            weights = np.full(
                values.size, 1.0 / (total_fixed_points * bin_width)
            )
            voltage_axis.hist(
                values,
                bins=bin_edges,
                weights=weights,
                histtype="step",
                linewidth=2,
                color=color,
                label=f"{label} (n={values.size:,})",
            )
    voltage_axis.set_xlabel("voltage [mV]")
    voltage_axis.set_ylabel("probability density")
    voltage_axis.set_title(
        "Simulated voltage versus fixed points calculated per neuron"
    )
    voltage_axis.legend()
    s0_axis.hist(
        s0,
        bins=min(50, max(10, int(np.sqrt(s0.size)))),
        color="tab:red",
        alpha=0.65,
    )
    s0_axis.set_xlabel(r"per-neuron mean $s_0^{(i)}=\langle s_{NMDA}^{(i)}\rangle$")
    s0_axis.set_ylabel("number of neurons")
    s0_axis.set_title(f"Distribution of all {s0.size:,} per-neuron mean s values")
    if experiment is not None:
        figure.suptitle(_experiment_nmda_time_constants_title(experiment))
        figure.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plots:
        plt.show(block=False)
    else:
        plt.close(figure)
    return output_path


def _run_one_summed_input_strength(
    parameter_name,
    value,
    base_experiment,
    output_dir,
    diagnostic_model_indices,
    plot_fixed_point_distributions,
    show_plots,
    animation_requests,
):
    """Pickle-safe worker for one independent Brian2 strength simulation."""
    experiment = base_experiment.with_property(**{parameter_name: value})
    result = simulate_ampa_nmda_single_compartment_ensemble_with_summed_nmda_input(
        experiment
    )
    display_value = (
        float(value / Hz)
        if parameter_name == "r_e"
        else float(value / nS)
        if parameter_name == "w_ampa"
        else float(value)
    )
    row = {
        "input_strength": display_value,
        "initial_voltage_distribution": "uniform",
        "initial_voltage_min_mV": float(experiment.initial_voltage_min / mV),
        "initial_voltage_max_mV": float(experiment.initial_voltage_max / mV),
        "initial_voltage_seed": experiment.random_seed,
        "realized_initial_voltage_min_mV": float(
            np.min(result["initial_voltage_mV"])
        ),
        "realized_initial_voltage_max_mV": float(
            np.max(result["initial_voltage_mV"])
        ),
        "realized_initial_voltage_mean_mV": float(
            np.mean(result["initial_voltage_mV"])
        ),
        "ensemble_mean_mV": result["ensemble_mean_mV"],
        "ensemble_variance_mV2": result["ensemble_variance_mV2"],
        "mean_x_nmda": float(np.mean(result["x_nmda"])),
        "mean_s_nmda": float(np.mean(result["s_nmda"])),
        "mean_g_ampa_nS": float(np.mean(result["g_ampa_nS"])),
        "mean_effective_g_nmda_nS": float(
            np.mean(result["effective_g_nmda_nS"])
        ),
        "tau_nmda_rise_ms": float(experiment.tau_nmda_rise / ms),
        "tau_nmda_decay_ms": float(experiment.tau_nmda_decay / ms),
    }
    saved_trace_units = set()
    for animation_request in animation_requests:
        if animation_request.unit_index not in saved_trace_units:
            save_summed_nmda_unit_trace_csv(
                result,
                experiment,
                animation_request.unit_index,
                Path(output_dir)
                / "trace_data"
                / summed_nmda_trace_filename(
                    display_value,
                    animation_request.unit_index,
                    experiment.random_seed,
                ),
            )
            saved_trace_units.add(animation_request.unit_index)
        animation_path = (
            Path(output_dir)
            / "animations"
            / summed_nmda_animation_filename(
                animation_request, experiment.random_seed
            )
        )
        generate_summed_nmda_moving_nullcline_animation(
            result,
            experiment,
            animation_request,
            animation_path,
        )
    if diagnostic_model_indices is not None:
        plot_summed_nmda_three_neuron_diagnostics(
            result,
            experiment,
            chosen_model_indices=diagnostic_model_indices,
            output_path=(
                Path(output_dir)
                / f"summed_nmda_{parameter_name}_{display_value:g}_diagnostics.png"
            ),
            show_plots=show_plots,
        )
        selected_indices = tuple(diagnostic_model_indices)[
            :NUMBER_OF_COMPARTMENT_DISTRIBUTIONS
        ]
        mean_g_ampa_nS = np.mean(result["g_ampa_nS"], axis=1)

        def fixed_points_from_compartment_s(compartment_index, s_samples):
            synthetic_result = {
                "s_nmda": np.asarray(s_samples)[:, None],
                "g_ampa_nS": np.full(
                    (len(s_samples), 1), mean_g_ampa_nS[compartment_index]
                ),
            }
            return compute_summed_per_unit_s0_and_fixed_points(
                synthetic_result, experiment
            )

        plot_compartment_s_voltage_fixed_point_distributions(
            result,
            selected_indices,
            fixed_points_from_compartment_s,
            Path(output_dir)
            / f"summed_nmda_{parameter_name}_{display_value:g}_compartment_distributions.png",
            show_plots=show_plots,
            figure_title=(
                "Initial V: uniform["
                f"{float(experiment.initial_voltage_min / mV):g}, "
                f"{float(experiment.initial_voltage_max / mV):g}] mV; "
                f"seed={experiment.random_seed}\n"
                f"{_experiment_nmda_time_constants_title(experiment)}"
            ),
        )
    if plot_fixed_point_distributions:
        fixed_point_result = compute_summed_per_unit_s0_and_fixed_points(
            result, experiment
        )
        fixed_point_name = (
            f"voltage_vs_fixed_point_distributions_{display_value:g}_Hz.png"
            if parameter_name == "r_e"
            else f"voltage_vs_fixed_point_distributions_{parameter_name}_{display_value:g}.png"
        )
        plot_voltage_and_fixed_point_distributions(
            result,
            fixed_point_result,
            Path(output_dir) / fixed_point_name,
            show_plots=show_plots,
            experiment=experiment,
        )
    return row


def _show_saved_plot(path, title):
    """Display a worker-generated PNG safely in the parent process."""
    figure, axis = plt.subplots(figsize=(14, 10))
    axis.imshow(plt.imread(path))
    axis.axis("off")
    axis.set_title(title)
    figure.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def _show_completed_strength_plots(
    row,
    parameter_name,
    output_dir,
    diagnostic_model_indices,
    plot_fixed_point_distributions,
):
    """Show one completed worker's plots while remaining jobs continue."""
    display_value = row["input_strength"]
    if diagnostic_model_indices is not None:
        path = (
            Path(output_dir)
            / f"summed_nmda_{parameter_name}_{display_value:g}_diagnostics.png"
        )
        _show_saved_plot(path, f"{parameter_name}={display_value:g}: diagnostics")
        compartment_path = (
            Path(output_dir)
            / f"summed_nmda_{parameter_name}_{display_value:g}_compartment_distributions.png"
        )
        _show_saved_plot(
            compartment_path,
            f"{parameter_name}={display_value:g}: per-compartment distributions",
        )
    if plot_fixed_point_distributions:
        fixed_point_name = (
            f"voltage_vs_fixed_point_distributions_{display_value:g}_Hz.png"
            if parameter_name == "r_e"
            else f"voltage_vs_fixed_point_distributions_{parameter_name}_{display_value:g}.png"
        )
        _show_saved_plot(
            Path(output_dir) / fixed_point_name,
            f"{parameter_name}={display_value:g}: fixed points",
        )


def screen_summed_nmda_input_strengths(
    parameter_name,
    values,
    *,
    base_experiment=None,
    output_dir=SUMMED_OUTPUT_DIR,
    show_plots=False,
    diagnostic_model_indices=None,
    plot_fixed_point_distributions=False,
    n_jobs=4,
    joblib_backend="loky",
    animation_requests=(),
):
    """Run a strength scan and save a compact CSV and summary plot.

    Supported axes are ``r_e``, ``w_ampa``, and ``w_nmda``.  Passing an
    explicit short ``base_experiment`` also makes this suitable for tests.
    """
    if parameter_name not in {"r_e", "w_ampa", "w_nmda"}:
        raise ValueError("parameter_name must be r_e, w_ampa, or w_nmda")
    if base_experiment is None:
        base_experiment = AmpaNmdaExperimentConfig.from_fitted_apic20()
    values = list(values)
    if not values:
        raise ValueError("values must not be empty")

    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    animation_requests = tuple(animation_requests)
    if animation_requests and parameter_name != "r_e":
        raise ValueError("animation_requests are only supported for an r_e scan")

    def requests_for(value):
        rate_hz = float(value / Hz)
        return tuple(
            request
            for request in animation_requests
            if np.isclose(float(request.r_e_hz), rate_hz)
        )

    worker_show_plots = show_plots and n_jobs == 1
    if n_jobs == 1:
        rows = [
            _run_one_summed_input_strength(
                parameter_name,
                value,
                base_experiment,
                output_dir,
                diagnostic_model_indices,
                plot_fixed_point_distributions,
                worker_show_plots,
                requests_for(value),
            )
            for value in values
        ]
    else:
        completed_jobs = Parallel(
            n_jobs=n_jobs,
            backend=joblib_backend,
            return_as="generator_unordered",
        )(
            delayed(_run_one_summed_input_strength)(
                parameter_name,
                value,
                base_experiment,
                output_dir,
                diagnostic_model_indices,
                plot_fixed_point_distributions,
                worker_show_plots,
                requests_for(value),
            )
            for value in values
        )
        rows = []
        for row in completed_jobs:
            rows.append(row)
            if show_plots:
                _show_completed_strength_plots(
                    row,
                    parameter_name,
                    output_dir,
                    diagnostic_model_indices,
                    plot_fixed_point_distributions,
                )
        rows.sort(key=lambda row: row["input_strength"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"summed_nmda_{parameter_name}_scan.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    strengths = [row["input_strength"] for row in rows]
    axes[0].plot(strengths, [row["ensemble_mean_mV"] for row in rows], "o-")
    axes[0].set_ylabel("mean V [mV]")
    axes[1].plot(strengths, [row["mean_s_nmda"] for row in rows], "o-")
    axes[1].set_ylabel("mean summed s_NMDA")
    axes[1].set_xlabel(parameter_name)
    figure.suptitle(_experiment_nmda_time_constants_title(base_experiment))
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = output_dir / f"summed_nmda_{parameter_name}_scan.png"
    figure.savefig(plot_path, dpi=200, bbox_inches="tight")
    if show_plots:
        plt.show(block=False)
    else:
        plt.close(figure)
    return rows, csv_path, plot_path


class SimulateSummedNMDASpikesUnderEIBalanceScriptTestCases(unittest.TestCase):
    """IDE runnables matching the original script's strength screens."""

    def test_simulate_one_input_rate(self, r_e=25 * Hz, show_plots=True):
        """Run one editable input rate and save its CSV and summary plot."""
        rate_hz = float(r_e / Hz)
        rows, csv_path, plot_path = screen_summed_nmda_input_strengths(
            "r_e",
            [r_e],
            output_dir=SUMMED_OUTPUT_DIR / "single_rate" / f"{rate_hz:g}_Hz",
            show_plots=show_plots,
            diagnostic_model_indices=(0, 1, 2),
            plot_fixed_point_distributions=True,
        )
        diagnostic_plot_path = (
            SUMMED_OUTPUT_DIR
            / "single_rate"
            / f"{rate_hz:g}_Hz"
            / f"summed_nmda_r_e_{rate_hz:g}_diagnostics.png"
        )
        fixed_point_plot_path = (
            SUMMED_OUTPUT_DIR
            / "single_rate"
            / f"{rate_hz:g}_Hz"
            / f"voltage_vs_fixed_point_distributions_{rate_hz:g}_Hz.png"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_strength"], rate_hz)
        self.assertTrue(csv_path.exists())
        self.assertTrue(plot_path.exists())
        self.assertTrue(diagnostic_plot_path.exists())
        self.assertTrue(fixed_point_plot_path.exists())
        return (
            rows[0], csv_path, plot_path,
            diagnostic_plot_path, fixed_point_plot_path,
        )

    def test_simulate_one_input_rate_nmda_only(self, r_e=20 * Hz, show_plots=True):
        """Run summed NMDA input with both AMPA conductance paths disabled."""
        rate_hz = float(r_e / Hz)
        experiment = AmpaNmdaExperimentConfig.from_fitted_apic20(r_e=r_e).with_property(
            w_ampa=0 * nS
        )
        output_dir = SUMMED_OUTPUT_DIR / "nmda_only" / f"{rate_hz:g}_Hz"
        rows, csv_path, plot_path = screen_summed_nmda_input_strengths(
            "r_e",
            [r_e],
            base_experiment=experiment,
            output_dir=output_dir,
            show_plots=show_plots,
            diagnostic_model_indices=(0, 1, 2),
            plot_fixed_point_distributions=True,
        )
        diagnostic_plot_path = (
            output_dir / f"summed_nmda_r_e_{rate_hz:g}_diagnostics.png"
        )
        fixed_point_plot_path = (
            output_dir
            / f"voltage_vs_fixed_point_distributions_{rate_hz:g}_Hz.png"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_strength"], rate_hz)
        self.assertAlmostEqual(rows[0]["mean_g_ampa_nS"], 0.0)
        self.assertTrue(csv_path.exists())
        self.assertTrue(plot_path.exists())
        self.assertTrue(diagnostic_plot_path.exists())
        self.assertTrue(fixed_point_plot_path.exists())
        return (
            rows[0], csv_path, plot_path,
            diagnostic_plot_path, fixed_point_plot_path,
        )

    def test_screen_input_rates(self):
        rows, csv_path, plot_path = screen_summed_nmda_input_strengths(
            "r_e", np.arange(1.0, 25.0) * Hz,
            show_plots=True,
            diagnostic_model_indices=(0, 1, 2),
            plot_fixed_point_distributions=True,
        )
        self.assertEqual(len(rows), 24)
        self.assertTrue(csv_path.exists())
        self.assertTrue(plot_path.exists())
        for rate_hz in range(1, 25):
            diagnostic_plot_path = (
                SUMMED_OUTPUT_DIR
                / f"summed_nmda_r_e_{rate_hz:g}_diagnostics.png"
            )
            self.assertTrue(diagnostic_plot_path.exists(), diagnostic_plot_path)
            fixed_point_plot_path = (
                SUMMED_OUTPUT_DIR
                / f"voltage_vs_fixed_point_distributions_{rate_hz:g}_Hz.png"
            )
            self.assertTrue(fixed_point_plot_path.exists(), fixed_point_plot_path)

    def test_screen_input_rates_nmda_only(self, show_plots=True):
        """Scan r_e from 1--49 Hz using three isolated Joblib workers."""
        rates = np.arange(25.0, 50.0) * Hz
        experiment = AmpaNmdaExperimentConfig.from_fitted_apic20().with_property(
            w_ampa=0 * nS
        )
        output_dir = SUMMED_OUTPUT_DIR / "nmda_only" / "rate_scan"
        # Edit this list to choose rate/unit traces and optional post-burn-in
        # time windows.  Omit t_min_ms/t_max_ms to animate the complete trace.
        # MP4 streams frames through FFmpeg and therefore supports far more
        # frames without retaining the whole animation in memory like GIF.
        video_settings = {
            "output_format": "mp4",
            "frame_count": 1_000,
            "fps": 40,
        }
        animation_requests = [
            SummedNmdaAnimationRequest(r_e_hz=30, unit_index=0, **video_settings),
            SummedNmdaAnimationRequest(r_e_hz=30, unit_index=2, **video_settings),
            SummedNmdaAnimationRequest(r_e_hz=28, unit_index=0, **video_settings),
            SummedNmdaAnimationRequest(r_e_hz=28, unit_index=1, **video_settings),
            SummedNmdaAnimationRequest(r_e_hz=28, unit_index=2, **video_settings),
            SummedNmdaAnimationRequest(r_e_hz=25, unit_index=0, **video_settings),
            SummedNmdaAnimationRequest(r_e_hz=25, unit_index=1, **video_settings),
            SummedNmdaAnimationRequest(r_e_hz=25, unit_index=2, **video_settings),
        ]
        rows, csv_path, plot_path = screen_summed_nmda_input_strengths(
            "r_e",
            rates,
            base_experiment=experiment,
            output_dir=output_dir,
            show_plots=show_plots,
            diagnostic_model_indices=COMPARTMENT_DISTRIBUTION_INDICES[
                :NUMBER_OF_COMPARTMENT_DISTRIBUTIONS
            ],
            plot_fixed_point_distributions=True,
            n_jobs=3,
            joblib_backend="loky",
            animation_requests=animation_requests,
        )
        self.assertEqual(len(rows), 49)
        self.assertTrue(all(row["mean_g_ampa_nS"] == 0.0 for row in rows))
        self.assertTrue(csv_path.exists())
        self.assertTrue(plot_path.exists())
        for animation_request in animation_requests:
            trace_path = (
                output_dir
                / "trace_data"
                / summed_nmda_trace_filename(
                    animation_request.r_e_hz,
                    animation_request.unit_index,
                    experiment.random_seed,
                )
            )
            animation_path = (
                output_dir
                / "animations"
                / summed_nmda_animation_filename(
                    animation_request, experiment.random_seed
                )
            )
            self.assertTrue(trace_path.exists(), trace_path)
            self.assertTrue(animation_path.exists(), animation_path)
        for rate_hz in range(1, 50):
            diagnostic_plot_path = (
                output_dir / f"summed_nmda_r_e_{rate_hz:g}_diagnostics.png"
            )
            fixed_point_plot_path = (
                output_dir
                / f"voltage_vs_fixed_point_distributions_{rate_hz:g}_Hz.png"
            )
            self.assertTrue(diagnostic_plot_path.exists(), diagnostic_plot_path)
            self.assertTrue(fixed_point_plot_path.exists(), fixed_point_plot_path)
            compartment_plot_path = (
                output_dir
                / f"summed_nmda_r_e_{rate_hz:g}_compartment_distributions.png"
            )
            self.assertTrue(compartment_plot_path.exists(), compartment_plot_path)
        return rows, csv_path, plot_path

    def test_screen_ampa_weights(self):
        rows, csv_path, plot_path = screen_summed_nmda_input_strengths(
            "w_ampa", np.linspace(0.01, 0.10, 10) * nS
        )
        self.assertEqual(len(rows), 10)
        self.assertTrue(csv_path.exists())
        self.assertTrue(plot_path.exists())

    def test_screen_nmda_weights(self):
        rows, csv_path, plot_path = screen_summed_nmda_input_strengths(
            "w_nmda", np.linspace(0.2, 2.0, 10)
        )
        self.assertEqual(len(rows), 10)
        self.assertTrue(csv_path.exists())
        self.assertTrue(plot_path.exists())


if __name__ == "__main__":
    unittest.main()

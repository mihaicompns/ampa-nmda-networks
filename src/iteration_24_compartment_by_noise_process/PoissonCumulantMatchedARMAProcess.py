"""Independent Poisson count increments and their first ten cumulants.

This module deliberately generates only ``dN``.  It does not filter the
increments and therefore does not generate the synaptic state ``s(t)``.
That state would be constructed elsewhere by convolving these increments
with the appropriate synaptic kernel.
"""

import unittest
from dataclasses import dataclass
from functools import cached_property
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from brian2 import Hz, ms
from scipy.optimize import lsq_linear
from scipy.signal import lfilter


OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output_noise_limit"


@dataclass(frozen=True)
class PoissonCumulantNoiseProcess:
    r"""Vectorized independent compound-Poisson increments.

    For every time bin,

    ``dN[n] ~ Poisson(lambda)``, with ``lambda = N * r_e * dt``.

    The bins and processes are mutually independent.  Consequently,

    ``kappa_p(dN) = lambda`` and
    ``kappa_p(a*dN) = a**p * lambda`` for every integer ``p >= 1``.
    """

    N_NMDA: int
    rate: object
    dt: object
    T: object
    jump_amplitude: object = 1.0
    random_seed: int = 0

    def __post_init__(self):
        if int(self.N_NMDA) != self.N_NMDA or self.N_NMDA <= 0:
            raise ValueError("N_NMDA must be a positive integer")
        if not np.isfinite(self.poisson_parameter) or self.poisson_parameter < 0:
            raise ValueError("N_NMDA * rate * dt must be finite and nonnegative")
        steps = float(self.T / self.dt)
        if steps < 1 or not np.isclose(steps, round(steps)):
            raise ValueError("T must be a positive integer multiple of dt")

    @property
    def poisson_parameter(self):
        r"""Return ``lambda = N r_e dt`` for one independent time bin."""
        return float(self.N_NMDA * self.rate * self.dt)

    @property
    def number_of_steps(self):
        return int(round(float(self.T / self.dt)))

    def increment_cumulant(self, order):
        r"""Return ``kappa_p(a dN) = a**p N r_e dt``."""
        if int(order) != order or order < 1:
            raise ValueError("cumulant order must be a positive integer")
        return self.jump_amplitude ** int(order) * self.poisson_parameter

    # Compatibility name: the innovation is exactly the unfiltered increment.
    innovation_cumulant = increment_cumulant

    @staticmethod
    def _sample_cumulants(values, maximum_order=10):
        """Estimate cumulants from central moments, through order ten."""
        if int(maximum_order) != maximum_order or not 1 <= maximum_order <= 10:
            raise ValueError("maximum_order must be an integer from 1 to 10")
        maximum_order = int(maximum_order)
        values = np.asarray(values, dtype=np.longdouble).ravel()
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("values must be nonempty and finite")
        mean = np.mean(values, dtype=np.longdouble)
        centered = values - mean
        moments = np.ones(maximum_order + 1, dtype=np.longdouble)
        moments[1] = 0
        for order in range(2, maximum_order + 1):
            moments[order] = np.mean(centered**order, dtype=np.longdouble)
        cumulants = np.zeros(maximum_order + 1, dtype=np.longdouble)
        cumulants[1] = mean
        for order in range(2, maximum_order + 1):
            cumulants[order] = moments[order] - sum(
                comb(order - 1, lower - 1)
                * cumulants[lower]
                * moments[order - lower]
                for lower in range(2, order)
            )
        return np.asarray(cumulants[1:], dtype=float)

    def generate(self, *, n_processes=1):
        """Return independent ``a*dN`` samples of shape ``(processes, bins)``."""
        if int(n_processes) != n_processes or n_processes < 1:
            raise ValueError("n_processes must be a positive integer")
        rng = np.random.default_rng(self.random_seed)
        increments = rng.poisson(
            self.poisson_parameter,
            size=(int(n_processes), self.number_of_steps),
        )
        return increments * self.jump_amplitude

    def sample_cumulants(self, *, n_processes=1_000, maximum_order=10):
        """Generate independent increments and compare empirical/theory."""
        if np.all(np.asarray(self.jump_amplitude) == 0):
            raise ValueError("jump_amplitude must be nonzero for normalization")
        samples = self.generate(n_processes=n_processes)
        normalized = np.asarray(samples / self.jump_amplitude, dtype=float)
        orders = np.arange(1, maximum_order + 1)
        return {
            "samples": samples,
            "normalized_samples": normalized,
            "empirical_cumulants": self._sample_cumulants(normalized, maximum_order),
            "theoretical_cumulants": np.full(maximum_order, self.poisson_parameter),
            "orders": orders,
        }

    def plot_cumulant_validation(
        self, *, n_processes=1_000, maximum_order=10,
        output_path=None, show_plot=True,
    ):
        result = self.sample_cumulants(
            n_processes=n_processes, maximum_order=maximum_order
        )
        output_path = Path(output_path or OUTPUT_DIR / "poisson_increment_cumulants.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, (trace_axis, cumulant_axis) = plt.subplots(2, 1, figsize=(13, 8))
        time_ms = np.arange(self.number_of_steps) * float(self.dt / ms)
        trace_axis.step(
            time_ms, result["normalized_samples"][0], where="post", linewidth=0.8
        )
        trace_axis.set(
            title=r"One independent Poisson increment realization $dN_n$",
            xlabel="time [ms]", ylabel=r"count increment $dN_n$",
        )
        orders = result["orders"]
        width = 0.36
        cumulant_axis.bar(
            orders - width / 2, result["theoretical_cumulants"], width,
            color="black", alpha=0.72, label=r"theory: $Nr_e\Delta t$",
        )
        cumulant_axis.bar(
            orders + width / 2, result["empirical_cumulants"], width,
            color="tab:orange", alpha=0.78, label="sampled increments",
        )
        cumulant_axis.set(
            title=r"Independent increment cumulants: $\kappa_p(dN)=Nr_e\Delta t$",
            xlabel="cumulant order p", ylabel=r"$\kappa_p(dN)$", xticks=orders,
        )
        if np.any(result["empirical_cumulants"] <= 0):
            cumulant_axis.set_yscale("symlog", linthresh=self.poisson_parameter)
        cumulant_axis.legend()
        figure.suptitle(
            rf"Poisson increments only (no kernel): $Nr_e\Delta t="
            rf"{self.poisson_parameter:.4f}$, $\Delta t={float(self.dt/ms):.2f}$ ms, "
            rf"$T={float(self.T/ms):.2f}$ ms, $N_{{paths}}={n_processes}$"
        )
        for axis in (trace_axis, cumulant_axis):
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        figure.tight_layout()
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        if show_plot:
            plt.show(block=False)
        else:
            plt.close(figure)
        result["output_path"] = output_path
        return result


@dataclass(frozen=True)
class KoutsoyiannisDimitriadisCumulantProcess:
    r"""Finite-MA cumulant-matching generator.

    Given desired output cumulants ``C_p`` and filter weights ``a_j``, the
    required i.i.d. innovation cumulants are

    ``kappa_p(v) = C_p / sum_j(a_j**p)``.

    A sampleable discrete innovation distribution is then fitted on
    ``innovation_support`` by nonnegative least squares on raw moments.  The
    generated output is ``X_n = sum_j a_j v_(n-j)``.  With the default
    one-tap filter ``(1,)``, this generates noise increments, not ``s(t)``.
    """

    target_cumulants: object
    filter_weights: object
    dt: object
    T: object
    innovation_support: object
    random_seed: int = 0

    def __post_init__(self):
        target = np.asarray(self.target_cumulants, dtype=float)
        weights = np.asarray(self.filter_weights, dtype=float)
        support = np.asarray(self.innovation_support, dtype=float)
        if target.ndim != 1 or not 1 <= target.size <= 10:
            raise ValueError("target_cumulants must contain orders 1 through K, K <= 10")
        if weights.ndim != 1 or weights.size < 1 or not np.all(np.isfinite(weights)):
            raise ValueError("filter_weights must be a finite nonempty vector")
        if support.ndim != 1 or support.size < target.size + 1:
            raise ValueError("innovation_support needs at least K + 1 values")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(support)):
            raise ValueError("cumulants and support must be finite")
        denominators = np.asarray(
            [np.sum(weights**order) for order in range(1, target.size + 1)]
        )
        if np.any(np.isclose(denominators, 0.0)):
            raise ValueError("filter power sum is zero for at least one cumulant order")
        steps = float(self.T / self.dt)
        if steps < 1 or not np.isclose(steps, round(steps)):
            raise ValueError("T must be a positive integer multiple of dt")
        object.__setattr__(self, "target_cumulants", target)
        object.__setattr__(self, "filter_weights", weights)
        object.__setattr__(self, "innovation_support", support)

    @property
    def number_of_steps(self):
        return int(round(float(self.T / self.dt)))

    @cached_property
    def required_innovation_cumulants(self):
        orders = range(1, self.target_cumulants.size + 1)
        power_sums = np.asarray(
            [np.sum(self.filter_weights**order) for order in orders]
        )
        return self.target_cumulants / power_sums

    @staticmethod
    def _cumulants_to_raw_moments(cumulants):
        cumulants = np.asarray(cumulants, dtype=np.longdouble)
        moments = np.ones(cumulants.size + 1, dtype=np.longdouble)
        padded = np.r_[np.longdouble(0), cumulants]
        for order in range(1, cumulants.size + 1):
            moments[order] = padded[order] + sum(
                comb(order - 1, lower - 1)
                * padded[lower]
                * moments[order - lower]
                for lower in range(1, order)
            )
        return np.asarray(moments, dtype=float)

    @staticmethod
    def _raw_moments_to_cumulants(raw_moments):
        moments = np.asarray(raw_moments, dtype=np.longdouble)
        cumulants = np.zeros(moments.size, dtype=np.longdouble)
        for order in range(1, moments.size):
            cumulants[order] = moments[order] - sum(
                comb(order - 1, lower - 1)
                * cumulants[lower]
                * moments[order - lower]
                for lower in range(1, order)
            )
        return np.asarray(cumulants[1:], dtype=float)

    @cached_property
    def innovation_probabilities(self):
        """Fit nonnegative probabilities to the required innovation moments."""
        moments = self._cumulants_to_raw_moments(
            self.required_innovation_cumulants
        )
        orders = np.arange(moments.size)[:, None]
        moment_matrix = self.innovation_support[None, :] ** orders
        row_scale = np.maximum(np.max(np.abs(moment_matrix), axis=1), np.abs(moments))
        row_scale[0] = 1.0
        fit = lsq_linear(
            moment_matrix / row_scale[:, None],
            moments / row_scale,
            bounds=(0.0, np.inf),
            tol=1e-14,
            lsmr_tol=1e-14,
            max_iter=10_000,
        )
        probabilities = np.maximum(fit.x, 0.0)
        probability_sum = probabilities.sum()
        if not fit.success or probability_sum <= 0:
            raise ValueError("no feasible discrete innovation fit was found")
        return probabilities / probability_sum

    @cached_property
    def fitted_innovation_cumulants(self):
        orders = np.arange(self.target_cumulants.size + 1)[:, None]
        raw_moments = (
            self.innovation_support[None, :] ** orders
        ) @ self.innovation_probabilities
        return self._raw_moments_to_cumulants(raw_moments)

    @cached_property
    def fitted_output_cumulants(self):
        orders = range(1, self.target_cumulants.size + 1)
        return self.fitted_innovation_cumulants * np.asarray(
            [np.sum(self.filter_weights**order) for order in orders]
        )

    def generate(self, *, n_processes=1):
        """Draw fitted innovations and apply the requested causal MA filter."""
        if int(n_processes) != n_processes or n_processes < 1:
            raise ValueError("n_processes must be a positive integer")
        warmup = self.filter_weights.size - 1
        rng = np.random.default_rng(self.random_seed)
        innovations = rng.choice(
            self.innovation_support,
            size=(int(n_processes), self.number_of_steps + warmup),
            p=self.innovation_probabilities,
        )
        output = lfilter(self.filter_weights, (1.0,), innovations, axis=1)
        return output[:, warmup:]

    def sample_cumulants(self, *, n_processes=1_000):
        """Generate KD samples and estimate all configured cumulant orders."""
        samples = self.generate(n_processes=n_processes)
        maximum_order = self.target_cumulants.size
        return {
            "samples": samples,
            "empirical_cumulants": PoissonCumulantNoiseProcess._sample_cumulants(
                samples, maximum_order
            ),
            "theoretical_cumulants": self.target_cumulants.copy(),
            "fitted_theoretical_cumulants": self.fitted_output_cumulants.copy(),
            "orders": np.arange(1, maximum_order + 1),
            "innovation_support": self.innovation_support.copy(),
            "innovation_probabilities": self.innovation_probabilities.copy(),
        }

    def plot_cumulant_validation(
        self, *, n_processes=1_000, output_path=None, show_plot=True
    ):
        """Plot one KD increment path and theoretical/empirical cumulants."""
        result = self.sample_cumulants(n_processes=n_processes)
        output_path = Path(
            output_path or OUTPUT_DIR / "kd_poisson_increment_cumulants.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, (trace_axis, cumulant_axis) = plt.subplots(
            2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": (1.2, 1.0)}
        )
        time_ms = np.arange(self.number_of_steps) * float(self.dt / ms)
        trace_axis.step(
            time_ms, result["samples"][0], where="post",
            color="tab:green", linewidth=0.8,
        )
        trace_axis.set(
            title="One Koutsoyiannis–Dimitriadis increment realization",
            xlabel="time [ms]", ylabel="noise increment",
        )

        orders = result["orders"]
        width = 0.36
        cumulant_axis.bar(
            orders - width / 2,
            result["theoretical_cumulants"],
            width,
            color="black",
            alpha=0.72,
            label="target theoretical",
        )
        cumulant_axis.bar(
            orders + width / 2,
            result["empirical_cumulants"],
            width,
            color="tab:green",
            alpha=0.78,
            label="empirical KD samples",
        )
        cumulant_axis.plot(
            orders,
            result["fitted_theoretical_cumulants"],
            "r--o",
            linewidth=1.2,
            markersize=4,
            label="fitted discrete-law theory",
        )
        cumulant_axis.set(
            title="KD Poisson-increment cumulants through order 10",
            xlabel="cumulant order p", ylabel=r"$\kappa_p$", xticks=orders,
        )
        if np.any(result["empirical_cumulants"] <= 0):
            cumulant_axis.set_yscale(
                "symlog",
                linthresh=max(float(np.min(np.abs(self.target_cumulants))), 1e-9),
            )
        cumulant_axis.legend()
        figure.suptitle(
            "Koutsoyiannis–Dimitriadis discrete innovation fit; "
            rf"$\Delta t={float(self.dt/ms):.2f}$ ms, "
            rf"$T={float(self.T/ms):.2f}$ ms, "
            rf"$N_{{paths}}={n_processes}$, "
            rf"$a={np.array2string(self.filter_weights, precision=2)}$"
        )
        for axis in (trace_axis, cumulant_axis):
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        figure.tight_layout()
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        if show_plot:
            plt.show(block=False)
        else:
            plt.close(figure)
        result["output_path"] = output_path
        return result


def run_fitted_nmda_poisson_cumulant_validation(
    *, r_e, dt, T, n_processes=1_000, output_path=None, show_plot=True
):
    """Sample fitted-model ``dN`` increments; do not generate ``s(t)``."""
    from iteration_21_nmda_spikes_under_ei_balance.SimulateSummedNMDASpikesUnderEIBalanceScripts import (
        AmpaNmdaExperimentConfig,
    )
    experiment = AmpaNmdaExperimentConfig.from_fitted_apic20().with_property(
        r_e=r_e, dt=dt
    )
    process = PoissonCumulantNoiseProcess(
        N_NMDA=experiment.n_nmda,
        rate=experiment.r_e,
        dt=experiment.dt,
        T=T,
        random_seed=experiment.random_seed,
    )
    if output_path is None:
        output_path = OUTPUT_DIR / (
            "poisson_increment_cumulants_1_to_10_"
            f"dt_{float(dt/ms):.2f}ms_T_{float(T/ms):.2f}ms.png"
        )
    return process.plot_cumulant_validation(
        n_processes=n_processes, maximum_order=10,
        output_path=output_path, show_plot=show_plot,
    )


def run_kd_poisson_increment_cumulant_validation(
    *, N_NMDA, rate, dt, T, n_processes=1_000,
    innovation_support=None, random_seed=0,
    output_path=None, show_plot=True,
):
    r"""Generate KD-fitted Poisson increments and validate cumulants 1--10."""
    poisson_parameter = float(N_NMDA * rate * dt)
    target_cumulants = np.full(10, poisson_parameter)
    if innovation_support is None:
        innovation_support = np.arange(11, dtype=float)
    process = KoutsoyiannisDimitriadisCumulantProcess(
        target_cumulants=target_cumulants,
        filter_weights=(1.0,),
        dt=dt,
        T=T,
        innovation_support=innovation_support,
        random_seed=random_seed,
    )
    if output_path is None:
        output_path = OUTPUT_DIR / (
            "kd_poisson_increment_cumulants_1_to_10_"
            f"dt_{float(dt/ms):.2f}ms_T_{float(T/ms):.2f}ms.png"
        )
    return process.plot_cumulant_validation(
        n_processes=n_processes,
        output_path=output_path,
        show_plot=show_plot,
    )


def run_poisson_vs_koutsoyiannis_dimitriadis_comparison(
    *, N_NMDA, rate, dt, T, n_processes=1_000,
    filter_weights=(1.0,), innovation_support=None,
    random_seed=0, output_path=None, show_plot=True,
):
    r"""Compare direct Poisson increments with KD cumulant matching.

    The target is ``C_p = N r_e dt`` for orders one through ten.  The default
    one-tap filter means both methods produce increments.  Supplying other
    weights explicitly requests a filtered KD output; it does not alter the
    direct Poisson reference.
    """
    direct = PoissonCumulantNoiseProcess(
        N_NMDA=N_NMDA, rate=rate, dt=dt, T=T, random_seed=random_seed
    )
    target = np.full(10, direct.poisson_parameter)
    if innovation_support is None:
        innovation_support = np.arange(11, dtype=float)
    kd = KoutsoyiannisDimitriadisCumulantProcess(
        target_cumulants=target,
        filter_weights=filter_weights,
        dt=dt,
        T=T,
        innovation_support=innovation_support,
        random_seed=random_seed + 1,
    )
    direct_samples = np.asarray(direct.generate(n_processes=n_processes), dtype=float)
    kd_samples = kd.generate(n_processes=n_processes)
    direct_empirical = direct._sample_cumulants(direct_samples, 10)
    kd_empirical = direct._sample_cumulants(kd_samples, 10)

    output_path = Path(
        output_path or OUTPUT_DIR / "poisson_vs_koutsoyiannis_dimitriadis.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(14, 11))
    time_ms = np.arange(direct.number_of_steps) * float(dt / ms)
    axes[0].step(
        time_ms, direct_samples[0], where="post", linewidth=0.8,
        label="direct Poisson increment",
    )
    axes[0].step(
        time_ms, kd_samples[0], where="post", linewidth=0.8, alpha=0.75,
        label="Koutsoyiannis–Dimitriadis",
    )
    axes[0].set(xlabel="time [ms]", ylabel="noise increment", title="Example paths")
    axes[0].legend()

    common_bins = np.arange(
        np.floor(min(direct_samples.min(), kd_samples.min())) - 0.5,
        np.ceil(max(direct_samples.max(), kd_samples.max())) + 1.5,
    )
    axes[1].hist(
        direct_samples.ravel(), bins=common_bins, density=True,
        histtype="stepfilled", alpha=0.35, label="direct Poisson",
    )
    axes[1].hist(
        kd_samples.ravel(), bins=common_bins, density=True,
        histtype="step", linewidth=2.0, label="KD fitted discrete law",
    )
    axes[1].set(xlabel="increment", ylabel="probability", title="Marginal distributions")
    axes[1].legend()

    orders = np.arange(1, 11)
    width = 0.25
    axes[2].bar(orders - width, target, width, color="black", label="target")
    axes[2].bar(orders, direct_empirical, width, label="direct Poisson")
    axes[2].bar(orders + width, kd_empirical, width, label="KD sampled")
    axes[2].plot(
        orders, kd.fitted_output_cumulants, "r--o", linewidth=1.2,
        label="KD fitted theory",
    )
    axes[2].set(
        xlabel="cumulant order p", ylabel=r"$\kappa_p$", xticks=orders,
        title="Target, theoretical fit, and empirical cumulants",
    )
    if np.any(np.r_[direct_empirical, kd_empirical] <= 0):
        axes[2].set_yscale("symlog", linthresh=max(direct.poisson_parameter, 1e-9))
    axes[2].legend(ncol=2)
    figure.suptitle(
        rf"Direct Poisson vs KD: $C_p=Nr_e\Delta t={direct.poisson_parameter:.4f}$; "
        rf"filter $a={np.array2string(np.asarray(filter_weights), precision=2)}$"
    )
    for axis in axes:
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show(block=False)
    else:
        plt.close(figure)
    return {
        "direct_samples": direct_samples,
        "kd_samples": kd_samples,
        "target_cumulants": target,
        "direct_empirical_cumulants": direct_empirical,
        "kd_empirical_cumulants": kd_empirical,
        "kd_fitted_cumulants": kd.fitted_output_cumulants,
        "kd_innovation_probabilities": kd.innovation_probabilities,
        "output_path": output_path,
    }


# Backward-compatible symbols; behavior is intentionally unfiltered now.
PoissonCumulantMatchedARMAProcess = PoissonCumulantNoiseProcess
run_fitted_nmda_poisson_arma_cumulant_validation = (
    run_fitted_nmda_poisson_cumulant_validation
)


class PoissonCumulantNoiseRunnables(unittest.TestCase):
    """Long, manually launched plotting workflow for PyCharm."""

    def test_generate_fitted_nmda_cumulants_1_to_10(self, show_plot=True):
        dt = 0.1 * ms
        T = 2000 * ms
        result = run_fitted_nmda_poisson_cumulant_validation(
            r_e=2500 * Hz, dt=dt, T=T, show_plot=show_plot
        )
        expected_steps = int(round(float(T / dt)))
        self.assertEqual(result["samples"].shape, (1_000, expected_steps))
        self.assertEqual(result["empirical_cumulants"].shape, (10,))
        self.assertTrue(result["output_path"].exists())
        return result

    def test_generate_kd_poisson_increment_cumulants_1_to_10(
        self, show_plot=True
    ):
        """Generate KD increments and plot theoretical/empirical cumulants."""
        dt = 0.1 * ms
        T = 20000 * ms
        result = run_kd_poisson_increment_cumulant_validation(
            N_NMDA=10,
            rate=2500 * Hz,
            dt=dt,
            T=T,
            n_processes=1_000,
            show_plot=show_plot,
        )
        expected_steps = int(round(float(T / dt)))
        self.assertEqual(result["samples"].shape, (1_000, expected_steps))
        self.assertEqual(result["empirical_cumulants"].shape, (10,))
        self.assertTrue(result["output_path"].exists())
        return result

    def test_compare_poisson_with_koutsoyiannis_dimitriadis(
        self, show_plot=True
    ):
        """Compare direct increments with a fitted KD discrete generator."""
        dt = 0.1 * ms
        T = 2000 * ms
        result = run_poisson_vs_koutsoyiannis_dimitriadis_comparison(
            N_NMDA=10,
            rate=250 * Hz,
            dt=dt,
            T=T,
            n_processes=1_000,
            filter_weights=(1.0,),
            show_plot=show_plot,
        )
        expected_steps = int(round(float(T / dt)))
        self.assertEqual(result["direct_samples"].shape, (1_000, expected_steps))
        self.assertEqual(result["kd_samples"].shape, (1_000, expected_steps))
        self.assertTrue(result["output_path"].exists())
        return result


if __name__ == "__main__":
    unittest.main()

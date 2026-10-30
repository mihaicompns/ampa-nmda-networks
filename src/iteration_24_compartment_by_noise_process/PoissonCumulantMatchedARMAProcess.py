"""Cumulant-matched ARMA processes driven by Poisson count innovations."""

import unittest
from dataclasses import dataclass
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from brian2 import Hz, ms


OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output_noise_limit"

@dataclass(frozen=True)
class PoissonCumulantMatchedARMAProcess:
    r"""Vectorized ARMA process driven by compound-Poisson innovations.

    The recursion uses the convention

    ``X[n] = sum_i phi[i] X[n-i-1] + epsilon[n]``
    ``       + sum_j theta[j] epsilon[n-j-1]``.

    For ``Delta N ~ Poisson(N r_e dt)`` and
    ``epsilon = jump_amplitude * Delta N``, every count cumulant is
    ``K_p = N r_e dt`` and

    ``kappa_p(epsilon) = jump_amplitude**p * N r_e dt``.

    If ``X[n] = sum_j psi[j] epsilon[n-j]``, the stationary output cumulant is

    ``kappa_p(X) = kappa_p(epsilon) * sum_j psi[j]**p``.

    Thus this class treats ``N r_e dt`` as the *input-count* cumulant.  It does
    not divide that value by the filter power sum.  Brian2 units carried by
    ``jump_amplitude`` are retained in generated samples and cumulants.
    """

    n_sources: int
    rate: object
    dt: object
    T: object
    ar_coefficients: tuple = ()
    ma_coefficients: tuple = ()
    jump_amplitude: object = 1.0
    random_seed: int = 0

    def __post_init__(self):
        if int(self.n_sources) != self.n_sources or self.n_sources <= 0:
            raise ValueError("n_sources must be a positive integer")
        poisson_parameter = float(self.n_sources * self.rate * self.dt)
        if not np.isfinite(poisson_parameter) or poisson_parameter < 0:
            raise ValueError("n_sources * rate * dt must be finite and nonnegative")
        number_of_steps_float = float(self.T / self.dt)
        number_of_steps = int(round(number_of_steps_float))
        if number_of_steps < 1 or not np.isclose(
            number_of_steps_float, number_of_steps
        ):
            raise ValueError("T must be a positive integer multiple of dt")

        ar = tuple(float(value) for value in self.ar_coefficients)
        ma = tuple(float(value) for value in self.ma_coefficients)
        if not np.all(np.isfinite(ar)) or not np.all(np.isfinite(ma)):
            raise ValueError("AR and MA coefficients must be finite")
        if ar:
            # Stability requires every eigenvalue of the AR companion
            # polynomial z**p - phi_1*z**(p-1) - ... - phi_p to lie inside
            # the unit circle.
            companion_roots = np.roots((1.0,) + tuple(-value for value in ar))
            if np.any(np.abs(companion_roots) >= 1.0):
                raise ValueError("AR coefficients do not define a stable process")
        object.__setattr__(self, "ar_coefficients", ar)
        object.__setattr__(self, "ma_coefficients", ma)

    @property
    def poisson_parameter(self):
        r"""Return ``lambda = N r_e dt`` for one time bin."""
        return float(self.n_sources * self.rate * self.dt)

    @property
    def number_of_steps(self):
        """Return the number of bins implied by total time ``T`` and ``dt``."""
        return int(round(float(self.T / self.dt)))

    def innovation_cumulant(self, order):
        r"""Return ``kappa_p(epsilon) = a**p N r_e dt``."""
        if int(order) != order or order < 1:
            raise ValueError("cumulant order must be a positive integer")
        return self.jump_amplitude**int(order) * self.poisson_parameter

    def impulse_response(self, number_of_terms=10_000):
        """Return the causal MA-infinity coefficients ``psi_j``."""
        if int(number_of_terms) != number_of_terms or number_of_terms < 1:
            raise ValueError("number_of_terms must be a positive integer")
        psi = np.zeros(int(number_of_terms), dtype=float)
        psi[0] = 1.0
        for lag in range(1, psi.size):
            if lag <= len(self.ma_coefficients):
                psi[lag] += self.ma_coefficients[lag - 1]
            for ar_lag, coefficient in enumerate(self.ar_coefficients, start=1):
                if ar_lag <= lag:
                    psi[lag] += coefficient * psi[lag - ar_lag]
        return psi

    def output_cumulant(self, order, *, impulse_response_terms=10_000):
        r"""Return ``kappa_p(X) = kappa_p(epsilon) sum_j psi_j**p``."""
        if int(order) != order or order < 1:
            raise ValueError("cumulant order must be a positive integer")
        order = int(order)
        psi = self.impulse_response(impulse_response_terms)
        return self.innovation_cumulant(order) * np.sum(psi**order)

    @staticmethod
    def _sample_cumulants(values, maximum_order=10):
        """Estimate marginal cumulants stably through order 10.

        The first cumulant is the sample mean.  Orders two and above are
        computed from central moments of ``Y = X - mean(X)`` using

        ``kappa_n(Y) = mu_n - sum_{j=2}^{n-1}
        C(n-1,j-1) kappa_j(Y) mu_(n-j)``.

        Centering before taking powers is essential here: the stationary
        Poisson AR process can have a large mean, and the equivalent
        raw-moment recurrence subtracts nearly equal values of order
        ``mean(X)**n``.  That causes catastrophic cancellation at high orders.

        These are plug-in (empirical-distribution) cumulants, rather than the
        finite-sample unbiased k-statistics.  ``longdouble`` is used for the
        intermediate moments and recurrence, where the platform supports it.
        """
        if int(maximum_order) != maximum_order or not 1 <= maximum_order <= 10:
            raise ValueError("maximum_order must be an integer from 1 to 10")
        maximum_order = int(maximum_order)
        values = np.asarray(values, dtype=np.longdouble).ravel()
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("values must be nonempty and finite")

        sample_mean = np.mean(values, dtype=np.longdouble)
        centered_values = values - sample_mean
        central_moments = np.ones(maximum_order + 1, dtype=np.longdouble)
        central_moments[1] = 0.0
        for order in range(2, maximum_order + 1):
            central_moments[order] = np.mean(
                centered_values**order, dtype=np.longdouble
            )

        centered_cumulants = np.zeros(
            maximum_order + 1, dtype=np.longdouble
        )
        for order in range(2, maximum_order + 1):
            correction = sum(
                comb(order - 1, lower_order - 1)
                * centered_cumulants[lower_order]
                * central_moments[order - lower_order]
                for lower_order in range(2, order)
            )
            centered_cumulants[order] = central_moments[order] - correction

        cumulants = centered_cumulants
        cumulants[1] = sample_mean
        return np.asarray(cumulants[1:], dtype=float)

    def sample_cumulants(
        self,
        *,
        n_processes=1_000,
        burn_in_steps=1_000,
        impulse_response_terms=10_000,
        maximum_order=10,
    ):
        """Generate paths and compare sampled and theoretical cumulants 1--10.

        Cumulants of different orders carry different physical dimensions.
        The comparison therefore uses ``Y = X / jump_amplitude``.  Its values
        are in jump-amplitude units and all four plotted numbers are ordinary
        numeric filter-count cumulants.
        """
        if np.all(np.asarray(self.jump_amplitude) == 0):
            raise ValueError("jump_amplitude must be nonzero for normalization")
        samples = self.generate(
            n_processes=n_processes,
            burn_in_steps=burn_in_steps,
        )
        normalized_samples = np.asarray(
            samples / self.jump_amplitude, dtype=float
        )
        empirical = self._sample_cumulants(normalized_samples, maximum_order)
        psi = self.impulse_response(impulse_response_terms)
        theoretical = np.asarray(
            [
                self.poisson_parameter * np.sum(psi**order)
                for order in range(1, maximum_order + 1)
            ]
        )
        return {
            "samples": samples,
            "normalized_samples": normalized_samples,
            "empirical_cumulants": empirical,
            "theoretical_cumulants": theoretical,
            "orders": np.arange(1, maximum_order + 1),
        }

    def plot_cumulant_validation(
        self,
        *,
        n_processes=1_000,
        burn_in_steps=1_000,
        maximum_order=10,
        output_path=None,
        show_plot=True,
    ):
        """Plot one sampled path and the first four cumulant comparisons."""
        result = self.sample_cumulants(
            n_processes=n_processes,
            burn_in_steps=burn_in_steps,
            maximum_order=maximum_order,
        )
        if output_path is None:
            output_path = OUTPUT_DIR / "poisson_cumulant_matched_arma.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure, (trace_axis, cumulant_axis) = plt.subplots(
            2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": (1.2, 1.0)}
        )
        time_ms = np.arange(self.number_of_steps) * float(self.dt / ms)
        trace_axis.plot(
            time_ms,
            result["normalized_samples"][0],
            color="tab:blue",
            linewidth=0.8,
        )
        trace_axis.set(
            title="One cumulant-matched ARMA noise realization",
            xlabel="time [ms]",
            ylabel=r"noise contribution $X_n/a$",
        )

        orders = result["orders"]
        width = 0.36
        cumulant_axis.bar(
            orders - width / 2,
            result["theoretical_cumulants"],
            width,
            color="black",
            alpha=0.72,
            label="theoretical",
        )
        cumulant_axis.bar(
            orders + width / 2,
            result["empirical_cumulants"],
            width,
            color="tab:orange",
            alpha=0.78,
            label="sampled",
        )
        cumulant_axis.set(
            title=(
                "Stationary marginal cumulants in jump-amplitude units "
                "(high orders have large sampling error)"
            ),
            xlabel="cumulant order p",
            ylabel=r"$\kappa_p(X/a)$",
            xticks=orders,
        )
        if np.any(result["empirical_cumulants"] <= 0):
            cumulant_axis.set_yscale("symlog", linthresh=1.0)
        cumulant_axis.legend()
        figure.suptitle(
            rf"Poisson ARMA: $Nr_e\Delta t={self.poisson_parameter:.4f}$, "
            rf"$\Delta t={float(self.dt / ms):.2f}$ ms, "
            rf"$T={float(self.T / ms):.2f}$ ms "
            rf"({self.number_of_steps} bins), $N_{{paths}}={n_processes}$"
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

    def generate(self, *, n_processes=1, burn_in_steps=1_000):
        """Generate independent stationary-approximation paths in vector form.

        The returned shape is ``(n_processes, number_of_steps)``.  A fresh RNG
        is constructed from ``random_seed`` on every call, making repeated
        calls reproducible.  Burn-in samples are simulated and discarded.
        """
        for name, value, minimum in (
            ("n_processes", n_processes, 1),
            ("burn_in_steps", burn_in_steps, 0),
        ):
            if int(value) != value or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        n_processes = int(n_processes)
        burn_in_steps = int(burn_in_steps)
        total_steps = self.number_of_steps + burn_in_steps

        rng = np.random.default_rng(self.random_seed)
        innovations = rng.poisson(
            self.poisson_parameter, size=(total_steps, n_processes)
        ).astype(float)
        output = np.zeros_like(innovations)
        for step in range(total_steps):
            value = innovations[step].copy()
            for lag, coefficient in enumerate(self.ar_coefficients, start=1):
                if lag <= step:
                    value += coefficient * output[step - lag]
            for lag, coefficient in enumerate(self.ma_coefficients, start=1):
                if lag <= step:
                    value += coefficient * innovations[step - lag]
            output[step] = value
        return output[burn_in_steps:].T * self.jump_amplitude


def run_fitted_nmda_poisson_arma_cumulant_validation(
    *, r_e, dt, T, n_processes=1_000, output_path=None, show_plot=True
):
    """Run the fitted-NMDA Poisson AR(1) cumulant validation and plot it."""
    from iteration_21_nmda_spikes_under_ei_balance.SimulateSummedNMDASpikesUnderEIBalanceScripts import (
        AmpaNmdaExperimentConfig,
    )

    experiment = AmpaNmdaExperimentConfig.from_fitted_apic20().with_property(
        r_e=r_e,
        dt=dt,
    )
    ar_coefficient = float(
        np.exp(-experiment.dt / experiment.tau_nmda_decay)
    )
    jump_amplitude = (
        experiment.alpha_nmda
        * experiment.w_nmda
        * experiment.tau_nmda_rise
    )
    burn_in_steps = int(
        np.ceil(10 * float(experiment.tau_nmda_decay / experiment.dt))
    )
    process = PoissonCumulantMatchedARMAProcess(
        n_sources=experiment.n_nmda,
        rate=experiment.r_e,
        dt=experiment.dt,
        T=T,
        ar_coefficients=(ar_coefficient,),
        jump_amplitude=jump_amplitude,
        random_seed=experiment.random_seed,
    )
    if output_path is None:
        output_path = OUTPUT_DIR / (
            "poisson_arma_cumulants_1_to_10_"
            f"dt_{float(dt / ms):.2f}ms_T_{float(T / ms):.2f}ms.png"
        )
    return process.plot_cumulant_validation(
        n_processes=n_processes,
        burn_in_steps=burn_in_steps,
        maximum_order=10,
        output_path=output_path,
        show_plot=show_plot,
    )


class PoissonCumulantMatchedARMARunnables(unittest.TestCase):
    """Long, manually launched plotting workflows for PyCharm."""

    def test_generate_fitted_nmda_cumulants_1_to_10(self, show_plot=True):
        """Generate ``T / dt`` noise bins and plot cumulants one through ten."""
        dt = 0.01 * ms
        T = 2000 * ms
        result = run_fitted_nmda_poisson_arma_cumulant_validation(
            r_e = 2500 * Hz,
            dt=dt,
            T=T,
            show_plot=show_plot,
        )
        expected_number_of_steps = int(round(float(T / dt)))
        self.assertEqual(
            result["samples"].shape,
            (1_000, expected_number_of_steps),
        )
        self.assertEqual(result["empirical_cumulants"].shape, (10,))
        self.assertTrue(result["output_path"].exists())
        return result


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
from brian2 import Hz, kHz, ms, mV, nF, nS
from brian2.units import second
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import factorized

from Plotting import show_plots_non_blocking
from iteration_21_nmda_spikes_under_ei_balance.SimulateNMDASpikesUnderEIBalanceScripts import (
    AmpaNmdaExperimentConfigFactoryBuilder,
    compute_per_unit_s0_and_fixed_points,
    simulate_ampa_nmda_single_compartment_ensemble,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"


def _conservative_upwind_advection(probability, velocity, axis, spacing):
    """Finite-volume advection with zero probability flux at the boundary."""
    velocity = np.broadcast_to(velocity, probability.shape)
    left = [slice(None)] * probability.ndim
    right = [slice(None)] * probability.ndim
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)
    left, right = tuple(left), tuple(right)
    face_velocity = 0.5 * (velocity[left] + velocity[right])
    interior_flux = np.maximum(face_velocity, 0.0) * probability[left]
    interior_flux += np.minimum(face_velocity, 0.0) * probability[right]
    faces_shape = list(probability.shape)
    faces_shape[axis] += 1
    faces = np.zeros(faces_shape, dtype=float)
    interior = [slice(None)] * probability.ndim
    interior[axis] = slice(1, -1)
    faces[tuple(interior)] = interior_flux
    return -np.diff(faces, axis=axis) / spacing


def _second_derivative_neumann(probability, axis, spacing):
    padding = [(0, 0)] * probability.ndim
    padding[axis] = (1, 1)
    padded = np.pad(probability, padding, mode="edge")
    center = [slice(None)] * probability.ndim
    lower = [slice(None)] * probability.ndim
    upper = [slice(None)] * probability.ndim
    center[axis] = slice(1, -1)
    lower[axis] = slice(None, -2)
    upper[axis] = slice(2, None)
    return (
        padded[tuple(upper)]
        - 2.0 * padded[tuple(center)]
        + padded[tuple(lower)]
    ) / spacing**2


def _mixed_derivative_neumann(probability, axis_a, spacing_a, axis_b, spacing_b):
    padding = [(0, 0)] * probability.ndim
    padding[axis_a] = (1, 1)
    padding[axis_b] = (1, 1)
    padded = np.pad(probability, padding, mode="edge")

    def shifted(offset_a, offset_b):
        slices = [slice(1, -1)] * probability.ndim
        slices[axis_a] = slice(1 + offset_a, -1 + offset_a or None)
        slices[axis_b] = slice(1 + offset_b, -1 + offset_b or None)
        return padded[tuple(slices)]

    return (
        shifted(1, 1) - shifted(1, -1)
        - shifted(-1, 1) + shifted(-1, -1)
    ) / (4.0 * spacing_a * spacing_b)


def _reconstruct_voltage_density(probability, shape, voltage_spacing_mV):
    """Return a smooth, normalized density from finite-volume cell masses.

    The raw marginal is a collection of probability masses at voltage-cell
    centers. Treating those masses as point samples produces a grid-locked
    comb when the deterministic voltage drift approaches its nullcline.
    Gaussian reconstruction represents each cell mass by a finite-width basis
    without modifying the evolved four-dimensional probability state.
    """
    cell_density = probability.reshape(shape).sum(axis=(1, 2, 3))
    cell_density /= voltage_spacing_mV
    bandwidth_mV = 3.0
    reconstructed = gaussian_filter1d(
        cell_density,
        sigma=bandwidth_mV / voltage_spacing_mV,
        mode="nearest",
    )
    reconstructed /= reconstructed.sum() * voltage_spacing_mV
    return reconstructed


def _theoretical_stationary_synaptic_moments(experiment):
    """Return diffusion-model stationary means and variances for g_A, x, and s."""
    nu_ampa_per_ms = float(experiment.n_e * experiment.r_e / kHz)
    nu_nmda_per_ms = float(experiment.n_nmda * experiment.r_e / kHz)
    tau_ampa_ms = float(experiment.tau_ampa / ms)
    tau_rise_ms = float(experiment.tau_nmda_rise / ms)
    tau_decay_ms = float(experiment.tau_nmda_decay / ms)
    alpha_per_ms = float(experiment.alpha_nmda / kHz)
    w_ampa_nS = float(experiment.w_ampa / nS)
    w_nmda = float(experiment.w_nmda)

    g_mean = nu_ampa_per_ms * w_ampa_nS * tau_ampa_ms
    g_variance = 0.5 * nu_ampa_per_ms * w_ampa_nS**2 * tau_ampa_ms
    x_mean = nu_nmda_per_ms * w_nmda * tau_rise_ms
    x_variance = 0.5 * nu_nmda_per_ms * w_nmda**2 * tau_rise_ms
    s_mean = alpha_per_ms * x_mean * tau_decay_ms
    s_variance = (
        alpha_per_ms**2
        * x_variance
        * tau_rise_ms
        * tau_decay_ms**2
        / (tau_rise_ms + tau_decay_ms)
    )
    return {
        "g_ampa": {"mean": g_mean, "variance": g_variance},
        "x_nmda": {"mean": x_mean, "variance": x_variance},
        "s_nmda": {"mean": s_mean, "variance": s_variance},
    }


def _stationary_synaptic_moments(probability, shape, synaptic_grids):
    """Measure marginal means and variances from finite-volume cell masses."""
    probability = np.asarray(probability, dtype=float).reshape(shape)
    moments = {}
    for name, axis, grid in (
        ("g_ampa", 1, synaptic_grids[0]),
        ("x_nmda", 2, synaptic_grids[1]),
        ("s_nmda", 3, synaptic_grids[2]),
    ):
        marginal = probability.sum(axis=tuple(i for i in range(4) if i != axis))
        mean = float(np.dot(marginal, grid))
        variance = float(np.dot(marginal, (grid - mean) ** 2))
        moments[name] = {"mean": mean, "variance": variance}
    return moments


def simulate_one_unit_for_duration(duration_seconds):
    """Joblib worker: simulate one unit and return plot-ready diagnostics."""
    experiment = (
        AmpaNmdaExperimentConfigFactoryBuilder.fitted_apic20()
        .with_property(
            n_models=1,
            simulation_time=float(duration_seconds) * second,
        )
        .build()
    )
    result = simulate_ampa_nmda_single_compartment_ensemble(experiment)
    fixed_points = compute_per_unit_s0_and_fixed_points(
        result,
        experiment=experiment,
    )
    voltage_mV = np.asarray(result["voltage_mV"][0], dtype=float)
    counts, edges = np.histogram(voltage_mV, bins=150)
    centers = 0.5 * (edges[:-1] + edges[1:])
    center_of_mass_mV = float(np.average(centers, weights=counts))
    return {
        "duration_seconds": float(duration_seconds),
        "voltage_mV": voltage_mV,
        "histogram_edges": edges,
        "center_of_mass_mV": center_of_mass_mV,
        "upper_fixed_point_mV": fixed_points["upper_fixed_point_mV"][0],
        "rate_hz": float(experiment.r_e / Hz),
        "burn_in_ms": float(experiment.burn_in / ms),
    }


def plot_one_vm_vs_upper_fixed_point(
    diagnostics,
    output_path,
    *,
    stationary_voltage_mV=None,
    stationary_density_per_mV=None,
):
    """Plot one simulated voltage histogram and optional stationary FP density."""
    voltage_mV = diagnostics["voltage_mV"]
    upper_fixed_point_mV = diagnostics["upper_fixed_point_mV"]
    center_of_mass_mV = diagnostics["center_of_mass_mV"]
    delta_voltage_mV = upper_fixed_point_mV - center_of_mass_mV

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(
        voltage_mV,
        bins=diagnostics["histogram_edges"],
        density=True,
        histtype="step",
        linewidth=2,
        color="tab:green",
        label=f"simulated $V_m$ (n={voltage_mV.size:,})",
    )
    if stationary_voltage_mV is not None:
        axis.plot(
            stationary_voltage_mV,
            stationary_density_per_mV,
            color="tab:red",
            linewidth=2.5,
            label="stationary Fokker–Planck density",
        )
    axis.axvline(
        upper_fixed_point_mV,
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        label=f"stable upper fixed point ({upper_fixed_point_mV:.3f} mV)",
    )
    axis.axvline(
        center_of_mass_mV,
        color="tab:blue",
        linestyle=":",
        linewidth=2,
        label=f"histogram center of mass ({center_of_mass_mV:.3f} mV)",
    )
    axis.set_title(
        "Single-unit post-burn-in membrane-voltage distribution\n"
        f"T={diagnostics['duration_seconds']:g} s, "
        f"r_e={diagnostics['rate_hz']:g} Hz, "
        f"burn-in={diagnostics['burn_in_ms']:g} ms, "
        r"$\Delta V=V_{upper\ FP}-\langle V_{emp} \rangle $"
        f"={delta_voltage_mV:.3f} mV"
    )
    axis.set_xlabel("membrane voltage [mV]")
    axis.set_ylabel("probability density [1/mV]")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


class UnderstandDistributionOfMembraneVoltageUnderNMDA(unittest.TestCase):

    def test_solve_fokker_planck_until_stationarity_and_make_video(self):
        experiment = AmpaNmdaExperimentConfigFactoryBuilder.fitted_apic20().build()

        # N_NMDA sources are shared by AMPA and NMDA; the remaining excitatory
        # sources drive AMPA only.  The diffusion tensor therefore contains
        # total AMPA variance, total NMDA variance, and their shared covariance.
        nu_ampa_per_ms = float(experiment.n_e * experiment.r_e / kHz)
        nu_nmda_per_ms = float(experiment.n_nmda * experiment.r_e / kHz)
        w_ampa_nS = float(experiment.w_ampa / nS)
        w_x = float(experiment.w_nmda)
        # Represent the covariance by diagonal grid transitions.  This
        # effective diagonal jump preserves AMPA variance and covariance; an
        # additional x-axis diffusion below restores the remaining NMDA variance.
        w_x_diagonal = (
            nu_nmda_per_ms * w_x / nu_ampa_per_ms
        )

        # Resolve voltage more finely than the compact synaptic grids. On the
        # old 24-point grid, one cell spanned 4.35 mV and grid-scale numerical
        # oscillations looked like additional modes in the marginal density.
        voltage = np.linspace(-80.0, 20.0, 121)
        # Resolve the synaptic marginals instead of representing each one with
        # only six cells.  Extending x is as important as refining it: the old
        # x_max=0.8 boundary was less than one theoretical standard deviation
        # above the mean and artificially compressed its stationary variance.
        # dg/dx remains 4, as required by the shared-noise diagonal below.
        g_ampa = np.linspace(0.0, 3.2, 11)   # dg = 0.32 nS
        x_nmda = np.linspace(0.0, 1.6, 21)   # dx = 0.08
        s_nmda = np.linspace(0.0, 20.0, 11)  # ds = 2
        grids = (voltage, g_ampa, x_nmda, s_nmda)
        spacings = tuple(grid[1] - grid[0] for grid in grids)
        if not np.isclose(
            spacings[1] / spacings[2], w_ampa_nS / w_x_diagonal
        ):
            raise ValueError(
                "g_AMPA and x_NMDA grid spacings do not follow shared-noise direction"
            )
        shape = tuple(grid.size for grid in grids)
        size = int(np.prod(shape))
        strides = (shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1)

        V, G, X, S = np.meshgrid(*grids, indexing="ij")
        sigma_mg = 1.0 / (
            1.0 + np.exp(-0.08 * V) * float(experiment.magnesium) / 3.57
        )
        g_nmda_scale_nS = float(experiment.g_nmda_max / nS) / experiment.n_nmda
        velocities = (
            (
                -float(experiment.g_L / nS) * (V - float(experiment.E_L / mV))
                - G * (V - float(experiment.E_E / mV))
                - g_nmda_scale_nS * S * sigma_mg
                * (V - float(experiment.E_E / mV))
            ) / float(experiment.C_m / nF),
            -G / float(experiment.tau_ampa / ms)
            + nu_ampa_per_ms * w_ampa_nS,
            -X / float(experiment.tau_nmda_rise / ms)
            + nu_nmda_per_ms * w_x,
            -S / float(experiment.tau_nmda_decay / ms)
            + float(experiment.alpha_nmda / kHz) * X,
        )

        flat_indices = np.arange(size).reshape(shape)
        rows, columns, data = [], [], []
        outgoing = np.zeros(size, dtype=float)

        def add_transitions(source, destination, rates):
            source = np.asarray(source, dtype=int).ravel()
            destination = np.asarray(destination, dtype=int).ravel()
            rates = np.asarray(rates, dtype=float).ravel()
            positive = rates > 0.0
            source, destination, rates = (
                source[positive], destination[positive], rates[positive]
            )
            rows.extend(destination.tolist())
            columns.extend(source.tolist())
            data.extend(rates.tolist())
            np.add.at(outgoing, source, rates)

        for axis, (velocity, spacing, stride) in enumerate(
            zip(velocities, spacings, strides)
        ):
            velocity = np.broadcast_to(velocity, shape)
            forward = [slice(None)] * 4
            forward[axis] = slice(None, -1)
            forward = tuple(forward)
            source = flat_indices[forward]
            add_transitions(
                source,
                source + stride,
                np.maximum(velocity[forward], 0.0) / spacing,
            )
            backward = [slice(None)] * 4
            backward[axis] = slice(1, None)
            backward = tuple(backward)
            source = flat_indices[backward]
            add_transitions(
                source,
                source - stride,
                np.maximum(-velocity[backward], 0.0) / spacing,
            )

        # Correlated diffusion lies along (w_A, w_x). The selected grid has
        # dg/dx = w_A/w_x, so it becomes diagonal nearest-neighbour diffusion.
        diffusion_rate = nu_ampa_per_ms * (w_ampa_nS / spacings[1]) ** 2
        plus = (slice(None), slice(None, -1), slice(None, -1), slice(None))
        source = flat_indices[plus]
        add_transitions(
            source,
            source + strides[1] + strides[2],
            np.full(source.shape, 0.5 * diffusion_rate),
        )
        minus = (slice(None), slice(1, None), slice(1, None), slice(None))
        source = flat_indices[minus]
        add_transitions(
            source,
            source - strides[1] - strides[2],
            np.full(source.shape, 0.5 * diffusion_rate),
        )
        represented_x_variance_rate = (
            diffusion_rate * spacings[2] ** 2
        )
        total_x_variance_rate = nu_nmda_per_ms * w_x**2
        residual_x_variance_rate = (
            total_x_variance_rate - represented_x_variance_rate
        )
        if residual_x_variance_rate < -1e-12:
            raise ValueError(
                "Synaptic grid cannot represent the shared diffusion tensor"
            )
        x_diffusion_rate = max(residual_x_variance_rate, 0.0) / (
            2.0 * spacings[2] ** 2
        )
        x_forward = (slice(None), slice(None), slice(None, -1), slice(None))
        source = flat_indices[x_forward]
        add_transitions(
            source,
            source + strides[2],
            np.full(source.shape, x_diffusion_rate),
        )
        x_backward = (slice(None), slice(None), slice(1, None), slice(None))
        source = flat_indices[x_backward]
        add_transitions(
            source,
            source - strides[2],
            np.full(source.shape, x_diffusion_rate),
        )
        diagonal = np.arange(size)
        rows.extend(diagonal.tolist())
        columns.extend(diagonal.tolist())
        data.extend((-outgoing).tolist())
        generator = coo_matrix((data, (rows, columns)), shape=(size, size)).tocsc()

        # Backward Euler is unconditionally stable; a 1 ms density step reaches
        # the same generator stationary state with one tenth as many sparse solves.
        dt_ms = 1.0
        identity = eye(size, format="csc")
        # Backward Euler is monotone for this continuous-time Markov
        # generator. Crank--Nicolson can oscillate around a delta initial
        # condition; clipping its negative cells changes the PDE and can
        # manufacture bumps in the voltage marginal.
        solve_implicit = factorized(identity - dt_ms * generator)

        g_mean = (
            nu_ampa_per_ms * float(experiment.tau_ampa / ms) * w_ampa_nS
        )
        x_mean = (
            nu_nmda_per_ms * float(experiment.tau_nmda_rise / ms) * w_x
        )
        s_mean = (
            float(experiment.alpha_nmda / kHz)
            * x_mean * float(experiment.tau_nmda_decay / ms)
        )
        initial_voltage_mV = -35.0
        upper_voltage_index = int(np.searchsorted(voltage, initial_voltage_mV))
        lower_voltage_index = upper_voltage_index - 1
        upper_voltage_weight = (
            initial_voltage_mV - voltage[lower_voltage_index]
        ) / (voltage[upper_voltage_index] - voltage[lower_voltage_index])
        synaptic_indices = (
            int(np.argmin(abs(g_ampa - g_mean))),
            int(np.argmin(abs(x_nmda - x_mean))),
            int(np.argmin(abs(s_nmda - s_mean))),
        )
        probability = np.zeros(size)
        probability[
            np.ravel_multi_index((lower_voltage_index, *synaptic_indices), shape)
        ] = 1.0 - upper_voltage_weight
        probability[
            np.ravel_multi_index((upper_voltage_index, *synaptic_indices), shape)
        ] = upper_voltage_weight

        frame_interval_ms = 5.0
        steps_per_frame = int(round(frame_interval_ms / dt_ms))
        maximum_time_ms = 400.0
        minimum_stationarity_time_ms = 150.0
        stationarity_tolerance = 3e-5
        stationary_checks_required = 5
        stationary_checks = 0
        last_change = float("inf")
        times_ms = [0.0]
        voltage_marginals = [
            _reconstruct_voltage_density(probability, shape, spacings[0])
        ]
        previous_marginal = voltage_marginals[-1]

        for step in range(1, int(maximum_time_ms / dt_ms) + 1):
            probability = solve_implicit(probability)
            if probability.min() < -1e-12:
                self.fail(
                    "Implicit Fokker--Planck step lost positivity: "
                    f"minimum probability={probability.min():g}"
                )
            # Remove only sparse-solver roundoff at a few ulps below zero.
            probability[probability < 0.0] = 0.0
            probability /= probability.sum()
            if step % steps_per_frame:
                continue
            time_ms = step * dt_ms
            marginal = _reconstruct_voltage_density(
                probability, shape, spacings[0]
            )
            change = np.sum(np.abs(marginal - previous_marginal)) * spacings[0]
            last_change = change
            times_ms.append(time_ms)
            voltage_marginals.append(marginal.copy())
            stationary_checks = (
                stationary_checks + 1
                if time_ms >= minimum_stationarity_time_ms
                and change < stationarity_tolerance
                else 0
            )
            previous_marginal = marginal
            if stationary_checks >= stationary_checks_required:
                break

        self.assertAlmostEqual(probability.sum(), 1.0, places=10)
        self.assertTrue(np.all(np.isfinite(probability)))
        fp_moments = _stationary_synaptic_moments(
            probability,
            shape,
            (g_ampa, x_nmda, s_nmda),
        )
        theoretical_moments = _theoretical_stationary_synaptic_moments(experiment)
        for variable_name in ("g_ampa", "x_nmda", "s_nmda"):
            for moment_name in ("mean", "variance"):
                with self.subTest(variable=variable_name, moment=moment_name):
                    np.testing.assert_allclose(
                        fp_moments[variable_name][moment_name],
                        theoretical_moments[variable_name][moment_name],
                        rtol=0.10,
                        atol=1e-12,
                        err_msg=(
                            f"stationary FP {variable_name} {moment_name} does not "
                            "match its analytical diffusion-model value"
                        ),
                    )
        stationary_marginal = voltage_marginals[-1]
        normalized_roughness = (
            np.max(np.abs(np.diff(stationary_marginal, n=2)))
            / np.max(stationary_marginal)
        )
        self.assertLess(
            normalized_roughness,
            0.08,
            "Stationary voltage marginal contains a grid-scale numerical kink",
        )
        self.assertGreater(
            stationary_checks,
            0,
            f"PDE did not reach stationarity; final marginal change={last_change:g}",
        )

        figure, axis = plt.subplots(figsize=(10, 6))
        line, = axis.plot(voltage, voltage_marginals[0], color="tab:green", linewidth=2)
        axis.axvline(-35.0, color="black", linestyle=":", label="initial delta: -35 mV")
        axis.set_xlim(voltage[0], voltage[-1])
        axis.set_ylim(0.0, 1.05 * max(np.max(frame) for frame in voltage_marginals))
        axis.set_xlabel("membrane voltage [mV]")
        axis.set_ylabel(r"$p_V(V,t)$ [1/mV]")
        axis.legend()

        def update(frame_index):
            line.set_ydata(voltage_marginals[frame_index])
            axis.set_title(
                "Four-dimensional Fokker–Planck evolution "
                "(positivity-preserving implicit step)\n"
                f"t={times_ms[frame_index]:.1f} ms; shared-event approximation"
            )
            return (line,)

        animation = FuncAnimation(
            figure, update, frames=len(times_ms), interval=1000 / 15, blit=False
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        video_path = OUTPUT_DIR / "fokker_planck_voltage_distribution_to_stationarity.mp4"
        animation.save(video_path, writer=FFMpegWriter(fps=15, bitrate=2400), dpi=150)
        plt.close(figure)

        self.assertTrue(video_path.exists(), str(video_path))
        self.assertGreater(video_path.stat().st_size, 0, str(video_path))

        simulated_20_seconds = simulate_one_unit_for_duration(20)
        comparison_path = OUTPUT_DIR / (
            "stationary_fokker_planck_vs_20_second_vm_histogram.png"
        )
        plot_one_vm_vs_upper_fixed_point(
            simulated_20_seconds,
            comparison_path,
            stationary_voltage_mV=voltage,
            stationary_density_per_mV=stationary_marginal,
        )
        self.assertTrue(comparison_path.exists(), str(comparison_path))
        self.assertGreater(comparison_path.stat().st_size, 0, str(comparison_path))

        return {
            "voltage_mV": voltage.copy(),
            "stationary_density_per_mV": stationary_marginal.copy(),
        }

    def test_plot_1_000_unit_two_second_vm_distribution_vs_stationary_fokker_planck(self):
        stationary = self.test_solve_fokker_planck_until_stationarity_and_make_video()
        experiment = (
            AmpaNmdaExperimentConfigFactoryBuilder.fitted_apic20()
            .with_property(
                n_models=1_000,
                simulation_time=2 * second,
                r_e = 17 * Hz
            )
            .build()
        )
        result = simulate_ampa_nmda_single_compartment_ensemble(experiment)
        simulated_voltage_mV = np.asarray(result["voltage_mV"], dtype=float).ravel()

        self.assertTrue(np.all(np.isfinite(simulated_voltage_mV)))
        self.assertGreater(simulated_voltage_mV.size, 0)

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.hist(
            simulated_voltage_mV,
            bins=150,
            density=True,
            histtype="step",
            linewidth=2,
            color="tab:green",
            label=(
                "1,000-unit Brian2 voltage distribution "
                f"(n={simulated_voltage_mV.size:,})"
            ),
        )
        axis.plot(
            stationary["voltage_mV"],
            stationary["stationary_density_per_mV"],
            color="tab:red",
            linewidth=2.5,
            label="stationary Fokker–Planck density",
        )
        axis.set_title(
            "Stationary Fokker–Planck density vs pooled simulated voltages\n"
            "1,000 units, T=2 s, burn-in=500 ms, "
            f"$r_e$={float(experiment.r_e / Hz):g} Hz"
        )
        axis.set_xlabel("membrane voltage [mV]")
        axis.set_ylabel("probability density [1/mV]")
        axis.legend()
        figure.tight_layout()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / (
            "stationary_fokker_planck_vs_1000_unit_2_second_vm_distribution.png"
        )
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        show_plots_non_blocking()

        self.assertTrue(output_path.exists(), str(output_path))
        self.assertGreater(output_path.stat().st_size, 0, str(output_path))

    def test_plot_one_vm_vs_upper_fixed_point(self):
        times = np.arange(9, 10, 1)
        completed = Parallel(n_jobs=-1, backend="loky")(
            delayed(simulate_one_unit_for_duration)(duration)
            for duration in times
        )
        completed.sort(key=lambda item: item["duration_seconds"])

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_paths = []
        for diagnostics in completed:
            voltage_mV = diagnostics["voltage_mV"]
            upper_fixed_point_mV = diagnostics["upper_fixed_point_mV"]
            center_of_mass_mV = diagnostics["center_of_mass_mV"]
            self.assertTrue(np.all(np.isfinite(voltage_mV)))
            self.assertIsNotNone(upper_fixed_point_mV)
            self.assertTrue(np.isfinite(center_of_mass_mV))
            delta_voltage_mV = upper_fixed_point_mV - center_of_mass_mV
            self.assertTrue(np.isfinite(delta_voltage_mV))

            output_path = OUTPUT_DIR / (
                "one_vm_vs_upper_fixed_point_"
                f"{diagnostics['duration_seconds']:g}_seconds.png"
            )
            plot_one_vm_vs_upper_fixed_point(diagnostics, output_path)
            output_paths.append(output_path)

        durations_seconds = np.asarray(
            [item["duration_seconds"] for item in completed], dtype=float
        )
        delta_voltages_mV = np.asarray(
            [
                item["upper_fixed_point_mV"] - item["center_of_mass_mV"]
                for item in completed
            ],
            dtype=float,
        )
        delta_figure, delta_axis = plt.subplots(figsize=(10, 6))
        delta_axis.plot(
            durations_seconds,
            delta_voltages_mV,
            color="tab:purple",
            marker="o",
            linewidth=2,
        )
        delta_axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        delta_axis.set_xlabel("simulation duration [s]")
        delta_axis.set_ylabel(
            r"$\Delta V=V_{\mathrm{upper\ FP}}-\langle V_{\mathrm{emp}} \rangle$ [mV]"
        )
        delta_axis.set_title(
            "Upper fixed point minus histogram center of mass\n"
            f"r_e={completed[0]['rate_hz']:g} Hz, "
            f"burn-in={completed[0]['burn_in_ms']:g} ms"
        )
        delta_figure.tight_layout()
        delta_output_path = OUTPUT_DIR / (
            "upper_fixed_point_minus_vm_center_of_mass_by_duration.png"
        )
        delta_figure.savefig(delta_output_path, dpi=200, bbox_inches="tight")

        show_plots_non_blocking()

        self.assertEqual(len(output_paths), len(times))
        for output_path in output_paths:
            self.assertTrue(output_path.exists(), str(output_path))
            self.assertGreater(output_path.stat().st_size, 0, str(output_path))
        self.assertTrue(delta_output_path.exists(), str(delta_output_path))
        self.assertGreater(
            delta_output_path.stat().st_size, 0, str(delta_output_path)
        )

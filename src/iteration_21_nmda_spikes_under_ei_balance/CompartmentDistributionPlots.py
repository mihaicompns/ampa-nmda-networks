"""Reusable per-compartment distribution diagnostics for iteration 21."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Fixed points are expensive root solves. Increase this if still smoother
# overlays are worth the additional runtime; 50,000 is 10x the old limit.
MAX_FIXED_POINT_S_SAMPLES_PER_COMPARTMENT = 50_000


def validate_compartment_indices(result, compartment_indices):
    """Return selected row indices after checking all recorded arrays."""
    indices = tuple(int(index) for index in compartment_indices)
    if not indices:
        raise ValueError("compartment_indices must contain at least one index")
    number_of_compartments = np.asarray(result["s_nmda"]).shape[0]
    invalid = [index for index in indices if not 0 <= index < number_of_compartments]
    if invalid:
        raise IndexError(
            f"compartment indices {invalid} are outside [0, {number_of_compartments})"
        )
    return indices


def plot_compartment_s_voltage_fixed_point_distributions(
    result,
    compartment_indices,
    fixed_points_from_s,
    output_path,
    *,
    show_plots=False,
    bins=100,
    max_fixed_point_samples=MAX_FIXED_POINT_S_SAMPLES_PER_COMPARTMENT,
    figure_title=None,
):
    """Plot s and V distributions, with s-conditioned fixed points over V.

    ``fixed_points_from_s(index, s_samples)`` must return the usual mapping
    containing ``lower_fixed_point_mV`` and ``upper_fixed_point_mV``.  This
    keeps plotting independent of the original and summed-input model types.
    """
    indices = validate_compartment_indices(result, compartment_indices)
    s_nmda = np.asarray(result["s_nmda"], dtype=float)
    voltage_mV = np.asarray(result["voltage_mV"], dtype=float)
    fixed_point_sample_count = min(
        s_nmda.shape[1], int(max_fixed_point_samples)
    )
    figure, axes = plt.subplots(
        len(indices),
        2,
        figsize=(13, max(3.5 * len(indices), 4.5)),
        squeeze=False,
    )

    for row, compartment_index in enumerate(indices):
        s_samples = s_nmda[compartment_index]
        voltage_samples = voltage_mV[compartment_index]
        fixed_point_s_samples = s_samples
        if s_samples.size > max_fixed_point_samples:
            sample_indices = np.random.default_rng(compartment_index).choice(
                s_samples.size, max_fixed_point_samples, replace=False
            )
            fixed_point_s_samples = s_samples[sample_indices]
        fixed_points = fixed_points_from_s(
            compartment_index, fixed_point_s_samples
        )
        lower = np.asarray(
            [value for value in fixed_points["lower_fixed_point_mV"] if value is not None],
            dtype=float,
        )
        upper = np.asarray(
            [value for value in fixed_points["upper_fixed_point_mV"] if value is not None],
            dtype=float,
        )

        s_axis, voltage_axis = axes[row]
        s_axis.hist(s_samples, bins=bins, density=True, color="tab:red", alpha=0.7)
        s_axis.set_title(f"Compartment {compartment_index}: s_NMDA distribution")
        s_axis.set_xlabel("s_NMDA [1]")
        s_axis.set_ylabel("probability density")

        voltage_axis.hist(
            voltage_samples,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color="tab:green",
            label="simulated V",
        )
        for values, color, label in (
            (lower, "tab:blue", "lower stable fixed point from s"),
            (upper, "tab:orange", "upper stable fixed point from s"),
        ):
            if values.size:
                voltage_axis.hist(
                    values,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=2,
                    color=color,
                    label=label,
                )
        voltage_axis.set_title(
            f"Compartment {compartment_index}: V and s-conditioned fixed points"
        )
        voltage_axis.set_xlabel("voltage [mV]")
        voltage_axis.set_ylabel("probability density")
        voltage_axis.legend(loc="best", fontsize=8)

    title = (
        "Per-compartment stationary distributions of s and V together with "
        f"stable fixed points (n={fixed_point_sample_count:,} s samples per compartment)"
    )
    if figure_title:
        title += f"\n{figure_title}"
    figure.suptitle(title, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plots:
        plt.show(block=False)
    else:
        plt.close(figure)
    return output_path

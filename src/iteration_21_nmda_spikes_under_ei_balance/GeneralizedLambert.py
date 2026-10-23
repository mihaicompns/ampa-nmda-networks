"""Fixed points and bifurcation diagram for instantaneous NMDA conductance."""

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw

# Model parameters
A = 0.25
B = 0.08  # 1/mV
E_L = -70.0  # mV
E_E = 0.0  # mV
G_L = 3.056  # nS

OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"
BIFURCATION_PLOT_PATH = OUTPUT_DIR / "generalized_lambert_g_n_bifurcation.png"
FIXED_POINT_EVOLUTION_PLOT_PATH = (
    OUTPUT_DIR / "generalized_lambert_fixed_point_evolution.png"
)
LOWER_UPPER_BRANCH_PLOT_PATH = (
    OUTPUT_DIR / "generalized_lambert_lower_upper_fixed_points.png"
)


def generalized_lambert_parameters(
    g_n, *, g_l=G_L, e_l=E_L, e_e=E_E, a=A, b=B
):
    r"""Return ``(r, z)`` for the generalized Lambert equation.

    The voltage solutions have the form

    .. math::

        V = E_L - B^{-1} W_r(z),

    where :math:`W_r(z)` is any real solution of
    :math:`x e^x + r x = z`.  For the current-balance equation used here,

    .. math::

        r = \frac{g_L+g_N}{A g_L}e^{B E_L}, \qquad
        z = -\frac{B}{A}\frac{g_N}{g_L}e^{B E_L}(E_E-E_L).

    Depending on ``(r, z)``, there is either one or three real branches.
    """
    g_n = float(g_n)
    r = (g_l + g_n) / (a * g_l) * np.exp(b * e_l)
    z = -(b / a) * (g_n / g_l) * np.exp(b * e_l) * (e_e - e_l)
    return r, z


def find_fixed_points(
    g_n=16.5, *, g_l=G_L, e_l=E_L, e_e=E_E, a=A, b=B
):
    """Return all membrane-voltage fixed points for one g_N value."""
    r, z = generalized_lambert_parameters(
        g_n, g_l=g_l, e_l=e_l, e_e=e_e, a=a, b=b
    )

    print(e_l)

    def generalized_lambert_residual(x):
        return x * np.exp(x) + r * x - z

    if r >= np.exp(-2):
        intervals = [(-100.0, 10.0)]
    else:
        alpha = float(lambertw(-np.e * r, k=-1).real - 1)
        beta = float(lambertw(-np.e * r, k=0).real - 1)
        intervals = [(-100.0, alpha), (alpha, beta), (beta, 10.0)]

    roots = []
    for left, right in intervals:
        f_left = generalized_lambert_residual(left)
        f_right = generalized_lambert_residual(right)
        if abs(f_left) < 1e-10:
            roots.append(left)
        if abs(f_right) < 1e-10:
            roots.append(right)
        if f_left * f_right < 0:
            roots.append(brentq(generalized_lambert_residual, left, right))

    voltages = sorted(e_l - x / b for x in roots)
    unique_voltages = []
    for voltage in voltages:
        if not unique_voltages or not np.isclose(voltage, unique_voltages[-1], atol=1e-8):
            unique_voltages.append(float(voltage))
    return unique_voltages


def fixed_point_current_slope(
    voltage_mV, g_n, *, g_l=G_L, e_e=E_E, a=A, b=B
):
    """Return dI_total/dV; a positive slope is stable for C dV/dt=-I."""
    sigma_nmda = 1.0 / (1.0 + a * np.exp(-b * voltage_mV))
    sigma_slope = b * sigma_nmda * (1.0 - sigma_nmda)
    return float(
        g_l + g_n * (sigma_nmda + (voltage_mV - e_e) * sigma_slope)
    )


def total_current_balance(
    voltage_mV, g_n, *, g_l=G_L, e_l=E_L, e_e=E_E, a=A, b=B
):
    """Return leak plus instantaneous NMDA current in nS*mV."""
    voltage_mV = np.asarray(voltage_mV, dtype=float)
    sigma_nmda = 1.0 / (1.0 + a * np.exp(-b * voltage_mV))
    return - g_l * (voltage_mV - e_l) - g_n * sigma_nmda * (voltage_mV - e_e)


def plot_fixed_point_evolution(
    g_n_values=(0.0, 10.0, 15.0, 16.5, 20.0, 30.0),
    voltage_min_mV=-80.0,
    voltage_max_mV=5.0,
    output_path=FIXED_POINT_EVOLUTION_PLOT_PATH,
    show_plot=True,
):
    """Plot current-balance curves and their fixed points as g_N increases."""
    voltage_grid = np.linspace(voltage_min_mV, voltage_max_mV, 2000)
    figure, axis = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(g_n_values)))
    for g_n, color in zip(g_n_values, colors):
        current = total_current_balance(voltage_grid, g_n)
        axis.plot(voltage_grid, current, color=color, label=rf"$g_N={g_n:g}$ nS")
        fixed_points = find_fixed_points(g_n)
        axis.scatter(
            fixed_points,
            np.zeros(len(fixed_points)),
            color=color,
            edgecolor="black",
            linewidth=0.5,
            s=45,
            zorder=3,
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("membrane voltage V [mV]")
    axis.set_ylabel(r"current balance $I_L+I_{NMDA}$ [nS mV]")
    axis.set_title("Evolution of fixed points with increasing NMDA conductance")
    axis.legend(ncol=2)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return output_path


def scan_fixed_points(g_n_values):
    """Evaluate fixed-point voltage and stability over a conductance grid."""
    records = []
    root_counts = []
    for g_n in np.asarray(g_n_values, dtype=float):
        fixed_points = find_fixed_points(g_n)
        root_counts.append(len(fixed_points))
        for voltage_mV in fixed_points:
            slope = fixed_point_current_slope(voltage_mV, g_n)
            records.append(
                {
                    "g_n_nS": float(g_n),
                    "voltage_mV": voltage_mV,
                    "current_slope_nS": slope,
                    "stable": slope > 0.0,
                }
            )
    return records, np.asarray(root_counts, dtype=int)


def lower_and_upper_fixed_point_branches(g_n_values):
    """Return lower/upper stable branches, with NaN where a branch does not exist."""
    g_n_values = np.asarray(g_n_values, dtype=float)
    all_fixed_points = [find_fixed_points(g_n) for g_n in g_n_values]
    root_counts = np.asarray([len(points) for points in all_fixed_points], dtype=int)
    three_root_indices = np.flatnonzero(root_counts == 3)
    if three_root_indices.size == 0:
        raise ValueError("The g_N grid does not enter the three-fixed-point region")

    first_three = int(three_root_indices[0])
    last_three = int(three_root_indices[-1])
    lower_branch = np.full(g_n_values.shape, np.nan)
    upper_branch = np.full(g_n_values.shape, np.nan)
    for index, fixed_points in enumerate(all_fixed_points):
        if index <= last_three:
            lower_branch[index] = fixed_points[0]
        if index >= first_three:
            upper_branch[index] = fixed_points[-1]
    return lower_branch, upper_branch, root_counts


def plot_lower_and_upper_fixed_points(
    g_n_min=0.0,
    g_n_max=30.0,
    number_of_g_n_values=2001,
    output_path=LOWER_UPPER_BRANCH_PLOT_PATH,
    show_plot=True,
):
    """Plot lower and upper stable fixed-point branches on separate axes."""
    g_n_values = np.linspace(g_n_min, g_n_max, number_of_g_n_values)
    lower_branch, upper_branch, root_counts = lower_and_upper_fixed_point_branches(
        g_n_values
    )
    transitions = np.flatnonzero(np.diff(root_counts) != 0)
    boundaries = [
        0.5 * (g_n_values[index] + g_n_values[index + 1])
        for index in transitions
    ]

    figure, (lower_axis, upper_axis) = plt.subplots(
        2, 1, figsize=(10, 9), sharex=True, sharey=True
    )
    lower_axis.plot(g_n_values, lower_branch, color="tab:blue", linewidth=2)
    lower_axis.set_title("Lower stable fixed point (terminates at upper saddle-node)")
    upper_axis.plot(g_n_values, upper_branch, color="tab:orange", linewidth=2)
    upper_axis.set_title("Upper stable fixed point (appears at lower saddle-node)")
    for axis in (lower_axis, upper_axis):
        for boundary_number, boundary in enumerate(boundaries):
            axis.axvline(
                boundary,
                color="0.35",
                linestyle=":",
                linewidth=1,
                label="approx. saddle-node" if boundary_number == 0 else None,
            )
        axis.set_ylabel("fixed-point voltage [mV]")
        axis.legend()
    upper_axis.set_xlabel(r"instantaneous $g_N$ [nS]")
    figure.suptitle("Lower and upper fixed-point branches versus NMDA conductance")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return output_path, g_n_values, lower_branch, upper_branch


def plot_fixed_point_bifurcation(
    g_n_min=0.0,
    g_n_max=30.0,
    number_of_g_n_values=2001,
    output_path=BIFURCATION_PLOT_PATH,
    show_plot=True,
):
    """Plot equilibrium voltage branches as instantaneous g_N increases."""
    g_n_values = np.linspace(g_n_min, g_n_max, number_of_g_n_values)
    records, root_counts = scan_fixed_points(g_n_values)
    stable = [record for record in records if record["stable"]]
    unstable = [record for record in records if not record["stable"]]

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.scatter(
        [record["g_n_nS"] for record in stable],
        [record["voltage_mV"] for record in stable],
        s=7,
        color="tab:blue",
        label="stable fixed point",
    )
    axis.scatter(
        [record["g_n_nS"] for record in unstable],
        [record["voltage_mV"] for record in unstable],
        s=11,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=0.7,
        label="unstable fixed point",
    )

    transitions = np.flatnonzero(np.diff(root_counts) != 0)
    for transition_number, index in enumerate(transitions):
        boundary = 0.5 * (g_n_values[index] + g_n_values[index + 1])
        axis.axvline(
            boundary,
            color="0.35",
            linestyle=":",
            linewidth=1,
            label="approx. saddle-node" if transition_number == 0 else None,
        )

    axis.set_xlabel(r"instantaneous $g_N$ [nS]")
    axis.set_ylabel("fixed-point voltage [mV]")
    axis.set_title("Fixed-point bifurcation diagram versus NMDA conductance")
    axis.legend()
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return output_path, records, root_counts


class GeneralizedLambertRunnableTestCases(unittest.TestCase):
    def test_plot_fixed_point_evolution(self):
        plot_path = plot_fixed_point_evolution()
        self.assertTrue(plot_path.exists(), str(plot_path))
        self.assertGreater(plot_path.stat().st_size, 0, str(plot_path))

    def test_plot_fixed_point_bifurcation(self):
        plot_path, records, root_counts = plot_fixed_point_bifurcation()
        self.assertTrue(plot_path.exists(), str(plot_path))
        self.assertGreater(len(records), 0)
        self.assertIn(3, root_counts)

    def test_plot_lower_and_upper_fixed_points(self):
        plot_path, _g_n, lower_branch, upper_branch = (
            plot_lower_and_upper_fixed_points()
        )
        self.assertTrue(plot_path.exists(), str(plot_path))
        self.assertTrue(np.isnan(lower_branch[-1]))
        self.assertTrue(np.isnan(upper_branch[0]))


if __name__ == "__main__":
    unittest.main()

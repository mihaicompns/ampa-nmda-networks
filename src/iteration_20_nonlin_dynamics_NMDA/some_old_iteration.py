import unittest

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ============================================================
# Model parameters
# Replace these with your actual parameter values
# ============================================================

g_L = 1.0
g_A = 2.0
g_N = 2.0
g_G = 2.0

E_L = -65.0
E_A = 0.0
E_N = 0.0
E_G = -70.0

Mg = 1.0  # [Mg2+]

alpha_N = 1.0
G_glu = 0.01
beta_N_values = [
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1
]

g_N_over_g_L_values = [
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0
]

# ============================================================
# Magnesium block
# ============================================================

def sigma_Mg(v):
    return 1.0 / (1.0 + (Mg / 3.75) * np.exp(-0.062 * v))


# ============================================================
# Effective NMDA conductance
#
# For now assumed constant.
# Replace this function if g_N_eff depends on v.
# ============================================================

def g_N_eff(v, g_N_fixed=g_N):
    return g_N_fixed * np.ones_like(v)


# ============================================================
# v-nullcline solved for m_N
#
# dv/dt = 0:
#
# 0 = -g_L(v-E_L)
#     -g_A*m_A*(v-E_A)
#     -g_N_eff*m_N*sigma_Mg(v)*(v-E_N)
#     -g_G*m_G*(v-E_G)
#
# therefore:
#
# m_N = -(leak + AMPA + GABA)
#       / (g_N_eff * sigma_Mg(v) * (v-E_N))
# ============================================================

def mN_v_nullcline(v, mA, mG, g_N_fixed=g_N):

    numerator = (
        g_L * (v - E_L)
        + g_A * mA * (v - E_A)
        + g_G * mG * (v - E_G)
    )

    denominator = (
        g_N_eff(v, g_N_fixed=g_N_fixed)
        * sigma_Mg(v)
        * (v - E_N)
    )

    return -numerator / denominator


def v_nullcline_residual(v, mA, mG, mN, use_Mg_block=True, g_N_fixed=g_N):
    Mg_block = sigma_Mg(v) if use_Mg_block else 1.0

    return (
        g_L * (v - E_L)
        + g_A * mA * (v - E_A)
        + g_N_eff(v, g_N_fixed=g_N_fixed) * mN * Mg_block * (v - E_N)
        + g_G * mG * (v - E_G)
    )


def mN_fixed_point(beta_N, alpha_N_fixed=alpha_N, G_glu_fixed=G_glu):
    return (
        alpha_N_fixed * G_glu_fixed
        / (alpha_N_fixed * G_glu_fixed + beta_N)
    )


# ============================================================
# Grid
# ============================================================

v_values = np.linspace(-90, -10, 300)
mA_values = np.linspace(0.0, 1.0, 150)

V, MA = np.meshgrid(v_values, mA_values)


# ============================================================
# Plotting helpers
# ============================================================


def physical_mN_v_nullcline(mG_fixed):
    MN = mN_v_nullcline(V, MA, mG_fixed)

    # --------------------------------------------------------
    # Physical gating-variable restriction:
    #
    # 0 <= m_N <= 1
    #
    # Values outside this interval are mathematically valid
    # solutions of dv/dt = 0, but are not physically meaningful
    # gating states.
    # --------------------------------------------------------

    return np.where(
        (MN >= 0.0) & (MN <= 1.0),
        MN,
        np.nan
    )


def plot_nullcline_surface_on_axis(ax, mG_fixed):
    MN_physical = physical_mN_v_nullcline(mG_fixed)

    surface = ax.plot_surface(
        V,
        MA,
        MN_physical,
        cmap="viridis",
        alpha=0.85,
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel(r"$v$ [mV]")
    ax.set_ylabel(r"$m_A$")
    ax.set_zlabel(r"$m_N$")

    ax.set_zlim(0, 1)
    ax.set_title(rf"$m_G={mG_fixed:.2f}$")
    ax.view_init(elev=25, azim=-135)

    return surface


def plot_nullcline_surface(mG_fixed, show_plot=True):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = plot_nullcline_surface_on_axis(ax, mG_fixed)

    ax.set_title(
        rf"$v$-nullcline, fixed $m_G={mG_fixed:.2f}$"
    )

    fig.colorbar(
        surface,
        ax=ax,
        shrink=0.6,
        label=r"$m_N$"
    )

    plt.tight_layout()
    if show_plot:
        plt.show()

    return fig, ax


def plot_mG_grid(show_plot=True):
    fig = plt.figure(figsize=(15, 12))
    grid_mG_values = np.linspace(0.0, 1.0, 9)

    for plot_index, mG_fixed in enumerate(grid_mG_values, start=1):
        ax = fig.add_subplot(3, 3, plot_index, projection="3d")
        plot_nullcline_surface_on_axis(ax, mG_fixed)

    fig.suptitle(
        r"$v$-nullcline for increasing fixed values of $m_G$",
        fontsize=16
    )

    plt.tight_layout()
    if show_plot:
        plt.show()

    return fig


def find_v_roots_for_mN(
        mN_fixed,
        mA_fixed=0.0,
        mG_fixed=0.0,
        use_Mg_block=True,
        g_N_fixed=g_N
):
    residuals = v_nullcline_residual(
        v_values,
        mA=mA_fixed,
        mG=mG_fixed,
        mN=mN_fixed,
        use_Mg_block=use_Mg_block,
        g_N_fixed=g_N_fixed
    )
    roots = []

    for left_index, right_index in zip(range(len(v_values) - 1), range(1, len(v_values))):
        v_left = v_values[left_index]
        v_right = v_values[right_index]
        f_left = residuals[left_index]
        f_right = residuals[right_index]

        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue

        if f_left == 0.0:
            roots.append(v_left)
        elif f_left * f_right < 0.0:
            solution = root_scalar(
                lambda v: v_nullcline_residual(
                    v,
                    mA=mA_fixed,
                    mG=mG_fixed,
                    mN=mN_fixed,
                    use_Mg_block=use_Mg_block,
                    g_N_fixed=g_N_fixed
                ),
                bracket=[v_left, v_right],
                method="brentq"
            )
            roots.append(solution.root)

        if f_right == 0.0:
            roots.append(v_right)

    return sorted(set(np.round(roots, decimals=10)))


def voltage_branch_for_mN_axis(
        mN_axis,
        mA_fixed=0.0,
        mG_fixed=0.0,
        use_Mg_block=True,
        g_N_fixed=g_N
):
    roots_by_mN = [
        find_v_roots_for_mN(
            mN_fixed,
            mA_fixed=mA_fixed,
            mG_fixed=mG_fixed,
            use_Mg_block=use_Mg_block,
            g_N_fixed=g_N_fixed
        )
        for mN_fixed in mN_axis
    ]
    max_roots = max(len(roots) for roots in roots_by_mN)

    branches = []
    for root_index in range(max_roots):
        branches.append(np.array([
            roots[root_index] if root_index < len(roots) else np.nan
            for roots in roots_by_mN
        ]))

    return branches


def plot_v_vs_mN_ignoring_AMPA_GABA(show_plot=True):
    mN_axis = np.linspace(0.0, 1.0, 300)

    fig, ax = plt.subplots(figsize=(8, 5))
    all_branch_values = []

    for use_Mg_block, label, linestyle in [
        (True, r"with $\sigma_{\mathrm{Mg}}(v)$", "-"),
        (False, r"without Mg block", "--")
    ]:
        branches = voltage_branch_for_mN_axis(
            mN_axis,
            mA_fixed=0.0,
            mG_fixed=0.0,
            use_Mg_block=use_Mg_block
        )

        for root_index, branch in enumerate(branches):
            all_branch_values.extend(branch[np.isfinite(branch)])
            ax.plot(
                mN_axis,
                branch,
                linewidth=2,
                linestyle=linestyle,
                label=label if root_index == 0 else None
            )

    ax.set_xlabel(r"$m_N$")
    ax.set_ylabel(r"$v$ [mV]")
    ax.set_xlim(0, 1)
    if all_branch_values:
        y_min = min(all_branch_values)
        y_max = max(all_branch_values)
        y_margin = 0.08 * (y_max - y_min)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_title(r"$v(m_N)$ nullcline with Mg sigmoid, $m_A=0$, $m_G=0$")
    ax.legend()

    plt.tight_layout()
    if show_plot:
        plt.show()

    return fig, ax


def plot_v_vs_mN_for_g_NMDA_strengths(show_plot=True):
    mN_axis = np.linspace(0.0, 1.0, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    all_branch_values = []

    for g_N_over_g_L in g_N_over_g_L_values:
        g_N_fixed = g_N_over_g_L * g_L
        branches = voltage_branch_for_mN_axis(
            mN_axis,
            mA_fixed=0.0,
            mG_fixed=0.0,
            use_Mg_block=True,
            g_N_fixed=g_N_fixed
        )

        for root_index, branch in enumerate(branches):
            all_branch_values.extend(branch[np.isfinite(branch)])
            ax.plot(
                mN_axis,
                branch,
                linewidth=2,
                label=(
                    rf"$g_{{NMDA}}/g_L={g_N_over_g_L:g}$"
                    if root_index == 0
                    else None
                )
            )

    ax.set_xlabel(r"$m_N$")
    ax.set_ylabel(r"$v$ [mV]")
    ax.set_xlim(0, 1)
    if all_branch_values:
        y_min = min(all_branch_values)
        y_max = max(all_branch_values)
        y_margin = 0.08 * (y_max - y_min)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_title(
        r"$v(m_N)$ nullcline with Mg sigmoid, varying $g_{NMDA}/g_L$"
    )
    ax.legend()

    plt.tight_layout()
    if show_plot:
        plt.show()

    return fig, ax


def plot_v_vs_mN_for_beta_N_values(show_plot=True):
    fig, ax = plot_v_vs_mN_ignoring_AMPA_GABA(show_plot=False)

    for beta_N in beta_N_values:
        mN_star = mN_fixed_point(beta_N)
        roots = find_v_roots_for_mN(mN_star, mA_fixed=0.0, mG_fixed=0.0)

        ax.axvline(
            mN_star,
            linestyle="--",
            linewidth=1.5,
            label=rf"$\beta_N={beta_N:.3f}\ \mathrm{{ms}}^{{-1}}$, $m_N^*={mN_star:.2f}$"
        )

        if roots:
            ax.scatter(
                [mN_star] * len(roots),
                roots,
                s=35,
                zorder=3
            )

    ax.set_title(
        r"$v$-nullcline with $m_N^*(\beta_N)$, $m_A=0$, $m_G=0$"
    )
    ax.legend()

    plt.tight_layout()
    if show_plot:
        plt.show()

    return fig, ax


def plot_beta_N_vs_v_fixed_points(show_plot=True):
    beta_N_axis = np.linspace(
        min(beta_N_values),
        max(beta_N_values),
        300
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for use_Mg_block, label, linestyle in [
        (True, r"with $\sigma_{\mathrm{Mg}}(v)$", "-"),
        (False, r"without Mg block", "--")
    ]:
        roots_by_beta_N = []

        for beta_N in beta_N_axis:
            mN_star = mN_fixed_point(beta_N)
            roots_by_beta_N.append(
                find_v_roots_for_mN(
                    mN_star,
                    mA_fixed=0.0,
                    mG_fixed=0.0,
                    use_Mg_block=use_Mg_block
                )
            )

        max_roots = max(len(roots) for roots in roots_by_beta_N)

        for root_index in range(max_roots):
            branch = [
                roots[root_index] if root_index < len(roots) else np.nan
                for roots in roots_by_beta_N
            ]
            ax.plot(
                beta_N_axis,
                branch,
                linewidth=2,
                linestyle=linestyle,
                label=label if root_index == 0 else None
            )

    for beta_N in beta_N_values:
        mN_star = mN_fixed_point(beta_N)
        roots = find_v_roots_for_mN(
            mN_star,
            mA_fixed=0.0,
            mG_fixed=0.0,
            use_Mg_block=True
        )
        if roots:
            ax.scatter(
                [beta_N] * len(roots),
                roots,
                s=35,
                zorder=3
            )

    mN_stars = [
        mN_fixed_point(beta_N)
        for beta_N in beta_N_values
    ]
    print(
        "m_N* range for beta_N_values:",
        min(mN_stars),
        max(mN_stars)
    )

    ax.set_xlabel(r"$\beta_N$ [$\mathrm{ms}^{-1}$]")
    ax.set_ylabel(r"$v^*$ [mV]")
    ax.set_title(
        r"Fixed voltage root as $\beta_N$ varies, $m_A=0$, $m_G=0$"
    )
    ax.legend()

    plt.tight_layout()
    if show_plot:
        plt.show()

    return fig, ax


class TestNonlinearDynamicsNMDAScripts(unittest.TestCase):
    def test_plot_single_mG_nullcline(self):
        plot_nullcline_surface(mG_fixed=0.2)

    def test_plot_v_vs_mN_ignoring_AMPA_GABA(self):
        plot_v_vs_mN_ignoring_AMPA_GABA()

    def test_plot_v_vs_mN_for_g_NMDA_strengths(self):
        plot_v_vs_mN_for_g_NMDA_strengths()

    def test_plot_beta_N_vs_voltage(self):
        plot_beta_N_vs_v_fixed_points()

    def test_plot_each_mG_nullcline(self):
        mG_values = [
            0.0,
            0.1,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0
        ]

        for mG_fixed in mG_values:
            plot_nullcline_surface(mG_fixed)

    def test_plot_mG_grid(self):
        plot_mG_grid()


if __name__ == "__main__":
    unittest.main()

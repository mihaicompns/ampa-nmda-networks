from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import lambertw


R_CRIT = np.exp(-2)

BRANCH_COLORS = {
    "-2": "tab:red",
    "-1": "tab:blue",
    "0": "tab:green",
}


@dataclass(frozen=True)
class RLambertBranch:
    name: str
    x_min: float
    x_max: float
    monotonicity: str

    @property
    def label(self):
        return fr"$W_{{r,{self.name}}}$ branch"


def f_r(x, r):
    return x * np.exp(x) + r * x


def real_branches_for_r(r, x_min=-10, x_max=0.2):
    """
    Return the real r-Lambert branches described by Mezo and Baricz.

    Branch colors are keyed by the returned branch names, so plot functions can
    keep a stable branch color across detail and combined figures.
    """
    if r < 0:
        gamma = lambertw(-r * np.e, 0).real - 1
        return [
            RLambertBranch("-1", x_min, gamma, "decreasing"),
            RLambertBranch("0", gamma, x_max, "increasing"),
        ]

    if np.isclose(r, 0):
        return [
            RLambertBranch("-1", x_min, -1.0, "decreasing"),
            RLambertBranch("0", -1.0, x_max, "increasing"),
        ]

    if np.isclose(r, R_CRIT):
        return [RLambertBranch("0", x_min, x_max, "increasing")]

    if 0 < r < R_CRIT:
        alpha = lambertw(-r * np.e, -1).real - 1
        beta = lambertw(-r * np.e, 0).real - 1
        return [
            RLambertBranch("-2", x_min, alpha, "increasing"),
            RLambertBranch("-1", alpha, beta, "decreasing"),
            RLambertBranch("0", beta, x_max, "increasing"),
        ]

    return [RLambertBranch("0", x_min, x_max, "increasing")]


def branch_count_for_r(r):
    return len(real_branches_for_r(r))


def points_for_branch(branch, r, n=1000):
    x = np.linspace(branch.x_min, branch.x_max, n)
    return f_r(x, r), x


def W_r_parametric(r, x_min=-10, x_max=0.2, n=1000):
    """
    Parametric representation of the real branches of W_r.

    Returns (a, W_r(a)), where

        a = f_r(x) = x exp(x) + r x,
        W_r(a) = x.

    NaNs separate distinct real branches.
    """
    branch_points = [
        points_for_branch(branch, r, n)
        for branch in real_branches_for_r(r, x_min=x_min, x_max=x_max)
    ]

    a_parts = []
    x_parts = []
    for a, x in branch_points:
        if a_parts:
            a_parts.append(np.array([np.nan]))
            x_parts.append(np.array([np.nan]))
        a_parts.append(a)
        x_parts.append(x)

    return np.concatenate(a_parts), np.concatenate(x_parts)


def plot_r_lambert_details(r, x_min=-10, x_max=0.2, n=1000, ax=None):
    """
    Plot one value of r, with each real branch colored and named in the legend.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for branch in real_branches_for_r(r, x_min=x_min, x_max=x_max):
        a, wr = points_for_branch(branch, r, n=n)
        ax.plot(
            a,
            wr,
            color=BRANCH_COLORS[branch.name],
            linewidth=2.5,
            label=branch.label,
        )

    ax.set_title(fr"Real branches of $W_r(a)$ for $r={r:g}$")
    ax.set_xlabel(r"$a = x e^x + r x$")
    ax.set_ylabel(r"$W_r(a)$")
    ax.legend(title="Branches")
    return ax


def plot_each_r_lambert_case(r_values, x_min=-10, x_max=0.2, n=1000):
    """
    Plot one figure per r value. These are the detail plots.
    """
    axes = []
    for r in r_values:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_r_lambert_details(r, x_min=x_min, x_max=x_max, n=n, ax=ax)
        fig.tight_layout()
        axes.append(ax)
    return axes


def plot_all_r_lambert_cases(r_values, x_min=-10, x_max=0.2, n=1000, ax=None):
    """
    Plot all r values in one figure.

    Branch colors remain stable, while the legend contains only whole functions.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    line_styles = ["-", "--", ":", "-."]
    legend_handles = []

    for index, r in enumerate(r_values):
        line_style = line_styles[index % len(line_styles)]
        for branch in real_branches_for_r(r, x_min=x_min, x_max=x_max):
            a, wr = points_for_branch(branch, r, n=n)
            ax.plot(
                a,
                wr,
                color=BRANCH_COLORS[branch.name],
                linestyle=line_style,
                linewidth=2,
                alpha=0.9,
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=line_style,
                linewidth=2,
                label=fr"$r={r:g}$",
            )
        )

    ax.set_title(r"Real branches of $W_r(a)$")
    ax.set_xlabel(r"$a = x e^x + r x$")
    ax.set_ylabel(r"$W_r(a)$")
    ax.legend(handles=legend_handles, title="Functions")
    return ax


def main():
    r_values = [0.2, 0.1, 0, -0.1]
    plot_each_r_lambert_case(r_values)
    plot_all_r_lambert_cases(r_values)
    plt.show()


if __name__ == "__main__":
    main()

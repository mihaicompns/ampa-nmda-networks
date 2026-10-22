import unittest

import numpy as np
from numpy.ma.extras import flatnotmasked_edges
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from Plotting import show_plots_non_blocking


# Definition: W_R(z) * exp(W_R(z)) + R * W_R(z) = z
def f(u, R, z):
    return u * np.exp(u) + R * u - z

def compute_two_roots(A, B, E_E):
    # Generalized Lambert parameter and argument
    R = -A * np.exp(-B * E_E - 2)
    z = -4 * R
    # u < -4
    root_negative_branch = brentq(lambda x: f(x, R=R, z=z), -10.0, -4.0)
    root_positive_branch = brentq(lambda x: f(x, R=R, z=z), 0.0, 10.0)
    return root_negative_branch, root_positive_branch

def u_to_E_L(u, E_E, B):
    return E_E + u / B

def compute_E_L(A, B, E_E=0.):

    R = -A * np.exp(-B * E_E - 2)
    z = -4* R
    u_minus, u_plus = compute_two_roots(A, B, E_E)
    E_L_minus = u_to_E_L(u_minus, E_E, B)
    E_L_plus = u_to_E_L(u_plus, E_E, B)

    print("Compute E_L ^ *")
    print(f"R = {R:.10f}")
    print(f"z = {z:.10f}")
    print()
    print(f"W_R,-1(-4R) = {u_minus:.10f}")
    print(f"W_R,0(-4R)  = {u_plus:.10f}")
    print()
    print(f"E_L,- = {E_L_minus:.6f} mV")
    print(f"E_L,+ = {E_L_plus:.6f} mV")
    print()
    print(f"Residual negative branch: {f(u_minus, R=R, z = z):.3e}")
    print(f"Residual positive branch: {f(u_plus, R=R, z=z):.3e}")

    return E_L_minus, E_L_plus

class ComputeMaxReversalForBistability(unittest.TestCase):
    """Runnables for iteration 21: NMDA spikes under EI balance."""

    def test_compute_E_L_B_008(self):
        compute_E_L(A=0.25, B=0.08, E_E=0.0)

    def test_compute_E_L_B_062(self):
        compute_E_L(A=0.25, B=0.062, E_E=0.0)

    def test_plot_E_L_against_Mg2_concentration(self):
        shade_alpha = 0.4  # Easily configurable

        # Approximate observed ranges from Ding et al. (2016), Fig. 4A–B
        # https://pmc.ncbi.nlm.nih.gov/articles/PMC5441687/pdf/nihms857315.pdf
        awake_range_mM = (0.45, 1.05)
        sleep_range_mM = (0.55, 1.20)

        C_s = np.linspace(0.4, 2, num=100)
        E_L_s_008 = [compute_E_L(C / 3.57, B=0.08)[0] for C in C_s]
        E_L_s_0062 = [compute_E_L(C / 3.57, B=0.062)[0] for C in C_s]

        fig, ax = plt.subplots()

        ax.axvspan(
            *awake_range_mM,
            color="gold",
            alpha=shade_alpha,
            linewidth=0,
            label="Awake: approx. plotted range",
            zorder=0,
        )
        ax.axvspan(
            *sleep_range_mM,
            color="cornflowerblue",
            alpha=shade_alpha,
            linewidth=0,
            label="Sleep: approx. plotted range",
            zorder=0,
        )

        ax.plot(C_s, E_L_s_008, label="B=0.08", lw=2)
        ax.plot(C_s, E_L_s_0062, label="B=0.062", lw=2)

        ax.set_xlabel(r"Extracellular $[\mathrm{Mg}^{2+}]_o$ (mM)")
        ax.set_ylabel(r"$E_{L}^{*}$ (mV)")
        ax.set_title(r"Upper Bound on $E_L$ for NMDA-Mediated Bistability")
        ax.set_xlim(C_s[0], C_s[-1])
        ax.legend()

        show_plots_non_blocking()

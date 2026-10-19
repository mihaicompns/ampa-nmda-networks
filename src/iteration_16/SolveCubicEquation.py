import sys
import unittest

import matplotlib
import numpy as np
import sympy as sp
from brian2 import mV, have_same_dimensions, nS, Hz, ms, farad, nF, second, is_dimensionless, Mohm, Quantity
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

from Plotting import show_plots_non_blocking, prepare_bigger_fonts
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import rate_LIF_whitenoise
from iteration_16.Preliminaries import compute_ampa_dv, compute_gaba_dv
from iteration_16.model import config_with_weak_synapses, ConductanceDiffusionSimulationConfig, chapter1Results, \
    config_with_medium_synapses, config_with_intermediate_synapses, wang_config_recurrent_synapses, \
    wang_config_external_ampa_synapses
from iteration_16.simpy import load_solutions, RichardsonSympyEquations, bind_config_to_sympy_values

logger.remove()  # remove default handler

logger.add(
    sys.stderr,
    level="DEBUG",
    format="{time:HH:mm:ss.SSS} | {level} | {message}",
    backtrace=True,
    diagnose=True
)

logger.add(
    "ui_debug.log",
    level="DEBUG",
    rotation="5 MB",
    retention="3 days",
    backtrace=True,
    diagnose=True
)


def polynomial_r(r, cfg: ConductanceDiffusionSimulationConfig, gamma, mu_v: Quantity, sigma_v: Quantity):
    if is_dimensionless(r):
        r = r * Hz

    """
    Returns P(x) where x = g r
    """

    g = cfg.g()
    x = g * r

    gL = cfg.g_L
    C = cfg.membrane_capacitance

    # synaptic weights
    K_e = cfg.w_ampa * cfg.tau_ampa * (cfg.e_ampa - mu_v) ** 2
    K_i = cfg.w_gaba * cfg.tau_gaba * (cfg.e_gaba - mu_v) ** 2

    assert have_same_dimensions(K_e, 1 * farad * mV ** 2)
    assert have_same_dimensions(K_i, 1 * farad * mV ** 2)

    # shorthand
    te = cfg.tau_ampa
    ti = cfg.tau_gaba

    # coefficients
    a3 = 2 * sigma_v ** 2 * te * ti * (1 + gamma) ** 3

    assert have_same_dimensions(a3, 1 * ms ** 2 * mV ** 2)

    a2 = (1 + gamma) * (
            2 * sigma_v ** 2 * (1 + gamma) * (3 * te * ti * gL + C * (te + ti)) - K_e * ti - gamma * K_i * te
    )
    assert have_same_dimensions(a2, 1 * nS * ms ** 2 * mV ** 2)

    a1 = (
            2 * sigma_v ** 2 * (1 + gamma) * (3 * te * ti * gL ** 2 + 2 * C * (te + ti) * gL + C ** 2) -
            K_e * (ti * gL + C) - gamma * K_i * (te * gL + C)
    )
    assert have_same_dimensions(a1, 1 * nS ** 2 * ms ** 2 * mV ** 2)

    a0 = (
            2 * sigma_v ** 2
            * (
                    te * ti * gL ** 3
                    + C * (te + ti) * gL ** 2
                    + C ** 2 * gL
            )
    )
    assert have_same_dimensions(a0, 1 * nS ** 3 * ms ** 2 * mV ** 2)

    result = a3 * x ** 3 + a2 * x ** 2 + a1 * x + a0
    assert have_same_dimensions(result[0], 1 * nS ** 3 * ms ** 2 * mV ** 2)

    coeffs = np.array([
        a3 / (ms ** 2 * mV ** 2),
        a2 / (nS * ms ** 2 * mV ** 2),
        a1 / (nS ** 2 * ms ** 2 * mV ** 2),
        a0 / (nS ** 3 * ms ** 2 * mV ** 2),
    ], dtype=float)

    roots = np.roots(coeffs) * nS

    for root in roots:
        residual = a3 * root ** 3 + a2 * root ** 2 + a1 * root ** 1 + a0
        scale = (
                abs(a3 * root ** 3)
                + abs(a2 * root ** 2)
                + abs(a1 * root)
                + abs(a0)
        )

        print(residual / scale)

    assert have_same_dimensions(roots[0], 1 * nS)
    # assert have_same_dimensions(roots[0], 1)
    return result / (nS ** 3 * ms ** 2 * mV ** 2), roots


def E_0(r, cfg: ConductanceDiffusionSimulationConfig, gamma):
    if is_dimensionless(r):
        r = r * Hz

    gL = cfg.g_L
    EL = cfg.e_L
    Ee = cfg.e_ampa
    Ei = cfg.e_gaba

    g0, ge, gi = comp_mean_g_s(cfg, gamma, r)

    num = gL * EL + ge * Ee + gi * Ei

    return num / g0


def sigmoid_v(vm, mg_concentration=1):
    return 1 / (1 + np.exp(-0.062 * vm / mV) * (mg_concentration / 3.57))

def E_0_with_full_nmda_activation(r, cfg: ConductanceDiffusionSimulationConfig, gamma, k=2):
    if is_dimensionless(r):
        r = r * Hz

    gL = cfg.g_L
    EL = cfg.e_L
    Ee = cfg.e_ampa
    Ei = cfg.e_gaba

    g0, ge, gi = comp_mean_g_s(cfg, gamma, r)

    num = gL * EL + ge * Ee + gi * Ei

    initial_e_0 = num / g0
    g_nmda_est = k * cfg.get_g_nmda_max() * sigmoid_v(initial_e_0)

    num_with_nmda = gL * EL + ge * Ee + gi * Ei + g_nmda_est * Ee
    return  num_with_nmda / (g0 + g_nmda_est)


def comp_mean_g_s(cfg, gamma, r):
    gL = cfg.g_L

    g = cfg.g()
    ge = g * r
    gi = gamma * g * r
    g0 = gL + ge + gi
    return g0, ge, gi


def sigma_sq(r, cfg: ConductanceDiffusionSimulationConfig, gamma, k=0):
    if is_dimensionless(r):
        r = r * Hz

    Ee = cfg.e_ampa
    Ei = cfg.e_gaba
    C = cfg.membrane_capacitance

    g0, ge, gi = comp_mean_g_s(cfg, gamma, r)
    assert have_same_dimensions(g0, 1 * nS)

    tau_0 = C / g0
    assert have_same_dimensions(tau_0, 1 * ms)

    sigma_e_sq = 0.5 * cfg.w_ampa * ge
    sigma_i_sq = 0.5 * cfg.w_gaba * gi
    assert have_same_dimensions(sigma_e_sq, 1 * nS ** 2)
    assert have_same_dimensions(sigma_i_sq, 1 * nS ** 2)

    E_0_comp = E_0(r, cfg, gamma)

    g_nmda_est = k * cfg.get_g_nmda_max() * sigmoid_v(E_0_comp)
    g0 = g0 + g_nmda_est

    sigma_term_e = sigma_e_sq / (g0 ** 2) * (Ee - E_0_comp) ** 2 * cfg.tau_ampa / (cfg.tau_ampa + tau_0)
    sigma_term_i = sigma_i_sq / (g0 ** 2) * (Ei - E_0_comp) ** 2 * cfg.tau_gaba / (cfg.tau_gaba + tau_0)

    assert is_dimensionless(cfg.tau_gaba / (cfg.tau_ampa + C / g0))

    if isinstance(r, list):
        assert have_same_dimensions(ge[1], 1 * nS)
        assert have_same_dimensions(gi[1], 1 * nS)
        assert have_same_dimensions(E_0_comp[1], 1 * mV)
        assert have_same_dimensions(sigma_term_e[1], 1 * mV ** 2)
        assert have_same_dimensions(sigma_term_i[1], 1 * mV ** 2)

    sigma_v_squared = sigma_term_e + sigma_term_i
    return sigma_v_squared


def plot_poly(gamma=0.5, sigma_v=2 * mV, cfg: ConductanceDiffusionSimulationConfig = config_with_weak_synapses):
    r = np.linspace(0, 50000, 1000) * Hz
    g = cfg.k * cfg.w_gaba * cfg.tau_gaba * cfg.N_I
    x = gamma * g * r

    y = polynomial_r(x, cfg, gamma, sigma_v)

    plt.figure(figsize=(6, 4))
    plt.plot(r, y)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("r (Hz)")
    plt.ylabel("P(r)")
    plt.title("Cubic intersection landscape")
    plt.show()


def evaluate_sympy_solution(solution, config: ConductanceDiffusionSimulationConfig, mu_v_target: Quantity = chapter1Results.mu_v,
                            sigma_target: Quantity = chapter1Results.sigma_v):
    from sympy.abc import x, y

    gL, Ee, Ei, EL, E_target = sp.symbols('gL Ee Ei EL E_target')
    g0 = gL + x * (1 + y)

    E0 = (gL * EL + x * Ee + y * x * Ei) / g0
    equation_e_0 = E0 - E_target

    y_expr = sp.solve(equation_e_0, y)[0]

    values = {
        'gL': float(config.g_L / nS),
        "EL": float(config.e_L / mV),
        "Ee": float(config.e_ampa / mV),
        "Ei": float(config.e_gaba / mV),
        "we": float(config.w_ampa / nS),
        "wi": float(config.w_gaba / nS),
        "taue": float(config.tau_ampa / second),
        "taui": float(config.tau_gaba / second),
        "C": float(config.membrane_capacitance / nF),
        'E_target': float(mu_v_target / mV),
        'sigma_target': float(sigma_target / mV)
    }
    x_val = solution.subs(values).evalf()
    if abs(sp.im(x_val)) < 1e-10:
        x_val = sp.re(x_val)
    else:
        print("Imaginary value of our solution: ", sp.im(x_val))
    y_val = y_expr.subs(x, x_val).subs(values).evalf()
    return float(x_val), float(y_val)


def read_solutions(config: ConductanceDiffusionSimulationConfig, solutions_file_name="solution.txt"):
    sols = []

    with open(solutions_file_name) as f:
        solutions = sp.sympify(f.read())
        for index, solution in enumerate(solutions):
            x_sol, y_sol = evaluate_sympy_solution(solution, config, mu_v_target=chapter1Results.mu_v,
                                                   sigma_target=chapter1Results.sigma_v)
            rate_sol = x_sol / (config.g() / nS)

            if rate_sol > 0 and y_sol > 0:
                print(rate_sol, y_sol)
                sols.append((rate_sol, y_sol))

    return sols


def compute_siegert_firing_rate_for_fitted_solution(config: ConductanceDiffusionSimulationConfig, gamma: Quantity, r: np.ndarray, means_vm: np.ndarray, vars_vm: np.ndarray):
    g0s, _, _ = comp_mean_g_s(config, gamma, r)
    taus_0 = config.membrane_capacitance / g0s

    result = np.zeros_like(r)
    for index, tau_0, mean, variance in zip(range(0, len(r)), taus_0, means_vm, vars_vm):
        # def rate_LIF_whitenoise(mu, tau_membrane, sigma_v, theta, V_reset, tau_ref):
        result[index] = rate_LIF_whitenoise(mu=mean, tau_membrane=tau_0, sigma_v=np.sqrt(variance), theta=config.theta, V_reset=config.v_reset, tau_ref=2 * ms)

    return result


def plot_e_0_and_sigma_sq_for_increasing_rate(config: ConductanceDiffusionSimulationConfig, sols,
                                              plot_title="Verify cubic solutions", r_max=100):
    prepare_bigger_fonts()

    r = np.linspace(0, r_max, 3000) * Hz
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14, 17),
        sharex=True,
    )
    colors = ["tab:blue", "tab:orange"]
    for (index, solution), color in zip(enumerate(sols[::-1]), colors):
        r_sol, gamma = solution
        mean_vm = E_0(r, config, gamma)
        mean_k2 =  E_0_with_full_nmda_activation(r, config, gamma, k=2)
        mean_k5 =  E_0_with_full_nmda_activation(r, config, gamma, k=5)
        mean_k10 =  E_0_with_full_nmda_activation(r, config, gamma, k=10)
        line, = ax1.plot(r, mean_vm / mV, label=r"$\gamma=$"f"{gamma:.2f}", color=color)

        ax1.plot(r, mean_k2/ mV, label=r"$\gamma=$"f"{gamma:.2f}, k=2", color=color, linestyle=":")
        ax1.plot(r, mean_k5 / mV, label=r"$\gamma=$"f"{gamma:.2f}, k=5", color=color, linestyle="-.")
        ax1.plot(r, mean_k10 / mV, label=r"$\gamma=$"f"{gamma:.2f}, k=10", color=color, linestyle="--")

        var_vm = sigma_sq(r, config, gamma)
        var_k2 = sigma_sq(r, config, gamma, k=2)
        var_k5 = sigma_sq(r, config, gamma, k=5)
        var_k10 = sigma_sq(r, config, gamma, k=10)
        ax2.plot(r, var_vm / mV ** 2, label=r"$\gamma=$"f"{gamma:.2f}", color=color)
        ax2.plot(r, var_k2 / mV ** 2, label=r"$\gamma=$"f"{gamma:.2f}, k=2", color=color, linestyle=":")
        ax2.plot(r, var_k5 / mV ** 2, label=r"$\gamma=$"f"{gamma:.2f}, k=5", color=color, linestyle="-.")
        ax2.plot(r, var_k10 / mV ** 2, label=r"$\gamma=$"f"{gamma:.2f}, k=10", color=color, linestyle="--")

        ax2.axhline((chapter1Results.sigma_v / mV) ** 2, color="black", lw=1, linestyle="--")

        ax3.plot(r, compute_siegert_firing_rate_for_fitted_solution(config, gamma, r, means_vm=mean_vm, vars_vm=var_vm), label=r"$\gamma=$"f"{gamma:.2f}", color=color)
        ax3.plot(r, compute_siegert_firing_rate_for_fitted_solution(config, gamma, r, means_vm=mean_k2, vars_vm=var_k2), label=r"$\gamma=$"f"{gamma:.2f}, k=2", linestyle=":", color=color)
        ax3.plot(r, compute_siegert_firing_rate_for_fitted_solution(config, gamma, r, means_vm=mean_k5, vars_vm=var_k5), label=r"$\gamma=$"f"{gamma:.2f}, k=5", linestyle="-.", color=color)
        ax3.plot(r, compute_siegert_firing_rate_for_fitted_solution(config, gamma, r, means_vm=mean_k10, vars_vm=var_k10), label=r"$\gamma=$"f"{gamma:.2f}, k=10", linestyle="--", color=color)

        print(f"E_0 of solution: {E_0(r_sol, config, gamma)}, sigma v of solution {sigma_sq(r_sol, config, gamma)}")
        ax1.axvline(x=r_sol / Hz, linestyle="--", label=f"Sol {index + 1}", color=line.get_color())
        ax2.axvline(x=r_sol / Hz, linestyle="--", label=f"Sol {index + 1}", color=line.get_color())
        ax3.axvline(x=r_sol / Hz, linestyle="--", label=f"Sol {index + 1}", color=line.get_color())

    ax1.axhline(chapter1Results.mu_v / mV, color="black", lw=1, linestyle="--")
    ax1.set_title(r"$E_0(r)$")
    ax1.set_ylabel("mV")
    ax2.set_title(r"$\sigma_v^2(r)$")
    ax2.set_ylabel(r"$mV^2$")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax3.set_xlabel(r"rate (Hz)")

    fig.suptitle(plot_title)
    fig.tight_layout()
    show_plots_non_blocking()


def plot_e_0_and_sigma_sq_for_increasing_rate_no_solution(config: ConductanceDiffusionSimulationConfig,
                                              plot_title="Numerical values of recurrent Wang config cannot produce a solution"):
    prepare_bigger_fonts()

    r = np.linspace(0, 300, 3000) * Hz
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 12),
        sharex=True,
    )

    for gamma in [0, 0.5, 1, 1.5, 2]:

        ax1.plot(r, E_0(r, config, gamma) / mV, label=r"$\gamma=$"f"{gamma:.2f}")

        ax2.plot(r, sigma_sq(r, config, gamma) / mV ** 2, label=r"$\gamma=$"f"{gamma:.2f}")
    ax2.axhline((chapter1Results.sigma_v / mV) ** 2, color="black", lw=1, linestyle="--")

    ax1.axhline(chapter1Results.mu_v / mV, color="black", lw=1, linestyle="--")
    ax1.set_title(r"$E_0(r)$")
    ax1.set_ylabel("mV")
    ax2.set_title(r"$\sigma_v^2(r)$")
    ax2.set_ylabel(r"$mV^2$")
    ax2.set_xlabel(r"rate (Hz)")
    ax1.legend()
    ax2.legend()
    fig.suptitle(plot_title)
    fig.tight_layout()
    show_plots_non_blocking()


class TripleExplorer:

    def __init__(self, cfg):

        self.cfg = cfg
        # parameters
        self.gamma = 0.5

        self.r = np.linspace(0, 2000, 2000) * Hz

        self.mu_v = - 47.61595645 * mV
        self.sigma_v = 1.90531046 * mV

        self.fig = plt.figure(figsize=(12, 12))

        graphs = 3
        sliders = 7

        gs = GridSpec(
            graphs + sliders, 1,
            height_ratios=[3] * graphs + [0.3] * sliders
        )

        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax3 = self.fig.add_subplot(gs[2])

        ax_gamma = self.fig.add_subplot(gs[graphs + 1])
        ax_C = self.fig.add_subplot(gs[graphs + 2])
        ax_g_L = self.fig.add_subplot(gs[graphs + 3])

        ax_we = self.fig.add_subplot(gs[graphs + 4])
        ax_wi = self.fig.add_subplot(gs[graphs + 5])

        ax_r_max = self.fig.add_subplot(gs[graphs + 6])

        # lines
        self.line_E, = self.ax1.plot([], [], lw=2)
        self.line_S, = self.ax2.plot([], [], lw=2)
        self.line_P, = self.ax3.plot([], [], lw=2)

        self.root_scatter = self.ax3.scatter([], [], color="red", zorder=10)

        self.ax1.set_title(r"$E_0(r, \gamma)$")
        self.ax2.set_title(r"$\sigma_v(r, \gamma)$")
        self.ax3.set_title("P(gr) with zero crossings")
        self.ax3.axhline(0, color="black", lw=1)

        self.ax1.set_ylabel("mV")
        self.ax2.set_ylabel(r"$mV^2$")
        self.ax3.set_ylabel("P(x)")
        self.ax3.set_xlabel("r (Hz)")

        self.ax1.axhline(self.mu_v / mV, color="black", lw=1, linestyle="--")
        self.ax2.axhline((self.sigma_v / mV) ** 2, color="black", lw=1, linestyle="--")
        self.ax3.axhline(0, color="black", lw=1, linestyle="--")

        self.s_gamma = Slider(ax_gamma, "γ", 0.1, 2.0, valinit=0.5)
        self.s_C = Slider(ax_C, "C", 0.1, valinit=0.5, valmax=10, valstep=0.1)
        self.s_g_L = Slider(ax_g_L, "$g_L$ (nS)", 5, valinit=20, valmax=100, valstep=1)
        self.s_we = Slider(ax_we, r"$w_e$ (nS)", 0.01, valinit=0.5, valmax=10)
        self.s_wi = Slider(ax_wi, r"$w_i$ (nS)", 0.01, valinit=0.5, valmax=10)
        self.s_r_max = Slider(ax_r_max, r"$r_\mathrm{max}$", valmin=0, valinit=1000, valstep=50, valmax=2000)

        self.fig_title = self.fig.suptitle(self.get_title())

        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.85)

        for s in [self.s_gamma, self.s_C, self.s_g_L, self.s_we, self.s_wi, self.s_r_max]:
            s.on_changed(self.update)

        self.update(None)

    def E0(self, r):
        return E_0(r, self.cfg, self.gamma)

    def sigma2(self, r):
        return sigma_sq(r, self.cfg, self.gamma)

    def P(self, r):
        return polynomial_r(r, self.cfg, self.gamma, self.sigma_v)

    def update(self, val):
        try:
            logger.debug("update triggered")
            self.gamma = self.s_gamma.val

            new_C = self.s_C.val * nF
            new_g_L = self.s_g_L.val * nS
            new_w_e = self.s_we.val * nS
            new_w_i = self.s_wi.val * nS

            r_max_new = self.s_r_max.val
            self.r = np.linspace(0, r_max_new, 2000) * Hz

            self.cfg = self.cfg.with_property(membrane_capacitance=new_C, w_ampa=new_w_e, w_gaba=new_w_i, g_L=new_g_L)

            E_vals = self.E0(self.r) / mV
            S_vals = self.sigma2(self.r) / (mV ** 2)
            P_vals = self.P(self.r)

            # plot 1: E0
            self.line_E.set_data(self.r, E_vals)

            self.ax1.relim()
            self.ax1.autoscale_view()

            # plot 2: sigma
            self.line_S.set_data(self.r, S_vals)
            self.ax2.relim()
            self.ax2.autoscale_view()

            # plot 3: polynomial
            self.line_P.set_data(self.r, P_vals)

            # zero crossings
            sign_changes = np.where(np.diff(np.sign(P_vals)))[0]
            roots_r = self.r[sign_changes]

            [coll.remove() for coll in list(self.ax3.collections)]
            self.ax3.scatter(roots_r, np.zeros_like(roots_r), color="red")

            self.ax3.relim()
            self.ax3.autoscale_view()

            self.fig_title.set_text(self.get_title())
            self.fig.canvas.draw_idle()
        except Exception as e:
            logger.exception("Exception inside update()", e)

    def get_title(self):
        return (
            "Search for parameters such that"r"$\mu_v=$"f"{self.mu_v / mV :.2f}, "r"$\sigma_v=$"f"{self.sigma_v / mV :.2f} \n"
            f"{cubic_solution_title(self.cfg)}")


def cubic_solution_title(cfg: ConductanceDiffusionSimulationConfig):
    return "Our model:" r"$R_{\mathrm{in}}=$" f"{(1 / cfg.g_L) / Mohm : .2f} MΩ, " r"$N_E$="f"{cfg.N_E} "r"$N_I$="f"{cfg.N_I}\n"

def plot_one_cubic_solution(solutions: list, config: ConductanceDiffusionSimulationConfig, r_max=10_000,
                            title="Check cubic solutions"):
    valid_solutions = []

    for index, solution in enumerate(solutions):
        x_sol, y_sol = evaluate_sympy_solution(solution, config, mu_v_target=chapter1Results.mu_v,
                                               sigma_target=chapter1Results.sigma_v)
        rate_sol = x_sol / (config.g() / nS)

        if rate_sol > 0 and y_sol > 0:
            print(rate_sol, y_sol)
            valid_solutions.append((rate_sol, y_sol))

    r = np.linspace(0.1, 100, r_max) * Hz

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    for index, solution in enumerate(valid_solutions):
        r_sol, gamma = solution
        ax1.plot(r, E_0(r, config, gamma) / mV, label=r"$\gamma=$"f"{gamma:.2f}")

        ax2.plot(r, sigma_sq(r, config, gamma) / mV ** 2, label=r"$\gamma=$"f"{gamma:.2f}")
        ax2.axhline((chapter1Results.sigma_v / mV) ** 2, color="black", lw=1, linestyle="--")

        print(f"E_0 of solution: {E_0(r_sol, config, gamma)}, sigma v of solution {sigma_sq(r_sol, config, gamma)}")
        ax1.axvline(x=r_sol / Hz, linestyle="--", label=f"Sol {index + 1}")
        ax2.axvline(x=r_sol / Hz, linestyle="--", label=f"Sol {index + 1}")

    ax1.axhline(chapter1Results.mu_v / mV, color="black", lw=1, linestyle="--")
    ax1.set_title(r"$E_0(r)$")
    ax1.set_ylabel("mV")

    ax2.set_title(r"$\sigma_v^2(r)$")
    ax2.set_ylabel(r"$mV^2$")

    ax2.set_xlabel(r"rate (Hz)")

    ax1.legend()
    ax2.legend()

    if title is None:
        title = "Solutions of cubic polynomial equation"
    fig.suptitle(f"{title} \n {cubic_solution_title(config)}")
    show_plots_non_blocking()


class MyTestCase(unittest.TestCase):

    def test_plot_e_0_sigma_0_p_of_x(self):
        cfg = config_with_weak_synapses.with_property(N_E=1000)

        gamma = 0.5
        sigma_v = 1.9 * mV

        r = np.linspace(0.1, 110, 1000) * Hz

        # x = (cfg.g() / (nS * second)) * r
        # x = (cfg.g() / (nS * second)) * r

        P_vals, roots = polynomial_r(r, cfg, gamma, mu_v=chapter1Results.mu_v, sigma_v=chapter1Results.sigma_v)

        a3 = (2 * sigma_v ** 2 * cfg.tau_ampa * cfg.tau_gaba * (1 + gamma) ** 3) / (ms ** 2 * mV ** 2)
        assert is_dimensionless(a3)

        x = cfg.g() * r

        print('XXXX')
        print(roots / cfg.g(), "Produce sigma sq ", sigma_sq(roots / cfg.g(), cfg, gamma))
        polyn_from_roots = a3 * (x - roots[0]) * (x - roots[1]) * (x - roots[2]) / (nS ** 3)

        E_vals = E_0(r, cfg, gamma)
        sigma_vals = sigma_sq(r, cfg, gamma)
        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(10, 10),
            sharex=True,
        )

        ax1.plot(r, E_vals / mV)

        ax1.set_title(r"$E_0(r)$")
        ax1.set_ylabel("mV")
        ax1.axhline(chapter1Results.mu_v / mV, color="black", lw=1, linestyle="--")

        ax2.plot(r, sigma_vals / mV ** 2)
        ax2.axhline((chapter1Results.sigma_v / mV) ** 2, color="black", lw=1, linestyle="--")

        ax2.set_title(r"$\sigma_v^2(r)$")
        ax2.set_ylabel(r"$mV^2$")
        # ax2.set_ylim(-1, 20)

        ax3.plot(r, P_vals, label="equation")
        ax3.plot(r, polyn_from_roots, label="poynomial from roots", lw=2, color="red", linestyle="--")

        ax3.axhline(
            0,
            color="black",
            lw=1,
        )

        ax3.set_title(r"$P(x)$")
        ax3.set_xlabel("r (Hz)")

        fig.suptitle(cubic_solution_title(cfg))

        plt.tight_layout()
        plt.show()

        g_unitless = cfg.g() / (nS * second)
        print(sigma_sq(roots / cfg.g(), cfg, gamma) / mV ** 2)
        # snp.testing.assert_allclose(sigma_v**2, sigma_sq(roots / cfg.g(), cfg, gamma))

    def test_run_ui(self):
        matplotlib.use("QtAgg")
        cfg = config_with_weak_synapses.with_property(N_E=800, N_I=200)
        ui = TripleExplorer(cfg)
        plt.tight_layout()
        plt.show()

    def test_check_E_0_units_match_brian2(self):
        cfg = config_with_weak_synapses
        gamma = 1

        self.assertEqual(-65 * mV, E_0(0 * Hz, cfg, gamma))
        self.assertEqual(-40 * mV, E_0(10 ** 100 * Hz, cfg, gamma))

        r = np.linspace(0, 1000, 101) * Hz

        e_0_np = E_0(r, cfg, gamma)
        test_compare = [None] * len(r)
        g = config_with_weak_synapses.g()
        one_rate = r[3]
        one_rate_no_unit = one_rate / Hz

        one_rate_computed_with_brian2_units = (g * one_rate) / nS
        one_rate_computed_unitless = (g / (nS * second)) * one_rate_no_unit
        self.assertAlmostEqual(one_rate_computed_with_brian2_units, one_rate_computed_unitless)

        for index, rate in enumerate(r):
            g = config_with_weak_synapses.g()
            assert have_same_dimensions(g, 1 * nS * ms)

            g_e0 = g * rate
            g_i0 = gamma * g * rate
            assert have_same_dimensions(g_e0, 1 * nS)
            assert have_same_dimensions(g_i0, 1 * nS)

            g_0 = cfg.g_L + g_e0 + g_i0
            current_e_0 = (cfg.g_L * cfg.e_L + g_e0 * cfg.e_ampa + g_i0 * cfg.e_gaba) / g_0
            assert have_same_dimensions(current_e_0, 1 * mV)
            test_compare[index] = current_e_0 / mV

        plt.plot(r / Hz, test_compare, label="Brian 2 units", alpha=0.6)
        plt.plot(r / Hz, e_0_np / mV, label="no units", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

        np.testing.assert_array_almost_equal(e_0_np / mV, test_compare)

    def test_check_sigma_v_sq_units_match_brian2(self):
        cfg = config_with_weak_synapses
        gamma = 1

        sq = sigma_sq([0, 0] * Hz, cfg, gamma) / mV ** 2
        np.testing.assert_array_almost_equal(sq, [0, 0])

        r = np.linspace(0, 1000, 101) * Hz

        one_rate = r[10]
        one_rate_no_unit = one_rate / Hz

        g = cfg.g()
        g_e0 = g * one_rate
        g_i0 = gamma * g * one_rate
        g_0 = cfg.g_L + g_e0 + g_i0
        assert have_same_dimensions(g_e0, 1 * nS)
        assert have_same_dimensions(g_i0, 1 * nS)
        assert have_same_dimensions(g_0, 1 * nS)

        current_e_0 = (cfg.g_L * cfg.e_L + g_e0 * cfg.e_ampa + g_i0 * cfg.e_gaba) / g_0
        tau_0 = cfg.membrane_capacitance / g_0

        assert have_same_dimensions(g_0, 1 * nS)
        assert have_same_dimensions(current_e_0, 1 * mV)
        assert have_same_dimensions(tau_0, 1 * ms)

        sigma_e_sq = 0.5 * cfg.w_ampa * g_e0
        sigma_i_sq = 0.5 * cfg.w_gaba * g_i0

        one_sigma_e_sq_with_units = sigma_e_sq / g_0 ** 2 * (cfg.e_ampa - current_e_0) ** 2 * (
                cfg.tau_ampa / (cfg.tau_ampa + tau_0))
        one_sigma_i_sq_with_units = sigma_i_sq / g_0 ** 2 * (cfg.e_gaba - current_e_0) ** 2 * (
                cfg.tau_gaba / (cfg.tau_gaba + tau_0))

        sigma_sq_test = one_sigma_e_sq_with_units + one_sigma_i_sq_with_units

        assert have_same_dimensions(sigma_sq_test, 1 * mV ** 2)

        g_e_0_no_units = (g / (second * nS)) * one_rate_no_unit
        g_i_0_no_units = gamma * (g / (second * nS)) * one_rate_no_unit
        g_0_no_units = cfg.g_L / nS + g_e_0_no_units + g_i_0_no_units

        assert is_dimensionless(g_e_0_no_units)
        assert is_dimensionless(g_i_0_no_units)
        assert is_dimensionless(g_0_no_units)

        self.assertAlmostEqual(g_0_no_units, g_0 / nS)

        tau_0_no_units = (cfg.membrane_capacitance / nF) / g_0_no_units * 1000  # because nF / nS returns 1 s
        tau_ampa_no_units = cfg.tau_ampa / ms
        tau_gaba_no_units = cfg.tau_gaba / ms

        self.assertAlmostEqual(tau_0_no_units, tau_0 / ms)

        sigma_e_sq_no_units = 0.5 * (cfg.w_ampa / nS) * g_e_0_no_units
        sigma_i_sq_no_units = 0.5 * (cfg.w_gaba / nS) * g_i_0_no_units

        E_e_no_units = cfg.e_ampa / mV
        E_i_no_units = cfg.e_gaba / mV
        E_L_no_units = cfg.e_L / mV
        E_0_no_units = ((
                                cfg.g_L / nS) * E_L_no_units + g_e_0_no_units * E_e_no_units + g_i_0_no_units * E_i_no_units) / g_0_no_units

        assert is_dimensionless(E_e_no_units)
        assert is_dimensionless(E_i_no_units)
        assert is_dimensionless(E_L_no_units)
        assert is_dimensionless(E_0_no_units)

        one_sigma_v_e_sq_no_units = sigma_e_sq_no_units / g_0_no_units ** 2 * (E_0_no_units - E_e_no_units) ** 2 * (
                tau_ampa_no_units / (tau_ampa_no_units + tau_0_no_units))
        one_sigma_v_i_sq_no_units = sigma_i_sq_no_units / g_0_no_units ** 2 * (E_0_no_units - E_i_no_units) ** 2 * (
                tau_gaba_no_units / (tau_gaba_no_units + tau_0_no_units))

        assert is_dimensionless(one_sigma_v_e_sq_no_units)
        assert is_dimensionless(one_sigma_v_i_sq_no_units)

        one_sigma_v_sq_no_units = one_sigma_v_e_sq_no_units + one_sigma_v_i_sq_no_units

        self.assertAlmostEqual(one_sigma_v_sq_no_units, sigma_sq_test / mV ** 2)

        test_compare = [None] * len(r)
        for index, rate in enumerate(r):
            g = config_with_weak_synapses.g()
            g_e0 = g * rate
            g_i0 = gamma * g * rate
            g_0 = cfg.g_L + g_e0 + g_i0
            assert have_same_dimensions(g_e0, 1 * nS)
            assert have_same_dimensions(g_i0, 1 * nS)
            assert have_same_dimensions(g_0, 1 * nS)

            current_e_0 = (cfg.g_L * cfg.e_L + g_e0 * cfg.e_ampa + g_i0 * cfg.e_gaba) / g_0
            tau_0 = cfg.membrane_capacitance / g_0

            assert have_same_dimensions(g_0, 1 * nS)
            assert have_same_dimensions(current_e_0, 1 * mV)
            assert have_same_dimensions(tau_0, 1 * ms)

            sigma_e_sq = 0.5 * cfg.w_ampa * g_e0
            sigma_i_sq = 0.5 * cfg.w_gaba * g_i0

            one_sigma_e_sq_with_units = sigma_e_sq / g_0 ** 2 * (cfg.e_ampa - current_e_0) ** 2 * (
                    cfg.tau_ampa / (cfg.tau_ampa + tau_0))
            one_sigma_i_sq_with_units = sigma_i_sq / g_0 ** 2 * (cfg.e_gaba - current_e_0) ** 2 * (
                    cfg.tau_gaba / (cfg.tau_gaba + tau_0))

            sigma_sq_test = one_sigma_e_sq_with_units + one_sigma_i_sq_with_units
            test_compare[index] = sigma_sq_test / mV ** 2

        sigma_np_result = sigma_sq(r, cfg, gamma)
        plt.plot(r / Hz, test_compare, label="Test computation")
        plt.plot(r / Hz, sigma_np_result / (mV ** 2), label="method")
        plt.legend()
        plt.tight_layout()
        plt.show()
        np.testing.assert_allclose(test_compare, sigma_np_result / mV ** 2)

    def test_config_with_large_N_E(self):
        with self.assertRaises(ValueError):
            config_with_weak_synapses.with_property(N_E=1000, k=0.5)

        object_under_test_1 = config_with_weak_synapses.with_property(N_E=1000, N_I=1000)

        self.assertEqual(1000, object_under_test_1.N_E)
        self.assertEqual(1000, object_under_test_1.N_I)
        self.assertEqual(2000, object_under_test_1.N)
        self.assertEqual(1, object_under_test_1.k)

        object_under_test_2 = config_with_weak_synapses.with_property(N=1000)
        self.assertEqual(1000, object_under_test_2.N)
        self.assertEqual(1, object_under_test_2.k)
        self.assertEqual(500, object_under_test_2.N_E)
        self.assertEqual(500, object_under_test_2.N_I)

    def test_numerical_simpy_solution(self):
        gr = 367.327808891460
        gamma = 1.44112462343384

        def compute_solution(cfg: ConductanceDiffusionSimulationConfig):
            rate_sol = gr / (cfg.g() / (nS * second)) * Hz

            print(E_0(rate_sol, cfg, gamma))
            print(sigma_sq(rate_sol, cfg, gamma))
            print(rate_sol)
            print("==================")

        compute_solution(config_with_weak_synapses)
        compute_solution(config_with_weak_synapses.with_property(N=1000))

    def test_sympy_solutions_solve_e_0_and_sigma_sq_0(self):

        cfgs = [config_with_weak_synapses, config_with_weak_synapses.with_property(N=100),
                config_with_weak_synapses.with_property(N=1000), config_with_weak_synapses.with_property(N=10000),
                wang_config_recurrent_synapses.with_property(N=2000, k=4),
                wang_config_external_ampa_synapses.with_property(N=2000, k=4)]

        with open("solution.txt") as f:
            solutions = sp.sympify(f.read())
            for cfg in cfgs:
                for index, solution in enumerate(solutions):
                    # compare_two_impl(solution, cfg, mu_v_target=Chapter1Results.mu_v, sigma_target=Chapter1Results.sigma_v)

                    x_sol, y_sol = evaluate_sympy_solution(solution, cfg, mu_v_target=chapter1Results.mu_v,
                                                           sigma_target=chapter1Results.sigma_v)
                    rate_sol = x_sol / (cfg.g() / nS)

                    print("Rate sol: ", rate_sol)
                    ge_0 = cfg.g() * rate_sol
                    print(f"g_e, 0 = x is {ge_0}")
                    print(
                        f"Lets do E_0: {(- 65 * 20 + 0 * ge_0 / nS - y_sol * ge_0 / nS * 80) / (20 + ge_0 / nS + ge_0 / nS * y_sol)}")

                    self.assertAlmostEqual(chapter1Results.mu_v / mV, E_0(rate_sol, cfg, y_sol) / mV)
                    self.assertAlmostEqual((chapter1Results.sigma_v / mV) ** 2,
                                           sigma_sq(rate_sol, cfg, y_sol) / (mV ** 2))

    # This is already moving in the direction of scripts. Do not leave it here and move it somewhere where it makes sense.
    def test_plot_sympy_solutions(self):

        def plot_title(synapse_label, config: ConductanceDiffusionSimulationConfig):
            ampa_dv = compute_ampa_dv(config)
            gaba_dv = compute_gaba_dv(config)

            return (r"$E_0=f_1$(rate) and $\sigma_v^2=f_2$(rate)" "\n" "for appropriate $\gamma$ able to fit "
                    r"$\bar E_0=$"f"{chapter1Results.mu_v / mV :.2f}, "r"$\sigma_v^2=$"f"{(chapter1Results.sigma_v / mV) ** 2 :.2f} \n"
                    f"{synapse_label} synapses: ""\n"r"$w_{\mathrm{AMPA}} = $" f"{config.w_ampa / nS: .2f} (nS), " r"$\Delta v_{\mathrm{AMPA}} = $" f"{ampa_dv: .2f} mV, "
                    r"$w_{\mathrm{GABA}} = $" f"{config.w_gaba / nS: .2f} (nS), " r"$\Delta v_{\mathrm{GABA}} = $" f"{gaba_dv: .2f} mV")

        '''
        config_wang_external_synapses = wang_config_external_ampa_synapses
        solutions_wang_recurrent_synapses = read_solutions(config_wang_external_synapses)
        plot_e_0_and_sigma_sq_for_increasing_rate(config_wang_external_synapses, solutions_wang_recurrent_synapses,
                                                  plot_title=plot_title("wang external", config_wang_external_synapses), r_max=20)

        config_weak_synapses = config_with_weak_synapses.with_property(N_E=1000, N_I=1000)
        solutions_weak_synapses = read_solutions(config_weak_synapses)
        plot_e_0_and_sigma_sq_for_increasing_rate(config_weak_synapses, solutions_weak_synapses, plot_title = plot_title("weak", config_weak_synapses))

        config_weak_synapses = config_with_weak_synapses.with_property(N_E=1000, N_I=1000)
        solutions_weak_synapses = read_solutions(config_weak_synapses)[1:]
        plot_e_0_and_sigma_sq_for_increasing_rate(config_weak_synapses, solutions_weak_synapses,
                                                  plot_title=plot_title("weak", config_weak_synapses))
        '''
        config_intermediate_synapses = config_with_intermediate_synapses.with_property(N_E=1600, N_I=400)
        solutions_intermediate_synapses = read_solutions(config_intermediate_synapses)
        plot_e_0_and_sigma_sq_for_increasing_rate(config_intermediate_synapses, solutions_intermediate_synapses,
                                                  plot_title=plot_title("intermediate", config_intermediate_synapses))

        config_moderate_synapses = config_with_medium_synapses.with_property(N_E=1000, N_I=1000)
        solutions_moderate_synapses = read_solutions(config_moderate_synapses)
        plot_e_0_and_sigma_sq_for_increasing_rate(config_moderate_synapses, solutions_moderate_synapses,
                                                  plot_title = plot_title("moderate", config_moderate_synapses))


    def test_evaluate_wang_solutions_for_cubic_solutions(self):
        try:
            load_solutions(wang_config_recurrent_synapses, load_negative_values=True)
        except ValueError, TypeError:
            print("Wang solutions for recurrent synapses cannot reach our desired values")
        print(load_solutions(wang_config_external_ampa_synapses, load_negative_values=True))

    def test_plot_e_0_sigma_sq_wang_config(self):
        plot_e_0_and_sigma_sq_for_increasing_rate_no_solution(wang_config_recurrent_synapses)


if __name__ == '__main__':
    unittest.main()

import unittest

import sympy as sp
from brian2 import nS, mV, nF, second
from sympy import Reals
from sympy.physics.control.control_plots import plt
from sympy.physics.quantum.identitysearch import np

from src.Plotting import prepare_bigger_fonts, show_plots_non_blocking
from src.iteration_16.model import chapter1Results, ConductanceDiffusionSimulationConfig, config_with_weak_synapses, \
    config_with_medium_synapses, wang_config_recurrent_synapses, wang_config_external_ampa_synapses


def bind_config_to_sympy_values(cfg):
    values = {
        'gL': float(cfg.g_L / nS),
        "EL": float(cfg.e_L / mV),
        "Ee": float(cfg.e_ampa / mV),
        "Ei": float(cfg.e_gaba / mV),
        "we": float(cfg.w_ampa / nS),
        "wi": float(cfg.w_gaba / nS),
        "taue": float(cfg.tau_ampa / second),
        "taui": float(cfg.tau_gaba / second),
        "C": float(cfg.membrane_capacitance / nF),
        'sigma_target': float(chapter1Results.sigma_v / mV),
        'E_target': float(chapter1Results.mu_v / mV)
    }
    return values

solutions_file_name = "solutions/solutions.txt"
polynomial_file_name = "polynomial.txt"

def load_solutions(config: ConductanceDiffusionSimulationConfig, file_name: str = solutions_file_name, load_negative_values = False):
    eqs = RichardsonSympyEquations()

    with open(file_name) as f:
        solutions = sp.sympify(f.read())

        values = bind_config_to_sympy_values(config)

        result = []

        for index, solution in enumerate(solutions):
            from sympy.abc import x
            x_val = solution.subs(values).evalf()
            if abs(sp.im(x_val)) < 1e-10:
                x_val = sp.re(x_val)

            y_val = eqs.y_expr.subs(values).subs(x, x_val)
            x_float_value = float(x_val)
            y_float_value = float(y_val)
            if load_negative_values or (x_float_value > 0 and y_float_value > 0):
                result.append((float(x_val), float(y_val)))

    return result

class RichardsonSympyEquations:

    def __init__(self):
        from sympy.abc import x, y

        gL, Ee, Ei, EL, E_target, we, wi, taue, taui, C, sigma_target = sp.symbols('gL Ee Ei EL E_target we wi taue taui C sigma_target')

        self.def_g0 = gL + x * (1 + y)
        self.def_E0 = (gL * EL + x * Ee + y * x * Ei) / self.def_g0

        self.def_sigma_sq = (
                we / 2 * x / self.def_g0 ** 2 * (Ee - self.def_E0) ** 2 *
                taue / (taue + C / self.def_g0)
                +
                wi / 2 * y * x / self.def_g0 ** 2 * (Ei - self.def_E0) ** 2 *
                taui / (taui + C / self.def_g0)
        )

        self.eq_sigma = self.def_sigma_sq - sigma_target**2

        self.y_expr = sp.solve(self.def_E0 - E_target, y)[0]

        self.x_polynomial = None

    def x_polyn(self, config=None):

        self.__init_x_polyn__()

        if config is None:
            return self.x_polynomial

        return self.x_polynomial.subs(bind_config_to_sympy_values(config))

    def __init_x_polyn__(self):
        if self.x_polynomial is None:
            from sympy.abc import y
            y_expr = self.y_expr
            eq_x = sp.simplify(
                self.eq_sigma.subs(y, y_expr)
            )
            # Cancel common factors first
            eq_x = sp.cancel(eq_x)

            num, _ = sp.fraction(eq_x)
            self.x_polynomial = sp.factor(num)



class SimpyTest(unittest.TestCase):

    def test_system_elimination_1(self):
        cfg = config_with_weak_synapses

        x = sp.symbols('x')

        gL, Ee, Ei, EL = sp.symbols('gL Ee Ei EL')
        gamma = sp.symbols('gamma')
        we, wi = sp.symbols('we wi')
        taue, taui = sp.symbols('taue taui')
        C = sp.symbols('C')

        g0 = gL + x * (1 + gamma)

        E0 = (gL * EL + x * Ee + gamma * x * Ei) / g0

        sigma2 = (
                we / 2 * x / g0 ** 2 * (Ee - E0) ** 2 *
                taue / (taue + C / g0)
                +
                wi / 2 * gamma * x / g0 ** 2 * (Ei - E0) ** 2 *
                taui / (taui + C / g0)
        )
        sigma2_simplified = sp.together(sp.simplify(sigma2))
        num, den = sp.fraction(sigma2_simplified)

        print(sp.factor(num))
        print(sp.factor(den))

        sigma_target = sp.symbols('sigma_target')

        E0_expr = E0.subs({
            gL: float(cfg.g_L / nS),
            EL: float(cfg.e_L / mV),
            Ee: float(cfg.e_ampa / mV),
            Ei: float(cfg.e_gaba / mV),
        })
        E_target = float(chapter1Results.mu_v / mV)

        eq1 = E0_expr - E_target

        eq2 = sigma2.subs({
            gL: float(cfg.g_L / nS),
            EL: float(cfg.e_L / mV),
            Ee: float(cfg.e_ampa / mV),
            Ei: float(cfg.e_gaba / mV),
            we: float(cfg.w_ampa / nS),
            wi: float(cfg.w_gaba / nS),
            taue: float(cfg.tau_ampa / second),
            taui: float(cfg.tau_gaba / second),
            C: float(cfg.membrane_capacitance / nF)
        }) - (chapter1Results.sigma_v / mV) ** 2

        sol = sp.nsolve(
            (eq1, eq2),
            (x, gamma),
            (100.0, 4.0)  # initial guesses
        )

        print(sol)

    def test_system_elimination_2(self, produce_latex = False):
        from sympy.abc import x, y

        # x = gr, y = gamma
        gL, Ee, Ei, EL, E_target = sp.symbols('gL Ee Ei EL E_target')
        we, wi = sp.symbols('we wi')
        taue, taui = sp.symbols('taue taui')
        C = sp.symbols('C')
        sigma_target = sp.symbols('sigma_target')

        g0 = gL + x * (1 + y)

        E0 = (gL * EL + x * Ee + y * x * Ei) / g0
        equation_e_0 = E0 - E_target
        sigma_sq = (
                we / 2 * x / g0 ** 2 * (Ee - E0) ** 2 *
                taue / (taue + C / g0)
                +
                wi / 2 * y * x / g0 ** 2 * (Ei - E0) ** 2 *
                taui / (taui + C / g0)
        )

        equation_sigma_sq = sigma_sq - sigma_target**2

        y_expr = sp.solve(equation_e_0, y)[0]
        print(sp.pretty(y_expr))

        eq_x = sp.simplify(
            equation_sigma_sq.subs(y, y_expr)
        )

        eq_x = sp.cancel(eq_x)

        expr = sp.together(eq_x)
        num, _ = expr.as_numer_denom()

        num = sp.collect(num, x)
        poly = sp.Poly.from_expr(num, x)

        if produce_latex:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(sp.latex(num, order='grlex'))
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        with open(polynomial_file_name, "w") as f:
            f.write(sp.srepr(poly.as_expr()))

        sol = sp.solve(num, x, domain=Reals)
        print(sol)

        with open(solutions_file_name, "w") as f:
            f.write(sp.srepr(sol))


    def test_solution_read(self):
        from sympy.abc import x, y

        gL, Ee, Ei, EL, E_target = sp.symbols('gL Ee Ei EL E_target')
        g0 = gL + x * (1 + y)

        E0 = (gL * EL + x * Ee + y * x * Ei) / g0

        we, taue, wi, taui, C = sp.symbols('we taue wi taui C')

        sigma_sq = (
                we / 2 * x / g0 ** 2 * (Ee - E0) ** 2 *
                taue / (taue + C / g0)
                +
                wi / 2 * y * x / g0 ** 2 * (Ei - E0) ** 2 *
                taui / (taui + C / g0)
        )
        equation_e_0 = E0 - E_target
        y_expr = sp.solve(equation_e_0, y)[0]

        #with open("solution.txt") as f:
        with open(solutions_file_name) as f:
            solutions = sp.sympify(f.read())
            cfg = config_with_weak_synapses

            values = bind_config_to_sympy_values(cfg)

            for index, solution in enumerate(solutions):
                x_val = solution.subs(values).evalf()
                if abs(sp.im(x_val)) < 1e-10:
                    x_val = sp.re(x_val)

                y_val = y_expr.subs(values).subs(x, x_val)

                print(f"x val {x_val}, y val {y_val}")

                self.assertAlmostEqual(-47.6159564500000, float(E0.subs(values).subs({"x": x_val, "y": y_val})), places=7)
                self.assertAlmostEqual(3.60999999999985, float(sigma_sq.subs(values).subs({"x": x_val, "y": y_val})), places=7)

    def test_polynomial_plot(self):
        from sympy.abc import x
        eqs = RichardsonSympyEquations()

        ylims = [(-0.5E9, 1.5E9), (-1E10, 1.5E10)]
        x_maxs = [300, 1500]
        for x_max, ylim in zip(x_maxs, ylims):
            from matplotlib.ticker import ScalarFormatter

            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))  # always use scientific notation

            prepare_bigger_fonts(zoom=2)
            x_minus = np.linspace(-200, 0, 1000)
            x_plus = np.linspace(0, x_max, 1000)

            fig, ax1 = plt.subplots(
                1, 1,
                figsize=(10, 12),
                sharex=True
            )

            for config in [config_with_weak_synapses, config_with_medium_synapses]:

                polyn = eqs.x_polyn(config)
                f_p = sp.lambdify(x, polyn, "numpy")

                ax1.plot(
                    x_minus, f_p(x_minus),
                    color="red", linewidth=2, linestyle="-.",
                    label=f"negative domain, {config.label} synapses"
                )

                ax1.plot(
                    x_plus, f_p(x_plus),
                    linewidth=2,
                    label=f"positive domain, {config.label} synapses"
                )

            ax1.axhline(y=0, linestyle='--', color='k')

            ax1.set_xlabel(r"$x \equiv gr$ (nS)")
            ax1.set_ylabel(r"$P(gr)$")

            ax1.set_title(
                "Solution for P(gr) = 0 after elimination of $\\gamma$\n"
                r"$g_0 = g_L + g \cdot r \cdot (1 + \gamma)$" "\n"
                r"$\bar E_0 = \frac{g_L \cdot E_L + g \cdot r \cdot \left( E_e + \gamma \cdot E_i \right)}{g_L + g \cdot r \cdot (1 + \gamma)}$" "\n"
                r"$\bar\sigma_0^2 = \frac{w_e}{2}\frac{g \cdot r}{g_0}\bar E_e^2\frac{\tau_e}{\tau_e g_0+C} + \frac{w_i}{2}\frac{\gamma \cdot g \cdot r}{g_0}\bar E_i^2\frac{\tau_i}{\tau_i g_0+C}$"
            )

            ax1.yaxis.set_major_formatter(formatter)
            ax1.set_ylim(ylim)
            ax1.legend()

            fig.tight_layout()
            show_plots_non_blocking()

    def test_solutions_can_be_loaded_from_file(self):
        solutions = load_solutions(config_with_weak_synapses)
        x_s, y_s = zip(*solutions)

        np.testing.assert_array_almost_equal(x_s, [8.18779600541636, 223.826982269890])
        np.testing.assert_array_almost_equal(y_s, [0.159110841915727, 1.42238599165563])

    def test_solutions_can_be_loaded_from_file_loads_negative_values(self):
        solutions = load_solutions(config_with_weak_synapses, load_negative_values=True)
        x_s, y_s = zip(*solutions)

        np.testing.assert_array_almost_equal(x_s, [-83.80041, 8.18779600541636, 223.826982269890])
        np.testing.assert_array_almost_equal(y_s, [1.5984684697754752, 0.159110841915727, 1.42238599165563])

    def test_see_why_wang_solutions_have_large_imaginary_parts(self):
        config = wang_config_recurrent_synapses

        with open("solution.txt") as f:
            solutions = sp.sympify(f.read())

            values = bind_config_to_sympy_values(config)

            for index, solution in enumerate(solutions):
                from sympy.abc import x
                x_val = solution.subs(values).evalf()
                print(f"Solution {index + 1}: {x_val}")

    def test_polynomial_plots_wang_numbers(self):
        from sympy.abc import x
        eqs = RichardsonSympyEquations()

        ylims = [(-0.5E9, 1.5E9)]
        x_maxs = [50, 80]
        for x_max in x_maxs:
            print(x_max)
            from matplotlib.ticker import ScalarFormatter

            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))  # always use scientific notation

            prepare_bigger_fonts(zoom=2)
            x_minus = np.linspace(-50, 0, 1000)
            x_plus = np.linspace(0, x_max, 1000)

            fig, ax1 = plt.subplots(
                1, 1,
                figsize=(10, 12),
                sharex=True
            )

            for config in [wang_config_recurrent_synapses, wang_config_external_ampa_synapses]:

                polyn = eqs.x_polyn(config)
                f_p = sp.lambdify(x, polyn, "numpy")

                ax1.plot(
                    x_minus, f_p(x_minus),
                    color="red", linewidth=2, linestyle="-.",
                    label=f"negative domain, {config.label} synapses"
                )

                ax1.plot(
                    x_plus, f_p(x_plus),
                    linewidth=2,
                    label=f"positive domain, {config.label} synapses"
                )

            ax1.axhline(y=0, linestyle='--', color='k')

            ax1.set_xlabel(r"$x \equiv gr$ (nS)")
            ax1.set_ylabel(r"$P(gr)$")

            ax1.set_title(
                "Solution for P(gr) = 0 after elimination of $\\gamma$\n"
                r"$g_0 = g_L + g \cdot r \cdot (1 + \gamma)$" "\n"
                r"$\bar E_0 = \frac{g_L \cdot E_L + g \cdot r \cdot \left( E_e + \gamma \cdot E_i \right)}{g_L + g \cdot r \cdot (1 + \gamma)}$" "\n"
                r"$\bar\sigma_0^2 = \frac{w_e}{2}\frac{g \cdot r}{g_0}\bar E_e^2\frac{\tau_e}{\tau_e g_0+C} + \frac{w_i}{2}\frac{\gamma \cdot g \cdot r}{g_0}\bar E_i^2\frac{\tau_i}{\tau_i g_0+C}$"
            )

            ax1.yaxis.set_major_formatter(formatter)
            # ax1.set_ylim(ylim)
            ax1.legend()

            fig.tight_layout()
            show_plots_non_blocking()

if __name__ == '__main__':
    unittest.main()

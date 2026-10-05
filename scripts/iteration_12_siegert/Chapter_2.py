import unittest
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, Hz, Quantity

rate_palmer_control = 0.18 * Hz
rate_palmer_mk_801 = 0.05 * Hz


class Chapter2Figures(unittest.TestCase):

    def test_figure_1_plot_lif_firing_with_zoom_in(self):
        mus = np.linspace(-65, -35, 1001) * mV
        sigmas = np.array([0, 2., 4., 6.]) * mV
        lif_config = default_diffusion_lif_config
        rates = compute_LIF_curves_for_mus_sigmas(mus, sigmas, lif_config=lif_config)

        sg = SiegertGradients.for_lif_config(lif_config)
        example_sigma = 4.0 * mV
        mu_rate_control = newton_fsolve_find_mu_for_fixed_sigma(siegert_gradient=sg, sigma_v=example_sigma,
                                                                r_target=rate_palmer_control)
        mu_rate_mk_801 = newton_fsolve_find_mu_for_fixed_sigma(siegert_gradient=sg, sigma_v=example_sigma,
                                                               r_target=rate_palmer_mk_801)

        fig = plt.figure(figsize=(16, 8))
        prepare_bigger_fonts()

        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

        ax_leg = fig.add_subplot(gs[:, 1])
        ax_leg.axis("off")  # hide legend axis frame

        for index, ax in enumerate([ax1, ax2]):

            ax.axhline(y=0.05, color='orange', linestyle='--',
                       label='rate from Palmer 2014, Figure 2e \n MK801, 0.05 Hz')
            ax.axhline(y=0.18, color='black', linestyle='-.',
                       label='rate from Palmer 2014, Figure 2e \n Control, 0.18 Hz')
            ax.axvline(x=lif_config.theta / mV, color='dimgray', linestyle='-.', label=r'Threshold $\theta$')

            for sigma, lif_values in zip(sigmas, rates):
                ax.plot(mus / mV, lif_values / Hz, label=fr'$\sigma={sigma / mV}$ mV', alpha=0.6, lw=3)

            ax.set_xlabel("Driving force $V_m$ ($\mu$) [mV]")
            ax.set_ylabel("Firing rate [Hz]")

            ax.text(
                0.02, 1.17, f"({chr(ord("A") + index)})",
                transform=ax.transAxes,
                fontsize=20,
                fontweight=1000,
                va="top",
                ha="left"
            )

        ax2.plot(mu_rate_control / mV, sg.firing_rate(mu_v=mu_rate_control, sigma_v=example_sigma) / Hz,
                 marker='x', color='C0', markeredgewidth=1, markersize=12, linestyle="", alpha=0.7,
                 label=fr"$\mu_{{\mathrm{{sol}}$ for $rate_{{\mathrm{{LIF}} = {rate_palmer_control / Hz : .2f}$")

        ax2.plot(mu_rate_mk_801 / mV, sg.firing_rate(mu_v=mu_rate_mk_801, sigma_v=example_sigma) / Hz,
                 marker='x', color='C0', alpha=1, markeredgewidth=0.8, markersize=12,
                 label=fr"$\mu_{{\mathrm{{sol}}$ for $rate_{{\mathrm{{LIF}} = {rate_palmer_mk_801 / Hz : .2f}$")

        ax2.set_ylim(0, 0.2)

        ax2.axvline(x=mu_rate_control / mV, ymin=0.1, ymax=0.95, color='C0', linestyle='--', alpha=0.8)
        ax2.axvline(x=mu_rate_mk_801 / mV, ymin=0.1, ymax=0.3, color='C0', linestyle='--', alpha=0.8)

        ax2.annotate(
            text="",
            xy=(mu_rate_mk_801 / mV, 0.03),
            xytext=(mu_rate_control / mV, 0.03),
            arrowprops=dict(
                arrowstyle="<->",
                color="black",
                lw=1.5
            )
        )
        ax2.text(
            x=(mu_rate_control / mV + mu_rate_mk_801 / mV) / 2,
            y=0.007,
            s=r"$\Delta \mu$",
            fontsize=14,
            ha="center",
            va="bottom"
        )

        handles, labels = ax1.get_legend_handles_labels()

        order = np.hstack((np.arange(0, len(sigmas)) + 3, [0, 1, 2]))

        ax_leg.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc="center left",
            frameon=False,
            labelspacing=1.6,
            fontsize=18
        )

        ax1.set_title("Plot of $r_0(\mu, \sigma)$, including experimental rates to be fitted")
        ax2.set_title("Detail of $r_0(\mu, \sigma)$ in the range of the rates to be fitted")
        fig.suptitle("$r_0(\mu, \sigma)$ as predicted by Siegert's first passage time formula")

        fig.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

    def test_plot_sigma_as_function_of_mu_v(self):

        nmda_block_mu_to_sigma = mu_to_sigma_for_constant_rate(
            default_diffusion_lif_config.with_label("MK-801"), r_target=0.05 * Hz
        )
        control_mu_to_sigma = mu_to_sigma_for_constant_rate(
            default_diffusion_lif_config.with_label("Control"), r_target=0.18 * Hz
        )

        large_rate_mu_to_sigma = mu_to_sigma_for_constant_rate(
            default_diffusion_lif_config.with_label("Example high rate"), r_target=25 * Hz
        )
        sg = SiegertGradients.for_lif_config(default_diffusion_lif_config)
        # find sigma closest to some value:
        example_sigma = 4.0

        mu_rate_control = newton_fsolve_find_mu_for_fixed_sigma(siegert_gradient=sg, sigma_v=example_sigma * mV,
                                                                r_target=rate_palmer_control)
        mu_rate_mk_801 = newton_fsolve_find_mu_for_fixed_sigma(siegert_gradient=sg, sigma_v=example_sigma * mV,
                                                               r_target=rate_palmer_mk_801)

        print(f"Delta mu: {mu_rate_control - mu_rate_mk_801}")

        results = [nmda_block_mu_to_sigma, control_mu_to_sigma]

        prepare_bigger_fonts()

        fig = plt.figure(figsize=(16, 8))
        prepare_bigger_fonts()

        gs = fig.add_gridspec(1, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)

        """Plot μ vs σ curves and linear fit. Used by script runners with caller_test_case=self for figure naming."""
        for result, color in zip(results, ["orange", "black", "blue"]):
            ax1.plot(result.mus, result.sigmas, color=color, label=f"{result.exp_label}, r = {result.r_target / Hz} Hz",
                     lw=2)

        ax1.plot(mu_rate_control / mV, example_sigma,
                 marker='x', color='C0', markeredgewidth=1, markersize=12, linestyle="", alpha=0.7,
                 label=r"$\mu_\mathrm{sol}$ for $rate_\mathrm{LIF} = " + f"{rate_palmer_control / Hz : .2f}$")

        ax1.plot(mu_rate_mk_801 / mV, example_sigma,
                 marker='x', color='C0', markeredgewidth=1, markersize=12, linestyle="", alpha=0.7,
                 label=r"$\mu_\mathrm{sol}$ for $rate_\mathrm{LIF} =" + f"{rate_palmer_mk_801 / Hz : .2f}$")

        ax1.axvline(x=mu_rate_control / mV, ymin=0.1, ymax=0.8, color='C0', linestyle='--', alpha=0.8)
        ax1.axvline(x=mu_rate_mk_801 / mV, ymin=0.1, ymax=0.8, color='C0', linestyle='--', alpha=0.8)

        ax1.axhline(y=example_sigma, xmin=0, xmax=0.4, color='C0', linestyle='--', alpha=0.8,
                    label=r"$\sigma = $" + f"{example_sigma: .2f} mV")

        ax1.annotate(
            text="",
            xy=(mu_rate_mk_801 / mV, 0.7),
            xytext=(mu_rate_control / mV, 0.7),
            arrowprops=dict(
                arrowstyle="<->",
                color="black",
                lw=1.5
            )
        )
        ax1.text(
            x=(mu_rate_control / mV + mu_rate_mk_801 / mV) / 2,
            y=0.2,
            s=r"$\Delta \mu$",
            fontsize=14,
            ha="center",
            va="bottom"
        )

        """Plot μ vs σ curves and linear fit. Used by script runners with caller_test_case=self for figure naming."""
        for result, color in zip([nmda_block_mu_to_sigma, control_mu_to_sigma, large_rate_mu_to_sigma],
                                 ["orange", "black", "blue"]):
            ax2.plot(result.mus, result.sigmas, color=color, label=f"{result.exp_label}, r = {result.r_target / Hz} Hz",
                     lw=2)

        for index, ax in enumerate([ax1, ax2]):
            ax.axvline(x=default_diffusion_lif_config.theta / mV, color='dimgray', linestyle='-.',
                       label=r'Threshold $\theta$')
            ax.set_xlabel(r"$\mu$ [mV]")
            ax.set_ylabel(r"$\sigma$ [mV]")

            ax.text(
                0.02, 1.1, f"({chr(ord("A") + index)})",
                transform=ax.transAxes,
                fontsize=20,
                fontweight=1000,
                va="top",
                ha="left"
            )

            ax.legend()

        ax1.set_title("For reported spontaneous firing rates")
        ax2.set_title("For comparison low vs high firing rates")

        fig.suptitle(
            r"Predicted $\sigma = f(\mu)$ curve for $r_0(\mu, \sigma)$ = constant"
            "\n"
            r"using Siegert's first passage time formula",
            ha='center'
        )
        fig.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

    def test_plot_line_computation_vs_fit(self, fitting_problems: list[LifParamFittingProblem] =
    (default_diffusion_lif_config.fitting(r_target=0.05 * Hz, label="MK801"),
     default_diffusion_lif_config.fitting(r_target=0.18 * Hz, label="Control"),
     default_diffusion_lif_config.fitting(r_target=25 * Hz, label="Example high rate"))):

        results = [mu_to_sigma_for_constant_rate(fitting_problem.lif_config, r_target=fitting_problem.r_target) for fitting_problem in fitting_problems]
        plot_line_computation_vs_fit(results,
                                     colors=("orange", "black", "blue"),
                                     caller_test_case=self,
                                     descriptor="linear_fit")

    def test_check_integral_limits_and_linear_fit(self):
        nmda_block_mu_to_sigma = mu_to_sigma_for_constant_rate(
            default_diffusion_lif_config.with_label("MK-801"), r_target=0.05 * Hz
        )
        control_mu_to_sigma = mu_to_sigma_for_constant_rate(
            default_diffusion_lif_config.with_label("Control"), r_target=0.18 * Hz
        )

        large_rate_mu_to_sigma = mu_to_sigma_for_constant_rate(
            default_diffusion_lif_config.with_label("Example high rate"), r_target=25 * Hz
        )
        results = [nmda_block_mu_to_sigma, control_mu_to_sigma, large_rate_mu_to_sigma]
        colors = ("orange", "black", "blue")
        ax_integral_limits_limit = [(-30, 1000), (-30, 1000), (-2, 50)]
        ax_zoom_limit = [(0, 100), (0, 100), (0, 5)]

        siegert_gradients = SiegertGradients.for_lif_config(default_diffusion_lif_config)

        firing_rates, integral_limits, integral_values = [None] * len(results), [None] * len(results), [None] * len(results)

        for index, result in enumerate(results):
            current_firing_rates = np.zeros_like(result.mus)
            current_integral_limits = np.zeros(shape=(2, len(result.mus)))
            current_integral_values = np.zeros_like(result.mus)
            for row_index, (mu, sigma) in enumerate(zip(result.mus, result.sigmas)):
                current_firing_rates[row_index] = siegert_gradients.firing_rate(mu_v=mu, sigma_v=sigma)
                current_integral_limits[:, row_index] = integration_limits(V_mean=mu, sigma_v=sigma,
                                                                           theta=default_diffusion_lif_config.theta,
                                                                           V_reset=default_diffusion_lif_config.V_r)
                current_integral_values[row_index] = I_mu_sigma(mu_v=mu, sigma_v=sigma,
                                                                theta=default_diffusion_lif_config.theta,
                                                                V_reset=default_diffusion_lif_config.V_r)
            firing_rates[index] = current_firing_rates
            integral_limits[index] = current_integral_limits
            integral_values[index] = current_integral_values

        prepare_bigger_fonts(zoom=1)
        lw = 2.5

        fig = plt.figure(figsize=(14, 18))
        gs = fig.add_gridspec(4, 2)

        ax_integral_limits_mmda_block = fig.add_subplot(gs[0, 0])
        ax_integral_limits_zoom_nmda_block = fig.add_subplot(gs[0, 1])

        ax_integral_limits_control = fig.add_subplot(gs[1, 0])
        ax_integral_limits_zoom_control = fig.add_subplot(gs[1, 1])

        ax_integral_limits_high_rate = fig.add_subplot(gs[2, 0])
        ax_integral_limits_zoom_high_rate = fig.add_subplot(gs[2, 1])

        axs_integral_limits = [ax_integral_limits_mmda_block, ax_integral_limits_control, ax_integral_limits_high_rate]
        axs_integral_limits_zoom = [ax_integral_limits_zoom_nmda_block, ax_integral_limits_zoom_control,
                                    ax_integral_limits_zoom_high_rate]

        ax_integral_values = fig.add_subplot(gs[3, 0])
        ax_rates_computation = fig.add_subplot(gs[3, 1], sharex=ax_integral_values)

        for index, result in enumerate(results):
            axs_integral_limits[index].plot(
                result.mus,
                integral_limits[index][0, :],
                label=f"lower limit",
                lw=lw,
                color=colors[index],
            )
            axs_integral_limits[index].plot(
                result.mus,
                integral_limits[index][1, :],
                linestyle='-.',
                lw=lw,
                color=colors[index],
                label=f"upper limit"
            )

            axs_integral_limits[index].set_ylim(ax_integral_limits_limit[index])

            delta = integral_limits[index][1, :] - integral_limits[index][0, :]
            axs_integral_limits_zoom[index].plot(result.mus,
                                                 ((default_diffusion_lif_config.theta - default_diffusion_lif_config.V_r) / mV) / (
                                                         np.sqrt(2) * result.sigmas),
                                                 label=r"Closed formula $\frac{V_R - \theta}{\sqrt{2} \cdot \sigma}$",
                                                 alpha=0.5, lw=lw, color="red")
            axs_integral_limits_zoom[index].plot(nmda_block_mu_to_sigma.mus,
                                                 delta,
                                                 label="upper limit - lower limit",
                                                 lw=lw, color=colors[index])
            axs_integral_limits_zoom[index].set_ylim(ax_zoom_limit[index])


            ax_integral_values.plot(result.mus, integral_values[index], label=f"{result.exp_label}", lw=lw,
                                    color=colors[index])
            ax_rates_computation.plot(result.mus, firing_rates[index].T, label=f"{result.exp_label}", lw=lw,
                                      color=colors[index])

        for index, result in enumerate(results):
            axs_integral_limits[index].set_title(f"Integral limits \n {result.exp_label}")
            axs_integral_limits[index].set_ylabel(r"Integral limits (unitless)")
            axs_integral_limits_zoom[index].set_title(f"upper limit -  lower limit \n {result.exp_label}")
            axs_integral_limits_zoom[index].set_ylabel(r"$\Delta$ limits (unitless)")

        ax_integral_values.set_ylabel(r"Integral value (unitless)")
        ax_rates_computation.set_ylabel(r"predicted rate [Hz]")

        for ax in [ax_integral_values, ax_rates_computation]:
            ax.set_xlabel(r"$\mu$ [mV]")
            ax.axvline(x=default_diffusion_lif_config.theta / mV, color='dimgray', linestyle='-.',
                       label=r'Threshold $\theta$')

        panels = ["A1", "A2", "A3", "B1", "B2", "B3", "C", "D"]
        for index, ax in enumerate(chain(axs_integral_limits, axs_integral_limits_zoom, [ax_integral_values, ax_rates_computation])):
            ax.text(
                0.02, 1.2, f"({panels[index]})",
                transform=ax.transAxes,
                fontsize=20,
                fontweight=1000,
                va="top",
                ha="left"
            )
            ax.legend()

        ax_integral_values.set_title(
            r"$I(\mu, \sigma)=\int_{\frac{\mu - \theta}{\sqrt{2}\sigma}}^{\frac{\mu - V_R}{\sqrt{2}\sigma}} "
            r"e^{x^2}\,\mathrm{erfc}(x)\,dx$" + "\n on the set \n" + r"$r_0(\mu, \sigma)$ = constant"
        )
        ax_rates_computation.set_title(
            "$r_0(\mu, \sigma)$" + " computed with actual numerical values \n returned by Newton's method", y=1.1)

        fig.suptitle("Plot of integral limits \n" +  "upper limit " + r"$\frac{\mu - V_R}{\sqrt{2}\sigma}$ and lower limit $\frac{\mu - \theta}{\sqrt{2}\sigma}$" + "\n with check that both integral and rate \n are constant" )
        fig.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

    def test_show_linear_fit_and_compute_mu_sigma(self, lif_config:DiffusionLIFConfig = default_diffusion_lif_config, rates=(0.05 * Hz, 0.18 * Hz)):

        #for delta_mu in [0.4, 0.7]:
        for delta_mu in [0.7]:

            lif_configs = [lif_config.with_label("MK-801"),
                           lif_config.with_label("Control")]
            target_rates = rates
            results = [mu_to_sigma_for_constant_rate(config, r_target=target_rate) for config, target_rate in
                       zip(lif_configs, target_rates)]
            nmda_block_mu_to_sigma = results[0]
            control_mu_to_sigma = results[1]

            m_mk801, b_mk801, _, _, _ = nmda_block_mu_to_sigma.linear_fit()
            m_control, b_control, _, _, _ = control_mu_to_sigma.linear_fit()

            sigma_v, mu_v = 1 / (m_mk801 - m_control) * np.array([[-m_control, m_mk801], [-1, 1]]) @ np.array(
                [[b_mk801], [m_control * delta_mu + b_control]])

            sigma_sol = sigma_v[0]
            mu_v_mk801 = mu_v[0]
            mu_v_control = mu_v_mk801 + delta_mu

            print(f"mu_v MK-801 = {mu_v_mk801}, mu_v Control = {mu_v_control}, sigma_v = {sigma_v}")

            siegert_gradient = SiegertGradients.for_lif_config(lif_config)
            rate_mk801 = siegert_gradient.firing_rate(mu_v=mu_v_mk801 * mV, sigma_v=sigma_sol * mV)
            rate_control = siegert_gradient.firing_rate(mu_v=mu_v_control * mV, sigma_v=sigma_sol * mV)
            print(f"Predicted MK801 rate: {rate_mk801}, Control rate: {rate_control}")

            prepare_bigger_fonts()

            fig, ax = plt.subplots(figsize=(10, 10))
            prepare_bigger_fonts()

            """Plot μ vs σ curves and linear fit. Used by script runners with caller_test_case=self for figure naming."""
            for result, color in zip(results, ["orange", "black"]):
                ax.plot(result.mus, result.sigmas, color=color, label=f"{result.exp_label}, r = {result.r_target / Hz} Hz",
                        lw=2)

            ax.plot(mu_v_control, sigma_sol,
                    marker='x', color='C0', markeredgewidth=1, markersize=12, linestyle="", alpha=0.7,
                    label=r"$\mu_\mathrm{sol} + \Delta \mu_{\mathrm{obs}}$="f"{mu_v_control: .3f} mV")

            ax.plot(mu_v_mk801, sigma_sol,
                    marker='x', color='C0', markeredgewidth=1, markersize=12, linestyle="", alpha=0.7,
                    label=r"$\mu_\mathrm{sol} =$"f"{mu_v_mk801 : .3f} mV")

            ax.axvline(x=mu_v_control, ymin=0.2, ymax=0.5, color='C0', linestyle='--', alpha=0.8)
            ax.axvline(x=mu_v_mk801, ymin=0.2, ymax=0.5, color='C0', linestyle='--', alpha=0.8)

            x_max = 0.9 - abs((lif_config.theta / mV - mu_v_control))/40
            ax.axhline(y=sigma_sol, xmin=0, xmax=x_max, color='C0', linestyle='--', alpha=0.8,
                       label=r"$\sigma_{\mathrm{sol}} = $" + f"{sigma_sol: .3f} mV")
            ax.annotate(
                text="",
                xy=(mu_v_mk801, 1.1),
                xytext=(mu_v_control, 1.1),
                arrowprops=dict(
                    arrowstyle="<->",
                    color="black",
                    lw=1.5
                )
            )
            ax.text(
                x=(mu_v_control + mu_v_mk801) / 2,
                y=0.6,
                s=r"$\Delta \mu$ = "f"{delta_mu: .1f} mV",
                fontsize=14,
                ha="center",
                va="bottom"
            )

            ax.set_xlabel(r"$\mu$ [mV]")
            ax.set_ylabel(r"$\sigma$ [mV]")

            ax.legend()
            fig.suptitle(f"Found solutions for {r"$r(\mu_{\mathrm{MK-801}}, \sigma) =$ "} {rate_mk801 / Hz: .2f} {" Hz"}, {r"$r(\mu_{\mathrm{Control}}, \sigma) =$"}{rate_control/Hz: .2f}{ " Hz"}\n"
                         r"$\mu_{\mathrm{MK-801}}$="f"{mu_v_mk801 :.3f} mV, "r"$\mu_{\mathrm{Control}}$="f"{mu_v_control :.3f} mV, "r"$\Delta \mu$="f"{mu_v_control - mu_v_mk801 : .3f} mV, "r"$\sigma$="f"{sigma_sol:.3f} mV \n"
                         r"$\theta - \mu_{\mathrm{Control}}$="f"{default_diffusion_lif_config.theta / mV - mu_v_control: .3f} mV")

            fig.tight_layout()
            show_plots_non_blocking(caller_test_case=self, descriptor=f"d_mu_{label_for_float(delta_mu)}")

            print(f"delta mu {delta_mu}: m mk801= {m_mk801}, m control = {m_control}"
                  f"b mk801 = {b_mk801}, b control = {b_control}")

            print(f"Empirical vs analytical: MK801: {b_mk801 - m_mk801 * lif_config.theta / mV}")
            print(f"Empirical vs analytical: Control: {b_control - m_control * lif_config.theta / mV}")


    def test_show_mu_to_sigma_curves_for_constant_rate_and_fit_mu_sigma(self, lif_config:DiffusionLIFConfig = default_diffusion_lif_config, rates=(0.05 * Hz, 0.18 * Hz), delta_mu: Quantity = 0.7 * mV):

        lif_configs = [lif_config.with_label("MK-801"),
                       lif_config.with_label("Control")]
        target_rates = rates
        results = [sigma_to_mu_for_constant_rate(config, r_target=target_rate) for config, target_rate in
                   zip(lif_configs, target_rates)]
        mk801_mu_to_sigma = results[0]
        control_mu_to_sigma = results[1]


        prepare_bigger_fonts()
        fig, (ax_mu_to_sigma, ax_delta_mu) = plt.subplots(1, 2, figsize=(8, 5))

        ax_mu_to_sigma.plot(mk801_mu_to_sigma.mus, mk801_mu_to_sigma.sigmas, color="orange", label="MK-801")
        ax_mu_to_sigma.plot(control_mu_to_sigma.mus, control_mu_to_sigma.sigmas, color="black", label="Control")

        delta_mus = control_mu_to_sigma.mus - mk801_mu_to_sigma.mus
        ax_delta_mu.plot(control_mu_to_sigma.sigmas,  delta_mus, label=r"$\Delta \mu(\sigma)$")

        ax_delta_mu.axvline(
            x=0.7,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=r"$\Delta (\mu)$ = 0.7 mV"
        )

        ax_mu_to_sigma.set_xlabel(r"$\mu [mV]")
        ax_mu_to_sigma.set_xlabel(r"$\sigma$ [mV]")

        ax_mu_to_sigma.set_title(r"Isorate curves $r_0(\mu, \sigma)$=constant")
        ax_delta_mu.set_title(r"$\Delta \mu(\sigma) = \mu_{\text{Control}}(\sigma) - \mu_{\text{MK801}}(\sigma)$")

        fig.tight_layout()
        fig.legend()

        show_plots_non_blocking()


    def test_show_linear_fit_and_compute_mu_sigma_updated_rates(self):
        rate_mk801_real = 0.05 * Hz / 0.59
        rate_control_real = 0.18 * Hz / 0.66
        self.test_show_linear_fit_and_compute_mu_sigma(rates=[rate_mk801_real, rate_control_real])

    def test_solve_graphically_for_rate_and_gain_coming_from_previous_solution(self):
        gain = 0.09699832465740052 * Hz / mV
        gain_computations = mu_to_sigma_for_constant_gain(
            default_diffusion_lif_config.with_label("MK-801"), gain=gain, mu_lims=None
        )

        rate_computations = mu_to_sigma_for_constant_rate(
            default_diffusion_lif_config.with_label("MK-801"), r_target=0.05 * Hz, mu_lims=None
        )

        sg = SiegertGradients.default()
        x_intersect, y_intersect = compute_intersection(rate_computations, gain_computations)
        error_nmda_block = np.abs(
            (rate_computations.r_target - sg.firing_rate(x_intersect * mV, y_intersect * mV)) / Hz)
        error_gain = np.abs(
            (gain_computations.r_target - sg.d_rate_d_mu(x_intersect * mV, y_intersect * mV)) / Hz * mV)

        prepare_bigger_fonts()

        plt.figure(figsize=(8, 6))

        plt.xlabel(r"$\mu$ [mV]")
        plt.ylabel(r"$\sigma$ [mV]")
        plt.title(
            "Model-based estimation of " r"($\mu$, $\sigma$)" "\n"
            r"for $r(\mu, \sigma) = $"f" {rate_palmer_mk_801 / Hz :.2f} Hz "" and "r"$\frac{d r}{d \mu }$ = " f"{gain / Hz * mV:.3f} Hz/mV\n"
            "predicted by first time passage formula \n"
            r"$r(\mu_{\mathrm{sol}}, \sigma_{\mathrm{sol}}) =$" f"{sg.firing_rate(x_intersect * mV, y_intersect * mV) / Hz : .3f} +{error_nmda_block: .0E} Hz, "
            r"$\frac{d r}{d \mu}(\mu_{\mathrm{sol}}, \sigma_{\mathrm{sol}}) =$" f"{sg.d_rate_d_mu(x_intersect * mV, y_intersect * mV) / Hz * mV : .3f} +{error_gain: .0E} Hz/mV")

        plt.vlines(x_intersect, ymin=0, ymax=y_intersect * 1.3, colors='b', linestyles='--', lw=0.9)
        plt.hlines(y_intersect,
                   xmin=np.min((rate_computations.mus[0], gain_computations.mus[0])),
                   xmax=x_intersect + 2, colors='b', linestyles='--', lw=0.9)

        plt.plot(rate_computations.mus, rate_computations.sigmas, color="orange",
                 label=r"$\sigma = f_1(\mu)$ for $r(\mu, f_1(\mu)) = $"f" {rate_computations.r_target / Hz} Hz")
        plt.plot(gain_computations.mus, gain_computations.sigmas, color="purple",
                 label=r"$\sigma = f_3(\mu)$ for $\frac{d r}{d \mu}(\mu, f_3(\mu)) = $"f" {gain / Hz * mV: .4f} Hz/mV")
        plt.scatter(x_intersect, y_intersect,  s=70, zorder=3, label=f"Solution ({x_intersect: .2f} mV, {y_intersect: .2f} mV)", alpha=0.6, color="blue")
        plt.legend()
        plt.subplots_adjust(top=0.75)
        show_plots_non_blocking(caller_test_case=self)

    def test_solve_graphical_for_rate_and_derivative_numerical_fit(self):

        delta_mu = 0.7 * mV
        rate_mk801 = 0.05 * Hz
        rate_control = 0.18 * Hz

        gain = 0.09699832465740052 * Hz / mV

        diffusion_lif_config = DiffusionLIFConfig(params={DiffusionLIFConfig.KEY_V_R: -55})
        siegert_gradients = SiegertGradients.for_lif_config(diffusion_lif_config)

        gain_computations = mu_to_sigma_for_constant_gain(
            diffusion_lif_config.with_label(r"$\frac{\Delta r}{ \Delta \mu}$ from numerical computation"),
            gain=gain
        )

        mk_801_rate_computations = mu_to_sigma_for_constant_rate(
            diffusion_lif_config.with_label("MK-801"), r_target=rate_mk801
        )

        control_rate_computations = mu_to_sigma_for_constant_rate(
            diffusion_lif_config.with_label("Control"), r_target=rate_control
        )

        control_rate_computations = control_rate_computations.with_delta_mu(delta_mu)

        plot_two_rates_and_one_gain(mk_801_rate_computations, control_rate_computations, gain_computations,
                                    siegert_gradients,
                                    plot_label=r"for $\frac{\Delta r}{\Delta \mu}$ based on numerical fitting", caller_test_case=self)

    def test_graphically(self):
        SolveByGraphicalSolutionScripts().test_solve_graphical_for_two_rates()


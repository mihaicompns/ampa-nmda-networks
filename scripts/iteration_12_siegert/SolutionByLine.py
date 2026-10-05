import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, Hz, Quantity, is_dimensionless
from loguru import logger

from Plotting import prepare_bigger_fonts, show_plots_non_blocking
from iteration_12_siegert.CorrelationSimulations import label_for_float
from iteration_12_transfer_function_of_lif_neurons.LookForAllSolutionsMuSigma import \
    compute_sigma_necessary_for_given_rate_and_mean, mu_to_sigma_for_constant_rate, MuToSigmaResult
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import SiegertGradients
from iteration_12_transfer_function_of_lif_neurons.config import DiffusionLIFConfig, default_diffusion_lif_config


class LineFit:
    def __init__(self, slope: float, intercept: float):
        self.slope = slope
        self.intercept = intercept

    def sigmas(self, mus):
        return self.slope * mus + self.intercept


def line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2:
        raise ValueError("Slope is undefined for vertical lines (x1 == x2).")

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    return LineFit(slope=m, intercept=b)


def plot_data_vs_line_fit(data: list[MuToSigmaResult], line_computation: list[LineFit], caller_test_case=None,
                          descriptor="linear_fit", axs=None, colors=("orange", "black")):
    should_create_figure = axs is None

    prepare_bigger_fonts()

    if should_create_figure:
        fig = plt.figure(figsize=(10, 16))

        # Outer grid: 2 rows
        outer_gs = fig.add_gridspec(
            nrows=2, ncols=1,
            height_ratios=[2, 3],
            hspace=0.15
        )

        # --- Top: linear fit only ---
        ax_linear_fit = fig.add_subplot(outer_gs[0])

        # --- Bottom: grouped table + residuals ---
        inner_gs_bottom = outer_gs[1].subgridspec(
            nrows=2, ncols=1,
            height_ratios=[1.5, 2],
            hspace=0.05
        )

        ax_table = fig.add_subplot(inner_gs_bottom[0])
        ax_table.axis('off')

        ax_residuals = fig.add_subplot(inner_gs_bottom[1], sharex=ax_linear_fit)

    r2_s = np.zeros_like(data)
    rows = []

    rmse_s = np.zeros_like(data)
    mae_s = np.zeros_like(data)

    for index, (result, line_data, color) in enumerate(zip(data, line_computation, colors)):
        x = result.mus
        y = result.sigmas

        # Calculate R² to show goodness of fit
        y_pred = line_data.slope * x + line_data.intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        residuals = y - y_pred

        r2_s[index] = r2
        rmse_s[index] = np.sqrt(np.mean((y - y_pred) ** 2))
        mae_s[index] = np.mean(np.abs(y - y_pred))

        ax_linear_fit.plot(x, y, alpha=0.5, label=f'{result.exp_label} data', linewidth=10, color=color)

        # Plot fitted lines across a common range
        x_plot = np.linspace(-65, -40, 1001)
        y_plot = line_data.slope * x_plot + line_data.intercept

        ax_linear_fit.plot(x_plot, y_plot, linewidth=3.5,
                           label=f'{result.exp_label} linear fit: \n $\sigma_v$={line_data.slope:.3f}$\mu${line_data.intercept:.2f}',
                           color=color)

        ax_residuals.scatter(x, residuals, color=color, s=5, alpha=0.5, label=f'{result.exp_label}')
        ax_residuals.axhline(0, color='black', lw=1, linestyle='--')

        residuals = y - y_pred
        max_error = np.max(np.abs(residuals))
        e_infinity = max_error / (np.max(y) - np.min(y))
        rows.append([result.exp_label, f"{max_error:.4f}", f"{e_infinity:.4f}"])

    ## labels and titles for ax linear fit ##
    labels = [r.exp_label for r in data]
    if len(labels) == 1:
        rmse_label = f"{labels[0]}: {rmse_s[0] :.4f}"
        mae_label = f"{labels[0]}: {mae_s[0]:.4f}"
    elif len(labels) == 2:
        rmse_label = f"{labels[0]}: {rmse_s[0] :.4f} and {labels[1]}: {rmse_s[1]:.4f}"
        mae_label = f"{labels[0]}: {mae_s[0] :.4f} and {labels[1]}: {mae_s[1]:.4f}"
    else:
        rmse_labels = [f"{label}: {rmse:.4f}" for label, rmse in zip(labels, rmse_s)]
        rmse_label = f"{', '.join(rmse_labels[:-1])} and {rmse_labels[-1]}"
        mae_labels = [f"{label}: {mae:.4f}" for label, mae in zip(labels, mae_s)]
        mae_label = f"{', '.join(mae_labels[:-1])} and {mae_labels[-1]}"

    ax_linear_fit.set_xlabel('$\mu_v$ [mV]')
    ax_linear_fit.set_ylabel('$\sigma_v$ [mV]')

    ax_linear_fit.text(
        0.5, 1.3,
        f'Verify errors of linear fits \n Root Mean Square Error [mV] \n {rmse_label} \n Mean Absolute Error [mV] \n {mae_label}',
        ha='center', va='top',
        transform=ax_linear_fit.transAxes,
        fontsize=18,
        clip_on=False
    )

    ## labels and title for table ##

    col_labels = ["", r"$E_{\max}$" + "\n" + "$\max |\mathrm{err}|$ [mV]",
                  r"$E_{\infty}$" + "\n" + r"$\frac{E_{\max}}{\max |\mathrm{err}| - \min |\mathrm{err}|}$"]
    table_title = r"Error ($\sigma_{v, i, \mathrm{found}} - \sigma_{v, i, \mathrm{linear\ est}}$) of linear fit" + "\n" + r"$\mathrm{err}_i = \sigma_{v, i} - \hat{\sigma}_{v, i}$"
    ax_table.text(
        0.5, 0.8, table_title,
        ha='center', va='bottom'
    )

    table = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        bbox=[0.1, 0, 0.8, 0.8]
    )

    table.scale(0.8, 3)  # adjust size

    table.auto_set_font_size(False)
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # header row
            cell.set_height(0.6)
            cell.get_text().set_fontsize(cell.get_text().get_fontsize() + 2)
            cell.set_text_props(weight='bold')
        else:  # body rows
            cell.set_height(0.2)

    ## labels for residuals axis ##
    ax_residuals.set_xlabel('$\mu_v$ [mV]')
    ax_residuals.set_ylabel('Error [mV]')

    for index, ax in enumerate([ax_linear_fit, ax_residuals]):
        ax.axvline(x=default_diffusion_lif_config.theta / mV, color='dimgray', linestyle='-.',
                   label=r'Threshold $\theta$')
        ax.text(
            0.02, 1.1, f"({chr(ord("A") + index)})",
            transform=ax.transAxes,
            fontsize=20,
            fontweight=1000,
            va="top",
            ha="left"
        )
    ax_linear_fit.legend(
        loc='upper right',
        bbox_to_anchor=(1.1, 1),
        borderaxespad=0.,
        fontsize=16,
    )
    ax_residuals.legend()

    # print(len(ax_linear_fit.texts))

    if should_create_figure:
        fig.tight_layout()
        show_plots_non_blocking(caller_test_case=caller_test_case, descriptor=descriptor)


def compute_data_and_fit(lif_config: DiffusionLIFConfig, r_target: Quantity, reference_mu=-55 * mV)-> (MuToSigmaResult, LineFit):
    line_fit_data = find_line_equation_from_one_point(lif_config, r_target, reference_mu)

    mus_to_sigmas = mu_to_sigma_for_constant_rate(lif_config=lif_config, r_target=r_target,
                                                                 mu_lims=(lif_config.theta / mV - 30, lif_config.theta / mV))

    return mus_to_sigmas, line_fit_data


def find_line_equation_from_one_point(lif_config, r_target, reference_mu =-55 * mV):
    if is_dimensionless(r_target):
        r_target = r_target * Hz
    sg = SiegertGradients.for_lif_config(lif_config)
    mu_det = sg.mu_det(r_target)

    first_point = (mu_det / mV, 0)
    sigma = compute_sigma_necessary_for_given_rate_and_mean(mu=reference_mu, r_target=r_target, lif_config=lif_config)
    second_point = (reference_mu / mV, sigma / mV)
    line_fit_data = line_from_points(first_point, second_point)

    logger.debug("Rate for deterministic model {} mV, {} mV: {}", first_point[0], first_point[1],
                sg.firing_rate(mu_v=first_point[0] * mV, sigma_v=first_point[1] * mV))
    logger.debug("Rate for {} mV, {} mV: {}", second_point[0], second_point[1],
                sg.firing_rate(mu_v=second_point[0] * mV, sigma_v=second_point[1] * mV))

    return line_fit_data

def fit_mu_and_sigma_two_rates_and_delta_mu(delta_mu, lif_config, target_rates):

    if not is_dimensionless(delta_mu):
        delta_mu = delta_mu  / mV

    mk801_line = find_line_equation_from_one_point(lif_config.with_label("MK-801"), r_target=target_rates[0],
                                                   reference_mu=-55 * mV)
    control_line = find_line_equation_from_one_point(lif_config.with_label("Control"), r_target=target_rates[1],
                                                     reference_mu=-55 * mV)
    sigma_v, mu_v = 1 / (mk801_line.slope - control_line.slope) * np.array(
        [[-control_line.slope, mk801_line.slope], [-1, 1]]) @ np.array(
        [[mk801_line.intercept], [control_line.slope * delta_mu + control_line.intercept]])
    return control_line, mk801_line, mu_v[0] * mV, sigma_v[0] * mV

def plot_two_rates_and_delta_solutions(lif_config, delta_mu, control_line, mk801_line, mu_v_control,
                                       mu_v_mk801, sigma_v, caller_test_case=None):

    sg = SiegertGradients.for_lif_config(lif_config=lif_config)
    rate_mk801 = sg.firing_rate(mu_v = mu_v_mk801, sigma_v = sigma_v)
    rate_control = sg.firing_rate(mu_v = mu_v_control, sigma_v = sigma_v)

    mu_v_mk801_unitless = mu_v_mk801 / mV
    mu_v_control_unitless = mu_v_control / mV
    sigma_v_unitless = sigma_v / mV

    prepare_bigger_fonts()
    mus = np.linspace(-60, -40, 1001)
    fig, ax = plt.subplots(figsize=(10, 10))
    prepare_bigger_fonts()
    ax.plot(mus, mk801_line.sigmas(mus), label=f"MK801", color="orange")
    ax.plot(mus, control_line.sigmas(mus), label="Control", color="black")
    ax.plot(mu_v_control_unitless, sigma_v_unitless,
            marker='x', color='C0', markeredgewidth=1, markersize=12, linestyle="", alpha=0.7,
            label=r"$\mu_\mathrm{sol} + \Delta \mu_{\mathrm{obs}}$="f"{mu_v_control_unitless: .3f} mV")
    ax.plot(mu_v_mk801_unitless, sigma_v_unitless,
            marker='x', color='C0', markeredgewidth=1, markersize=12, linestyle="", alpha=0.7,
            label=r"$\mu_\mathrm{sol} =$"f"{mu_v_mk801_unitless : .3f} mV")
    ax.axvline(x=mu_v_control_unitless, ymin=0.2, ymax=0.5, color='C0', linestyle='--', alpha=0.8)
    ax.axvline(x=mu_v_mk801_unitless, ymin=0.2, ymax=0.5, color='C0', linestyle='--', alpha=0.8)
    x_max = 0.9 - abs((lif_config.theta - mu_v_control) / mV) / 40
    ax.axhline(y=sigma_v_unitless, xmin=0, xmax=x_max, color='C0', linestyle='--', alpha=0.8,
               label=r"$\sigma_{\mathrm{sol}} = $" + f"{sigma_v / mV: .3f} mV")
    ax.annotate(
        text="",
        xy=(mu_v_mk801_unitless, 1.1),
        xytext=(mu_v_control_unitless, 1.1),
        arrowprops=dict(
            arrowstyle="<->",
            color="black",
            lw=1.5
        )
    )
    ax.text(
        x=(mu_v_control_unitless + mu_v_mk801_unitless) / 2,
        y=0.6,
        s=r"$\Delta \mu$ = "f"{delta_mu / mV: .1f} mV",
        fontsize=14,
        ha="center",
        va="bottom"
    )
    ax.set_xlabel(r"$\mu$ [mV]")
    ax.set_ylabel(r"$\sigma$ [mV]")
    ax.legend()
    fig.suptitle(
        f"Two points: Found solutions for {r"$r(\mu_{\mathrm{MK-801}}, \sigma) =$ "} {rate_mk801 / Hz: .2f} {" Hz"}, {r"$r(\mu_{\mathrm{Control}}, \sigma) =$"}{rate_control/Hz: .2f}{ " Hz"}\n"
        r"$\mu_{\mathrm{MK-801}}$="f"{mu_v_mk801 / mV :.3f} mV, "r"$\mu_{\mathrm{Control}}$="f"{mu_v_control / mV :.3f} mV, "r"$\Delta \mu$="f"{mu_v_control / mV - mu_v_mk801 / mV : .3f} mV, "r"$\sigma$="f"{sigma_v / mV:.3f} mV \n"
        r"$\theta - \mu_{\mathrm{Control}}$="f"{default_diffusion_lif_config.theta / mV - mu_v_control / mV: .3f} mV")
    fig.tight_layout()
    show_plots_non_blocking(caller_test_case=caller_test_case, descriptor=f"d_mu_{label_for_float(delta_mu)}")

def scan_lif_configs(lif_configs=list[DiffusionLIFConfig], x_axis=np.empty((0,)), label=r"$V_R$", unit="mV", delta_mu: Quantity = 0.7 * mV, caller_test_case=None):

    mus_s = [None] * len(lif_configs)
    sigmas_s = [None] * len(lif_configs)
    rates_mk = [None] * len(lif_configs)
    rates_control = [None] * len(lif_configs)
    for index, config in enumerate(lif_configs):

        _, _, mu, sigma = fit_mu_and_sigma_two_rates_and_delta_mu(delta_mu=delta_mu, lif_config=config,
                                                                  target_rates=[0.05, 0.18])
        mus_s[index] = (mu + delta_mu) / mV
        sigmas_s[index] = sigma / mV
        rates_mk[index] = SiegertGradients.for_lif_config(config).firing_rate(mu_v=mu, sigma_v=sigma) / Hz
        rates_control[index] = SiegertGradients.for_lif_config(config).firing_rate(mu_v=mu + delta_mu,
                                                                                   sigma_v=sigma) / Hz

    prepare_bigger_fonts()

    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))

    ax_mu = ax[0, 0]
    ax_sigma = ax[0, 1]
    ax_mk_rates = ax[1, 0]
    ax_control_rates = ax[1, 1]
    ax_control_rates.sharey(ax_mk_rates)

    ax_mu.plot(x_axis, mus_s, lw=2)
    ax_mu.set_title("$\mu_{sol}$")
    ax_mu.set_ylabel("[mV]")

    ax_sigma.plot(x_axis, sigmas_s, lw=2, color="orange")
    ax_sigma.set_title("$\sigma{sol}$")
    ax_sigma.set_ylabel("[mV]")

    ax_mk_rates.plot(x_axis, rates_mk, lw=2)
    ax_mk_rates.set_title("$r_0(\mu, \sigma)$ MK801")
    ax_mk_rates.set_ylabel("[Hz]")

    ax_control_rates.plot(x_axis, rates_control, lw=2, color="black")
    ax_control_rates.set_title("$r_0(\mu + \Delta \mu, \sigma)$ Control")
    ax_control_rates.set_ylabel("[Hz]")

    for axs in ax.flatten():
        axs.set_xlabel(fr"{label} [{unit}]")

    fig.suptitle(f"Effect of {label} \n on parameter fitting")
    fig.tight_layout()
    show_plots_non_blocking(caller_test_case=caller_test_case)

class SolutionsByTwoPointsScripts(unittest.TestCase):
    def test_plot_control(self):
        lif_config = default_diffusion_lif_config.with_label("Control")

        sg = SiegertGradients.default()

        one_point = (-40, 0)

        sigma = compute_sigma_necessary_for_given_rate_and_mean(mu=-55, r_target=0.18, lif_config=lif_config)

        print(f"intercept: {sg.firing_rate(-40 * mV, 0 * mV)}")
        print(f"Found point: {sg.firing_rate(-55 * mV, sigma)}")

        second_point = (-55, sigma / mV)

        line_fit_data = line_from_points(one_point, second_point)
        x = np.linspace(-65, -40, 1001)

        # Apply the line equation
        y = line_fit_data.slope * x + line_fit_data.intercept

        mu_to_sigma_for_control_rate = mu_to_sigma_for_constant_rate(lif_config=lif_config, r_target=0.18 * Hz,
                                                                     mu_lims=(-65, -40))

        # Plot
        plt.plot(x, y, label=f"y = {line_fit_data.slope:.2f}x - {np.abs(line_fit_data.intercept):.2f}", lw=2)
        plt.scatter(*one_point, color='red')
        plt.scatter(*second_point, color='green')

        plt.plot(mu_to_sigma_for_control_rate.mus, mu_to_sigma_for_control_rate.sigmas, label=f"Data", alpha=0.7)

        plt.legend()
        plt.xlabel("$\mu$")
        plt.ylabel("$\sigma$")
        plt.title("Line through two points")

        plt.show()

        plot_data_vs_line_fit([mu_to_sigma_for_control_rate], [line_fit_data], caller_test_case=self)

    def test_plot_mk801(self):
        lif_config = default_diffusion_lif_config.with_label("MK801")

        sg = SiegertGradients.default()

        one_point = (-40, 0)

        sigma = compute_sigma_necessary_for_given_rate_and_mean(mu=-55, r_target=0.05, lif_config=lif_config)

        print(f"intercept: {sg.firing_rate(-40 * mV, 0 * mV)}")
        print(f"Found point: {sg.firing_rate(-55 * mV, sigma)}")

        second_point = (-55, sigma / mV)

        line_fit_data = line_from_points(one_point, second_point)
        x = np.linspace(-65, -40, 1001)
        mu_to_sigma_mk801 = mu_to_sigma_for_constant_rate(lif_config=lif_config, r_target=0.05 * Hz,
                                                                     mu_lims=(-65, -40))

        # Plot
        plt.plot(x, line_fit_data.sigmas(x), label=f"y = {line_fit_data.slope:.2f}x - {np.abs(line_fit_data.intercept):.2f}", lw=3)
        plt.scatter(*one_point, color='red')
        plt.scatter(*second_point, color='green')

        plt.plot(mu_to_sigma_mk801.mus, mu_to_sigma_mk801.sigmas, label=f"Data", alpha=0.7)

        plt.legend()
        plt.xlabel("$\mu$")
        plt.ylabel("$\sigma$")
        plt.title(f"Line through two points {lif_config.label}")

        plt.show()

        plot_data_vs_line_fit([mu_to_sigma_mk801], [line_fit_data], caller_test_case=self)


    def test_check_solution_coming_from_one_point(self):
        target_rates = [0.05 * Hz, 0.18 * Hz]
        lif_config = default_diffusion_lif_config
        delta_mu = 0.7 * mV

        control_line, mk801_line, mu_v, sigma_v = fit_mu_and_sigma_two_rates_and_delta_mu(delta_mu, lif_config, target_rates)

        mu_v_mk801 = mu_v
        mu_v_control = mu_v_mk801 + delta_mu

        siegert_gradient = SiegertGradients.for_lif_config(lif_config)
        rate_mk801 = siegert_gradient.firing_rate(mu_v=mu_v_mk801, sigma_v=sigma_v)
        rate_control = siegert_gradient.firing_rate(mu_v=mu_v_control, sigma_v=sigma_v)

        print(f"mu_v MK-801 = {mu_v_mk801}, mu_v Control = {mu_v_control}, sigma_v = {sigma_v}")
        print(f"Predicted MK801 rate: {rate_mk801}, Control rate: {rate_control}")
        print(f"delta mu {delta_mu}: m mk801= {mk801_line.slope}, m control = {control_line.slope}"
              f"b mk801 = {mk801_line.intercept}, b control = {control_line.intercept}")

        plot_two_rates_and_delta_solutions(lif_config, delta_mu, control_line, mk801_line, mu_v_control,
                                                mu_v_mk801, sigma_v, caller_test_case=self)


    def test_show_influence_of_v_r_on_mean_sigma_scan_VRs(self):
        V_Rs = np.linspace(-80, -41, num=1000)
        lif_configs = [default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_V_R, V_R) for V_R in V_Rs]

        scan_lif_configs(lif_configs=lif_configs, x_axis=V_Rs, label=r"$V_R$", unit="mV", caller_test_case=self)

    def test_show_influence_of_v_r_on_mean_sigma_scan_tau_ms(self):
        tau_ms = np.linspace(1, 101, num=1000)
        lif_configs = [default_diffusion_lif_config.with_property(DiffusionLIFConfig.KEY_TAU_MEMBRANE, tau_m) for tau_m in tau_ms]

        scan_lif_configs(lif_configs=lif_configs, x_axis=tau_ms, label=r"$\tau_m$", unit="ms", caller_test_case=self)


    def test_quality_of_line_through_two_points_fit(self, lif_config = default_diffusion_lif_config):

        mus_to_sigmas_mk801, line_fit_data_mk801 = compute_data_and_fit(lif_config=lif_config.with_label("MK801"),
                                                                        r_target=0.05 * Hz)
        mus_to_sigmas_control, line_fit_data_control = compute_data_and_fit(
            lif_config=lif_config.with_label("Control"),
            r_target=0.18 * Hz)

        x = np.linspace(-65, -40, 1001)

        # Apply the line equation
        y = line_fit_data_control.slope * x + line_fit_data_control.intercept

        plt.plot(x, y,
                 label=f"y = {line_fit_data_control.slope:.2f}x - {np.abs(line_fit_data_control.intercept):.2f}", lw=5,
                 alpha=0.6)
        plt.plot(mus_to_sigmas_control.mus, mus_to_sigmas_control.sigmas, label="Data", lw=5, alpha=0.6)

        plt.legend()
        plt.xlabel("$\mu$")
        plt.ylabel("$\sigma$")
        plt.title(r"Data (pointwise $\{ (\mu, \sigma) search \}$) \\ vs \\ line computed through two points")
        plt.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

        show_plots_non_blocking()

        plot_data_vs_line_fit([mus_to_sigmas_mk801, mus_to_sigmas_control],
                              [line_fit_data_mk801, line_fit_data_control], caller_test_case=self)



if __name__ == '__main__':
    unittest.main()

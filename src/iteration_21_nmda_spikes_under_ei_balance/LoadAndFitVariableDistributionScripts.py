"""Load and fit one saved simulation-variable histogram at a time."""

from pathlib import Path

import matplotlib.pyplot as plt

from Plotting import show_plots_non_blocking
from iteration_21_nmda_spikes_under_ei_balance.SimulateNMDASpikesUnderEIBalanceScripts import (
    DEBUG_DISTRIBUTION_VARIABLES,
    fit_saved_debug_sample,
)


def load_and_fit_variable_distribution(
    fit_sample_path,
    variable_name,
    *,
    number_of_gaussians=2,
    output_path=None,
    show_plot=True,
):
    """Load, fit, and plot original observations saved by the simulation."""
    fit_sample_path = Path(fit_sample_path)
    if variable_name not in DEBUG_DISTRIBUTION_VARIABLES:
        raise ValueError(
            f"Unknown variable {variable_name!r}; choose from "
            f"{tuple(DEBUG_DISTRIBUTION_VARIABLES)}"
        )
    if not fit_sample_path.exists():
        raise FileNotFoundError(
            f"Raw fit sample not found: {fit_sample_path}. Run "
            "test_simulate_1_000_ampa_nmda_single_compartments with "
            f"saveDebugFile.enabled=True and the {variable_name} save flag enabled."
        )

    plot_path, fit = fit_saved_debug_sample(
        fit_sample_path,
        variable_name,
        number_of_gaussians=number_of_gaussians,
        output_path=output_path,
    )
    if show_plot:
        show_plots_non_blocking()
    else:
        plt.close("all")
    return plot_path, fit

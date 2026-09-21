import logging
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
from brian2 import Hz, second, nS, ms
from joblib import Parallel, delayed

from brian2 import BrianLogger

from src.Plotting import prepare_bigger_fonts, show_plots_non_blocking
from src.iteration_16.Preliminaries import compute_nmda_dv
from src.iteration_16.model import ConductanceDiffusionSimulationConfig, MeanfieldScaling, config_with_weak_synapses, \
    config_with_medium_synapses, wang_config_external_ampa_synapses, config_with_intermediate_synapses
from src.iteration_16.nmda_compartment_model import NMDASimulationWangCompartments
from src.iteration_16.simpy import load_solutions

BrianLogger.log_level_error()
BrianLogger.suppress_name("brian2.codegen")
BrianLogger.log_level_error()

def gen_plot_title_multi_compartment_run(config: ConductanceDiffusionSimulationConfig):

    return (f"{config.label} {" with " + config.scaling.value + " meanfield scaling of " + r"$\bar g_{\mathrm{N}}$" if config.scaling != MeanfieldScaling.NONE else ""}: ""\n"r"$w_{\mathrm{AMPA}} = $" f"{config.w_ampa / nS: .2f} (nS), "r"$w_{\mathrm{GABA}} = $" f"{config.w_gaba / nS: .2f} (nS) "
            r"$N_E=$"f"{config.N_E}, "r"$N_I=$"f"{config.N_I}, "r"$r_e=$"f"{config.r_e / Hz:.2f} Hz, "r"$r_i=$"f"{config.r_i / Hz:.2f} Hz, "r"$\gamma=$"f"{config.g_i0() / config.g_e0():.2f}")

def gen_plot_title(config: ConductanceDiffusionSimulationConfig):

    return (f"{config.label} with {config.k_comp} NMDA compartments: ""\n"r"$w_{\mathrm{AMPA}} = $" f"{config.w_ampa / nS: .2f} (nS), "r"$w_{\mathrm{GABA}} = $" f"{config.w_gaba / nS: .2f} (nS) "
            r"$N_E=$"f"{config.N_E}, "r"$N_I=$"f"{config.N_I}, "r"$r_e=$"f"{config.r_e / Hz:.2f} Hz, "r"$r_i=$"f"{config.r_i / Hz:.2f} Hz, "r"$\gamma=$"f"{config.g_i0() / config.g_e0():.2f}")

#import pandas as pd

def save_comparison_run(df: pd.DataFrame, config: ConductanceDiffusionSimulationConfig):
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    filename = (
        f"{config.label.replace(" ", "_")}_up_to_{df['k_comp'].max()}_compartments_mf_{config.scaling.value}_"
        f"{int(config.simulation_time/ms)}_ms_simulation.csv"
    )

    filepath = output_dir / filename

    df.to_csv(
        filepath,
        index=False,
        float_format="%.10f"
    )

    print(f"Saved results to {filepath}")

class CompartmentSimulationsWithFittedSolution(unittest.TestCase):

    def test_config_for_weak_configuration_loads(self):
        cfg = config_with_weak_synapses.with_property(N_E=1000)
        solutions = load_solutions(config=cfg)
        print(solutions)

        for gr, gamma in solutions:
            print(gr, gamma)
            object_under_test = cfg.with_fitted_solution(gr, gamma)
            self.assertAlmostEqual(gamma, object_under_test.g_i0() / object_under_test.g_e0())

            print(object_under_test.r_e)

    def test_for_albert(self):
        cfg = config_with_weak_synapses.with_property(N_E=10, simulation_time = 2 * second, seed=200, r_e=5 * Hz, r_i=5 * Hz)

        low_inhibition_ratio = cfg
        k_s = [1, 5, cfg.N_E]
        run_simulations_in_parallel_and_compare(low_inhibition_ratio, k_s)

    def test_for_plested(self):
        cfg = config_with_weak_synapses.with_property(N_E=2000, simulation_time = 2 * second, seed=200)
        solutions = load_solutions(config=cfg)
        gr, gamma = solutions[0]

        low_inhibition_ratio = cfg.with_fitted_solution(gr = gr, gamma=gamma)

        k_s = [1, 5, low_inhibition_ratio.N_E]
        run_simulations_in_parallel_and_compare(low_inhibition_ratio, k_s)

        config_weak_synapses_strong_scaling = low_inhibition_ratio.with_property(scaling=MeanfieldScaling.STRONG, label="Weak synapses, strong NMDA meanfield scaling")
        run_simulations_in_parallel_and_compare(config_weak_synapses_strong_scaling, k_s)

        config_weak_synapses_weak_scaling = low_inhibition_ratio.with_property(scaling=MeanfieldScaling.WEAK,
                                                                         label="Weak synapses, weak NMDA meanfield scaling")
        run_simulations_in_parallel_and_compare(config_weak_synapses_weak_scaling, k_s)

    def test_simulate_with_fitted_config_first_solution(self):
        cfg = config_with_weak_synapses.with_property(N_E=2000, simulation_time = 2 * second, seed=200)
        solutions = load_solutions(config=cfg)
        gr, gamma = solutions[0]

        low_inhibition_ratio = cfg.with_fitted_solution(gr = gr, gamma=gamma)

        k_s = [1, 10, 100, 500, low_inhibition_ratio.N_E]
        run_simulations_in_parallel_and_compare(low_inhibition_ratio, k_s)

        config_weak_synapses_strong_scaling = low_inhibition_ratio.with_property(scaling=MeanfieldScaling.STRONG, label="Weak synapses, strong NMDA meanfield scaling")
        run_simulations_in_parallel_and_compare(config_weak_synapses_strong_scaling, k_s)

        config_weak_synapses_weak_scaling = low_inhibition_ratio.with_property(scaling=MeanfieldScaling.WEAK,
                                                                         label="Weak synapses, weak NMDA meanfield scaling")
        run_simulations_in_parallel_and_compare(config_weak_synapses_weak_scaling, k_s)

    def test_simulate_with_fitted_config_second_solution(self):
        cfg = config_with_weak_synapses.with_property(N_E=2000, simulation_time = 2 * second, seed=200, dt=0.01 * ms)
        solutions = load_solutions(config=cfg)
        gr, gamma = solutions[1]

        high_inhibition_ratio = cfg.with_fitted_solution(gr = gr, gamma=gamma)

        k_s = [1, 10, 100, 500, high_inhibition_ratio.N_E]
        run_simulations_in_parallel_and_compare(high_inhibition_ratio, k_s)


        config_weak_synapses_strong_scaling = high_inhibition_ratio.with_property(scaling=MeanfieldScaling.STRONG, label="Weak synapses, strong NMDA meanfield scaling")
        run_simulations_in_parallel_and_compare(config_weak_synapses_strong_scaling, k_s)

        config_weak_synapses_weak_scaling = high_inhibition_ratio.with_property(scaling=MeanfieldScaling.WEAK,
                                                                         label="Weak synapses, weak NMDA meanfield scaling")
        run_simulations_in_parallel_and_compare(config_weak_synapses_weak_scaling, k_s)

    def test_simulate_with_fitted_strongsynapses(self):
        cfg = config_with_medium_synapses.with_property(N_E=1600, N_I= 400, simulation_time = 2 * second, seed=200, dt=0.01 * ms)
        solutions = load_solutions(config=cfg)
        gr, gamma = solutions[0]

        high_inhibition_ratio = cfg.with_fitted_solution(gr = gr, gamma=gamma)

        k_s = [1, 10, 100, 500, high_inhibition_ratio.N_E]
        run_simulations_in_parallel_and_compare(high_inhibition_ratio, k_s)


        config_moderate_synapses_strong_scaling = high_inhibition_ratio.with_property(scaling=MeanfieldScaling.STRONG, label="Weak synapses, strong NMDA meanfield scaling")
        run_simulations_in_parallel_and_compare(config_moderate_synapses_strong_scaling, k_s)

        config_weak_synapses_weak_scaling = high_inhibition_ratio.with_property(scaling=MeanfieldScaling.WEAK,
                                                                         label="Weak synapses, weak NMDA meanfield scaling")
        run_simulations_in_parallel_and_compare(config_weak_synapses_weak_scaling, k_s)

    def test_simulate_with_wang_external_synapses(self):
        cfg = wang_config_external_ampa_synapses.with_property(simulation_time = 2 * second, seed=200, dt=0.01 * ms)
        solutions = load_solutions(config=cfg)
        gr, gamma = solutions[1]

        high_inhibition_ratio = cfg.with_fitted_solution(gr = gr, gamma=gamma)

        k_s = [1, 10, 100, 500, high_inhibition_ratio.N_E]
        run_simulations_in_parallel_and_compare(high_inhibition_ratio, k_s)


        config_weak_synapses_strong_scaling = high_inhibition_ratio.with_property(scaling=MeanfieldScaling.STRONG, label="Moderate synapses, strong NMDA meanfield scaling")
        run_simulations_in_parallel_and_compare(config_weak_synapses_strong_scaling, k_s)

        config_weak_synapses_weak_scaling = high_inhibition_ratio.with_property(scaling=MeanfieldScaling.WEAK,
                                                                         label="Moderate synapses, weak NMDA meanfield scaling")
        run_simulations_in_parallel_and_compare(config_weak_synapses_weak_scaling, k_s)


    def test_try_simulation_with_numerical_instabilities(self):
        cfg = wang_config_external_ampa_synapses.with_property(N_E=2000, simulation_time=500 * ms, seed=200, dt=0.01 * ms, k_comp=100)
        solutions = load_solutions(config=cfg)
        gr, gamma = solutions[1]

        high_inhibition_ratio = cfg.with_fitted_solution(gr=gr, gamma=gamma)
        NMDASimulationWangCompartments.run_and_plot(high_inhibition_ratio)

    def test_create_statistics_for_compartments_wang_external_model(self, time=10*ms):
        config = wang_config_external_ampa_synapses.with_property(label="wang external second solution", simulation_time=time)
        self.test_create_statistics_for_compartments(config=config, solution_index=1)

    def test_create_statistics_for_compartments_wang_external_model_first_solution(self, time=10*ms):
        config = wang_config_external_ampa_synapses.with_property(label="wang external first solution", simulation_time=time)
        self.test_create_statistics_for_compartments(config=config, solution_index=0)

    def test_create_statistics_for_compartments_moderate_synapses_first_solution(self, time=10*ms):
        config = config_with_medium_synapses.with_property(label="medium synapses first solution", simulation_time=time)
        self.test_create_statistics_for_compartments(config=config, solution_index=0)


    def test_create_statistics_for_compartments_intermediate_synapses_first_solution(self, time=10*ms):
        config = config_with_intermediate_synapses.with_property(label="intermediate synapses second solution",
                                                           simulation_time=time)
        self.test_create_statistics_for_compartments(config=config, solution_index=0)

    def test_create_statistics_for_compartments_intermediade_synapses_second_solution(self, time=10*ms):
        config = config_with_intermediate_synapses.with_property(label="intermediate synapses second solution",
                                                           simulation_time=time)
        self.test_create_statistics_for_compartments(config=config, solution_index=1)

    def test_aggregate_long_run(self):
        time= 5*second
        try:
            self.test_create_statistics_for_compartments_moderate_synapses_first_solution(time)
        except Exception as e:
            print("test_create_statistics_for_compartments_moderate_synapses_first_solution failed")
            print(e)
        try:
            self.test_create_statistics_for_compartments_wang_external_model_first_solution(time)
        except Exception as e:
            print("test_create_statistics_for_compartments_wang_external_model_first_solution failed")
            print(e)
        try:
            self.test_create_statistics_for_compartments_wang_external_model(time)
        except Exception as e:
            print("test_create_statistics_for_compartments_wang_external_model failed")
            print(e)
        try:
            self.test_create_statistics_for_compartments_intermediate_synapses_first_solution()
        except Exception as e:
            print("test_create_statistics_for_compartments_intermediate_synapses_first_solution failed")
            print(e)
        try:
            self.test_create_statistics_for_compartments_intermediade_synapses_second_solution()
        except Exception as e:
            print("test_create_statistics_for_compartments_intermediade_synapses_second_solution failed")
            print(e)

    def test_create_statistics_for_compartments(self, config=config_with_weak_synapses, solution_index=1, start_from_item=0):

        for scaling in [MeanfieldScaling.NONE, MeanfieldScaling.WEAK, MeanfieldScaling.STRONG]:
            # interested in the evolution of mean membrane v and variance membrane v for a configuration and an increase
            # in number of compartments. Up to 500, lets say
            config = config.with_property(N_E=2000, dt=0.01 * ms, scaling=scaling)
            solutions = load_solutions(config=config)
            gr, gamma = solutions[solution_index]

            base_config = config.with_fitted_solution(gr=gr, gamma=gamma)

            def comp_stats(config: ConductanceDiffusionSimulationConfig):
                run_results = NMDASimulationWangCompartments.run(config, detailed_statistics=False)
                mean_soma_v, var_soma_v = run_results.mean_variance_soma_v()
                mean_soma_v_no_nmda, var_soma_v_no_nmda = run_results.mean_variance_soma_v_no_nmda()
                return {
                    "k_comp": config.k_comp,
                    "mean_soma_v": mean_soma_v,
                    "var_soma_v": var_soma_v,
                    "mean_soma_v_no_nmda": mean_soma_v_no_nmda,
                    "var_soma_v_no_nmda": var_soma_v_no_nmda
                }

            configs = [base_config.with_property(k_comp = k_comp) for k_comp in range(1, 501)]
            results_all = []

            chunk_size = 50

            for i in range(start_from_item, len(configs), chunk_size):
                chunk = configs[i:i + chunk_size]

                results_chunk = Parallel(n_jobs=-1, verbose=10)(
                    delayed(comp_stats)(config) for config in chunk
                )

                results_all.extend(results_chunk)

                # Save incremental progress
                df = pd.DataFrame(results_all).sort_values("k_comp")
                save_comparison_run(df, config=base_config)

                print(f"Saved progress: {min(i + chunk_size, len(configs))}/{len(configs)}")

            prepare_bigger_fonts()
            fig, (ax_mean, ax_std) = plt.subplots(
                1, 2,
                figsize=(12, 8),
                sharex=True
            )

            # Mean membrane voltage
            ax_mean.plot(
                df["k_comp"],
                df["mean_soma_v"],
                label="NMDA"
            )

            ax_mean.plot(
                df["k_comp"],
                df["mean_soma_v_no_nmda"],
                linestyle=":",
                label="No NMDA"
            )

            ax_mean.set_title("Mean soma membrane voltage")
            ax_mean.set_xlabel(r"\# NMDA compartments")
            ax_mean.set_ylabel("Mean $V_m$ (mV)")
            ax_mean.legend()

            # Standard deviation (convert from variance)
            ax_std.plot(
                df["k_comp"],
                df["var_soma_v"],
                label="NMDA"
            )

            ax_std.plot(
                df["k_comp"],
                df["var_soma_v_no_nmda"],
                linestyle=":",
                label="No NMDA"
            )

            ax_std.set_title("Variance of soma membrane voltage")
            ax_std.set_xlabel(r"\# NMDA compartments")
            ax_std.set_ylabel("Var($V_m$) (mV)")
            ax_std.legend()

            fig.suptitle("Mean and Variance of the Somatic Membrane Potential \n in the Wang Model with NMDA Compartments \n"
                         f"{gen_plot_title_multi_compartment_run(config=base_config)}")
            plt.tight_layout()
            show_plots_non_blocking(caller_test_case=self)

    def test_plot_existing_simulation(self, file="weak synapses_up_to_500_compartments_mf_none_2000_ms_simulation", config=config_with_weak_synapses, solution_index=0):
        output_dir = Path("results")

        df = pd.read_csv(output_dir / f"{file}.csv")

        from scipy.signal import savgol_filter

        prepare_bigger_fonts()

        fig, (ax_mean, ax_std) = plt.subplots(
            1, 2,
            figsize=(12, 8),
            sharex=True
        )

        ax_mean.plot(df["k_comp"], df["mean_soma_v"], label="NMDA")
        ax_mean.plot(
            df["k_comp"],
            df["mean_soma_v_no_nmda"],
            ":",
            label="No NMDA"
        )

        ax_std.plot(df["k_comp"], savgol_filter(df["var_soma_v"],window_length=31,  # must be odd
            polyorder=3) , label="NMDA")
        ax_std.plot(
            df["k_comp"],
            savgol_filter(df["var_soma_v_no_nmda"], window_length=31,  # must be odd
                          polyorder=3),
            ":",
            label="No NMDA"
        )

        ax_std.set_title("Variance of soma membrane voltage")
        ax_std.set_xlabel(r"\# NMDA compartments")
        ax_std.set_ylabel("Var($V_m$) (mV)")
        ax_std.legend()

        config = config.with_property(N_E=2000, simulation_time=15 * second, seed=200, dt=0.01 * ms)
        solutions = load_solutions(config=config)
        gr, gamma = solutions[solution_index]

        base_config = config.with_fitted_solution(gr=gr, gamma=gamma)

        fig.suptitle(
            "Mean and Variance of the Somatic Membrane Potential \n in the Wang Model with NMDA Compartments \n"
            f"{gen_plot_title_multi_compartment_run(config=base_config)}")
        plt.tight_layout()

        ax_mean.legend()
        ax_std.legend()

        plt.tight_layout()
        plt.show()

    def test_check_one_compartment_is_mean_field_model_of_max_compartments(self):
        config = config_with_medium_synapses.with_property(N_E=2000, dt=0.01 * ms, scaling=MeanfieldScaling.WEAK, simulation_time=2 * second, seed=200)
        solutions = load_solutions(config=config)
        gr, gamma = solutions[0]

        config = config.with_fitted_solution(gr=gr, gamma=gamma)

        k_s = [1, 1000, 2000]
        results = run_simulations_in_parallel_and_compare(config, k_s, plot_comparrison=True)

        prepare_bigger_fonts()
        fig = plt.figure(figsize=(12, 8))
        outer = fig.add_gridspec(2, 1, height_ratios=[1, 1])

        # =========================================================
        # ROW 1: VOLTAGES (all k on same axes)
        # =========================================================
        ax_v = fig.add_subplot(outer[0])

        for k in k_s:
            r = results[k]

            t = r.neuron_monitor.t
            v = r.neuron_monitor.v[0]

            ax_v.plot(t, v, label=f"k={k}", alpha=0.6)

        last_result = results[k_s[-1]]
        baseline_v_m = last_result.neuron_monitor.v[1]

        ax_v.plot(last_result.neuron_monitor.t, baseline_v_m, label=f"No NMDA", linestyle=":", color="blue")
        ax_v.set_title("$V_m$ - Membrane voltage comparison")
        ax_v.set_ylabel("$V_m$ (mV)")
        # ax_v.legend()

        ax_v.legend()

        ax_delta_v = fig.add_subplot(outer[1])

        for k in k_s:
            r = results[k]

            t = r.neuron_monitor.t
            v = r.neuron_monitor.v[0] - baseline_v_m

            ax_delta_v.plot(t, v, label=f"k={k}", alpha=0.6)

        ax_delta_v.set_title(r"$\Delta V_m = V_m(t) - V_{m, \mathrm{No NMDA}}(t)$ ")
        ax_delta_v.set_ylabel(r"$ \Delta V_m$ (mV)")
        ax_delta_v.legend()

        fig.suptitle("Verify meanfield limit")

        fig.tight_layout()
        prepare_bigger_fonts()
        show_plots_non_blocking()

    def test_what_is_the_dv_of_one_nmda_spike(self):
        config = config_with_medium_synapses.with_property(N_E=2000, dt=0.01 * ms, scaling=MeanfieldScaling.WEAK,
                                                           simulation_time=5 * second, seed=200)
        solutions = load_solutions(config=config)
        gr, gamma = solutions[0]
        print(f"NMDA dv before: {compute_nmda_dv(config.with_property(r_e = 0 * Hz, r_i = 0 * Hz))}")
        config = config.with_fitted_solution(gr=gr, gamma=gamma)
        g_e0 = config.g_e0()
        g_i0 = config.g_i0()
        g_tot = config.g_L + g_e0 + g_i0
        config_without_ampa_and_gaba_spikes = config.with_property(r_e = 0 * Hz, r_i = 0 * Hz)
        print(f"NMDA dv at rest: {compute_nmda_dv(config_without_ampa_and_gaba_spikes)}")
        config_under_shunt = config_without_ampa_and_gaba_spikes.with_property(N_E=2000, g_L = g_tot)
        print(f"NMDA dv under shunt rest: {compute_nmda_dv(config_under_shunt)}")



    def test_test_plot_existing_simulations(self):
        # self.test_plot_existing_simulation(file="medium_synapses_first_solution_up_to_500_compartments_mf_none_5000_ms_simulation", config=config_with_medium_synapses)
        # self.test_plot_existing_simulation(file="wang_external_first_solution_up_to_500_compartments_mf_none_5000_ms_simulation", config=wang_config_external_ampa_synapses, solution_index=0)
        # self.test_plot_existing_simulation(file="wang_external_second_solution_up_to_500_compartments_mf_none_5000_ms_simulation", config=wang_config_external_ampa_synapses, solution_index=1)


        self.test_plot_existing_simulation(file="weak synapses_up_to_500_compartments_mf_none_2000_ms_simulation", config=config_with_weak_synapses, solution_index=0)


def run_simulations_in_parallel_and_compare(base_config: ConductanceDiffusionSimulationConfig, k_s, plot_comparrison=True):
    #run_one = lambda k: NMDASimulationWangCompartments.run_and_plot(base_config.with_property(k_comp=k), title=gen_plot_title(base_config.with_property(k_comp=k)))
    run_one = lambda k: NMDASimulationWangCompartments.run_and_plot(base_config.with_property(k_comp=k), detailed_statistics=plot_comparrison, title=gen_plot_title(base_config.with_property(k_comp=k)), testing=False)

    results = Parallel(n_jobs=-2)(
        delayed(run_one)(k) for k in k_s
    )

    result_by_k = dict(zip(k_s, results))

    if plot_comparrison:
        plot_k_sweep_results(result_by_k, k_s, experiment_title=gen_plot_title_multi_compartment_run(base_config))

    return result_by_k


if __name__ == '__main__':
    unittest.main()

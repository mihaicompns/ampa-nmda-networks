import unittest
import numpy as np
from brian2 import ms, Hz, nS

from iteration_16.model import ConductanceDiffusionSimulationConfig, WANG_MODEL_FOR_FULL_NMDA_INPUT
from iteration_16.simulation import WangSimulation, plot


class ConductanceDiffusionTestCase(unittest.TestCase):

    def test_default_config_runs_with_zero_spikes(self):
        config = ConductanceDiffusionSimulationConfig(
            simulation_time=10 * ms,
            dt=0.1 * ms,
            seed=12345,
        )

        result = WangSimulation.run(config)

        assert len(result.time_ms) > 0

        plot(result)

        self.assertEqual(100, len(result.time_ms))
        self.assertEqual(100, len(result.membrane_voltage_mV))
        self.assertEqual(100, len(result.g_ampa_nS))
        self.assertEqual(100, len(result.g_gaba_nS))
        self.assertEqual(100, len(result.s_nmda))
        self.assertEqual(100, len(result.x_nmda))

        self.assertEqual(0, len(result.ampa_presyn_spikes))
        self.assertEqual(0, len(result.gaba_presyn_spikes))
        self.assertEqual(0, len(result.nmda_presyn_spikes))

        np.testing.assert_allclose(
            result.membrane_voltage_mV,
            -70.0,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            result.g_ampa_nS,
            0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            result.g_gaba_nS,
            0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            result.s_nmda,
            0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            result.x_nmda,
            0.0,
            atol=1e-12,
        )

    def test_a_diffusion_simulation_runs(self):
        config = ConductanceDiffusionSimulationConfig(
            simulation_time=10 * ms,
            dt=0.1 * ms,
            seed=12345,
            r_e=5 * Hz,
            r_i=5 * Hz,
            r_n=5 * Hz
        )

        result = WangSimulation.run(config)

        assert len(result.time_ms) > 0

        plot(result)

        self.assertEqual(100, len(result.time_ms))
        self.assertEqual(100, len(result.membrane_voltage_mV))
        self.assertEqual(100, len(result.g_ampa_nS))
        self.assertEqual(100, len(result.g_gaba_nS))
        self.assertEqual(100, len(result.s_nmda))
        self.assertEqual(100, len(result.x_nmda))

        self.assertEqual(-69.86652233080414, result.membrane_voltage_mV.max())
        self.assertEqual(-69.96326659099682, result.membrane_voltage_mV.mean())
        self.assertEqual(0.05118931893115729, result.membrane_voltage_mV.std())

        self.assertEqual(1, len(result.ampa_presyn_spikes), "this seed is selected to have only 1 ampa spike")
        self.assertEqual(0, len(result.gaba_presyn_spikes))
        self.assertEqual(0, len(result.nmda_presyn_spikes))

        np.testing.assert_allclose(
            result.membrane_voltage_mV[: 50],
            -70.0,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            result.g_ampa_nS[: 50],
            0.0,
            atol=1e-12,
        )

        self.assertEqual(0.5, result.g_ampa_nS.max())

        np.testing.assert_allclose(
            result.g_gaba_nS[-1],
            0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            result.s_nmda[-1],
            0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            result.x_nmda[-1],
            0.0,
            atol=1e-12,
        )

    def test_one_ampa_spike(self):
        config = ConductanceDiffusionSimulationConfig(
            simulation_time=100 * ms,
            dt=0.1 * ms,
            ampa_spike_times=np.array([50]),
            r_i = 0 * Hz,
            r_n = 0 * Hz,
            seed=12345)

        result = WangSimulation.run(config)
        plot(result)

        self.assertEqual(1, len(result.ampa_presyn_spikes))
        self.assertEqual(0, len(result.gaba_presyn_spikes))
        self.assertEqual(0, len(result.nmda_presyn_spikes))

        self.assertIsNone(result.gaba_spike_delta_v())
        self.assertIsNone(result.nmda_spike_delta_v())

        self.assertAlmostEqual(0.10485572212694194, result.ampa_spike_delta_v())

    def test_one_gaba_spike(self):
        config = ConductanceDiffusionSimulationConfig(
            simulation_time=100 * ms,
            dt=0.1 * ms,
            gaba_spike_times=np.array([50]),
            r_e=0 * Hz,
            r_n=0 * Hz,
            seed=12345)

        result = WangSimulation.run(config)
        plot(result)

        self.assertEqual(0, len(result.ampa_presyn_spikes))
        self.assertEqual(1, len(result.gaba_presyn_spikes))
        self.assertEqual(0, len(result.nmda_presyn_spikes))

        self.assertIsNone(result.ampa_spike_delta_v())
        self.assertIsNone(result.nmda_spike_delta_v())
        self.assertAlmostEqual(-0.05025750639670434, result.gaba_spike_delta_v())


    def test_one_nmda_spike(self):
        config = ConductanceDiffusionSimulationConfig(
            simulation_time=200 * ms,
            model=WANG_MODEL_FOR_FULL_NMDA_INPUT,
            dt=0.1 * ms,
            nmda_spike_times=np.array([20]),
            r_e=0 * Hz,
            r_i=0 * Hz,
            seed=12345)

        result = WangSimulation.run(config)
        plot(result)

        self.assertEqual(0, len(result.ampa_presyn_spikes))
        self.assertEqual(0, len(result.gaba_presyn_spikes))
        self.assertEqual(1, len(result.nmda_presyn_spikes))

        self.assertIsNone(result.ampa_spike_delta_v())
        self.assertIsNone(result.gaba_spike_delta_v())

        self.assertAlmostEqual(1.2946114420009067, result.nmda_spike_delta_v())

    def test_nmda_spike_with_full_activation_has_to_be_considerably_lerger(self):
        usual_config = ConductanceDiffusionSimulationConfig(
            simulation_time=200 * ms,
            dt=0.1 * ms,
            nmda_spike_times=np.array([20]),
            r_e=0 * Hz,
            r_i=0 * Hz,
            seed=12345)

        result = WangSimulation.run_and_plot(usual_config)
        full_activation_result = WangSimulation.run_and_plot(usual_config.with_property(model=WANG_MODEL_FOR_FULL_NMDA_INPUT))

        print("Full NMDA ", full_activation_result.nmda_spike_delta_v())
        print("Usual NMDA ", result.nmda_spike_delta_v())

        self.assertLess(result.nmda_spike_delta_v(), full_activation_result.nmda_spike_delta_v())


    def test_can_change_one_property(self):
        config = ConductanceDiffusionSimulationConfig(
            simulation_time=100 * ms,
            dt=0.1 * ms,
            ampa_spike_times=np.array([50]),
            r_i = 0 * Hz,
            r_n = 0 * Hz,
            seed=12345)

        result = WangSimulation.run(config)
        plot(result)

        copy_config = config.with_property(w_ampa =1.0 * nS)
        result_of_copy_config = WangSimulation.run(copy_config)
        plot(result_of_copy_config)

        print(result.ampa_spike_delta_v())
        print(result_of_copy_config.ampa_spike_delta_v())

        self.assertEqual(0.10485572212694194, result.ampa_spike_delta_v())
        self.assertEqual(0.20952561298761907, result_of_copy_config.ampa_spike_delta_v())



if __name__ == '__main__':
    unittest.main()

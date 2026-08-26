import unittest
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from brian2 import have_same_dimensions, farad, meter, ohm, second, msecond, uF, cm, um, siemens, ms, mvolt, volt, \
    ufarad, us, mm, uvolt
from brian2.units.allunits import pampere, ampere, nampere, mampere, uampere
from numpy.testing import assert_array_equal

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ITERATION_SRC = SRC_ROOT / "iteration_19_tapered_dendrites"
for path in (REPO_ROOT, SRC_ROOT, ITERATION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Plotting import show_plots_non_blocking
from iteration_19_tapered_dendrites.data import CableParameters, to_SI


def cable_params():

    return CableParameters.from_SI(
        c_m=0.01,          # F/m²
        rm=2000,           # ohm*m²
        ra=1,              # ohm*m
        L=500e-6,          # m
        N=200,
        r0=2e-6,
        I_e=1.5e-12
    )

def assert_close(a, b):
    assert np.isclose(
        float(a / b),
        1.0
    ), f"{a} != {b}"

def assert_quantity_equal(a, b):
    assert have_same_dimensions(a, b)
    assert np.isclose(float(a / b), 1.0)

class DataClassesTestCase(unittest.TestCase):

    @staticmethod
    def test_override_equals_rebuild():
        params = CableParameters.from_SI(
            c_m=0.01,
            rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )
        p1 = params.with_SI_properties(rm=4000)

        p2 = CableParameters.from_SI(
            c_m=0.01,
            rm=4000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        assert p1.tau == p2.tau
        assert p1.b == p2.b

    def test_from_SI_units(self):
        p = CableParameters.from_SI(
            c_m=0.01,
            rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        assert have_same_dimensions(
            p.c_m,
            farad / meter ** 2
        )

        assert have_same_dimensions(
            p.rm,
            ohm * meter ** 2
        )

        assert have_same_dimensions(
            p.tau,
            second
        )

    @staticmethod
    def test_gL():
        p = CableParameters.from_SI(
            c_m=0.01,
            rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        assert_close(
            p.gL,
            1 / p.rm
        )

    @staticmethod
    def test_tau():
        p = CableParameters.from_SI(
            c_m=0.01,
            rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        assert_close(
            p.tau,
            p.rm * p.c_m
        )

    @staticmethod
    def test_dx():
        p = CableParameters.from_SI(
            c_m=0.01,
            rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        assert_close(
            p.dx,
            p.L / (p.N - 1)
        )

    @staticmethod
    def test_b():
        p = CableParameters.from_SI(
            c_m=0.01,
            rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        expected = (
                p.r_at_0 /
                (2 * p.c_m * p.ra)
        )

        assert_close(
            p.b,
            expected
        )

    @staticmethod
    def test_override():
        p = CableParameters.from_SI(
            c_m=0.01,
            rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        q = p.with_SI_properties(
            rm=4000
        )

        assert_close(
            q.rm,
            4000 * ohm * meter ** 2
        )

        assert_close(
            q.tau,
            q.rm * q.c_m
        )

    @staticmethod
    def test_override_does_not_modify_original():
        p = CableParameters.from_SI(
            c_m=0.01,
            rm=2000,
            ra=1,
            L=500e-6,
            N=200,
            r0=2e-6,
            I_e=1.5e-12
        )

        old = p.rm

        q = p.with_SI_properties(
            rm=5000
        )

        assert_close(
            p.rm,
            old
        )

        assert_close(
            q.rm,
            5000 * ohm * meter ** 2
        )

    def test_from_units_to_isi(self):
        p = CableParameters(c_m=1 * uF / cm **2, rm=2 * 1E4 * ohm * cm ** 2)

        self.assertAlmostEqual(20, p.tau / msecond)
        self.assertAlmostEqual(0.02, p.tau / second)
        numerical = p.to_numerical()
        self.assertAlmostEqual(0.02, numerical.tau)

    def test_lamdb(self):
        Rm = 2 * 1E4 * ohm * cm ** 2

        p = CableParameters(c_m=1 * uF / cm ** 2,
                            rm=Rm,
                            gL=1 / Rm,
                            ra=100 * ohm * cm,
                            L=500.0 * um,
                            N=101,
                            r0=2 * um,
                            I_e=150 * pampere)


        self.assertTrue(have_same_dimensions(p.lambd(), meter))
        self.assertEqual(np.sqrt(2) / 10 * cm, p.lambd())
        self.assertAlmostEqual(np.sqrt(2) * 1E-3, p.lambd() / meter)
        self.assertAlmostEqual(np.sqrt(2) * 1E-3, p.to_numerical().lambd())

    def test_r_lambda(self):
        Rm = 2 * 1E4 * ohm * cm ** 2

        p = CableParameters(c_m=1 * uF / cm ** 2,
                            rm=Rm,
                            gL=1 / Rm,
                            ra=100 * ohm * cm,
                            L=500.0 * um,
                            N=101,
                            r0=2 * um,
                            I_e=150 * pampere)

        self.assertTrue(have_same_dimensions(p.R_lambda(), ohm))
        self.assertEqual(p.R_lambda(), p.ra * p.lambd() / (np.pi * p.r0**2))
        p.lambd()

    def test_numbers_used_in_simulation(self):
        Rm = 2 * 1E4 * ohm * cm ** 2

        p = CableParameters(c_m=1 * uF / cm ** 2,
                            rm= Rm,
                            gL = 1 / Rm,
                            ra = 100 * ohm * cm,
                            L = 500.0 * um,
                            N = 101,
                            r0 = 2 * um,
                            I_e = 150 * pampere)

        p_si = p.to_numerical()

        self.assertAlmostEqual(0.01, p_si.c_m)  # F/m²
        self.assertAlmostEqual(2.0, p_si.rm)  # Ω·m²
        self.assertAlmostEqual(1.0, p_si.ra)  # Ω·m
        self.assertAlmostEqual(500e-6, p_si.L)  # m
        self.assertAlmostEqual( 5e-6, p_si.dx)  # m
        self.assertAlmostEqual(2e-6, p_si.r_at_0)  # m
        self.assertAlmostEqual( 1.5e-12, p_si.I_e) # Ampere

        p_from_numerical = CableParameters.from_numerical(p_si)

        self.assertAlmostEqual(1, p_from_numerical.c_m / (uF / cm ** 2))
        self.assertAlmostEqual(2e4, p_from_numerical.rm / (ohm * cm ** 2))
        self.assertAlmostEqual(5e-5, p_from_numerical.gL / (siemens / cm ** 2))
        self.assertAlmostEqual(100.0, p_from_numerical.ra / (ohm * cm))
        self.assertAlmostEqual(500.0, p_from_numerical.L / um)
        self.assertAlmostEqual(5.0, p_from_numerical.dx / um)
        self.assertAlmostEqual(20.0, p_from_numerical.tau / ms)
        self.assertAlmostEqual(2.0, p_from_numerical.r_at_0 / um)
        self.assertAlmostEqual(1.0, p_from_numerical.b / (cm ** 2 / second))
        self.assertAlmostEqual(1E5, p_from_numerical.b / (um ** 2 / ms))
        self.assertAlmostEqual(150.0, p_from_numerical.I_e / pampere)

    def test_r_lambda_and_prefactor(self):
        rm = 2 * 1E4 * ohm * cm ** 2

        p = CableParameters(
            c_m=1 * uF / cm ** 2,
            rm=rm,
            gL=1 / rm,
            ra=100 * ohm * cm,
            L=500.0 * um,
            N=101,
            r0=2 * um,
            I_e=150 * pampere
        )

        p_si = p.to_numerical()

        # Base SI conversions
        self.assertAlmostEqual(0.01, p_si.c_m)  # F/m²
        self.assertAlmostEqual(2.0, p_si.rm)  # Ω·m²
        self.assertAlmostEqual(0.5, p_si.gL)  # S/m²
        self.assertAlmostEqual(1.0, p_si.ra)  # Ω·m
        self.assertAlmostEqual(500e-6, p_si.L)  # m
        self.assertAlmostEqual(5e-6, p_si.dx)  # m
        self.assertAlmostEqual(2e-6, p_si.r_at_0)  # m
        self.assertAlmostEqual(1.5e-12, p_si.I_e)  # A

        # Derived cable quantities

        # tau_m = rm * cm
        tau_m = p_si.rm * p_si.c_m
        self.assertAlmostEqual(0.02, tau_m)  # s

        # lambda = sqrt(r0*rm/(2*ra))
        lambda_ = np.sqrt(
            p_si.r_at_0 * p_si.rm / (2 * p_si.ra)
        )
        self.assertAlmostEqual(1.414213562e-3, lambda_)  # m
        # R_lambda = rm/(2*pi*r0*lambda)
        R_lambda = (
                p_si.rm /
                (2 * np.pi * p_si.r_at_0 * lambda_)
        )
        self.assertAlmostEqual(1.125395, R_lambda * 1E-8, places=6)  # ohm

        # Maximum prefactor at t_rel = 0.00080 ms
        t_rel = 0.00080e-3  # seconds
        prefactor = (
                p_si.I_e * R_lambda /
                np.sqrt(4 * np.pi * t_rel / tau_m)
        )

        self.assertAlmostEqual(0.753, prefactor, places=3)  # V


        tau_m = tau_m * second / ms
        t_rel = t_rel * second / ms
        prefactor = (
                p_si.I_e * R_lambda /
                np.sqrt(4 * np.pi * t_rel / tau_m)
        )

        self.assertAlmostEqual(0.753, prefactor, places=3)  # V

    def test_plot_prefactor(self):
        rm = 2 * 1E4 * ohm * cm ** 2

        p = CableParameters(
            c_m=1 * uF / cm ** 2,
            rm=rm,
            gL=1 / rm,
            ra=100 * ohm * cm,
            L=500.0 * um,
            N=101,
            r0=2 * um,
            I_e=150 * pampere
        ).to_numerical()

        Ie = p.I_e
        R_lambda = p.R_lambda()
        tau_m = p.tau

        # Time range
        t0 = 0.00080e-3 * second / ms  # 0.00080 ms = 8e-7 s
        t_end = t0 + 10  # t0 + 10 ms

        times = np.linspace(t0, t_end, 1000)

        # Prefactor
        prefactor = Ie * R_lambda / np.sqrt(4 * np.pi * times / tau_m)

        print("XXXX ", p.lambd() ** 2 / tau_m)

        # Plot
        plt.figure(figsize=(7, 4))
        plt.plot(times, prefactor * volt/mvolt, lw=2)

        plt.xlabel(r"$t$ (ms)")
        plt.ylabel(r"$I_e R_\lambda/\sqrt{4\pi t/\tau_m}$ (mV)")
        plt.title("Tuckwell prefactor")

        show_plots_non_blocking()

    @staticmethod
    def test_x_roundtrip():
        p = CableParameters(
            L=500 * um,
            N=101,
            x=np.linspace(0, 500, 101) * um
        )

        p_si = p.to_numerical()

        assert isinstance(p_si.x, np.ndarray)
        assert np.isclose(p_si.x[-1], 500e-6)

        p2 = CableParameters.from_numerical(p_si)

        assert have_same_dimensions(p2.x, meter)
        assert np.allclose(
            p.x / meter,
            p2.x / meter
        )

    def test_change_N_recomputes_spatial_quantities(self):
        Rm = 2 * 1E4 * ohm * cm ** 2

        # Initial coarse discretization
        p11 = CableParameters(
            c_m=1 * uF / cm ** 2,
            rm=Rm,
            gL=1 / Rm,
            ra=100 * ohm * cm,
            L=500.0 * um,
            N=11,
            r0=2 * um,
            I_e=150 * pampere,
        )

        # Check derived quantities exist
        self.assertIsNotNone(p11.gL)
        self.assertIsNotNone(p11.tau)
        self.assertIsNotNone(p11.b)
        self.assertIsNotNone(p11.dx)

        # Create x manually if this is not done in __post_init__
        x11 = np.linspace(0, p11.L, p11.N)

        # Store physical quantities
        c_m_11 = p11.c_m
        Rm_11 = p11.rm
        gL_11 = p11.gL
        tau_11 = p11.tau
        b_11 = p11.b
        L_11 = p11.L
        r0_11 = p11.r0

        # Refine discretization
        p101 = p11.with_property(N=101)

        # Recompute x after changing N
        x101 = np.linspace(0, p101.L, p101.N)

        p101 = p101.with_property(x=x101)

        # --- Physical parameters must be unchanged ---
        self.assertAlmostEqual(1, p101.c_m / c_m_11)
        self.assertAlmostEqual(1, p101.rm / Rm_11)
        self.assertAlmostEqual(1, p101.gL / gL_11)
        self.assertAlmostEqual(1, p101.tau / tau_11)
        self.assertAlmostEqual(1, p101.b / b_11)
        self.assertAlmostEqual(1, p101.L / L_11)
        self.assertAlmostEqual(1, p101.r_at_0 / r0_11)

        self.assertEqual(11, p11.N)
        self.assertEqual(101, p101.N)

        self.assertAlmostEqual(50, p11.dx / um)
        self.assertAlmostEqual(5, p101.dx / um)

        self.assertEqual(11, len(p11.x))
        self.assertEqual(101, len(p101.x))

        self.assertAlmostEqual(0, p11.x[0] / um)
        self.assertAlmostEqual(500, p11.x[-1] / um)
        self.assertAlmostEqual(0, p101.x[0] / um)
        self.assertAlmostEqual(500, p101.x[-1] / um)

class TestToSI(unittest.TestCase):

    def test_time_inference_without_unit(self):
        """Seconds, milliseconds, microseconds are automatically detected."""

        self.assertAlmostEqual(
            to_SI(1 * second),
            1.0
        )

        self.assertAlmostEqual(
            to_SI(1 * ms),
            1e-3
        )

        self.assertAlmostEqual(
            to_SI(500 * ms),
            0.5
        )

        self.assertAlmostEqual(
            to_SI(250 * us),
            250e-6
        )


    def test_length_inference_without_unit(self):
        """Length units should automatically convert to meters."""

        self.assertAlmostEqual(
            to_SI(1 * meter),
            1.0
        )

        self.assertAlmostEqual(
            to_SI(1 * cm),
            1e-2
        )

        self.assertAlmostEqual(
            to_SI(10 * mm),
            1e-2
        )

        self.assertAlmostEqual(
            to_SI(250 * um),
            250e-6
        )

        self.assertAlmostEqual(
            to_SI(5 * cm),
            0.05
        )


    def test_voltage_inference_without_unit(self):
        """Voltage conversion to volts."""

        self.assertAlmostEqual(
            to_SI(1 * volt),
            1.0
        )

        self.assertAlmostEqual(
            to_SI(1 * mvolt),
            1e-3
        )

        self.assertAlmostEqual(
            to_SI(1000 * mvolt),
            1.0
        )

        self.assertAlmostEqual(
            to_SI(50 * uvolt),
            50e-6
        )

        self.assertAlmostEqual(
            to_SI(-70 * mvolt),
            -0.07
        )


    def test_current_inference_without_unit(self):
        """Current conversion to amperes."""

        self.assertAlmostEqual(
            to_SI(1 * ampere),
            1.0
        )

        self.assertAlmostEqual(
            to_SI(1 * mampere),
            1e-3
        )

        self.assertAlmostEqual(
            to_SI(1 * pampere),
            1e-12
        )

        self.assertAlmostEqual(
            to_SI(150 * pampere),
            150e-12
        )

        self.assertAlmostEqual(
            to_SI(500 * nampere),
            500e-9
        )

    def test_current_density_inference_without_unit(self):
        """Current density conversion."""

        self.assertAlmostEqual(
            1.0,
            to_SI(1 * ampere / meter ** 2)
        )

        self.assertAlmostEqual(
            1e-2,
            to_SI(1 * uampere / cm ** 2)
        )

        # 10 nA / um² = 10 * 1e-9 A / (1e-6 m)^2 = 1e4 A/m²
        self.assertAlmostEqual(
            1e4,
            to_SI(10 * nampere / um ** 2)
        )

        self.assertAlmostEqual(
            150.0,
            to_SI(150 * pampere / (um ** 2))
        )

        self.assertAlmostEqual(
            5.0,
            to_SI(0.5 * mampere / cm ** 2)
        )

    def test_capacitance_density_inference_without_unit(self):
        """Membrane capacitance density."""

        self.assertAlmostEqual(
            to_SI(1 * farad / meter**2),
            1.0
        )

        self.assertAlmostEqual(
            to_SI(1 * ufarad / cm**2),
            1e-2
        )

        self.assertGreater(
            to_SI(10 * ufarad / cm**2),
            0
        )

        self.assertIsInstance(
            to_SI(5 * ufarad / cm**2),
            float
        )


    def test_resistance_density_inference_without_unit(self):
        """Membrane resistance conversion."""

        self.assertAlmostEqual(
            to_SI(1 * ohm * meter**2),
            1.0
        )

        self.assertAlmostEqual(
            to_SI(1e4 * ohm * cm**2),
            1.0
        )

        self.assertAlmostEqual(
            to_SI(2e4 * ohm * cm**2),
            2.0
        )

        self.assertGreater(
            to_SI(1000 * ohm * cm**2),
            0
        )


    def test_axial_resistivity_inference_without_unit(self):
        """Axial resistivity conversion."""

        self.assertAlmostEqual(
            to_SI(1 * ohm * meter),
            1.0
        )

        self.assertAlmostEqual(
            to_SI(100 * ohm * cm),
            1.0
        )

        self.assertAlmostEqual(
            to_SI(200 * ohm * cm),
            2.0
        )

        self.assertGreater(
            to_SI(50 * ohm * cm),
            0
        )


    def test_diffusion_coefficient_inference_without_unit(self):
        """Diffusion coefficient conversion."""

        self.assertAlmostEqual(
            to_SI(1 * meter**2 / second),
            1.0
        )

        self.assertAlmostEqual(
            to_SI(1 * cm**2 / second),
            1e-4
        )

        self.assertAlmostEqual(
            to_SI(100 * cm**2 / second),
            1e-2
        )

        self.assertGreater(
            to_SI(0.1 * cm**2 / second),
            0
        )


    def test_dimensionless_values(self):
        """Dimensionless values remain unchanged."""

        self.assertEqual(
            to_SI(1),
            1.0
        )

        self.assertEqual(
            to_SI(5),
            5.0
        )

        assert_array_equal(
            to_SI(np.array([1, 2, 3])),
            np.array([1, 2, 3])
        )

        self.assertEqual(
            to_SI(0),
            0.0
        )


    def test_array_conversion(self):
        """Arrays preserve shape and convert element-wise."""

        result = to_SI(
            np.array([1, 10, 100]) * cm
        )

        np.testing.assert_allclose(
            result,
            np.array([0.01, 0.1, 1.0])
        )

        self.assertEqual(
            result.shape,
            (3,)
        )

        self.assertIsInstance(
            result,
            np.ndarray
        )


    def test_none_input(self):
        self.assertIsNone(
            to_SI(None)
        )


if __name__ == '__main__':
    unittest.main()

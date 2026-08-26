import math
import unittest
import sys
from pathlib import Path

import numpy as np
from brian2 import (
    have_same_dimensions,
    farad,
    meter,
    ohm,
    second,
    uF,
    cm,
    um,
    ms,
    ufarad,
    us, cmeter, )
from brian2.units.allunits import (
    pampere,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ITERATION_SRC = SRC_ROOT / "iteration_19_tapered_dendrites"
for path in (REPO_ROOT, SRC_ROOT, ITERATION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from conical_data import (
    ConicalCableParameters,
    ConicalNumericalCableParameters,
    to_SI,
)
from iteration_19_tapered_dendrites.TaperredDendritesPDE import file_name_prefix_for_uniform, experiment_label_for_uniform


def assert_close(a, b, rtol=1e-10, atol=1e-15):
    np.testing.assert_allclose(
        float(a / b),
        1.0,
        rtol=rtol,
        atol=atol,
    )


class ConicalDataClassesTestCase(unittest.TestCase):

    # ================================================================
    # Factory
    # ================================================================

    @staticmethod
    def make_parameters(
            N=101,
            r0=2 * um,
            rL=0.5 * um,
            L=500 * um,
    ):
        Rm = 2e4 * ohm * cm ** 2

        return ConicalCableParameters(
            c_m=1 * uF / cm ** 2,
            rm=Rm,
            gL=1 / Rm,
            ra=100 * ohm * cm,
            L=L,
            N=N,
            r_at_0=r0,
            r_at_L=rL,
            I_e=150 * pampere,
        )

    # ================================================================
    # Basic construction
    # ================================================================

    def test_is_conical_type(self):
        p = self.make_parameters()

        self.assertIsInstance(
            p,
            ConicalCableParameters,
        )

        numerical = p.to_numerical()

        self.assertIsInstance(
            numerical,
            ConicalNumericalCableParameters,
        )

    def test_required_geometry(self):
        p = self.make_parameters()

        self.assertEqual(
            p.r_at_0,
            2 * um,
        )

        self.assertEqual(
            p.r_at_L,
            0.5 * um,
        )

        self.assertEqual(
            p.L,
            500 * um,
        )

    # ================================================================
    # Units
    # ================================================================

    def test_units(self):
        p = self.make_parameters()

        self.assertTrue(
            have_same_dimensions(
                p.c_m,
                farad / meter ** 2,
            )
        )

        self.assertTrue(
            have_same_dimensions(
                p.rm,
                ohm * meter ** 2,
            )
        )

        self.assertTrue(
            have_same_dimensions(
                p.ra,
                ohm * meter,
            )
        )

        self.assertTrue(
            have_same_dimensions(
                p.L,
                meter,
            )
        )

        self.assertTrue(
            have_same_dimensions(
                p.r_at_0,
                meter,
            )
        )

        self.assertTrue(
            have_same_dimensions(
                p.r_at_L,
                meter,
            )
        )

        self.assertTrue(
            have_same_dimensions(
                p.dx,
                meter,
            )
        )

        self.assertTrue(
            have_same_dimensions(
                p.tau,
                second,
            )
        )

    # ================================================================
    # Derived parameters
    # ================================================================

    def test_gL(self):
        p = self.make_parameters()

        assert_close(
            p.gL,
            1 / p.rm,
        )

    def test_tau(self):
        p = self.make_parameters()

        assert_close(
            p.tau,
            p.c_m * p.rm,
        )

    def test_dx(self):
        for N in [2, 3, 11, 101, 301]:

            p = self.make_parameters(N=N)

            assert_close(
                p.dx,
                p.L / (N - 1),
            )

    # ================================================================
    # Taper parameter
    # ================================================================

    def test_k_known_value(self):
        """
        r0 = 2 um
        rL = 0.5 um
        L  = 500 um

        k = (2 - 0.5)/(2*500) [1/um]
          = 0.0015 / um
          = 1500 / m
        """

        p = self.make_parameters()

        expected = 1500.0 / meter

        self.assertAlmostEqual(
            float(p.k / (1 / meter)),
            1500.0,
        )

    def test_k_units(self):
        p = self.make_parameters()

        self.assertTrue(
            have_same_dimensions(
                p.k,
                1 / meter,
            )
        )

    # ================================================================
    # Radius
    # ================================================================

    def test_radius_at_left_boundary(self):
        p = self.make_parameters()

        assert_close(
            p.radius(0 * um),
            p.r_at_0,
        )

    def test_radius_at_right_boundary(self):
        p = self.make_parameters()

        assert_close(
            p.radius(p.L),
            p.r_at_L,
        )

    def test_radius_is_linear(self):
        p = self.make_parameters()

        x = np.linspace(
            0,
            500,
            11,
        ) * um

        r = p.radius(x)

        expected = (
            p.r_at_0
            +
            (p.r_at_L - p.r_at_0)
            * x
            / p.L
        )

        np.testing.assert_allclose(
            r / um,
            expected / um,
        )

    def test_radius_midpoint(self):
        p = self.make_parameters()

        r = p.radius(250 * um)

        self.assertAlmostEqual(
            float(r / um),
            1.25,
        )

    def test_radius_uniform_change(self):
        p = self.make_parameters()

        x = np.linspace(
            0,
            500,
            101,
        ) * um

        r = p.radius(x)

        dr = np.diff(r / um)

        # Every spatial step has the same radius change.
        np.testing.assert_allclose(
            dr,
            np.ones(100) * dr[0],
        )

    # ================================================================
    # Lambda
    # ================================================================

    def test_lambda_at_left(self):
        p = self.make_parameters()
        r0 = p.r_at_0
        k = p.k

        expected = np.sqrt(
            p.r_at_0 * p.rm / (2 * p.ra * math.sqrt(1 + (k * r0) ** 2))
        )

        expected_approx = np.sqrt(
            p.r_at_0 * p.rm / (2 * p.ra)
        )

        assert_close(
            p.lambd(0 * um),
            expected,
        )

        self.assertAlmostEqual(math.sqrt(2) * 1E-3,  p.lambd(0 * um) / meter, places=6)
        self.assertAlmostEqual(math.sqrt(2) / 10,  p.lambd(0 * um) / cmeter, places=6)

        self.assertLess(p.lambd(0 * um) / meter, expected_approx / meter)

    def test_lambda_at_right(self):
        p = self.make_parameters()

        r0 = p.r_at_0
        k = p.k
        expected = np.sqrt(
            p.r_at_L * p.rm / (2 * p.ra * math.sqrt(1 + (k * r0) ** 2))
        )

        expected_approx = np.sqrt(p.r_at_L * p.rm / (2 * p.ra))

        assert_close(
            p.lambd(p.L),
            expected,
        )

        self.assertAlmostEqual(math.sqrt(2) / 20, p.lambd(p.L) / cmeter, places=6)
        self.assertAlmostEqual(1/ math.sqrt(2) * 1E-3, p.lambd(p.L) / meter, places=6, msg="Manuscript says 1/sqrt 2, but ignored 1 + k**2 r**2")
        self.assertLess(p.lambd(p.L) / meter, expected_approx / meter,
                               msg="Manuscript says 1/sqrt 2, but ignored 1 + k**2 r**2")

    def test_lambda_decreases_for_narrowing_cone(self):
        p = self.make_parameters()

        lam0 = p.lambd(0 * um)
        lamL = p.lambd(p.L)

        self.assertGreater(
            float(lam0 / meter),
            float(lamL / meter),
        )

    def test_lambda_known_values(self):
        """
        For the chosen parameters:

            lambda(r0) = sqrt(2e-6 * 2 / 2)
                       = sqrt(2e-6)
                       ~= 1.41421356e-3 m

        At rL = 0.5 um:

            lambda(rL)
            = sqrt(0.5e-6)
            ~= 7.07106781e-4 m
        """

        p = self.make_parameters()

        lambda_0 = p.lambd(0 * um)
        lambda_L = p.lambd(p.L)

        self.assertAlmostEqual(
            float(lambda_0 / meter),
            np.sqrt(2) * 1e-3,
        )

        self.assertAlmostEqual(
            float(lambda_L / meter),
            np.sqrt(0.5) * 1e-3,
        )

    # ================================================================
    # a coefficient
    # ================================================================

    def test_a_is_zero_for_cylinder(self):
        p = self.make_parameters(
            r0=2 * um,
            rL=2 * um,
        )

        self.assertAlmostEqual(
            float(p.a()),
            0.0,
        )

    def test_a_known_value(self):
        p = self.make_parameters()

        expected = (
            p.k
            * p.r_at_0
            /
            (
                p.c_m
                * p.ra
                * np.sqrt(
                1
                +
                (p.k * p.r_at_0) ** 2
                )
            )
        )

        assert_close(
            expected,
            p.a()
        )

        self.assertAlmostEqual(0.3, p.a() / (meter/second), places=5)

    def test_a_units(self):
        p = self.make_parameters()

        self.assertTrue(
            have_same_dimensions(
                p.a(),
                meter / second,
            )
        )

    # ================================================================
    # b(x)
    # ================================================================

    def test_b_is_array(self):
        p = self.make_parameters()

        b = p.b()

        self.assertIsInstance(
            b,
            np.ndarray,
        )

        self.assertEqual(
            len(b),
            p.N,
        )

    def test_b_at_left(self):
        p = self.make_parameters()

        expected = (
            p.r_at_0
            /
            (
                2
                * p.c_m
                * p.ra
                * np.sqrt(
                1
                +
                (p.k * p.r_at_0) ** 2
                )
            )
        )

        assert_close(
            p.b(0 * um),
            expected,
        )

    def test_b_at_right(self):
        p = self.make_parameters()

        expected = (
            p.r_at_L
            /
            (
                2
                * p.c_m
                * p.ra
                * np.sqrt(
                1
                +
                (p.k * p.r_at_0) ** 2
                )
            )
        )

        assert_close(
            p.b(p.L),
            expected,
        )

    def test_b_decreases_for_narrowing_cone(self):
        p = self.make_parameters()

        b = p.b()

        self.assertGreater(
            b[0],
            b[-1],
        )

    def test_b_is_linear(self):
        p = self.make_parameters()

        x = np.linspace(
            0,
            500,
            101,
        ) * um

        b = p.b(x)

        db = np.diff(b)

        np.testing.assert_allclose(
            db,
            np.ones(100) * db[0],
        )

    # ================================================================
    # Conversion to numerical
    # ================================================================

    def test_to_numerical_type(self):
        p = self.make_parameters()

        numerical = p.to_numerical()

        self.assertIsInstance(
            numerical,
            ConicalNumericalCableParameters,
        )

    def test_to_numerical_values(self):
        p = self.make_parameters()

        numerical = p.to_numerical()

        self.assertAlmostEqual(
            numerical.c_m,
            0.01,
        )

        self.assertAlmostEqual(
            numerical.rm,
            2.0,
        )

        self.assertAlmostEqual(
            numerical.gL,
            0.5,
        )

        self.assertAlmostEqual(
            numerical.ra,
            1.0,
        )

        self.assertAlmostEqual(
            numerical.L,
            500e-6,
        )

        self.assertAlmostEqual(
            numerical.dx,
            5e-6,
        )

        self.assertAlmostEqual(
            numerical.r_at_0,
            2e-6,
        )

        self.assertAlmostEqual(
            numerical.r_at_L,
            0.5e-6,
        )

        self.assertAlmostEqual(
            numerical.I_e,
            150e-12,
        )

    def test_numerical_k(self):
        p = self.make_parameters()

        numerical = p.to_numerical()

        self.assertAlmostEqual(
            numerical.k,
            1500.0,
        )

    def test_numerical_radius(self):
        p = self.make_parameters()

        numerical = p.to_numerical()

        self.assertAlmostEqual(
            numerical.radius(0),
            2e-6,
        )

        self.assertAlmostEqual(
            numerical.radius(500e-6),
            0.5e-6,
        )

    def test_numerical_lambda(self):
        p = self.make_parameters()

        numerical = p.to_numerical()

        expected = np.sqrt(
            numerical.r_at_0
            * numerical.rm
            /
            (2 * numerical.ra)
        )

        self.assertAlmostEqual(
            numerical.lambd(0),
            expected,
        )

    def test_numerical_a(self):
        p = self.make_parameters()

        numerical = p.to_numerical()

        expected = (
            numerical.k
            * numerical.r_at_0
            /
            (
                numerical.c_m
                * numerical.ra
                * np.sqrt(
                    1
                    +
                    (numerical.k * numerical.r_at_0) ** 2
                )
            )
        )

        self.assertAlmostEqual(
            numerical.a(),
            expected,
        )

    def test_numerical_b(self):
        p = self.make_parameters()

        numerical = p.to_numerical()

        b = numerical.b(
            numerical.x
        )

        expected = (
            numerical.radius(numerical.x)
            /
            (
                2
                * numerical.c_m
                * numerical.ra
                * np.sqrt(
                    1
                    +
                    (numerical.k * numerical.r_at_0) ** 2
                )
            )
        )

        np.testing.assert_allclose(
            b,
            expected,
        )

    # ================================================================
    # Round trip
    # ================================================================

    def test_round_trip(self):
        p = self.make_parameters()

        numerical = p.to_numerical()

        p2 = (
            ConicalCableParameters
            .from_numerical(numerical)
        )

        assert_close(
            p2.r_at_0,
            p.r_at_0,
        )

        assert_close(
            p2.r_at_L,
            p.r_at_L,
        )

        assert_close(
            p2.L,
            p.L,
        )

        assert_close(
            p2.c_m,
            p.c_m,
        )

        assert_close(
            p2.rm,
            p.rm,
        )

        assert_close(
            p2.ra,
            p.ra,
        )

        assert_close(
            p2.tau,
            p.tau,
        )

        np.testing.assert_allclose(
            p2.x / um,
            p.x / um,
        )

    # ================================================================
    # with_property
    # ================================================================

    def test_with_property_returns_conical_type(self):
        p = self.make_parameters()

        q = p.with_property(
            rL=1 * um
        )

        self.assertIsInstance(
            q,
            ConicalCableParameters,
        )

        self.assertEqual(
            q.r_at_L,
            1 * um,
        )

    def test_with_property_does_not_modify_original(self):
        p = self.make_parameters()

        q = p.with_property(
            rL=1 * um
        )

        self.assertEqual(
            p.r_at_L,
            0.5 * um,
        )

        self.assertEqual(
            q.r_at_L,
            1 * um,
        )

    def test_with_property_recomputes_dx(self):
        p = self.make_parameters(N=11)

        q = p.with_property(
            N=101,
        )

        self.assertAlmostEqual(
            float(p.dx / um),
            50.0,
        )

        self.assertAlmostEqual(
            float(q.dx / um),
            5.0,
        )

    def test_with_property_recomputes_x(self):
        p = self.make_parameters(N=11)

        q = p.with_property(
            N=101,
        )

        self.assertEqual(
            len(p.x),
            11,
        )

        self.assertEqual(
            len(q.x),
            101,
        )

        self.assertAlmostEqual(
            float(q.x[-1] / um),
            500,
        )

    def test_with_property_recomputes_tau(self):
        p = self.make_parameters()

        q = p.with_property(
            rm=4e4 * ohm * cm ** 2
        )

        self.assertAlmostEqual(
            float(q.tau / ms),
            20.0,
        )

    def test_with_property_recomputes_taper(self):
        p = self.make_parameters()

        q = p.with_property(
            rL=1 * um
        )

        self.assertNotEqual(
            float(p.k / (1 / meter)),
            float(q.k / (1 / meter)),
        )

    # ================================================================
    # with_SI_properties
    # ================================================================

    def test_with_SI_properties_returns_conical_type(self):
        p = self.make_parameters()

        q = p.with_SI_properties(
            rL=1e-6,
        )

        self.assertIsInstance(
            q,
            ConicalCableParameters,
        )

        self.assertAlmostEqual(
            float(q.r_at_L / um),
            1.0,
        )

    def test_with_SI_properties_radius(self):
        p = self.make_parameters()

        q = p.with_SI_properties(
            r0=3e-6,
            rL=1e-6,
        )

        self.assertAlmostEqual(
            float(q.r_at_0 / um),
            3.0,
        )

        self.assertAlmostEqual(
            float(q.r_at_L / um),
            1.0,
        )

    def test_with_SI_properties_recomputes_k(self):
        p = self.make_parameters()

        q = p.with_SI_properties(
            r0=3e-6,
            rL=1e-6,
            L=500e-6,
        )

        expected_k = (
            (3e-6 - 1e-6)
            /
            (3e-6 * 500e-6)
        )

        self.assertAlmostEqual(
            q.k / (1 / meter),
            expected_k,
        )

    # ================================================================
    # Different discretizations
    # ================================================================

    def test_different_discretizations_same_geometry(self):

        p11 = self.make_parameters(N=11)
        p101 = self.make_parameters(N=101)
        p301 = self.make_parameters(N=301)

        # Physical geometry is identical.
        self.assertAlmostEqual(
            float(p11.r_at_0 / um),
            float(p101.r_at_0 / um),
        )

        self.assertAlmostEqual(
            float(p11.r_at_L / um),
            float(p301.r_at_L / um),
        )

        self.assertAlmostEqual(
            float(p11.L / um),
            float(p301.L / um),
        )

        # k is independent of N.
        self.assertAlmostEqual(
            float(p11.k / (1 / meter)),
            float(p101.k / (1 / meter)),
        )

        self.assertAlmostEqual(
            float(p101.k / (1 / meter)),
            float(p301.k / (1 / meter)),
        )

        # dx changes.
        self.assertAlmostEqual(
            float(p11.dx / um),
            50.0,
        )

        self.assertAlmostEqual(
            float(p101.dx / um),
            5.0,
        )

        self.assertAlmostEqual(
            float(p301.dx / um),
            500 / 300,
        )

        # b(x) sampled at the same physical location
        # should be independent of discretization.
        for x in [0, 100, 250, 400, 500]:

            xq = x * um

            b11 = p11.b(xq)
            b101 = p101.b(xq)
            b301 = p301.b(xq)

            assert_close(
                b11,
                b101,
            )

            assert_close(
                b101,
                b301,
            )

    def test_discrete_endpoints_for_different_N(self):

        for N in [2, 3, 11, 101, 301]:

            p = self.make_parameters(N=N)

            self.assertAlmostEqual(
                float(p.x[0] / um),
                0.0,
            )

            self.assertAlmostEqual(
                float(p.x[-1] / um),
                500.0,
            )

            self.assertEqual(
                len(p.x),
                N,
            )

            self.assertAlmostEqual(
                float(p.radius(p.x[0]) / um),
                2.0,
            )

            self.assertAlmostEqual(
                float(p.radius(p.x[-1]) / um),
                0.5,
            )

    # ================================================================
    # Cylindrical limit
    # ================================================================

    def test_cylindrical_limit(self):

        p = self.make_parameters(
            r0=2 * um,
            rL=2 * um,
        )

        self.assertAlmostEqual(
            float(p.k / (1 / meter)),
            0.0,
        )

        self.assertAlmostEqual(
            float(p.a()),
            0.0,
        )

        r = p.radius(p.x)

        np.testing.assert_allclose(
            r / um,
            np.ones(p.N) * 2,
        )

        b = p.b(p.x)

        expected = (
            p.r_at_0
            /
            (2 * p.c_m * p.ra)
        )

        np.testing.assert_allclose(
            b / (meter ** 2 / second),
            float(expected / (meter ** 2 / second)),
        )


class TestConicalToSI(unittest.TestCase):

    def test_none(self):
        self.assertIsNone(
            to_SI(None)
        )

    def test_length(self):

        self.assertAlmostEqual(
            to_SI(1 * meter),
            1.0,
        )

        self.assertAlmostEqual(
            to_SI(1 * cm),
            1e-2,
        )

        self.assertAlmostEqual(
            to_SI(500 * um),
            500e-6,
        )

    def test_time(self):

        self.assertAlmostEqual(
            to_SI(1 * second),
            1.0,
        )

        self.assertAlmostEqual(
            to_SI(20 * ms),
            20e-3,
        )

        self.assertAlmostEqual(
            to_SI(250 * us),
            250e-6,
        )

    def test_capacitance_density(self):

        self.assertAlmostEqual(
            to_SI(
                1 * farad / meter ** 2
            ),
            1.0,
        )

        self.assertAlmostEqual(
            to_SI(
                1 * ufarad / cm ** 2
            ),
            1e-2,
        )

    def test_resistance_density(self):

        self.assertAlmostEqual(
            to_SI(
                1e4 * ohm * cm ** 2
            ),
            1.0,
        )

        self.assertAlmostEqual(
            to_SI(
                2e4 * ohm * cm ** 2
            ),
            2.0,
        )

    def test_axial_resistivity(self):

        self.assertAlmostEqual(
            to_SI(
                100 * ohm * cm
            ),
            1.0,
        )

    def test_dimensionless(self):

        self.assertEqual(
            to_SI(1),
            1.0,
        )

        self.assertEqual(
            to_SI(5),
            5.0,
        )

    def test_array(self):

        result = to_SI(
            np.array([1, 10, 100]) * cm
        )

        np.testing.assert_allclose(
            result,
            np.array([
                0.01,
                0.1,
                1.0,
            ]),
        )

    def test_k_must_be_always_positive(self):
        rm = 2 * 1E4 * ohm * cm ** 2
        x_N = 101
        t_max = to_SI(0.1 * second)

        dt = to_SI(1E-7 * second)
        L = to_SI(2000 * um)
        default_params = ConicalCableParameters(c_m=1 * uF / cm ** 2,
                                                rm=rm,
                                                gL=1 / rm,
                                                ra=100 * ohm * cm,
                                                L=500.0 * um,
                                                N=101,
                                                r_at_0=2 * um,
                                                r_at_L=0.5 * um,
                                                I_e=150 * pampere)
        simulation_params = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt, L=L,
                                                              I_e=to_SI(150 * pampere))

        self.assertEqual(2, simulation_params.radius(0 * um) / um)
        self.assertEqual(0.5, simulation_params.radius(2000 * um) / um)
        self.assertEqual(0.000375, simulation_params.k * um)

        self.assertAlmostEqual(500, default_params.L / um)
        self.assertAlmostEqual(2000, simulation_params.L / um)

        self.assertEqual(0, default_params.x[0] / um)
        self.assertAlmostEqual(500, default_params.x[-1] / um)

        self.assertEqual(0, simulation_params.x[0] / um)
        self.assertAlmostEqual(2000, simulation_params.x[-1] / um)

    def test_filename_generation(self):
        self.assertEqual("Uniform x ~ [0 um - 500 um]", experiment_label_for_uniform(to_SI(0 * um), to_SI(500 * um)))
        self.assertEqual("uniform_x_0_um_500_um_", file_name_prefix_for_uniform(to_SI(0 * um), to_SI(500 * um)))

if __name__ == "__main__":
    unittest.main()

import unittest
import sys
from pathlib import Path

import numpy as np
from brian2 import ms, um, uamp, cm, have_same_dimensions, ufarad, ohm, second, mV, volt, Hz, meter, uF, coulomb, uvolt, \
    is_dimensionless
from brian2.units.allunits import mampere, pampere, ampere
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ITERATION_SRC = SRC_ROOT / "iteration_19_tapered_dendrites"
for path in (REPO_ROOT, SRC_ROOT, ITERATION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from CylindricalDendritesPDE import dirac_delta as dirac_cylindrical
from iteration_19_tapered_dendrites.CylindricalDendritesPDE import dirac_delta_unitless, \
    plot_tuckwell_solution_closed_cable, plot_tuckwell_solution_closed_cable_unitless_separation_of_variables, \
    plot_difussion_unitless, plot_tuckwell_solution_closed_cable_unitless_method_of_images, \
    compute_v_theory_tuckwell_closed_rod_unitless_method_of_images
from iteration_19_tapered_dendrites.TaperredDendritesPDE import dirac_delta
from iteration_19_tapered_dendrites.data import to_SI, CableParameters, NumericalCableParameters

from numpy.testing import assert_allclose, assert_array_equal


class TestPDECases(unittest.TestCase):

    def setUp(self):
        self.dx = 1 * um
        self.dt = 1 * ms

        self.x = np.arange(10) * self.dx

        self.t0 = 5 * ms
        self.w = 1 * uamp / cm
        self.L = 500.0 * um
        self.N = 11
        self.r_0 = 2 * um
        self.r_L = 0.5 * um

        k = (1 - self.r_L / self.r_0) / self.L
        self.r_of_x = self.r_0 * (1 - k * self.x)

    def test_before_impulse(self):
        result = dirac_delta(
            t=4 * ms,
            t0=self.t0,
            dt=self.dt,
            x0=5 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        self.assertEqual(result.shape, (10,))
        assert_allclose(result / (uamp / um), 0)

    def test_after_impulse(self):
        result = dirac_delta(
            t=6 * ms,
            t0=5.99999999 * ms,
            dt=self.dt,
            x0=5 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        self.assertEqual(result.shape, (10,))
        assert_allclose(result / (uamp / um), 0)

    def test_exact_node(self):
        result = dirac_delta(
            t=self.t0,
            t0=self.t0,
            dt=self.dt,
            x0=4 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        expected = np.zeros(10)
        expected[4] = 1.0

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        assert_allclose(result / (self.w / self.dx), expected, atol=1e-6)

    def test_halfway_between_nodes(self):
        result = dirac_delta(
            t=self.t0,
            t0=self.t0,
            dt=self.dt,
            x0=4.5 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        expected = np.zeros(10)
        expected[4] = 0.5
        expected[5] = 0.5

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        assert_allclose(result / (self.w / self.dx), expected, atol=1e-6)

    def test_quarter_between_nodes(self):
        result = dirac_delta(
            t=self.t0,
            t0=self.t0,
            dt=self.dt,
            x0=4.25 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        expected = np.zeros(10)
        expected[4] = 0.75
        expected[5] = 0.25

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))

        assert_allclose(result / (self.w / self.dx), expected)

    def test_left_boundary(self):
        result = dirac_delta(
            t=self.t0,
            t0=self.t0,
            dt=self.dt,
            x0=-0.00001 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        expected = np.zeros(10)
        expected[0] = 1.0
        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        assert_allclose(result / (self.w / self.dx), expected)

    def test_right_boundary(self):
        result = dirac_delta(
            t=self.t0,
            t0=self.t0,
            dt=self.dt,
            x0=10 * um,
            x=self.x,
            r_of_x=self.r_of_x,
            dx=self.dx,
            w=self.w,
        )

        expected = np.zeros(10)
        expected[-1] = 1.0
        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        assert_allclose(result / (self.w / self.dx), expected)

    def test_conservation(self):
        positions = np.linspace(0, 9, 41) * um

        for x0 in positions:
            result = dirac_delta(
                t=self.t0,
                t0=self.t0,
                dt=self.dt,
                x0=x0,
                x=self.x,
                r_of_x=self.r_of_x,
                dx=self.dx,
                w=self.w,
            )
            self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
            total = np.sum(result) * self.dx
            self.assertAlmostEqual(
                float(total / self.w),
                1.0,
                places=12,
            )

    def test_only_one_or_two_nonzero_entries(self):
        positions = [
            0.5,
            2.3,
            5.8,
            8.1,
        ]

        for p in positions:
            result = dirac_delta(
                t=self.t0,
                t0=self.t0,
                dt=self.dt,
                x0=p * um,
                x=self.x,
                r_of_x=self.r_of_x,
                dx=self.dx,
                w=self.w,
            )

            self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
            nnz = np.count_nonzero(result / (self.w / self.dx))
            self.assertEqual(nnz, 2)

    def test_exact_node_has_single_nonzero(self):
        for x0 in self.x:
            result = dirac_delta(
                t=self.t0,
                t0=self.t0,
                dt=self.dt,
                x0=x0,
                x=self.x,
                r_of_x=self.r_of_x,
                dx=self.dx,
                w=self.w,
            )

            self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
            nnz = np.count_nonzero(np.abs(result / (mampere / cm ** 2)) > 1e-12)
            self.assertEqual(nnz, 1, msg=f"Node {x0} has 2 nonzero entries: {result[np.where(result != 0)]}")

    def test_dirac_delta_units(self):
        c_m = 1 * uF / cm ** 2
        Rm = 2 * 1E4 * ohm * cm ** 2
        gL = 1 / Rm
        # 1. Parameters
        L = 500.0 * um
        N = 11
        x = np.linspace(0, L, N)
        dx = L / (N - 1)  # um
        dt = 0.01 * ms

        tau = c_m / gL  # ms

        assert have_same_dimensions(tau, 1 * second)

        print("tau=", tau)

        w = 100 * pampere
        r_0 = 2 * um

        result = dirac_cylindrical(
            x0=-5 * um,
            t0=6.00000009 * ms,
            t=6 * ms,
            dt=dt,
            tau_m=tau,
            x=x,
            r_of_x=r_0,
            dx=dx,
            I_e=w
        )

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        self.assertEqual(result[1:].shape, (10,))
        self.assertAlmostEqual(31830.98861837907, result[0] / (uamp / cm ** 2), places=8)
        self.assertAlmostEqual(318.3098861837907, result[0] / (ampere / meter ** 2), places=8)
        assert_allclose(result[1:] / (uamp / cm ** 2), 0)

        # ------------------------------------------------------------------
        # Charge conservation:
        # ∑ i_e · (2πr) · dx · dt = I_e τ_m
        # ------------------------------------------------------------------

        injected_charge = np.sum(result * (2 * np.pi * r_0) * dx * dt)

        self.assertTrue(have_same_dimensions(injected_charge, coulomb))
        self.assertAlmostEqual(
            injected_charge / coulomb,
            (w * tau) / coulomb,
            places=12
        )

        # Equivalent invariant:
        self.assertAlmostEqual(
            np.sum(result) / (ampere / meter ** 2),
            (w * tau / (2 * np.pi * r_0 * dx * dt)) / (ampere / meter ** 2),
            places=12
        )

    def test_dirac_delta_units_for_no_input(self):
        c_m = 1 * uF / cm ** 2
        Rm = 2 * 1E4 * ohm * cm ** 2
        gL = 1 / Rm
        # 1. Parameters
        L = 500.0 * um
        N = 11
        x = np.linspace(0, L, N)
        dx = L / (N - 1)  # um

        tau = c_m / gL  # ms

        assert have_same_dimensions(tau, 1 * second)

        print("tau=", tau)

        w = 100 * pampere
        r_0 = 2 * um

        result = dirac_cylindrical(
            x0=-5 * um,
            t0=5 * ms,
            t=6 * ms,
            dt=0.1 * ms,
            tau_m=tau,
            x=x,
            r_of_x=r_0,
            dx=dx,
            I_e=w
        )

        self.assertTrue(have_same_dimensions(result[0], 1 * mampere / cm ** 2))
        self.assertEqual(result.shape, (11,))
        assert_allclose(result / (uamp / cm ** 2), 0)

    def test_diract_components_with_units(self):

        c_m = 1 * uF / cm ** 2
        Rm = 2 * 1E4 * ohm * cm ** 2
        gL = 1 / Rm
        # 1. Parameters
        L = 500.0 * um
        N = 11
        x = np.linspace(0, L, N)
        dx = L / (N - 1)  # um
        dt = 0.1 * ms

        tau = c_m / gL  # ms

        w = 100 * pampere
        r0 = 2 * um

        delta_xt = 1.0 / (dx * dt)

        i_e = w * tau / (2 * np.pi * r0) * delta_xt

        self.assertTrue(have_same_dimensions(i_e, ampere / meter ** 2))
        self.assertAlmostEqual(31.830988618379067, i_e / pampere * um ** 2)
        self.assertAlmostEqual(31.830988618379067, to_SI(i_e))

    def test_dirac_unitless_returns_float_array(self):

        c_m = 1 * uF / cm ** 2
        Rm = 2 * 1E4 * ohm * cm ** 2
        gL = 1 / Rm
        # 1. Parameters
        L = 500.0 * um
        N = 11
        x = np.linspace(0, L, N)
        dx = L / (N - 1)  # um

        tau = c_m / gL  # ms

        w = 100 * pampere
        r_0 = 2 * um

        result = dirac_delta_unitless(
            x0=to_SI(-5 * um, meter),
            t0=to_SI(6.00000009 * ms, second),
            t=to_SI(6 * ms, second),
            dt=to_SI(0.1 * ms, second),
            tau_m=to_SI(tau, second),
            x=to_SI(x, meter),
            r_of_x=to_SI(r_0, meter),
            dx=to_SI(dx, meter),
            I_e=to_SI(w, ampere)
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

        self.assertEqual(result[0], 31.830988618379067)
        assert_allclose(result[1:] / (uamp / cm ** 2), 0)


class TestLinearTaperCableOneStep(unittest.TestCase):

    def setUp(self):
        # Small grid for testing
        self.N = 11

        self.L = 500 * um
        self.x = np.linspace(0, self.L, self.N)
        self.dx = self.L / (self.N - 1)

        # Parameters
        self.cm = 1 * ufarad / cm ** 2
        Rm = 2e4 * ohm * cm ** 2
        self.gL = 1 / Rm
        self.ra = 100 * ohm * cm

        self.tau = self.cm / self.gL

        self.r0 = 2 * um
        self.rL = 0.5 * um

        self.k = (1 - self.rL / self.r0) / self.L
        self.r_of_x = self.r0 * (1 - self.k * self.x)

        self.a = (
                self.k * self.r0 /
                (self.cm * self.ra *
                 np.sqrt(1 + self.r0 ** 2 * self.k ** 2))
        )

        self.b = (
                self.r0 * (1 - self.k * self.x) /
                (2 * self.cm * self.ra *
                 np.sqrt(1 + self.r0 ** 2 * self.k ** 2))
        )

        self.dt = 0.01 * ms

        self.A = self.build_matrix()

    def build_matrix(self):
        lower = self.b[1:] / self.dx ** 2 + self.a / (2 * self.dx)
        main = -1 / self.tau - 2 * self.b / self.dx ** 2
        upper = self.b[:-1] / self.dx ** 2 - self.a / (2 * self.dx)

        # Sparse tridiagonal matrix
        A = diags(
            diagonals=[lower, main, upper],
            offsets=[-1, 0, 1],
            format="lil"
        )

        return A.tocsr().toarray() * (1 / second)

    def test_matrix_initialization(self):
        self.assertAlmostEqual(50, self.dx / um)

        A = self.A
        self.assertEqual(
            self.A.shape,
            (self.N, self.N)
        )

        self.assertTrue(have_same_dimensions(self.A[0, 0], ms ** -1))

        self.assertAlmostEqual(self.a / (cm / second), 29.999865000911253,
                               msg="Manuscript say 30 cm / s but that is an approximation."
                                   "we are ignoring in the manuscript 1/sqrt(1 + k**2r0**2)")

        self.assertAlmostEqual(self.a / (meter / second), 0.29999865000911253,
                               msg="Manuscript say 0.3 m / s but that is an approximation."
                                   "we are ignoring in the manuscript 1/sqrt(1 + k**2r0**2)")

        self.assertAlmostEqual(self.b[0] / (cm ** 2 / second), 0.9999955000303749,
                               msg="Manuscript say 1 cm^2 / s but that is an approximation."
                                   "we are ignoring in the manuscript 1/sqrt(1 + k**2r0**2)")
        self.assertAlmostEqual(self.b[-1] / (cm ** 2 / second), 0.24999888,
                               msg="Manuscript say 0.25 cm^2 / s but that is an approximation."
                                   "we are ignoring in the manuscript 1/sqrt(1 + k**2r0**2)")

        self.assertAlmostEqual(self.b[0] / (meter ** 2 / second), 0.00009999955000303749)
        self.assertAlmostEqual(self.b[-1] / (meter ** 2 / second), 0.000024999888)

        self.assertAlmostEqual((self.b[0] / self.dx ** 2) / Hz, 4E4,
                               msg="Manuscript says 3.96 x 10E6 BUT our dx there is 5 um while here it is 50! So, 10^2 difference",
                               places=0)

        self.assertAlmostEqual((self.b[-1] / self.dx ** 2) / Hz, 1E4,
                               msg="Manuscript says 10E6 BUT our dx there is 5 um while here it is 50! So, 10^2 difference",
                               places=0)

        self.assertAlmostEqual(50, 1 / self.tau * second)

    def test_one_forward_euler_step_no_input(self):
        # Simple voltage profile
        V = np.linspace(
            -70,
            20,
            self.N
        ) * mV

        dVdt = self.A @ V

        V_next = V + self.dt * dVdt

        self.assertTrue(have_same_dimensions(dVdt[0], volt / second))
        self.assertTrue(have_same_dimensions(V_next[0], volt))
        # The step should change voltage
        self.assertFalse(np.allclose(V_next / mV, V / mV))

    def test_constant_voltage_decay(self):
        # If V is spatially constant, diffusion terms vanish
        V0 = -70 * mV

        V = np.ones(self.N) * V0

        dVdt = self.A @ V

        expected = - V0 / self.tau

        assert_allclose(
            dVdt[1:-1] / (mV / ms),
            np.ones(self.N - 2) *
            (expected / (mV / ms)),
            rtol=1e-10,
            atol=1e-10
        )

    def test_synaptic_input_one_step(self):
        V = np.zeros(self.N) * mV

        I_syn = dirac_delta(
            x0=250 * um,
            t0=0 * ms,
            t=0 * ms,
            dx=self.dx,
            dt=self.dt,
            x=self.x,
            r_of_x=self.r_of_x,
            w=1 * uamp / cm
        )

        dVdt = self.A @ V + I_syn / self.cm

        V_next = V + self.dt * dVdt

        # Some voltage must appear
        self.assertGreater(
            np.max(np.abs(V_next)),
            0 * mV
        )

        # Check dimensions
        self.assertTrue(
            have_same_dimensions(
                dVdt[0],
                volt / second
            )
        )


Rm = 2 * 1E4 * ohm * cm ** 2
default_params = CableParameters(c_m=1 * uF / cm ** 2,
                                 rm=Rm,
                                 gL=1 / Rm,
                                 ra=100 * ohm * cm,
                                 L=500.0 * um,
                                 N=101,
                                 r0=2 * um,
                                 I_e=150 * pampere)


def create_difussion_matrix(p: NumericalCableParameters):
    x = p.x
    dx = p.dx

    tau = p.tau
    b = p.b

    difussion = np.ones(len(x) - 1) * b / dx ** 2
    difussion_decay = np.ones(len(x)) * (-1 / tau - 2 * b / dx ** 2)

    # Sparse tridiagonal matrix
    A = diags(
        diagonals=[difussion, difussion_decay, difussion],
        offsets=[-1, 0, 1],
        format="lil"
    )

    # ensure boundary conditions automatically in A matrix
    # A[0, 0] = -1 / tau - 2 * b / dx ** 2
    A[0, 1] = 2 * b / dx ** 2
    A[-1, -2] = 2 * b / dx ** 2
    # A[-1, -1] = -1 / tau - 2 * b / dx ** 2
    return A


class TestForwardEulerInCylinderOneStep(unittest.TestCase):

    def test_parameter_values_small_N(self):
        simulation_params = default_params.with_property(t=10 * ms, N=11)
        p = simulation_params.to_numerical()

        A = create_difussion_matrix(p)
        self.assertEqual(1E-4, p.b)
        self.assertEqual(1, simulation_params.b / cm ** 2 * second)

        self.assertAlmostEqual(np.sqrt(2) * 1E-3, p.lambd())
        self.assertAlmostEqual(np.sqrt(2) * 1E-3, simulation_params.lambd() / meter)
        self.assertAlmostEqual(np.sqrt(2) * 1E3, simulation_params.lambd() / um)
        self.assertAlmostEqual(np.sqrt(2) / 10, simulation_params.lambd() / cm)

        diag = -1 / p.tau - 2 / 25 * 1E6
        diag_cp = -1 / 0.02 - 2 / 25 * 1E6
        assert_array_equal(np.ones(9) * diag, A.diagonal()[1:10])

        diff = 1 / 25 * 1E6

        assert_allclose(np.ones(9) * diff, A.diagonal(-1)[:-1])
        self.assertAlmostEqual(2 / 25 * 1E6, A.diagonal(-1)[-1])

        assert_allclose(np.ones(9) * diff, A.diagonal(1)[1:])
        self.assertAlmostEqual(2 / 25 * 1E6, A.diagonal(1)[0])

    def test_parameter_values_moderate_N(self):
        simulation_params = default_params.with_property(t=10 * ms, N=101)
        p = simulation_params.to_numerical()

        A = create_difussion_matrix(p)
        self.assertEqual(1E-4, p.b)
        self.assertEqual(1, simulation_params.b / cm ** 2 * second)
        self.assertAlmostEqual(np.sqrt(2) * 1E-3, simulation_params.lambd() / meter)
        self.assertAlmostEqual(np.sqrt(2) * 1E3, simulation_params.lambd() / um)
        self.assertAlmostEqual(np.sqrt(2) / 10, simulation_params.lambd() / cm)

        self.assertAlmostEqual(5E-6, p.dx)
        self.assertAlmostEqual(5, simulation_params.dx / um)

        diag = -1 / p.tau - 2 / 25 * 1E8
        diag_cp = -1 / 0.02 - 2 / 25 * 1E8
        assert_allclose(np.ones(101) * diag, A.diagonal())
        assert_allclose(np.ones(101) * diag_cp, A.diagonal())

        diff = 1 / 25 * 1E8

        assert_allclose(np.ones(99) * diff, A.diagonal(-1)[:-1])
        self.assertAlmostEqual(2 / 25 * 1E8, A.diagonal(-1)[-1])

        assert_allclose(np.ones(99) * diff, A.diagonal(1)[1:])
        self.assertAlmostEqual(2 / 25 * 1E8, A.diagonal(1)[0])

    def test_parameter_values_large_N(self):
        simulation_params = default_params.with_property(t=10 * ms, N=1001)
        p = simulation_params.to_numerical()

        A = create_difussion_matrix(p)
        self.assertEqual(1E-4, p.b)
        self.assertEqual(1, simulation_params.b / cm ** 2 * second)
        self.assertAlmostEqual(np.sqrt(2) * 1E-3, simulation_params.lambd() / meter)
        self.assertAlmostEqual(np.sqrt(2) * 1E3, simulation_params.lambd() / um)
        self.assertAlmostEqual(np.sqrt(2) / 10, simulation_params.lambd() / cm)

        self.assertEqual(5E-7, p.dx)
        self.assertEqual(0.5, simulation_params.dx / um)

        diag = -1 / p.tau - 2 / 25 * 1E10
        diag_cp = -1 / 0.02 - 2 / 25 * 1E10
        assert_allclose(np.ones(1001) * diag, A.diagonal())
        assert_allclose(np.ones(1001) * diag_cp, A.diagonal())

        diff = 1 / 25 * 1E10

        assert_allclose(np.ones(999) * diff, A.diagonal(-1)[:-1])
        self.assertAlmostEqual(2 / 25 * 1E10, A.diagonal(-1)[-1])

        assert_allclose(np.ones(999) * diff, A.diagonal(1)[1:])
        self.assertAlmostEqual(2 / 25 * 1E10, A.diagonal(1)[0])

    def test_one_dirac_step(self):
        simulation_params = default_params.with_property(t=10 * ms, N=6)
        p = simulation_params.to_numerical()

        # 1. Parameters
        dx = p.dx
        dt = to_SI(1E-8 * second)

        A = create_difussion_matrix(p)

        V = np.zeros(len(p.x))

        x0 = to_SI(200 * um)
        t0 = to_SI(1 * ms)
        I_e = to_SI(1.5 * pampere)

        dx = p.dx

        def synaptic_input_profile(t, x0, dt):
            return dirac_delta_unitless(x0=x0, t0=t0, x=p.x, t=t, dx=dx, dt=dt, I_e=I_e,
                                        tau_m=p.tau, r_of_x=p.r_at_0)

        t = t0 - 0.1 * dt

        i_of_t = synaptic_input_profile(t, x0, dt)
        inputed_current = dx * dt * np.sum(i_of_t)

        self.assertAlmostEqual(I_e * p.tau / (2 * np.pi * p.r_at_0), inputed_current, places=20)

        V_n_euler = np.copy(V)
        V_n_plus_1_euler = V_n_euler + dt * (A @ V_n_euler + i_of_t)

        print(f"{I_e * p.tau / (2 * np.pi * p.r_at_0 * dx) : .6e}")
        print(f"{V_n_plus_1_euler[2] : .6e}")

        self.assertAlmostEqual(I_e * p.tau / (2 * np.pi * p.r_at_0 * dx), float(V_n_plus_1_euler[2]))
        self.assertAlmostEqual(I_e * p.tau / (2 * np.pi * p.r_at_0), float(V_n_plus_1_euler[2] * dx))

    def test_current_injection_increased_x_discretization(self):
        simulation_params = default_params.with_property(t=10 * ms, N=101)
        p = simulation_params.to_numerical()

        # 1. Parameters
        dx = p.dx
        dt = to_SI(1E-8 * second)

        A = create_difussion_matrix(p)

        V = np.zeros(len(p.x))

        x0 = to_SI(250 * um)
        t0 = to_SI(1 * ms)
        I_e = to_SI(1.5 * pampere)

        dx = p.dx

        def synaptic_input_profile(t, x0, dt):
            return dirac_delta_unitless(x0=x0, t0=t0, x=p.x, t=t, dx=dx, dt=dt, I_e=I_e,
                                        tau_m=p.tau, r_of_x=p.r_at_0)

        t = t0 - 0.1 * dt

        x0_index = 50

        i_of_t = synaptic_input_profile(t, x0, dt)
        inputed_current = dx * dt * np.sum(i_of_t)

        self.assertAlmostEqual(I_e * p.tau / (2 * np.pi * p.r_at_0), float(inputed_current), places=20)

        V_n_euler = np.copy(V)
        V_n_plus_1_euler = V_n_euler + dt * (A @ V_n_euler + i_of_t)

        print(f"{I_e * p.tau / (2 * np.pi * p.r_at_0 * dx) : .6e}")
        print(f"{V_n_plus_1_euler[2] : .6e}")

        self.assertAlmostEqual(I_e * p.tau / (2 * np.pi * p.r_at_0 * dx), V_n_plus_1_euler[x0_index])
        self.assertAlmostEqual(I_e * p.tau / (2 * np.pi * p.r_at_0), V_n_plus_1_euler[x0_index] * dx)


def load_simulation(filename):
    data = np.load(filename)

    times = data["times"]
    V_s = data["V_s"]

    simulation_params = default_params.with_SI_properties(
        t=times[-1],
        N=int(data["N"]),
        dt=data["dt"],
        L=data["L"],
        I_e=data["I_e"],
    )

    p = simulation_params.to_numerical()

    x0 = data["x0"] * meter
    t0 = 0.1 * ms

    return times, V_s, p, x0, t0


def saved_simulations_exist(*filenames):
    return all((REPO_ROOT / filename).exists() for filename in filenames)


REFERENCE_N1501_L500_DT50PS = (
    "saved_simulations/cable_sim_N1501_L500_t_max30ms_dt50ps_x09.999999999999999e-05um.npz"
)
REFERENCE_N1501_L500_DT5000PS = (
    "saved_simulations/cable_sim_N1501_L500_t_max30ms_dt5000ps_x09.999999999999999e-05um.npz"
)
REFERENCE_N1501_L1500_DT50PS = (
    "saved_simulations/cable_sim_N1501_L1500_t_max0_500ms_dt50ps_x00.00025um.npz"
)
REFERENCE_DIVERGENCE_FILES = (
    "saved_simulations/cable_sim_N101_L500_t_max0_300ms_dt10000ps_x00.00025um.npz",
    "saved_simulations/cable_sim_N1001_L500_t_max0_300ms_dt10000ps_x00.00025um.npz",
    "saved_simulations/cable_sim_N2001_L500_t_max0_300ms_dt10000ps_x00.00025um.npz",
)


class TestCrankNicolsonOneStep(unittest.TestCase):

    def setUp(self):
        simulation_params = default_params.with_property(t=10 * ms, N=1001)
        self.si_units = simulation_params.to_numerical()
        self.dt = to_SI(1E-8 * second)

        # 1. Parameters
        dx = self.si_units.dx
        tau = self.si_units.tau
        dt = self.dt

        # Spatial domain and initial condition
        x = self.si_units.x
        b = self.si_units.b
        difussion = np.ones(len(x) - 1) * b / dx ** 2
        difussion_decay = np.ones(len(x)) * (-1 / tau - 2 * b / dx ** 2)

        # Sparse tridiagonal matrix
        A = diags(
            diagonals=[difussion, difussion_decay, difussion],
            offsets=[-1, 0, 1],
            format="lil"
        )

        # ensure boundary conditions automatically in A matrix
        A[0, 0] = -1 / tau - 2 * b / dx ** 2
        A[0, 1] = 2 * b / dx ** 2
        A[-1, -2] = 2 * b / dx ** 2
        A[-1, -1] = -1 / tau - 2 * b / dx ** 2

        I = eye(A.shape[0], format="csc")

        self.A = A
        # Crank-Nicolson matrices
        self.L = (I - 0.5 * dt * A).tocsc()
        self.R = (I + 0.5 * dt * A).tocsc()

        # Factorize once
        self.solve = factorized(self.L)

    def test_one_step(self):
        V = np.zeros(len(self.si_units.x))
        dt = self.dt
        self.x0 = to_SI(250 * um)
        I_e = to_SI(1.5 * pampere)
        dx = self.si_units.dx

        def synaptic_input_profile(t, x0, dt):
            return 1 / self.si_units.c_m *  dirac_delta_unitless(x0=x0, t0=to_SI(1 * ms), x=self.si_units.x, t=t, dx=dx, dt=dt, I_e=I_e,
                                                                 tau_m=self.si_units.tau, r_of_x=self.si_units.r_at_0)

        t = to_SI(1 * ms) - 1E-10
        x0 = to_SI(250 * um)

        id_x0 = np.searchsorted(self.si_units.x, x0)
        # RHS
        # dt/2 is a bug It is incorrect!!. Why? We always multiply by dt. So normalizing by dt/2 actually doubles the area
        syn_input_t = synaptic_input_profile(t=t, x0=x0, dt=dt)
        syn_input_t_plus_one = synaptic_input_profile(t=t + dt, x0=x0, dt=dt)
        input_t_and_t_half = 1 / 2 * (syn_input_t + syn_input_t_plus_one)

        inputed_charge = I_e * self.si_units.tau
        inputed_voltage = dx * dt * np.sum(syn_input_t)

        self.assertAlmostEqual(I_e * self.si_units.tau / (2 * np.pi * self.si_units.r_at_0) / self.si_units.c_m, float(inputed_voltage), places=20)
        self.assertAlmostEqual(0.5 * I_e * self.si_units.tau / (2 * np.pi * self.si_units.r_at_0) / self.si_units.c_m, float(dx * dt * np.sum(input_t_and_t_half)), places=20)

        self.assertEqual(0, np.sum(syn_input_t_plus_one))

        V_n_euler = np.copy(V)
        V_n_plus_1_euler = V_n_euler + dt * (self.A @ V_n_euler + input_t_and_t_half)

        print(f"{I_e * self.si_units.tau / (2 * np.pi * self.si_units.r_at_0) : .6e}")
        print(f"{V_n_plus_1_euler[500] : .6e}")

        self.assertAlmostEqual(0.5 * I_e * self.si_units.tau / self.si_units.c_m / (2 * np.pi * self.si_units.r_at_0), V_n_plus_1_euler[500] * dx, places=20)

        # cmn iteration is tested somewhere else

    def test_prefactor(self):
        numerical_prefactor = self.si_units.I_e * self.si_units.tau / (2 * np.pi * self.si_units.r_at_0)

        theory_prefactor = self.si_units.I_e * self.si_units.R_lambda()
        # r_lambda =  self.rm / (2 * np.pi * self.r0 * self.lambd())
        # lambd = math.sqrt(self.r0 * self.rm / (2 * self.ra))
        print(numerical_prefactor / theory_prefactor)

        print(self.si_units.tau * self.si_units.lambd() / self.si_units.rm)

    @unittest.skipUnless(
        saved_simulations_exist(REFERENCE_N1501_L500_DT50PS),
        "Requires saved Crank-Nicolson reference simulation data.",
    )
    def test_tuckwell_theory_vs_simulation(self):
        times_unitless, V_s_unitless, p, x0, t0 = load_simulation(
            "saved_simulations/cable_sim_N1501_L500_t_max30ms_dt50ps_x09.999999999999999e-05um.npz")
        desired_positions = np.array([100, 250, 500]) * um
        desired_positions_unitless = desired_positions / um

        self.assertTrue(is_dimensionless(times_unitless[-1]))
        self.assertTrue(is_dimensionless(V_s_unitless[-1, -1]))
        self.assertEqual(NumericalCableParameters, p.__class__)
        self.assertTrue(have_same_dimensions(x0, meter))
        self.assertTrue(have_same_dimensions(t0, second))

        times = np.array(times_unitless) * second
        V_s = np.array(V_s_unitless) * volt
        p_units = CableParameters.from_numerical(p)

        # TODO: maybe here, V_s should be, actually, unitless!
        plot_tuckwell_solution_closed_cable_unitless_method_of_images(times=times_unitless, V_s=V_s_unitless,
                                                                      p=p,
                                                                      desired_positions=desired_positions_unitless,
                                                                      t0=t0, x0=x0,
                                                                      sim_type="Crank-Nicolson")
        plot_tuckwell_solution_closed_cable(times=times, V_s=V_s, p=p_units, desired_positions=desired_positions, t0=t0,
                                            x0=x0, sim_type="Crank-Nicolson")
        plot_tuckwell_solution_closed_cable_unitless_separation_of_variables(times=times_unitless, V_s=V_s_unitless, p=p,
                                                                             desired_positions=desired_positions_unitless, t0=t0, x0=x0,
                                                                             sim_type="Crank-Nicolson")
        '''
        for offset in np.arange(5, 100, step=50):
            plot_tuckwell_solution_closed_cable_difference(V_s=V_s, desired_positions=desired_positions, p=p, t0=t0,
                                                           times=times, x0=x0, t0_offset=offset)
        '''

    @unittest.skipUnless(
        saved_simulations_exist(REFERENCE_N1501_L500_DT50PS, REFERENCE_N1501_L500_DT5000PS),
        "Requires saved Crank-Nicolson reference simulation data.",
    )
    def test_tuckwell_theory_vs_simulation_method_of_images(self):
        times_unitless, V_s_unitless, p, x0, t0 = load_simulation(
            "saved_simulations/cable_sim_N1501_L500_t_max30ms_dt50ps_x09.999999999999999e-05um.npz")
        times_unitless, V_s_unitless, p, x0, t0 = load_simulation(
            "saved_simulations/cable_sim_N1501_L500_t_max30ms_dt5000ps_x09.999999999999999e-05um.npz")
        desired_positions_unitless = np.array([100])

        plot_tuckwell_solution_closed_cable_unitless_method_of_images(times=times_unitless, V_s=V_s_unitless,
                                                                      p=p,
                                                                      desired_positions=desired_positions_unitless,
                                                                      t0=t0, x0=x0,
                                                                      sim_type="Crank-Nicolson")


    @unittest.skipUnless(
        saved_simulations_exist(REFERENCE_N1501_L1500_DT50PS),
        "Requires saved Crank-Nicolson reference simulation data.",
    )
    def test_plot_simulation(self):
        times, V_s, p, x0, t0 = load_simulation(
            "saved_simulations/cable_sim_N1501_L1500_t_max0_500ms_dt50ps_x00.00025um.npz")
        plot_difussion_unitless(times=times, V_s=V_s, p=p, t0=t0, x0=x0, sim_type="Crank-Nicolson", save=False)

    def test_CN_steps(self):
        for N in [11, 101, 1001]:
            simulation_params = default_params.with_SI_properties(t=to_SI(1 * ms), N=N, dt=1E-8, L=to_SI(500 * um),
                                                                  I_e=to_SI(150 * pampere))
            p = simulation_params.to_numerical()

            # 1. Parameters
            dx = p.dx
            dt = p.dt
            tau = p.tau
            x = p.x
            x0 = to_SI(250 * um)
            I_e = p.I_e

            x_inj_index = 5

            b = p.b
            difussion = np.ones(len(x) - 1) * b / dx ** 2
            difussion_decay = np.ones(len(x)) * (-1 / tau - 2 * b / dx ** 2)

            # Sparse tridiagonal matrix
            A = diags(
                diagonals=[difussion, difussion_decay, difussion],
                offsets=[-1, 0, 1],
                format="lil"
            )

            # ensure boundary conditions automatically in A matrix
            A[0, 0] = -1 / tau - 2 * b / dx ** 2
            A[0, 1] = 2 * b / dx ** 2
            A[-1, -2] = 2 * b / dx ** 2
            A[-1, -1] = -1 / tau - 2 * b / dx ** 2

            # Identity matrix
            I = eye(A.shape[0], format="csc")

            # Crank-Nicolson matrices
            L = (I - 0.5 * p.dt * A).tocsc()
            R = (I + 0.5 * p.dt * A).tocsc()

            # Factorize once
            solve = factorized(L)

            # I want to say Vi-1 => V_i => V_i+1. Delta pulse should be at time step i. I.e. close to t_i
            V = np.zeros(len(x))

            t0 = to_SI(0.1 * ms)
            t_i = t0 - 0.1 * dt
            t_i_minus_3 = t_i - 3 * dt
            t_i_minus_2 = t_i - 2 * dt
            t_i_minus_1 = t_i - dt
            t_i_plus_1 = t_i + dt

            def synaptic_input_profile(t, x0, dt, p):
                return dirac_delta_unitless(x0=x0, t0=t0, x=x, t=t, dx=dx, dt=dt, I_e=I_e,
                                            tau_m=p.tau, r_of_x=p.r_at_0)

            def one_CN_step(V, t, dt):
                # V = V_i
                delta_i = synaptic_input_profile(t=t, x0=x0, dt=dt, p=p)
                delta_i_plus_1 = synaptic_input_profile(t=t + p.dt, x0=x0, dt=dt, p=p)
                source = 1 / 2 * dt * 1 / p.c_m * (delta_i + delta_i_plus_1)

                rhs = R @ V + source
                return solve(rhs), delta_i, delta_i_plus_1, source

            V0 = V.copy()
            V_i_minus_2, S_i_minus_2, S_i_minus_1, source_0 = one_CN_step(V, t=t_i_minus_3, dt=p.dt)
            V_i_minus_1, S_i_minus_1_again, S_i, source_1 = one_CN_step(V_i_minus_2, t=t_i_minus_2, dt=p.dt)
            V_i, S_i_again, S_i_plus_1, source_2 = one_CN_step(V_i_minus_1, t=t_i_minus_1, dt=p.dt)
            V_i_plus_1, S_i_plus_1_again, S_i_plus_2, source_3 = one_CN_step(V_i, t=t_i, dt=p.dt)
            V_i_plus_2, S_i_plus_2_again, S_i_plus_3, source_4 = one_CN_step(V_i_plus_1, t=t_i_plus_1, dt=p.dt)

            # verify delta pulses
            self.assertEqual(0, np.sum(source_0), "nothing injected")
            self.assertEqual(0, np.sum(V_i_minus_2), "no voltage update")
            self.assertEqual(0, np.sum(source_1), "nothing injected")
            self.assertEqual(0, np.sum(V_i_minus_1), "no voltage update")
            self.assertAlmostEqual(p.I_e * p.tau / 2, np.sum(source_2) * p.dx * 2 * np.pi * p.r_at_0 * p.c_m, msg=f"N={N}: Q/2 inserted at interation i", places=24)
            self.assertAlmostEqual(p.I_e * p.tau / 2, np.sum(source_3) * p.dx * 2 * np.pi * p.r_at_0 * p.c_m, msg=f"N={N}:Q/2 inserted at interation i+1", places=24)
            self.assertEqual(0, np.sum(source_4), "nothing injected")



            assert_array_equal(S_i_minus_1, S_i_minus_1_again)
            assert_array_equal(S_i, S_i_again)
            assert_array_equal(S_i_plus_1, S_i_plus_1_again)

            # dQ = cm * V d Area, d Area = 2 pi r0 dx, Q = int 2 pi a cm int v dx
            # also, around injection point, x0, V(x, t0) = Ie rm / 2 pi a * delta (x-x0). So we have to integrate
            # thus, int v dx = Ie rm / 2 pi a => we haev integral Vdx expressed in 2 ways
            # from dQ = cm V dA, by integrating we get Q = 2 pi a int dx v(x)  * c_m
            # \int v(x, t) dx = \int I_e r_m / (2 pi a) \delta(x) dx
            total_charge = p.I_e * p.tau

            def total_charge_on_membrane(V, p):
                return 2 * np.pi * p.r_at_0 * p.c_m * np.sum(V) * p.dx

            leak_q_decay_approx_in_one_half_step = 0.5 * total_charge * (dt / (2*p.tau)) / (1 + (dt/2*p.tau))

            self.assertAlmostEqual(total_charge / 2 - total_charge_on_membrane(V_i, p), leak_q_decay_approx_in_one_half_step, places=24)
            self.assertAlmostEqual(total_charge / (2 * (1 + dt/(2 * tau))), total_charge_on_membrane(V_i, p), places=24,
                                   msg="1 step of solving CN explicitly")

            # dQ = cm * V d Area, d Area = 2 pi r0 dx, Q = int 2 pi a cm int v dx
            # also, around injection point, x0, V(x, t0) = Ie rm / 2 pi a * delta (x-x0). So we have to integrate
            # thus, int v dx = Ie rm / 2 pi a => we haev integral Vdx expressed in 2 ways

            expected_total_leak = total_charge * (
                    1 - 1 / (1 + dt/(2 * p.tau)) ** 2
            )

            self.assertAlmostEqual(
                total_charge - total_charge_on_membrane(V_i_plus_1, p),
                expected_total_leak,
                places=20,
            )
            '''
            Q_{\text{leak}, t_{i} \rightarrow t_{i+1}} = \frac{\bar Q}{2} 
            \frac{3 \frac{\Delta t}{2\tau} + \left(\frac{\Delta t}{2 \tau}\right)^2}{\left(1 + \frac{\Delta t}{2 \tau}\right)^2}
            '''
            leak_q_decay_approx_in_i_to_i_dt_half = 0.5 * total_charge * (3*dt/(2 * p.tau) + (dt/(2 * p.tau))**2) / (1 + dt/(2 * p.tau))**2
            self.assertAlmostEqual(total_charge, total_charge_on_membrane(V_i_plus_1, p)
                                   + leak_q_decay_approx_in_one_half_step
                                   + leak_q_decay_approx_in_i_to_i_dt_half, places=20)
            self.assertAlmostEqual(total_charge / 2 - np.sum(V_i_plus_1 - V_i) * p.dx * 2 * np.pi * p.r_at_0 * p.c_m,
                                   leak_q_decay_approx_in_i_to_i_dt_half,
                                   msg="total charge input increases voltage in the whole system", places=20)

    def test_no_difussion_on_uniform_rod(self):
        simulation_params = default_params.with_SI_properties(t=to_SI(1 * ms), N=1001, dt=1E-8, L=to_SI(500 * um),
                                                              I_e=to_SI(150 * pampere))
        p = simulation_params.to_numerical()

        # 1. Parameters
        dx = p.dx
        dt = p.dt
        tau = p.tau
        x = p.x

        b = p.b
        difussion = np.ones(len(x) - 1) * b / dx ** 2
        difussion_decay = np.ones(len(x)) * (-1 / tau - 2 * b / dx ** 2)

        # Sparse tridiagonal matrix
        A = diags(
            diagonals=[difussion, difussion_decay, difussion],
            offsets=[-1, 0, 1],
            format="lil"
        )

        # ensure boundary conditions automatically in A matrix
        A[0, 0] = -1 / tau - 2 * b / dx ** 2
        A[0, 1] = 2 * b / dx ** 2
        A[-1, -2] = 2 * b / dx ** 2
        A[-1, -1] = -1 / tau - 2 * b / dx ** 2

        # Identity matrix
        I = eye(A.shape[0], format="csc")

        # Crank-Nicolson matrices
        L = (I - 0.5 * p.dt * A).tocsc()
        R = (I + 0.5 * p.dt * A).tocsc()
        solve = factorized(L)

        V0 = np.ones(len(x)) * 1.0

        V1 = solve(R @ V0)

        cn_factor = (1 - dt / (2 * tau)) / (1 + dt / (2 * tau))

        np.testing.assert_allclose(
            V1,
            cn_factor * V0,
            rtol=1e-13,
            atol=1e-15,
            err_msg="Constant solution does not decay with the CN leak factor",
        )

        n_steps = 100
        V = np.ones(len(x))
        for _ in range(n_steps):
            V = solve(R @ V)

        expected_factor = ((1 - dt / (2 * tau)) / (1 + dt / (2 * tau))) ** n_steps

        np.testing.assert_allclose(
            V,
            expected_factor * np.ones(len(x)),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_error_convergence(self):
        simulation_params = default_params.with_SI_properties(
            t=to_SI(1 * ms),
            N=1001,
            dt=1e-5,
            L=to_SI(500 * um),
            I_e=to_SI(150 * pampere),
        )
        p = simulation_params.to_numerical()

        tau = p.tau
        V0 = 1.0

        T = 0.1 * tau

        def cn_decay(dt):
            n = int(round(T / dt))
            V = V0

            cn_factor = (1 - dt / (2 * tau)) / (
                                1 + dt / (2 * tau)
                        )

            for _ in range(n):
                V *= cn_factor

            return V

        V_exact = np.exp(-T / tau)

        E_dt = abs(cn_decay(1e-5) - V_exact)
        E_dt2 = abs(cn_decay(5e-6) - V_exact)

        observed_order = np.log2(E_dt / E_dt2)

        self.assertAlmostEqual(
            observed_order,
            2.0,
            places=1,
        )

    @unittest.skipUnless(
        saved_simulations_exist(*REFERENCE_DIVERGENCE_FILES),
        "Requires saved Crank-Nicolson reference simulation data.",
    )
    def test_try_to_understand_divergence_at_t_0(self):

        one_file = ["saved_simulations/cable_sim_N101_L500_t_max0_300ms_dt10000ps_x00.00025um.npz"]
        files = ["saved_simulations/cable_sim_N101_L500_t_max0_300ms_dt10000ps_x00.00025um.npz",
                 "saved_simulations/cable_sim_N1001_L500_t_max0_300ms_dt10000ps_x00.00025um.npz",
                 "saved_simulations/cable_sim_N2001_L500_t_max0_300ms_dt10000ps_x00.00025um.npz"]

        for file in files:

            times_unitless, V_s_unitless, p, x0, t0 = load_simulation(file)
            #desired_positions = np.array([100, 250, 500]) * um
            desired_positions = np.array([250]) * um
            desired_positions_unitless = desired_positions / um



            times = np.array(times_unitless) * second
            V_s = np.array(V_s_unitless) * volt
            p_units = CableParameters.from_numerical(p)

            V_s_theory = compute_v_theory_tuckwell_closed_rod_unitless_method_of_images(times=times_unitless,
                                                                                        desired_positions=desired_positions_unitless,
                                                                                        t0=t0, x0=x0, p=p, n_max=101)

            plot_tuckwell_solution_closed_cable_unitless_method_of_images(times=times_unitless, V_s=V_s_unitless,
                                                                          p=p,
                                                                          desired_positions=desired_positions_unitless,
                                                                          t0=t0, x0=x0,
                                                                          sim_type="Crank-Nicolson")
            injection_time_id = np.searchsorted(times, 0.1 * ms)
            injection_possition_id = np.searchsorted(p.x, 250 * um)
            print(V_s[injection_time_id, injection_possition_id])

            # how much charge do we have:
            print(np.sum(V_s[injection_time_id+1, :]) * p.dx * 2 * np.pi * p.r_at_0 * p.c_m / (p.I_e * p.tau))


            # the issue: in t0, tuckwell's closed form formula is not "spread" while our spread is always
            # dx. But, when comparing the integral over dx of the 2 functions, we should obtain the same quantity.
            # hence this test.
            int_cn = np.sum(V_s[injection_time_id, :] / volt) * p.dx
            int_tuckwell = (
                    p.I_e * p.rm / (2 * np.pi * p.r_at_0)
                    * np.exp(-(times[injection_time_id] - t0) / p_units.tau)
            )

            self.assertAlmostEqual(1, float(int_cn / int_tuckwell), places=12)

if __name__ == '__main__':
    unittest.main()

import math
from dataclasses import dataclass, field, replace

import numpy as np
from brian2 import (
    Quantity,
    farad,
    meter,
    ohm,
    siemens,
    second,
)
from brian2.units.allunits import ampere, pampere

from iteration_19_tapered_dendrites.data import to_SI, CableParameters


@dataclass(frozen=True)
class ConicalNumericalCableParameters:
    """
    Numerical representation of a conical cable.

    All values are floats in SI units.

    The radius changes linearly from r0 at x=0 to rL at x=L:

        r(x) = r0 * (1 - k*x)

    where

        k = (r0 - rL) / (r0 * L)

    Parameters
    ----------
    c_m : float
        Membrane capacitance density [F/m^2].
    rm : float
        Membrane resistance [ohm*m^2].
    gL : float
        Leak conductance density [S/m^2].
    ra : float
        Axial resistivity [ohm*m].

    N : int
        Number of spatial nodes.
    L : float
        Cable length [m].
    dx : float
        Spatial step [m].

    tau : float
        Membrane time constant [s].

    r_at_0 : float
        Radius at x=0 [m].
    r_at_L : float
        Radius at x=L [m].

    I_e : float
        External current [A].
    I_i : float
        Inhibitory current [A].

    t : float
        Simulation time [s].
    dt : float
        Time step [s].

    x : ndarray
        Spatial grid [m].
    """

    c_m: float          # F/m^2
    rm: float            # ohm*m^2
    gL: float            # S/m^2
    ra: float            # ohm*m

    N: int
    L: float             # m
    dx: float            # m

    tau: float           # s

    r_at_0: float            # m
    r_at_L: float            # m

    I_e: float            # A
    I_i: float = to_SI(30 * pampere)  # A

    t: float = 0.003      # s
    dt: float = 3e-6      # s

    x: np.ndarray | None = None

    def __str__(self):
        return (
            "ConicalNumericalCableParameters (SI)\n"
            f"  c_m = {self.c_m:.4e} F/m²\n"
            f"  rm  = {self.rm:.4e} Ω·m²\n"
            f"  gL  = {self.gL:.4e} S/m²\n"
            f"  ra  = {self.ra:.4e} Ω·m\n"
            f"  N   = {self.N}\n"
            f"  L   = {self.L:.4e} m\n"
            f"  dx  = {self.dx:.4e} m\n"
            f"  tau = {self.tau:.4e} s\n"
            f"  r0  = {self.r_at_0:.4e} m\n"
            f"  rL  = {self.r_at_L:.4e} m\n"
            f"  k   = {self.k:.4e} 1/m\n"
            f"  Ie  = {self.I_e:.4e} A\n"
            f"  Ii  = {self.I_i:.4e} A"
        )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    @property
    def k(self):
        """
        Linear taper parameter [1/m].

            r(x) = r0 * (1 - k*x)
        """
        return (
            (self.r_at_0 - self.r_at_L)
            /
            (self.r_at_0 * self.L)
        )

    def radius(self, x):
        """
        Local radius r(x) [m].
        """
        x = np.asarray(x)

        return self.r_at_0 * (1.0 - self.k * x)

    # --------------------------------------------------------
    # Cable properties
    # --------------------------------------------------------

    def lambd(self, x):
        """
        Local space constant lambda(x) [m].

            lambda(x) = sqrt(
                r(x) * rm / (2 * ra)
            )
        """
        r = self.radius(x)

        return np.sqrt(
            r * self.rm / (2.0 * self.ra)
        )

    def R_lambda(self, x=0.0):
        """
        Local axial resistance over one space constant [ohm].
        """
        r = self.radius(x)
        lambd = self.lambd(x)

        return self.rm / (
            2.0 * np.pi * r * lambd
        )

    # --------------------------------------------------------
    # PDE coefficients
    # --------------------------------------------------------

    def a(self):
        """
        First derivative coefficient.

        The PDE contains

            -a * dV/dx

        with

            a =
                k*r0 /
                (
                    c_m*ra*sqrt(1 + k^2*r0^2)
                )

        Units: m/s.
        """
        return (
            self.k * self.r_at_0
            /
            ( self.c_m
                * self.ra
                * np.sqrt(
                1.0 + (self.k * self.r_at_0) ** 2
                )
            )
        )

    def b(self, x=None):
        """
        Second derivative coefficient b(x).

            b(x) =
                r(x) /
                (
                    2*c_m*ra*sqrt(1 + k^2*r0^2)
                )

        Units: m^2/s.
        """
        if x is None:
            x = self.x

        r = self.radius(x)

        return (
            r
            /
            (
                2.0
                * self.c_m
                * self.ra
                * np.sqrt(
                1.0 + (self.k * self.r_at_0) ** 2
                )
            )
        )

    def dt_stability(self):
        """
        Explicit diffusion stability limit.

        dt <= dx^2 / (2*max(b)).
        """
        return (
            0.5
            * self.dx ** 2
            / np.max(self.b())
        )

    def to_numerical_cable_params_at(self, x0):
        """
        Convert the conical cable into a local cylindrical cable at x0.

        The returned NumericalCableParameters represents a cylinder whose
        radius is equal to the local conical radius r(x0).

        All other cable parameters are copied unchanged.

        Parameters
        ----------
        x0 : Brian2 length quantity
            Position along the cone at which the local cylinder is defined.

        Returns
        -------
        NumericalCableParameters
            Numerical SI representation of the local cylinder.
        """

        # Local radius of the cone
        r_local = self.radius(x0)

        # Construct an ordinary cylindrical CableParameters object
        local = CableParameters(
            N=self.N,
            c_m=self.c_m,
            rm=self.rm,
            gL=self.gL,
            ra=self.ra,

            L=self.L,
            dx=self.dx,

            tau=self.tau,
            r0=r_local,

            b=None,

            I_e=self.I_e,
            I_i=self.I_i,

            t=self.t,
            dt=self.dt,

            x=self.x,
        )

        return local.to_numerical()

    def to_numerical_cylindrical_params(self, x0=0.0):
        """
        Convert this conical cable to a cylindrical cable for comparison.

        The cylinder keeps the same electrical properties, length, grid, time
        step, and input strength. Its radius is constant and equal to the cone
        radius at x0. By default x0=0, so the cylinder uses the proximal radius.
        """
        return self.to_numerical_cable_params_at(x0)


# ============================================================
# Unit-aware conical cable
# ============================================================

@dataclass(frozen=True)
class ConicalCableParameters:
    """
    Unit-aware representation of a conical cable.

    The radius changes uniformly from r0 to rL:

        r(x) = r0 * (1 - k*x)

    where

        k = (r0-rL)/(r0*L)

    All physical quantities carry Brian2 units.
    """

    N: int = 101

    c_m: Quantity | None = None
    rm: Quantity | None = None
    gL: Quantity | None = None
    ra: Quantity | None = None

    L: Quantity | None = None
    dx: Quantity | None = None

    tau: Quantity | None = None

    r_at_0: Quantity | None = None
    r_at_L: Quantity | None = None

    I_e: Quantity | None = None
    I_i: Quantity | None = field(default_factory=lambda: 30 * pampere)

    t: Quantity | None = None
    dt: Quantity | None = None

    x: np.ndarray | None = None

    def __post_init__(self):

        # ----------------------------------------------------
        # Rm -> gL
        # ----------------------------------------------------

        if self.rm is not None and self.gL is None:
            object.__setattr__(
                self,
                "gL",
                1 / self.rm
            )

        # ----------------------------------------------------
        # c_m + gL -> tau
        # ----------------------------------------------------

        if (
            self.c_m is not None
            and self.gL is not None
            and self.tau is None
        ):
            object.__setattr__(
                self,
                "tau",
                self.c_m / self.gL
            )

        # ----------------------------------------------------
        # If rL is not specified, make it cylindrical
        # ----------------------------------------------------

        if self.r_at_L is None and self.r_at_0 is not None:
            object.__setattr__(
                self,
                "rL",
                self.r_at_0
            )

        # ----------------------------------------------------
        # L + N -> dx and x
        # ----------------------------------------------------

        if (
            self.L is not None
            and self.N is not None
        ):
            object.__setattr__(
                self,
                "dx",
                self.L / (self.N - 1)
            )

            if (
                self.x is None
                or len(self.x) != self.N
            ):
                object.__setattr__(
                    self,
                    "x",
                    np.linspace(
                        0,
                        float(self.L / meter),
                        self.N
                    ) * meter
                )

    def __str__(self):

        s = (
            "ConicalCableParameters:\n"
            f"  c_m = {self.c_m}\n"
            f"  Rm  = {self.rm}\n"
            f"  gL  = {self.gL}\n"
            f"  ra  = {self.ra}\n"
            f"  L   = {self.L}\n"
            f"  N   = {self.N}\n"
            f"  dx  = {self.dx}\n"
            f"  tau = {self.tau}\n"
            f"  r0  = {self.r_at_0}\n"
            f"  rL  = {self.r_at_L}\n"
            f"  k   = {self.k}\n"
            f"  I_e = {self.I_e}\n"
            f"  I_i = {self.I_i}\n"
            f"  t   = {self.t}\n"
            f"  dt  = {self.dt}\n"
        )

        if self.x is not None:
            s += (
                f"  x   = "
                f"[{self.x[0]} -> {self.x[-1]}] "
                f"({len(self.x)} points)\n"
            )

        return s

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    @property
    def k(self):
        """
        Linear taper parameter [1/m].
        """
        return (
            (self.r_at_0 - self.r_at_L)
            /
            (self.r_at_0 * self.L)
        )

    def radius(self, x):
        """
        Local radius r(x).
        """
        return self.r_at_0 * (
            1.0 - self.k * x
        )

    # --------------------------------------------------------
    # Cable properties
    # --------------------------------------------------------

    def lambd(self, x):
        """
        Local space constant lambda(x).
        """
        r = self.radius(x)

        return np.sqrt(
            r * self.rm / (2.0 * self.ra * math.sqrt(1 + (self.k * self.r_at_0) ** 2))
        )

    def R_lambda(self, x=0.0):
        """
        Local axial resistance over one space constant.
        """
        r = self.radius(x)
        lambd = self.lambd(x)

        return self.rm / (
            2.0 * np.pi * r * lambd
        )

    # --------------------------------------------------------
    # PDE coefficients
    # --------------------------------------------------------

    def a(self):
        """
        First derivative coefficient.
        """
        return (
            self.k * self.r_at_0
            /
            ( self.c_m
                * self.ra
                * np.sqrt(
                        1.0 + (self.k * self.r_at_0) ** 2
                )
            )
        )

    def b(self, x=None):
        """
        Local second derivative coefficient b(x).
        """
        if x is None:
            x = self.x

        r = self.radius(x)

        return (
            r
            /
            (
                2.0
                * self.c_m
                * self.ra
                * np.sqrt(
                1.0 + (self.k * self.r_at_0) ** 2
                )
            )
        )

    # --------------------------------------------------------
    # Conversions
    # --------------------------------------------------------

    def to_numerical(self):
        """
        Convert to ConicalNumericalCableParameters.

        All returned values are plain SI floats.
        """

        return ConicalNumericalCableParameters(
            c_m=to_SI(
                self.c_m,
                farad / meter**2
            ),

            rm=to_SI(
                self.rm,
                ohm * meter**2
            ),

            gL=to_SI(
                self.gL,
                siemens / meter**2
            ),

            ra=to_SI(
                self.ra,
                ohm * meter
            ),

            N=self.N,

            L=to_SI(
                self.L,
                meter
            ),

            dx=to_SI(
                self.dx,
                meter
            ),

            tau=to_SI(
                self.tau,
                second
            ),

            r_at_0=to_SI(
                self.r_at_0,
                meter
            ),

            r_at_L=to_SI(
                self.r_at_L,
                meter
            ),

            I_e=to_SI(
                self.I_e,
                ampere
            ),
            I_i=to_SI(
                self.I_i,
                ampere
            ),

            t=to_SI(
                self.t,
                second
            ),

            dt=to_SI(
                self.dt,
                second
            ),

            x=to_SI(
                self.x,
                meter
            ),
        )

    # --------------------------------------------------------
    # Construction from SI
    # --------------------------------------------------------

    @classmethod
    def from_SI(
        cls,
        *,
        c_m=None,
        rm=None,
        gL=None,
        ra=None,

        L=None,
        N=101,

        x=None,
        dx=None,

        tau=None,

        r0=None,
        rL=None,

        I_e=None,
        I_i=to_SI(30 * pampere),

        t=None,
        dt=None):
        """
        Construct from plain SI values.
        """

        c_m = (
            None
            if c_m is None
            else c_m * farad / meter**2
        )

        rm = (
            None
            if rm is None
            else rm * ohm * meter**2
        )

        gL = (
            None
            if gL is None
            else gL * siemens / meter**2
        )

        ra = (
            None
            if ra is None
            else ra * ohm * meter
        )

        L = (
            None
            if L is None
            else L * meter
        )

        x = (
            None
            if x is None
            else np.asarray(x) * meter
        )

        dx = (
            None
            if dx is None
            else dx * meter
        )

        tau = (
            None
            if tau is None
            else tau * second
        )

        r0 = (
            None
            if r0 is None
            else r0 * meter
        )

        rL = (
            None
            if rL is None
            else rL * meter
        )

        I_e = (
            None
            if I_e is None
            else I_e * ampere
        )
        I_i = (
            30 * pampere
            if I_i is None
            else I_i * ampere
        )

        t = (
            None
            if t is None
            else t * second
        )

        dt = (
            None
            if dt is None
            else dt * second
        )

        return cls(
            N=N,

            c_m=c_m,
            rm=rm,
            gL=gL,
            ra=ra,

            L=L,
            dx=dx,

            tau=tau,

            r_at_0=r0,
            r_at_L=rL,

            I_e=I_e,
            I_i=I_i,

            t=t,
            dt=dt,

            x=x,
        )

    @classmethod
    def from_numerical(
        cls,
        other: ConicalNumericalCableParameters,
    ):
        """
        Construct a unit-aware object from a numerical
        conical parameter object.
        """

        return cls.from_SI(
            c_m=other.c_m,
            rm=other.rm,
            gL=other.gL,
            ra=other.ra,

            L=other.L,
            N=other.N,

            x=other.x,
            dx=other.dx,

            tau=other.tau,

            r0=other.r_at_0,
            rL=other.r_at_L,

            I_e=other.I_e,
            I_i=getattr(other, "I_i", to_SI(30 * pampere)),

            t=other.t,
            dt=other.dt,
        )

    # --------------------------------------------------------
    # Updating
    # --------------------------------------------------------

    def with_property(self, **changes):
        """
        Return a new ConicalCableParameters object.

        Values must be supplied with Brian2 units.
        """

        return replace(
            self,
            **changes
        )

    def with_SI_properties(self, **changes):
        """
        Return a new ConicalCableParameters object.

        Values are supplied as plain SI values.
        """

        converted = {}

        for key, value in changes.items():

            if value is None:
                converted[key] = None

            elif key == "c_m":
                converted[key] = (
                    value * farad / meter**2
                )

            elif key == "rm":
                converted[key] = (
                    value * ohm * meter**2
                )

            elif key == "gL":
                converted[key] = (
                    value * siemens / meter**2
                )

            elif key == "ra":
                converted[key] = (
                    value * ohm * meter
                )

            elif key == "L":
                converted[key] = value * meter

            elif key == "dx":
                converted[key] = value * meter

            elif key == "tau":
                converted[key] = value * second

            elif key == "r_at_0":
                converted[key] = value * meter

            elif key == "r_at_L":
                converted[key] = value * meter

            elif key == "I_e":
                converted[key] = value * ampere

            elif key == "I_i":
                converted[key] = value * ampere

            elif key == "N":
                converted[key] = value

            elif key == "t":
                converted[key] = value * second

            elif key == "dt":
                converted[key] = value * second

            elif key == "x":
                converted[key] = (
                    np.asarray(value) * meter
                )

            else:
                raise ValueError(
                    f"Unknown conical cable parameter "
                    f"'{key}'"
                )

        values = self.__dict__.copy()
        values.update(converted)

        # ----------------------------------------------------
        # Derived quantities
        # ----------------------------------------------------

        # Rm -> gL
        if (
            "rm" in converted
            or "gL" in converted
        ):
            if values["rm"] is not None:
                values["gL"] = (
                    1 / values["rm"]
                )

        # c_m + gL -> tau
        if (
            values["c_m"] is not None
            and values["gL"] is not None
        ):
            values["tau"] = (
                values["c_m"]
                /
                values["gL"]
            )

        # L + N -> dx + x
        if (
            values["L"] is not None
            and values["N"] is not None
        ):
            values["dx"] = (
                values["L"]
                /
                (values["N"] - 1)
            )


            values["x"] = (
                np.linspace(
                    0,
                    float(
                        values["L"] / meter
                    ),
                    values["N"]
                )
                * meter
            )

        return replace(
            self,
            **values
        )

import numpy as np


def create_delta_pulses(
        t_max,
        x_distribution,
        t_distribution,
        seed=None):
    """
    Generate a set of spatial-temporal delta pulses.

    The continuous stimulus is

        I(x,t) = sum_j delta(x-x_j) delta(t-t_j)

    Parameters
    ----------

    t_max : float
        Maximum simulation time [s].

    x_distribution :
        Distribution object providing ``rvs()`` for spatial locations.

        Example:
            scipy.stats.uniform(loc=a, scale=b-a)

    t_distribution :
        Distribution of inter-arrival times.

        Example:
            scipy.stats.expon(scale=1/r_i)

    Returns
    -------
    pulses : ndarray
        Array with shape ``(n_time, n_space)``.

        Each event is represented by a discretized delta function.

    event_times : ndarray
        Actual event times [s].

    event_positions : ndarray
        Actual event positions [m].
    """

    rng = np.random.default_rng(seed)

    # Generate spike times using the same RNG
    event_times = generate_spike_times(
        t_distribution=t_distribution,
        t_max=t_max,
        random_state=rng,
    )

    # Generate corresponding spatial positions
    event_positions = x_distribution.rvs(
        size=len(event_times),
        random_state=rng,
    )

    return np.vstack(
        (event_times, event_positions)
    )


def generate_event_positions(n_events, x_distribution, ):
    if len(n_events) == 0:
        return (
            np.zeros(n_events),
            np.array([]),
            np.array([])
        )

    return x_distribution.rvs(size=len(n_events))

def generate_spike_times(t_distribution, t_max, random_state=None):
    event_times = []
    t = 0.0
    # Expected number of events over the simulation interval.
    expected_events = t_max / t_distribution.mean()
    # Generate at least twice the expected number at once.
    batch_size = max(1000, int(2 * np.ceil(expected_events)))
    while True:

        # Draw a batch of inter-arrival times
        delta_t = t_distribution.rvs(size=batch_size, random_state=random_state)

        # Convert inter-arrival times to absolute event times
        times = np.cumsum(delta_t)

        # Keep only events before t_max
        valid = times < t_max

        if np.any(valid):
            event_times.append(times[valid])

        # If the first event beyond t_max has occurred, we're done
        if np.any(~valid):
            break

        # Continue from the last generated event
        t_current = times[-1]
    event_times = np.concatenate(event_times) if event_times else np.empty(0)
    return event_times

def find_multiple_events_per_cell(events, dt, dx):
    occupied = set()

    for i, (t, x) in enumerate(zip(events[0], events[1])):

        time_index = int(np.floor(t / dt))
        space_index = int(np.floor(x / dx))

        cell = (time_index, space_index)

        if cell in occupied:
            return {
                "event_index": i,
                "time": t,
                "position": x,
                "time_index": time_index,
                "space_index": space_index,
            }

        occupied.add(cell)

    return None

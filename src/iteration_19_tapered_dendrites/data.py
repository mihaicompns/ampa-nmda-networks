import math
from dataclasses import dataclass, field

from brian2 import Quantity, farad, meter, ohm, siemens, second, have_same_dimensions, volt, is_dimensionless
from brian2.units.allunits import ampere, pampere
from dataclasses import replace
import numpy as np


def to_SI(value, unit = None):
    if value is None:
        return None

    if np.ndim(value) == 0:
        sample = value
    else:
        sample = value.flat[0]

    if unit is None:
        # Time
        if have_same_dimensions(sample, second):
            unit = second

        # Length
        elif have_same_dimensions(sample, meter):
            unit = meter

        # Voltage
        elif have_same_dimensions(sample, volt):
            unit = volt

        # Current
        elif have_same_dimensions(sample, ampere):
            unit = ampere

        # Current density
        elif have_same_dimensions(sample, ampere / meter ** 2):
            unit = ampere / meter ** 2

        # Capacitance density
        elif have_same_dimensions(sample, farad / meter ** 2):
            unit = farad / meter ** 2

        # Conductance density
        elif have_same_dimensions(sample, siemens / meter ** 2):
            unit = siemens / meter ** 2

        # Membrane resistance
        elif have_same_dimensions(sample, ohm * meter ** 2):
            unit = ohm * meter ** 2

        # Axial resistivity
        elif have_same_dimensions(sample, ohm * meter):
            unit = ohm * meter

        # Diffusion coefficient
        elif have_same_dimensions(sample, meter ** 2 / second):
            unit = meter ** 2 / second

        # Frequency / rate
        elif have_same_dimensions(sample, 1 / second):
            unit = 1 / second

        # Dimensionless
        elif is_dimensionless(sample):
            unit = 1

    value_in_unit = value / unit

    if np.isscalar(value_in_unit):
        return float(value_in_unit)

    return np.asarray(value_in_unit, dtype=float)


@dataclass(frozen=True)
class NumericalCableParameters:
    """
    Solver representation.
    All values are floats in SI base units.
    """

    c_m: float  # F/m^2
    rm: float  # ohm*m^2
    gL: float  # S/m^2
    ra: float  # ohm*m. This one is called rL in Dayan

    N: int
    L: float  # m
    dx: float  # m
    tau: float  # s
    r0: float  # m
    b: float  # m^2/s

    I_e: float  # A
    I_i: float = to_SI(30 * pampere)  # A
    t: float = 0.003  # seconds
    dt: float = 3E-6  # seconds
    x: np.ndarray | None = None

    def __str__(self):
        return (
            "NumericalCableParameters (SI)\n"
            f"  c_m = {self.c_m:.4e} F/m²\n"
            f"  rm  = {self.rm:.4e} Ω·m²\n"
            f"  ra  = {self.ra:.4e} Ω·m\n"
            f"  N   = {self.N}\n"
            f"  L   = {self.L:.4e} m\n"
            f"  dx  = {self.dx:.4e} m\n"
            f"  tau = {self.tau:.4e} s\n"
            f"  b   = {self.b:.4e} m²/s\n"
            f"  Ie  = {self.I_e:.4e} A\n"
            f"  Ii  = {self.I_i:.4e} A"
        )

    def radius(self, x):
        """
        Local radius r(x) [m]. Constant along the cable for a cylinder.
        """
        return np.full_like(np.asarray(x, dtype=float), self.r0)

    def lambd(self):
        return math.sqrt(self.r0 * self.rm / (2 * self.ra))

    def R_lambda(self): # Ohm
        return self.rm / (2 * np.pi * self.r0 * self.lambd())


@dataclass(frozen=True)
class CableParameters:
    N: int = 101
    c_m: Quantity | None = None
    rm: Quantity | None = None
    gL: Quantity | None = None
    ra: Quantity | None = None

    L: Quantity | None = None
    dx: Quantity | None = None

    tau: Quantity | None = None
    r0: Quantity | None = None
    b: Quantity | None = None

    I_e: Quantity | None = None
    I_i: Quantity | None = field(default_factory=lambda: 30 * pampere)

    # simulation parameters
    t: Quantity | None = None
    dt: Quantity | None = None

    x: np.ndarray | None = None

    def __str__(self):
        s = (
            "CableParameters:\n"
            f"  c_m = {self.c_m}\n"
            f"  Rm  = {self.rm}\n"
            f"  ra  = {self.ra}\n"
            f"  L   = {self.L}\n"
            f"  N   = {self.N}\n"
            f"  dx  = {self.dx}\n"
            f"  tau = {self.tau}\n"
            f"  r0  = {self.r0}\n"
            f"  b   = {self.b}\n"
            f"  I_e = {self.I_e}\n"
            f"  I_i = {self.I_i}\n"
            f"  t   = {self.t}\n"
            f"  dt  = {self.dt}\n"
        )

        if self.x is not None:
            s += f"  x   = [{self.x[0]} -> {self.x[-1]}] ({len(self.x)} points)\n"

        return s

    def lambd(self):
        lambda_sq = self.r0 * self.rm / (2 * self.ra)
        return np.sqrt(lambda_sq)

    def R_lambda(self): # Ohm
        return self.rm / (2 * np.pi * self.r0 * self.lambd())

    def __post_init__(self):

        # Rm -> gL
        if self.rm is not None and self.gL is None:
            object.__setattr__(
                self,
                "gL",
                1 / self.rm
            )

        # c_m + gL -> tau
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

        # L + N -> dx and x
        if (
                self.L is not None
                and self.N is not None
        ):
            object.__setattr__(
                self,
                "dx",
                self.L / (self.N - 1)
            )

            if self.x is None or len(self.x) != self.N:
                object.__setattr__(
                    self,
                    "x",
                    np.linspace(
                        0,
                        float(self.L / meter),
                        self.N
                    ) * meter
                )

        # r0 + c_m + ra -> b
        if (
                self.r0 is not None
                and self.c_m is not None
                and self.ra is not None
                and self.b is None
        ):
            object.__setattr__(
                self,
                "b",
                self.r0 / (2 * self.c_m * self.ra)
            )

    def to_numerical(self):
        return NumericalCableParameters(
            c_m=to_SI(self.c_m, farad / meter ** 2),
            rm=to_SI(self.rm, ohm * meter ** 2),
            gL=to_SI(self.gL, siemens / meter ** 2),
            ra=to_SI(self.ra, ohm * meter),
            N=self.N,
            L=to_SI(self.L, meter),
            dx=to_SI(self.dx, meter),
            x=to_SI(self.x, meter),
            tau=to_SI(self.tau, second),
            r0=to_SI(self.r0, meter),

            b=to_SI(self.b, meter ** 2 / second),

            t=to_SI(self.t, second),
            dt=to_SI(self.dt, second),

            I_e=to_SI(self.I_e, ampere),
            I_i=to_SI(self.I_i, ampere)
        )

    @classmethod
    def from_numerical(cls, other: NumericalCableParameters):
        """
        Construct CableParameters from a NumericalCableParameters object.
        All numerical values are assumed to be in SI units.
        """
        return cls.from_SI(
            c_m=other.c_m,
            rm=other.rm,
            gL=other.gL,
            ra=other.ra,
            L=other.L,
            N=other.N,
            dx=other.dx,
            x=other.x,
            tau=other.tau,
            r0=other.r0,
            b=other.b,

            t=other.t,
            dt=other.dt,

            I_e=other.I_e,
            I_i=getattr(other, "I_i", to_SI(30 * pampere)),
        )

    @classmethod
    def from_SI(
            cls,
            *,
            c_m=None,  # F/m^2
            rm=None,  # ohm*m^2
            gL=None,  # S/m^2
            ra=None,  # ohm*m

            L=None,  # m
            N=101,

            x=None,
            dx=None,  # m

            tau=None,  # s
            r0=None,  # m

            b=None,  # m^2/s

            I_e=None,  # A
            I_i=to_SI(30 * pampere),  # A
            t=None,  # s
            dt=None  # s
    ):
        # Attach Brian2 units
        c_m = None if c_m is None else c_m * farad / meter ** 2
        rm = None if rm is None else rm * ohm * meter ** 2
        gL = None if gL is None else gL * siemens / meter ** 2
        ra = None if ra is None else ra * ohm * meter

        L = None if L is None else L * meter
        x = None if x is None else np.asarray(x) * meter
        dx = None if dx is None else dx * meter

        tau = None if tau is None else tau * second
        r0 = None if r0 is None else r0 * meter

        b = None if b is None else b * meter ** 2 / second

        I_e = None if I_e is None else I_e * ampere
        I_i = 30 * pampere if I_i is None else I_i * ampere

        # Derived quantities
        if gL is None and rm is not None:
            gL = 1 / rm

        if tau is None and c_m is not None and gL is not None:
            tau = c_m / gL

        if dx is None and L is not None and N is not None:
            dx = L / (N - 1)

        if b is None and r0 is not None and c_m is not None and ra is not None:
            b = r0 / (2 * c_m * ra)

        t = None if t is None else t * second
        dt = None if dt is None else dt * second

        return cls(
            c_m=c_m,
            rm=rm,
            gL=gL,
            ra=ra,

            L=L,
            N=N,
            x=x,
            dx=dx,

            tau=tau,
            r0=r0,
            b=b,

            I_e=I_e,
            I_i=I_i,
            t=t,
            dt=dt
        )

    def with_property(self, **changes):
        """
        Return a new CableParameters object with updated Brian2 quantities.

        Derived quantities are recomputed automatically by __post_init__.
        """

        return replace(self, **changes)

    def with_SI_properties(self, **changes):

        converted = {}

        for key, value in changes.items():

            if value is None:
                converted[key] = None

            elif key == "c_m":
                converted[key] = value * farad / meter ** 2

            elif key == "rm":
                converted[key] = value * ohm * meter ** 2

            elif key == "gL":
                converted[key] = value * siemens / meter ** 2

            elif key == "ra":
                converted[key] = value * ohm * meter

            elif key == "L":
                converted[key] = value * meter

            elif key == "dx":
                converted[key] = value * meter

            elif key == "tau":
                converted[key] = value * second

            elif key == "r0":
                converted[key] = value * meter

            elif key == "b":
                converted[key] = value * meter ** 2 / second

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

            else:
                raise ValueError(
                    f"Unknown cable parameter '{key}'"
                )

        values = self.__dict__.copy()
        values.update(converted)

        # Recompute derived quantities
        if "rm" in converted or "gL" in converted:
            if values["rm"] is not None:
                values["gL"] = 1 / values["rm"]

        if (
                values["c_m"] is not None
                and values["gL"] is not None
        ):
            values["tau"] = (
                    values["c_m"] /
                    values["gL"]
            )

        if (
                values["L"] is not None
                and values["N"] is not None
        ):
            values["dx"] = (
                    values["L"] /
                    (values["N"] - 1)
            )

        if (
                values["r0"] is not None
                and values["c_m"] is not None
                and values["ra"] is not None
        ):
            values["b"] = (
                    values["r0"]
                    /
                    (2 * values["c_m"] * values["ra"])
            )

        return replace(self, **values)

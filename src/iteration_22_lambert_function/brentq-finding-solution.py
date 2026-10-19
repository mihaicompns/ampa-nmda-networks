import numpy as np
from scipy.special import lambertw
from scipy.optimize import brentq

# Biophysical parameters
g_L = 3.056       # nS
A = 0.25          # dimensionless
B = 0.08          # mV^-1
E_L = -70.0       # mV
E_E = 0.0         # mV

# Exact saddle-node condition for W_+
def H(r):
    W = lambertw(-np.e * r, k=0).real

    D_physical = B * (E_E - E_L) * (
        1 - np.exp(B * E_L) / (A * r)
    )

    D_SN = -(W - 1)**2 / W

    return D_physical - D_SN


# Admissible interval: positive NMDA conductance and real W_+
r_min = np.exp(B * E_L) / A
r_max = np.nextafter(np.exp(-2), 0.0)

# Solve exact saddle-node equation
r_b = brentq(H, r_min, r_max, xtol=1e-14)

# Recover bifurcation parameters
D_b = B * (E_E - E_L) * (
    1 - np.exp(B * E_L) / (A * r_b)
)

g_NMDA_b = g_L * (
    A * r_b * np.exp(-B * E_L) - 1
)

V_b = E_L - (lambertw(-np.e * r_b, k=0).real - 1) / B

print(f"r_b       = {r_b:.12f}")
print(f"D_b       = {D_b:.12f}")
print(f"g_NMDA_b  = {g_NMDA_b:.12f} nS")
print(f"V_b       = {V_b:.12f} mV")
print(f"Residual  = {H(r_b):.3e}")
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import lambertw

def W_r_parametric(r, x_min=-10, x_max=0.2, n=1000):
    """
        Parametric representation of the real branches of W_r.

        Returns (a, W_r(a)), where

            a = f_r(x) = x exp(x) + r x,
            W_r(a) = x.

        NaNs separate distinct real branches.
        """

    r_crit = np.exp(-2)

    # ---------------------------------------------------------
    # Case 1: r < 0
    # One minimum -> two real inverse branches
    # ---------------------------------------------------------
    if r < 0:
        gamma = lambertw(-r * np.e, 0).real - 1

        x1 = np.linspace(x_min, gamma, n)
        x2 = np.linspace(gamma, x_max, n)

        x = np.concatenate((x1, [np.nan], x2))

    # ---------------------------------------------------------
    # Case 2: r = 0
    # Classical Lambert W:
    # f_0(x) = x exp(x)
    # minimum at x = -1
    # ---------------------------------------------------------
    elif np.isclose(r, 0):
        gamma = -1.0

        x1 = np.linspace(x_min, gamma, n)
        x2 = np.linspace(gamma, x_max, n)

        x = np.concatenate((x1, [np.nan], x2))

    # ---------------------------------------------------------
    # Case 3: 0 < r < exp(-2)
    # Maximum + minimum -> three real inverse branches
    # ---------------------------------------------------------
    elif r < r_crit and not np.isclose(r, r_crit):
        alpha = lambertw(-r * np.e, -1).real - 1
        beta = lambertw(-r * np.e, 0).real - 1

        x1 = np.linspace(x_min, alpha, n)
        x2 = np.linspace(alpha, beta, n)
        x3 = np.linspace(beta, x_max, n)

        x = np.concatenate(
            (x1, [np.nan], x2, [np.nan], x3)
        )

    # ---------------------------------------------------------
    # Case 4: r = exp(-2)
    # Degenerate stationary point at x = -2.
    # The inverse is continuous but not differentiable there.
    # ---------------------------------------------------------
    elif np.isclose(r, r_crit):
        xc = -2.0

        x1 = np.linspace(x_min, xc, n)
        x2 = np.linspace(xc, x_max, n)

        # No NaN: continuously connected
        x = np.concatenate((x1, x2))

    # ---------------------------------------------------------
    # Case 5: r > exp(-2)
    # Strictly monotonic -> one real branch
    # ---------------------------------------------------------
    else:
        x = np.linspace(x_min, x_max, n)

    # Parametric inverse:
    # horizontal coordinate = a = f_r(x)
    # vertical coordinate   = W_r(a) = x
    a = x * np.exp(x) + r * x

    return a, x

for r in [0.2, 0.1, 0, -0.1]:
    a, Wr = W_r_parametric(r)
    plt.plot(a, Wr, label=fr"$r={r}$")

plt.xlabel(r"$a$")
plt.ylabel(r"$W_r(a)$")
plt.legend()
plt.show()
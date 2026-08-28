"""Raw SVI slice algebra: JW <-> raw conversion, analytic derivatives,
single-slice arbitrage diagnostics.

Total implied variance of a raw SVI slice, in log forward moneyness
y = ln(K / F):

    w(y) = a + b * ( rho * (y - m) + sqrt( (y - m)^2 + sigma^2 ) )

Everything here is expiry-independent: a slice does not know its own tau.
Term structure lives in `surface.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# raw SVI slice
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SVIRaw:
    """Raw SVI parameters (a, b, rho, m, sigma) for one expiry."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    # -- total variance and its analytic y-derivatives -------------------- #
    def w(self, y):
        y = np.asarray(y, dtype=float)
        d = y - self.m
        return self.a + self.b * (self.rho * d + np.sqrt(d * d + self.sigma ** 2))

    def dw_dy(self, y):
        y = np.asarray(y, dtype=float)
        d = y - self.m
        return self.b * (self.rho + d / np.sqrt(d * d + self.sigma ** 2))

    def d2w_dy2(self, y):
        y = np.asarray(y, dtype=float)
        d = y - self.m
        return self.b * self.sigma ** 2 / np.power(d * d + self.sigma ** 2, 1.5)

    # -- shape ------------------------------------------------------------ #
    @property
    def w_min(self) -> float:
        """Minimum total variance over the slice (attained at the vertex)."""
        return self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho ** 2)

    def is_well_posed(self) -> tuple[bool, list[str]]:
        """Domain constraints from Gatheral & Jacquier, section 3.1."""
        msgs = []
        if not (self.b >= 0):
            msgs.append(f"b = {self.b:.6g} < 0")
        if not (abs(self.rho) < 1):
            msgs.append(f"|rho| = {abs(self.rho):.6g} >= 1")
        if not (self.sigma > 0):
            msgs.append(f"sigma = {self.sigma:.6g} <= 0")
        if not (self.w_min >= 0):
            msgs.append(f"a + b*sigma*sqrt(1-rho^2) = {self.w_min:.6g} < 0")
        return (len(msgs) == 0), msgs

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.a, self.b, self.rho, self.m, self.sigma)


# --------------------------------------------------------------------------- #
# Step 1 -- SVI-JW  ->  SVI-raw
# --------------------------------------------------------------------------- #
def jw_to_raw(atm_var: float, skew: float, putwing: float, callwing: float,
              min_imp_var: float, tau: float,
              beta_clamp: float = 0.0) -> SVIRaw:
    """Convert the five SVI-JW quotes of one expiry into raw SVI parameters.

    Implements Lemma 3.2 of Gatheral & Jacquier (2012).

    Parameters
    ----------
    atm_var     : ATMVol ** 2
    skew        : ATM skew  psi_t = d(sigma_BS)/dk  at k = 0
    putwing     : left wing slope   p_t
    callwing    : right wing slope  c_t
    min_imp_var : Kurt ** 2, the minimum implied VARIANCE
    tau         : dt_vol(VolDate), Business/260

    Skew convention
    ---------------
    Gatheral & Jacquier's psi_t is the TOTAL-VARIANCE skew

        psi_t = (1 / (2 sqrt(w_t))) dw/dk |_{k=0},   beta = rho - 2 psi_t / b * sqrt(w_t)

    whereas the desk quotes `Skew` as the VOLATILITY skew d(sigma_BS)/dk at
    k = 0.  The two differ by exactly sqrt(tau):

        sigma(k) = sqrt(w(k)/tau)  =>  d(sigma)/dk|_0 = psi_t / sqrt(tau)

    so psi_t = Skew * sqrt(tau) and

        beta = rho - 2 * Skew * sqrt(omg * tau) / b.

    The sqrt(omg * tau) in the task statement is therefore CORRECT, not a typo:
    the extra sqrt(tau) converts the quoted volatility skew into Gatheral's
    total-variance skew.  `tests.py::test_quoted_skew_is_the_volatility_skew`
    pins this down by differentiating the calibrated smile numerically.

    Note that `raw_to_jw` returns psi_t, NOT the quoted Skew; divide by
    sqrt(tau) to get back to the desk's field.

    beta_clamp
    ----------
    |beta| <= 1 is required (it is the convexity condition
    -putwing <= 2 psi <= callwing).  A handful of real quote days breach it by
    a fraction of a percent.  `beta_clamp` is the tolerance within which such a
    breach is pulled back onto the boundary instead of raising:

        beta_clamp = 0.0   raise on any breach (default, nothing is hidden)
        beta_clamp = 0.05  clamp |beta| in (1, 1.05] to 1 - 1e-12

    Clamping lands on the degenerate sigma -> 0 corner of the slice: a smile
    with a sharp vertex.  It keeps a day usable but it is a repair, so it is
    off unless asked for.
    """
    if tau <= 0:
        raise ValueError("tau must be strictly positive")

    omg = atm_var * tau                                   # ATM total variance w_t = JW v_t
    sqrt_omg = np.sqrt(omg)

    b = 0.5 * sqrt_omg * (putwing + callwing)
    if b <= 0:
        raise ValueError("b <= 0: putwing + callwing must be positive")

    rho = 1.0 - putwing * sqrt_omg / b
    # skew is d(sigma)/dk; psi_t = skew * sqrt(tau).  See the docstring.
    # These are internal Lemma 3.2 shape variables.  Keep their names distinct
    # from empirical hedge beta and stickiness alpha in the dynamic-alpha study.
    jw_beta = rho - 2.0 * skew * np.sqrt(omg * tau) / b

    if abs(jw_beta) > 1.0:
        if beta_clamp > 0.0 and abs(jw_beta) <= 1.0 + beta_clamp:
            jw_beta = float(np.sign(jw_beta)) * (1.0 - 1e-12)
        else:
            raise ValueError(
                f"SVI-JW beta = {jw_beta:.6g} outside [-1, 1]: the quoted smile is not "
                "convex (requires -putwing <= 2*skew <= callwing)"
            )

    jw_shape = np.sign(jw_beta) * np.sqrt(1.0 / jw_beta ** 2 - 1.0)

    num = (atm_var - min_imp_var) * tau
    #ATM total variance - mimimum implied variance, scaled by tau.  This is the numerator of m = num / den.
    den = b * (-rho + np.sign(jw_shape) * np.sqrt(1.0 + jw_shape ** 2)
               - jw_shape * np.sqrt(1.0 - rho ** 2))
    # m = num / den, where den is a function of the SVI-JW shape variables.

    if abs(den) < 1e-14:                                  # degenerate: m = 0
        m = 0.0
        if abs(abs(rho) - 1.0) < 1e-12:
            sigma = num / b
        else:
            sigma = num / (b * (1.0 - np.sqrt(1.0 - rho ** 2)))
    else:
        m = num / den
        sigma = jw_shape * m
        if m == 0.0:
            sigma = (num / b if abs(abs(rho) - 1.0) < 1e-12
                     else num / (b * (1.0 - np.sqrt(1.0 - rho ** 2))))
    #see the paper lemma 3.2 for when m=0

    a = min_imp_var * tau - b * sigma * np.sqrt(1.0 - rho ** 2)
    return SVIRaw(a=float(a), b=float(b), rho=float(rho),
                  m=float(m), sigma=float(sigma))
#checked


def raw_to_jw(raw: SVIRaw, tau: float) -> dict:
    """Inverse of `jw_to_raw` (equation 3.5).  Used for the round-trip self-check."""
    a, b, rho, m, sg = raw.as_tuple()
    root = np.sqrt(m * m + sg * sg)
    w_t = a + b * (-rho * m + root)
    sqrt_w = np.sqrt(w_t)
    return {
        "atm_vol": np.sqrt(w_t / tau),
        "skew": (b / (2.0 * sqrt_w)) * (-m / root + rho),
        "putwing": b * (1.0 - rho) / sqrt_w,
        "callwing": b * (1.0 + rho) / sqrt_w,
        "kurt": np.sqrt((a + b * sg * np.sqrt(1.0 - rho ** 2)) / tau),
    }


# --------------------------------------------------------------------------- #
# single-slice arbitrage diagnostics
# --------------------------------------------------------------------------- #
def g_function(w, dw, d2w, y):
    """Gatheral's g, equation (2.1).

    Identical to the denominator D of the Dupire-Gatheral local variance
    formula.  g > 0 everywhere  <=>  the risk-neutral density is positive
    <=>  the slice is free of butterfly arbitrage.
    """
    w = np.asarray(w, dtype=float)
    dw = np.asarray(dw, dtype=float)
    d2w = np.asarray(d2w, dtype=float)
    y = np.asarray(y, dtype=float)
    return (
        (1.0 - y * dw / (2.0 * w)) ** 2
        - 0.25 * dw ** 2 * (1.0 / w + 0.25)
        + 0.5 * d2w
    )


def butterfly_diagnostics(raw: SVIRaw, y_grid: np.ndarray) -> dict:
    """Evaluate g on a grid and report where (if anywhere) it turns negative."""
    y = np.asarray(y_grid, dtype=float)
    g = g_function(raw.w(y), raw.dw_dy(y), raw.d2w_dy2(y), y)
    bad = y[g <= 0.0]
    return {
        "min_g": float(g.min()),
        "arb_free": bool(g.min() > 0.0),
        "y_bad_lo": float(bad.min()) if bad.size else None,
        "y_bad_hi": float(bad.max()) if bad.size else None,
        "g": g,
    }


def crossedness(raw_lo: SVIRaw, raw_hi: SVIRaw, y_grid: np.ndarray) -> float:
    """Maximum amount by which the near slice pokes above the far slice.

    Definition 5.1 of the paper, evaluated on a grid instead of at the exact
    quartic roots.  Zero means no calendar spread arbitrage on the grid.
    """
    y = np.asarray(y_grid, dtype=float)
    gap = raw_lo.w(y) - raw_hi.w(y)
    return float(max(0.0, gap.max()))


def sufficient_butterfly_free_callwing(putwing: float, skew: float) -> float:
    """c' = p + 2 * psi, section 5.1: the call wing that guarantees no butterfly."""
    return putwing + 2.0 * skew

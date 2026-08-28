"""Black-Scholes with cost of carry.

The core routines are parameterised by TOTAL VARIANCE w rather than by
(sigma, T).  That removes the single most common bug in this project: mixing
the Business/260 volatility clock with the Act/365 discount clock.  The caller
supplies

    F  = forward           (built on dt_r)
    w  = total variance    (built on dt_vol)
    df = discount factor   (built on dt_r)

and the formulas never have to pick a time convention themselves.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# price
# --------------------------------------------------------------------------- #
def bs_price_w(F, K, w, df, is_call: bool = True):
    """Undiscounted-forward Black formula, discounted by df."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    w = np.maximum(np.asarray(w, dtype=float), 1e-300)
    sw = np.sqrt(w)
    d1 = np.log(F / K) / sw + 0.5 * sw
    d2 = d1 - sw
    if is_call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def bs_price(S, K, sigma, tau_vol, tau_r, r, b, is_call: bool = True):
    """Convenience wrapper taking (sigma, tau_vol) and (r, b, tau_r)."""
    F = S * np.exp(b * tau_r)
    w = np.asarray(sigma, dtype=float) ** 2 * tau_vol
    return bs_price_w(F, K, w, np.exp(-r * tau_r), is_call)


# --------------------------------------------------------------------------- #
# greeks
# --------------------------------------------------------------------------- #
def _d1_d2(F, K, w):
    sw = np.sqrt(np.maximum(w, 1e-300))
    d1 = np.log(np.asarray(F, float) / np.asarray(K, float)) / sw + 0.5 * sw
    return d1, d1 - sw


def bs_delta_w(F, K, w, df, carry_factor, is_call: bool = True):
    """dV/dS.  carry_factor = dF/dS = exp(b * tau_r)."""
    d1, _ = _d1_d2(F, K, w)
    n = norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0
    return df * carry_factor * n


def bs_vega_sqrt_w(F, K, w, df):
    """dV/d(sqrt(w)).  Multiply by sqrt(tau_vol) to get the usual dV/dsigma."""
    d1, _ = _d1_d2(F, K, w)
    return df * np.asarray(F, float) * norm.pdf(d1)


def bs_vega(F, K, w, df, tau_vol):
    """dV/dsigma with sigma = sqrt(w / tau_vol)."""
    return bs_vega_sqrt_w(F, K, w, df) * np.sqrt(tau_vol)


def bs_gamma_w(F, K, w, df, carry_factor):
    """d2V/dS2."""
    d1, _ = _d1_d2(F, K, w)
    sw = np.sqrt(np.maximum(w, 1e-300))
    return df * carry_factor ** 2 * norm.pdf(d1) / (np.asarray(F, float) * sw)


def bs_vanna_w(F, K, w, df, carry_factor):
    """d2V/dS d(sqrt(w))."""
    d1, d2 = _d1_d2(F, K, w)
    sw = np.sqrt(np.maximum(w, 1e-300))
    return -df * carry_factor * norm.pdf(d1) * d2 / sw


def bs_volga_w(F, K, w, df):
    """d2V/d(sqrt(w))^2."""
    d1, d2 = _d1_d2(F, K, w)
    sw = np.sqrt(np.maximum(w, 1e-300))
    return bs_vega_sqrt_w(F, K, w, df) * d1 * d2 / sw


# --------------------------------------------------------------------------- #
# inversion
# --------------------------------------------------------------------------- #
def implied_total_variance(price, F, K, df, is_call: bool = True,
                           tol: float = 1e-12, max_iter: int = 100):
    """Invert the Black formula for total variance by bisection (robust, vectorised)."""
    price = np.asarray(price, dtype=float)
    lo = np.full_like(price, 1e-12)
    hi = np.full_like(price, 25.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = bs_price_w(F, K, mid, df, is_call)
        too_low = val < price
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
        if np.all(hi - lo < tol):
            break
    return 0.5 * (lo + hi)

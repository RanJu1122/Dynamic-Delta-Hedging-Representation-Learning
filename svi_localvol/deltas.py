"""The delta family.

With a smile the option value is  V = C_BS(S, K, sigma_imp(K, T; S)), so

    Delta = dC_BS/dS  +  dC_BS/dsigma * dsigma_imp/dS
            ^ unambiguous   ^ pure modelling assumption

Today's surface pins down the first term and says nothing at all about the
second: it is a statement about how the surface MOVES, and the surface only
contains information about where it IS.  Delta is therefore a range, not a
number.

This is the LEGACY analytic extension of the original pricing task.  It
parameterises that range with a single stickiness ratio R.  The
surface is evaluated at

    y_eval(S) = ln(K / F_ref) - R * ln(S / S_ref)

    R =  0  sticky strike        (surface pinned in strike space)
    R =  1  sticky moneyness     (surface rides with the spot)
    R = -1  local-vol-like       (ATM vol moves by 2x the skew)

TWO CONVENTIONS -- DO NOT CONFLATE
----------------------------------
The dynamic-hedging study calls its stickiness parameter `alpha` and anchors it
differently:

    alpha = 0  sticky local vol   (sigma_loc frozen, local vol model delta)
    alpha = 1  sticky strike      <- the quoted StickinessRatio in the data
    alpha = 2  sticky moneyness

so on the shared anchors  **alpha = R + 1**  (`alpha_from_R` / `R_from_alpha`).

They are however different OBJECTS, not just different labels:

  * R lives here, in the ANALYTIC implied-surface shift.  It is exact and
    closed-form: y_eval = ln(K/F_ref) - R ln(S/S_ref), and delta follows by
    differentiating the Black-Scholes price.
  * alpha lives in `VolSurface.local_vol`, in the DUPIRE DENOMINATOR, and its
    delta only exists through a Monte Carlo bump-and-reprice.

Measured on the shipped test surface (ATM call, furthest VolDate, 200k paths,
common random numbers):

    alpha   MC delta (denominator)   analytic delta at R = alpha - 1
      0.0            0.3728                     0.4160
      1.0            0.5256                     0.5284      <- agree
      2.0            0.6759                     0.6408

They coincide at the sticky-strike anchor and drift apart at the ends, so use
the converters to line up NAMES, never to substitute one delta for the other.
Empirically R is fitted per regime by `backtest.py`.  New dynamic-alpha research
must not use that backtester or treat ``alpha = R + 1`` as a numerical model;
it should use ``alpha_mc_delta_curve`` and the local-vol engine directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .blackscholes import (bs_gamma_w, bs_price_w, bs_vanna_w, bs_vega_sqrt_w,
                           bs_volga_w)
from .params import validate_stickiness_alpha
from .surface import VolSurface

STICKY_STRIKE = 0.0
STICKY_MONEYNESS = 1.0
LOCAL_VOL_LIKE = -1.0

#: Offset between the two stickiness conventions: alpha = R + ALPHA_R_OFFSET.
ALPHA_R_OFFSET = 1.0


def alpha_from_R(R):
    """Hedging-study alpha for an analytic stickiness ratio R."""
    return np.asarray(R, dtype=float) + ALPHA_R_OFFSET


def R_from_alpha(alpha):
    """Analytic stickiness ratio R for a hedging-study alpha."""
    return np.asarray(alpha, dtype=float) - ALPHA_R_OFFSET


# --------------------------------------------------------------------------- #
# analytic smile greeks
# --------------------------------------------------------------------------- #
def smile_greeks(surface: VolSurface, T, K, spot: float | None = None,
                 stickiness: float = STICKY_MONEYNESS,
                 is_call: bool = True,
                 surface_shift: float | None = None) -> dict:
    """Price and greeks of a vanilla under a given stickiness assumption.

    Two distinct things are easy to conflate here, so they are separate
    arguments:

    surface_shift
        WHERE on the surface to read today's volatility.  Defaults to
        stickiness * ln(spot / ref_spot), i.e. the surface has already ridden
        with the spot since it was fitted.  Pass 0.0 when the surface handed in
        is already anchored on today's spot (a live daily quote, as in the
        backtest) -- otherwise the shift is applied twice.

    stickiness
        HOW the surface will move for the next infinitesimal spot move.  This
        is what turns Delta into a range; it never affects today's price.

    `spot` is the LIVE spot and always drives the forward.
    """
    S = surface.ref_spot if spot is None else float(spot)
    K = np.asarray(K, dtype=float)

    tau_r = surface.tau_r(T)
    tau_v = surface.tau_vol(T)
    carry = float(np.exp(surface.market.cost_of_carry * tau_r))
    df = surface.discount_factor(T)
    F = S * carry                                    # forward off the LIVE spot

    if surface_shift is None:
        surface_shift = stickiness * float(np.log(S / surface.ref_spot))
    res = surface.total_variance(T, K, spot_adj=surface_shift,
                                 spot=surface.ref_spot, order=2)
    w = np.maximum(res["w"], 1e-12)
    dw_dy = res["dw_dy"]
    d2w_dy2 = res["d2w_dy2"]

    sw = np.sqrt(w)
    d1 = np.log(F / K) / sw + 0.5 * sw
    d2 = d1 - sw

    price = bs_price_w(F, K, w, df, is_call)
    vega_sw = bs_vega_sqrt_w(F, K, w, df)            # dV / d(sqrt(w))
    dC_dw = vega_sw / (2.0 * sw)                     # dV / dw
    delta_bs = df * carry * (norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0)

    # dy_eval / dS = -R / S
    delta_smile = delta_bs - (stickiness / S) * dC_dw * dw_dy

    return {
        "price": price,
        "forward": F,
        "w": w,
        "sigma_imp": np.sqrt(w / tau_v),
        "dw_dy": dw_dy,
        "d2w_dy2": d2w_dy2,
        "delta_bs": delta_bs,
        "delta": delta_smile,
        "vega_sqrt_w": vega_sw,
        "vega_sigma": vega_sw * np.sqrt(tau_v),
        "gamma_bs": bs_gamma_w(F, K, w, df, carry),
        "vanna": bs_vanna_w(F, K, w, df, carry),
        "volga": bs_volga_w(F, K, w, df),
        "d1": d1, "d2": d2,
        "stickiness": stickiness,
        "spot": S,
    }


def delta(surface: VolSurface, T, K, spot: float | None = None,
          stickiness: float = STICKY_MONEYNESS, is_call: bool = True):
    """Smile delta under a given stickiness ratio."""
    return smile_greeks(surface, T, K, spot, stickiness, is_call)["delta"]


def delta_range(surface: VolSurface, T, K, spot: float | None = None,
                stickiness_grid: Sequence[float] = (-1.0, 0.0, 1.0),
                is_call: bool = True) -> dict:
    """Delta across a set of stickiness assumptions -- the 'delta is a range'."""
    out = {float(R): delta(surface, T, K, spot, R, is_call)
           for R in stickiness_grid}
    vals = np.array([np.atleast_1d(v) for v in out.values()])
    out["_min"] = vals.min(axis=0)
    out["_max"] = vals.max(axis=0)
    out["_width"] = vals.max(axis=0) - vals.min(axis=0)
    return out


def delta_finite_difference(surface: VolSurface, T, K, spot: float | None = None,
                            stickiness: float = STICKY_MONEYNESS,
                            is_call: bool = True, h_rel: float = 1e-4):
    """Bump-and-reprice on the analytic price.  Validates the closed form above."""
    S = surface.ref_spot if spot is None else float(spot)
    h = h_rel * S
    up = smile_greeks(surface, T, K, S + h, stickiness, is_call)["price"]
    dn = smile_greeks(surface, T, K, S - h, stickiness, is_call)["price"]
    return (up - dn) / (2.0 * h)


# --------------------------------------------------------------------------- #
# Monte Carlo delta as a function of alpha
# --------------------------------------------------------------------------- #
def alpha_mc_delta_curve(surface: VolSurface, strikes, maturity,
                         alphas: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
                         n_paths: int = 200_000, seed: int = 20260807,
                         n_substeps: int = 4, h_rel: float = 0.01,
                         **grid_kwargs):
    """Local-vol Monte Carlo delta for each alpha, on one strike grid.

    This is a delta diagnostic, NOT the research Step 3 beta(alpha) converter.
    Step 3 must bump spot, invert model prices back to implied vol, and measure
    ``beta = -dIV/dlogS``.  This helper only maps alpha to MC delta and owns no
    numerics of its own: every row comes from
    `LocalVolMC.step4_delta_diagnostics`, so the antithetic pair-mean standard
    errors, the common random numbers across the two bumps and the control
    variate are all shared with Step 4.

    Returns a long DataFrame: one row per (alpha, strike).
    """
    from .montecarlo import LocalVolGrid, LocalVolMC

    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    s0 = float(surface.market.spot)
    h = h_rel * s0
    up_shift = float(np.log((s0 + h) / surface.ref_spot))
    dn_shift = float(np.log((s0 - h) / surface.ref_spot))

    frames = []
    for a in alphas:
        a = validate_stickiness_alpha(a)
        base = LocalVolGrid.build(surface, maturity, alpha=a, **grid_kwargs)
        g_up = LocalVolGrid.build(surface, maturity, spot_adj=up_shift,
                                  alpha=a, **grid_kwargs)
        g_dn = LocalVolGrid.build(surface, maturity, spot_adj=dn_shift,
                                  alpha=a, **grid_kwargs)
        mc = LocalVolMC(surface, base, n_paths=n_paths, seed=seed,
                        antithetic=True, n_substeps=n_substeps)
        tbl = mc.step4_delta_diagnostics(strikes, maturity, grid_up=g_up,
                                         grid_down=g_dn, bump=h)
        tbl.insert(0, "alpha", float(a))
        frames.append(tbl)
    return pd.concat(frames, ignore_index=True)


def alpha_delta_curve(*args, **kwargs):
    """Backward-compatible alias for :func:`alpha_mc_delta_curve`."""
    return alpha_mc_delta_curve(*args, **kwargs)


# --------------------------------------------------------------------------- #
# empirical backbone
# --------------------------------------------------------------------------- #
@dataclass
class BackboneFit:
    """Legacy R regression; not the new study's empirical beta definition."""

    slope: float          # d sigma_ATM / d ln S
    r_squared: float
    n_obs: int
    implied_stickiness: float

    def __str__(self) -> str:
        return (f"backbone slope={self.slope:+.4f}  R2={self.r_squared:.3f}  "
                f"n={self.n_obs}  =>  R*={self.implied_stickiness:+.2f}")


def fit_backbone(d_log_spot: np.ndarray, d_atm_vol: np.ndarray,
                 skew: float) -> BackboneFit:
    """Estimate R from history.

    Under stickiness R the ATM implied volatility satisfies
        d sigma_ATM / d ln S = (1 - R) * skew,
    so  R = 1 - slope / skew.
    """
    x = np.asarray(d_log_spot, dtype=float)
    y = np.asarray(d_atm_vol, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        raise ValueError("need at least three observations")

    slope = float(np.sum(x * y) / np.sum(x * x))        # through the origin
    resid = y - slope * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return BackboneFit(slope=slope, r_squared=r2, n_obs=int(x.size),
                       implied_stickiness=1.0 - slope / skew)

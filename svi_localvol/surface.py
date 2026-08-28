"""SVI volatility surface, implied volatility and Dupire local volatility.

A `VolSurface` owns
  * the market data (pricing date, rates, holidays, reference spot),
  * one raw SVI slice per VolDate together with its tau = dt_vol(VolDate),
and exposes the two functions with the signatures fixed by the task:

    ImpliedVol(T, K)
    localvol(T, K, spot_adj)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .conventions import dt_r, dt_vol, gen_schedule, to_date
from .params import MarketData, VolQuoteSet, validate_stickiness_alpha
from .svi import SVIRaw, crossedness, g_function, jw_to_raw, raw_to_jw


@dataclass(frozen=True)
class Slice:
    vol_date: dt.date
    tau: float                 # dt_vol(vol_date), Business/260
    raw: SVIRaw
    alpha: float               # StickinessRatio for this expiry

    @property
    def stickiness_ratio(self) -> float:
        """Backward-compatible alias for `alpha`."""
        return self.alpha


class VolSurface:
    """Arbitrage-checked SVI volatility surface with a Dupire local vol map."""

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(self, market: MarketData, quotes: VolQuoteSet,
                 calendar_repair: bool = False, beta_clamp: float = 0.0):
        """
        calendar_repair
            False -- use the quoted slices as they are.  Where the quotes cross
                     in total variance, dw/dT < 0 and `local_vol` returns NaN.
                     Use this mode for diagnostics so arbitrage is visible.
            True  -- make w non-decreasing in tau at every log-moneyness by a
                     running maximum over slices.  dw/dT >= 0 by construction,
                     so the local volatility exists everywhere.  Use this for
                     the Monte Carlo, and compare against implied vols read off
                     the SAME repaired surface so the check stays honest.
        """
        self.market = market
        self.quotes = quotes
        self.ref_spot = quotes.ref_spot
        self.calendar_repair = bool(calendar_repair)
        self.beta_clamp = float(beta_clamp)

        self.slices: list[Slice] = []
        for q in quotes.quotes:
            tau = dt_vol(market.pricing_date, q.vol_date, market.holidays)
            #tau defined use business_days_between(pricing_date, vol_date) / 260.0
            raw = jw_to_raw(
                atm_var=q.atm_var,
                skew=q.skew,
                putwing=q.putwing,
                callwing=q.callwing,
                min_imp_var=q.min_imp_var,
                tau=tau,
                beta_clamp=self.beta_clamp,
            )
            self.slices.append(
                Slice(q.vol_date, tau, raw, q.alpha)
            )

        self.taus = np.array([s.tau for s in self.slices])
        self.alphas = np.array([s.alpha for s in self.slices])
        if np.any(np.diff(self.taus) <= 0):
            raise ValueError("VolDates must produce a strictly increasing dt_vol axis")

    @classmethod
    def from_dict(cls, market: MarketData, vol_params: dict,
                  calendar_repair: bool = False) -> "VolSurface":
        return cls(market, VolQuoteSet.from_dict(vol_params), calendar_repair)

    def repaired(self) -> "VolSurface":
        """A copy of this surface with the calendar arbitrage ironed out."""
        return VolSurface(self.market, self.quotes, calendar_repair=True,
                          beta_clamp=self.beta_clamp)

    # ------------------------------------------------------------------ #
    # time axes and forward
    # ------------------------------------------------------------------ #
    def tau_vol(self, T) -> float:
        """Business/260 volatility time."""
        return dt_vol(self.market.pricing_date, T, self.market.holidays)

    def tau_r(self, T) -> float:
        """Act/365 discount time."""
        return dt_r(self.market.pricing_date, T)

    def alpha_at(self, T) -> float:
        """Stickiness parameter alpha at an arbitrary date.

        alpha is quoted per VolDate (the `StickinessRatio` list).  Between
        quoted expiries it is interpolated linearly on the SAME dt_vol axis the
        total variance uses, and held flat outside, so that `local_vol` has a
        well defined alpha at every date on the simulation grid.
        """
        tau = self.tau_vol(T)
        if len(self.taus) == 1:
            return float(self.alphas[0])
        return float(np.interp(tau, self.taus, self.alphas,
                               left=self.alphas[0], right=self.alphas[-1]))

    def forward(self, T, spot: float | None = None) -> float:
        """F = spot * exp(b * dt_r).

        The surface is calibrated off `ref_spot`, so that is the default.
        Pass an explicit `spot` only when repricing under a bumped spot.
        """
        s = self.ref_spot if spot is None else spot
        return s * np.exp(self.market.cost_of_carry * self.tau_r(T))

    def log_moneyness(self, T, K, spot: float | None = None):
        """y = ln(K / F)."""
        return np.log(np.asarray(K, dtype=float) / self.forward(T, spot))

    def discount_factor(self, T) -> float:
        return float(np.exp(-self.market.rate * self.tau_r(T)))

    # ------------------------------------------------------------------ #
    # total variance in (y, tau)
    # ------------------------------------------------------------------ #
    def _locate(self, tau: float) -> tuple[str, int, int]:
        """Return ('flat'|'interp', idx_lo, idx_hi) for a volatility time."""
        n = len(self.taus)
        if tau <= self.taus[0]:
            return "flat", 0, 0
        if tau >= self.taus[-1]:
            return "flat", n - 1, n - 1
        j = int(np.searchsorted(self.taus, tau, side="left"))
        return "interp", j - 1, j

    def _slice_curves(self, y_eval, order: int) -> dict:
        """Evaluate every slice at y_eval, applying the calendar repair.

        Returns arrays of shape (n_slices,) + y_eval.shape.  When the repair is
        on, slice i is replaced by the running maximum over slices 0..i; the
        y-derivatives are taken from whichever slice supplied that maximum, so
        the repaired surface stays a genuine SVI slice at every point.
        """
        y = np.atleast_1d(np.asarray(y_eval, dtype=float))
        w = np.stack([s.raw.w(y) for s in self.slices])
        out = {"w": w}

        if order >= 1:
            out["dw_dy"] = np.stack([s.raw.dw_dy(y) for s in self.slices])
        if order >= 2:
            out["d2w_dy2"] = np.stack([s.raw.d2w_dy2(y) for s in self.slices])

        if self.calendar_repair:
            n = w.shape[0]
            src = np.zeros_like(w, dtype=int)
            best = w[0].copy()
            bidx = np.zeros(w.shape[1:], dtype=int)
            for i in range(n):
                take = w[i] >= best
                best = np.where(take, w[i], best)
                bidx = np.where(take, i, bidx)
                w[i] = best
                src[i] = bidx
            for key in ("dw_dy", "d2w_dy2"):
                if key in out:
                    out[key] = np.take_along_axis(out[key], src, axis=0)
        return out

    def total_variance(self, T, K, spot: float | None = None, order: int = 0):
        """Total implied variance w(y, tau) and, optionally, its derivatives.

        Returns a dict with keys 'w', 'tau', 'y' and, when order >= 1,
        'dw_dtau', 'dw_dy'; when order >= 2 also 'd2w_dy2'.

        Implied variance is always evaluated on the quoted surface.  Dynamic
        alpha enters only the Dupire denominator in :meth:`local_vol`.
        """
        tau = self.tau_vol(T)
        if tau <= 0:
            raise ValueError(f"T = {to_date(T)} is not after the pricing date")

        y = np.asarray(self.log_moneyness(T, K, spot), dtype=float)
        scalar = (y.ndim == 0)
        y_eval = np.atleast_1d(y)

        cur = self._slice_curves(y_eval, order)
        mode, lo, hi = self._locate(tau)

        def _fin(a):
            a = np.asarray(a)
            return float(a.reshape(-1)[0]) if scalar and a.size == 1 else a

        if mode == "flat":
            sl = self.slices[lo]
            scale = tau / sl.tau
            out = {"w": _fin(cur["w"][lo] * scale), "tau": tau,
                   "y": _fin(np.atleast_1d(y)), "y_eval": _fin(y_eval)}
            if order >= 1:
                out["dw_dtau"] = _fin(cur["w"][lo] / sl.tau)   # boundary rule: w / tau
                out["dw_dy"] = _fin(cur["dw_dy"][lo] * scale)
            if order >= 2:
                out["d2w_dy2"] = _fin(cur["d2w_dy2"][lo] * scale)
            return out

        sd, su = self.slices[lo], self.slices[hi]
        dtau = su.tau - sd.tau
        theta = (tau - sd.tau) / dtau
        slope = (cur["w"][hi] - cur["w"][lo]) / dtau

        #interpolation

        out = {"w": _fin(cur["w"][lo] + slope * (tau - sd.tau)), "tau": tau,
               "y": _fin(np.atleast_1d(y)), "y_eval": _fin(y_eval)}
        if order >= 1:
            out["dw_dtau"] = _fin(slope)
            dd, du = cur["dw_dy"][lo], cur["dw_dy"][hi]
            out["dw_dy"] = _fin(dd + (du - dd) * theta)
        if order >= 2:
            dd2, du2 = cur["d2w_dy2"][lo], cur["d2w_dy2"][hi]
            out["d2w_dy2"] = _fin(dd2 + (du2 - dd2) * theta)
        return out

    # ------------------------------------------------------------------ #
    # implied volatility
    # ------------------------------------------------------------------ #
    def implied_vol(self, T, K, spot: float | None = None):
        """Black-Scholes implied volatility for any date T and strike(s) K.

        T need not be a VolDate.  Total variance is interpolated linearly on
        the dt_vol axis and extrapolated flat in volatility outside the quotes.
        """
        res = self.total_variance(T, K, spot=spot, order=0)
        w = np.maximum(res["w"], 0.0)
        vol = np.sqrt(w / res["tau"])
        return float(vol) if np.isscalar(K) or np.ndim(K) == 0 else vol

    def implied_total_variance(self, T, K):
        return self.total_variance(T, K, order=0)["w"]

    # ------------------------------------------------------------------ #
    # Dupire local volatility
    # ------------------------------------------------------------------ #
    def local_vol(self, T, K, spot_adj: float = 0.0, spot: float | None = None,
                  return_diagnostics: bool = False, alpha: float | None = None):
        """Dupire-Gatheral local volatility, same shape as K.

            sigma_loc^2 = (dw/dT) / D
            D = 1 - (y/w) dw/dy + 1/4 (-1/4 - 1/w + y^2/w^2) (dw/dy)^2
                + 1/2 d2w/dy2

        D is algebraically identical to Gatheral's g (equation 2.1), so
            numerator < 0  ->  calendar spread arbitrage
            D <= 0         ->  butterfly arbitrage
        NaN is returned at those points; use `return_diagnostics=True` to get
        boolean masks instead of silently swallowing them.

        alpha
        -----
        Stickiness parameter, the `StickinessRatio` field.  The shift applied is

            y_adj = y - alpha * spot_adj

        `None` (the default) takes alpha from the quotes, interpolated on the
        dt_vol axis by `alpha_at`.  Pass a float to override every expiry at
        once -- that is what the alpha sweep in the hedging study needs.

        Measured on the calibration test surface:
        alpha = 0 reproduces the frozen-local-vol delta, alpha = 1 reproduces
        the sticky-strike Black-Scholes delta, alpha = 2 moves further toward
        sticky moneyness.

        The shift enters only the Dupire denominator, exactly as specified by
        both project documents; the quoted implied surface is not shifted.
        """
        a = (self.alpha_at(T) if alpha is None
             else validate_stickiness_alpha(alpha))
        shift = a * float(spot_adj)
        res = self.total_variance(T, K, spot=spot, order=2)

        w = res["w"]
        dw_dT = res["dw_dtau"]
        dw_dy = res["dw_dy"]
        d2w_dy2 = res["d2w_dy2"]
        y_adj = res["y"] - shift

        with np.errstate(divide="ignore", invalid="ignore"):
            D = (
                1.0
                - (y_adj / w) * dw_dy
                + 0.25 * (-0.25 - 1.0 / w + y_adj ** 2 / w ** 2) * dw_dy ** 2
                + 0.5 * d2w_dy2
            )
            var_loc = np.where((D > 0.0) & (dw_dT >= 0.0) & (w > 0.0),
                               dw_dT / D, np.nan)
            vol_loc = np.sqrt(var_loc)

        if return_diagnostics:
            diag = {
                "D": np.asarray(D),
                "dw_dT": np.broadcast_to(np.asarray(dw_dT), np.shape(vol_loc)).copy(),
                "w": np.asarray(w),
                "calendar_arb": np.asarray(np.broadcast_to(
                    np.asarray(dw_dT) < 0.0, np.shape(vol_loc))).copy(),
                "butterfly_arb": np.asarray(D) <= 0.0,
                "alpha": a,
                "shift": shift,
            }
            return vol_loc, diag
        return vol_loc

    # ------------------------------------------------------------------ #
    # matrices
    # ------------------------------------------------------------------ #
    def date_axis(self, end, period: str = "1D") -> list[dt.date]:
        return gen_schedule(self.market.pricing_date, end, period=period,
                            bizconv="Following", hol=self.market.holidays)

    def implied_vol_matrix(self, dates: Sequence[dt.date],
                           levels: np.ndarray) -> pd.DataFrame:
        levels = np.asarray(levels, dtype=float)
        K = levels * self.market.spot
        rows = []
        idx = []
        for T in dates:
            if self.tau_vol(T) <= 0:
                continue
            rows.append(np.atleast_1d(self.implied_vol(T, K)))
            idx.append(to_date(T))
        return pd.DataFrame(rows, index=pd.Index(idx, name="date"),
                            columns=pd.Index(levels, name="level"))

    def local_vol_matrix(self, dates: Sequence[dt.date], levels: np.ndarray,
                         spot_adj: float = 0.0,
                         alpha: float | None = None) -> pd.DataFrame:
        levels = np.asarray(levels, dtype=float)
        K = levels * self.market.spot
        rows, idx = [], []
        for T in dates:
            if self.tau_vol(T) <= 0:
                continue
            rows.append(np.atleast_1d(
                self.local_vol(T, K, spot_adj=spot_adj, alpha=alpha)))
            idx.append(to_date(T))
        return pd.DataFrame(rows, index=pd.Index(idx, name="date"),
                            columns=pd.Index(levels, name="level"))

    # ------------------------------------------------------------------ #
    # reporting / self-checks
    # ------------------------------------------------------------------ #
    def slice_table(self) -> pd.DataFrame:
        rows = []
        for s, q in zip(self.slices, self.quotes.quotes):
            ok, msgs = s.raw.is_well_posed()
            back = raw_to_jw(s.raw, s.tau)
            rows.append({
                "VolDate": s.vol_date,
                "dt_vol": s.tau,
                "dt_r": self.tau_r(s.vol_date),
                "forward": self.forward(s.vol_date),
                "a": s.raw.a, "b": s.raw.b, "rho": s.raw.rho,
                "m": s.raw.m, "sigma": s.raw.sigma,
                "w_atm_target": q.atm_var * s.tau,
                "w_at_y0": float(s.raw.w(0.0)),
                "skew_target": q.skew,
                # raw_to_jw returns Gatheral's psi_t; the desk quotes
                # d(sigma)/dk = psi_t / sqrt(tau).  See svi.jw_to_raw.
                "skew_roundtrip": back["skew"] / np.sqrt(s.tau),
                "well_posed": ok,
                "notes": "; ".join(msgs),
            })
        return pd.DataFrame(rows)

    def self_check(self, tol: float = 1e-8) -> pd.DataFrame:
        """Round-trip checks required by the task statement plus the skew check.

        The w(y=0) == atm_var * tau identity holds for ANY beta, so on its own
        it cannot detect a wrong skew convention.  The skew round-trip can --
        provided it compares like with like: `raw_to_jw` returns Gatheral's
        total-variance skew psi_t, while the quoted `Skew` field is the
        volatility skew d(sigma)/dk = psi_t / sqrt(tau).
        """
        rows = []
        for s, q in zip(self.slices, self.quotes.quotes):
            back = raw_to_jw(s.raw, s.tau)
            F = self.forward(s.vol_date)
            rows.append({
                "VolDate": s.vol_date,
                "err_w_atm": float(s.raw.w(0.0) - q.atm_var * s.tau),
                "err_impliedvol_atm": float(
                    self.implied_vol(s.vol_date, F) - q.atm_vol),
                "err_skew": float(back["skew"] / np.sqrt(s.tau) - q.skew),
                "err_putwing": float(back["putwing"] - q.putwing),
                "err_callwing": float(back["callwing"] - q.callwing),
                "err_kurt": float(back["kurt"] - q.kurt),
            })
        df = pd.DataFrame(rows)
        df["pass"] = df.drop(columns=["VolDate"]).abs().max(axis=1) < tol
        return df
    def arbitrage_report(self, y_grid: np.ndarray | None = None) -> dict:
        """Butterfly per slice, calendar crossedness between adjacent slices.

        `crossedness` and the reported `y_bad_*` are grid quantities, so they
        only mean something together with the window they were measured on.
        Two extra columns state the analytic behaviour beyond that window:

            unbounded_call_wing : b_lo (1 + rho_lo) > b_hi (1 + rho_hi)
            unbounded_put_wing  : b_lo (1 - rho_lo) > b_hi (1 - rho_hi)

        w(y) is asymptotically linear with slope b(1 +/- rho), so when either
        flag is True the near slice stays above the far one for ALL large |y|
        and the true crossedness is unbounded -- no grid can measure it.  The
        `truncated` flag says the violation was still live at the grid edge.
        """
        if y_grid is None:
            # wide enough to cover the whole traded band plus both wings; the
            # old default (-1.0, 0.6) cut the call-wing violations in half and
            # under-reported crossedness by up to a factor of five.
            y_grid = np.linspace(-1.5, 1.5, 601)
        y_grid = np.asarray(y_grid, dtype=float)
        y_lo, y_hi = float(y_grid.min()), float(y_grid.max())

        #here is finite arbitrage check, not strictly mathematically

        butterfly = []
        for s in self.slices:
            g = g_function(s.raw.w(y_grid), s.raw.dw_dy(y_grid),
                           s.raw.d2w_dy2(y_grid), y_grid)
            bad = y_grid[g <= 0]
            butterfly.append({
                "VolDate": s.vol_date,
                "min_g": float(g.min()),
                "butterfly_free": bool(g.min() > 0),
                "y_bad_lo": float(bad.min()) if bad.size else np.nan,
                "y_bad_hi": float(bad.max()) if bad.size else np.nan,
                "y_grid_lo": y_lo,
                "y_grid_hi": y_hi,
                "truncated": bool(bad.size and (bad.min() <= y_lo
                                                or bad.max() >= y_hi)),
            })

        calendar = []
        for i in range(len(self.slices) - 1):
            lo, hi = self.slices[i], self.slices[i + 1]
            gap = lo.raw.w(y_grid) - hi.raw.w(y_grid)
            bad = y_grid[gap > 0]
            rl, rh = lo.raw, hi.raw
            calendar.append({
                "from": lo.vol_date,
                "to": hi.vol_date,
                "crossedness": crossedness(rl, rh, y_grid),
                "calendar_free": bool(gap.max() <= 0),
                "y_bad_lo": float(bad.min()) if bad.size else np.nan,
                "y_bad_hi": float(bad.max()) if bad.size else np.nan,
                "y_grid_lo": y_lo,
                "y_grid_hi": y_hi,
                "truncated": bool(bad.size and (bad.min() <= y_lo
                                                or bad.max() >= y_hi)),
                "unbounded_call_wing": bool(rl.b * (1.0 + rl.rho)
                                            > rh.b * (1.0 + rh.rho) + 1e-15),
                "unbounded_put_wing": bool(rl.b * (1.0 - rl.rho)
                                           > rh.b * (1.0 - rh.rho) + 1e-15),
            })

        return {"butterfly": pd.DataFrame(butterfly),
                "calendar": pd.DataFrame(calendar)}

    def local_vol_arbitrage_map(self, dates: Sequence[dt.date],
                                levels: np.ndarray) -> pd.DataFrame:
        """Per-grid-point label: 'ok', 'calendar', 'butterfly' or 'both'."""
        levels = np.asarray(levels, dtype=float)
        K = levels * self.market.spot
        rows, idx = [], []
        for T in dates:
            if self.tau_vol(T) <= 0:
                continue
            _, d = self.local_vol(T, K, return_diagnostics=True)
            lab = np.where(d["calendar_arb"] & d["butterfly_arb"], "both",
                   np.where(d["calendar_arb"], "calendar",
                    np.where(d["butterfly_arb"], "butterfly", "ok")))
            rows.append(lab)
            idx.append(to_date(T))
        return pd.DataFrame(rows, index=pd.Index(idx, name="date"),
                            columns=pd.Index(levels, name="level"))

"""Monte Carlo primitives for a calibrated local-volatility surface.

The engine is deliberately split in two:

  LocalVolGrid   pre-computes Sigma[date, spot_ratio] once, sanitises it, and
                 serves interpolated volatilities during the simulation.
  LocalVolMC     evolves the paths and prices payoffs.

Pre-computing the grid matters: calling `localvol` inside the time loop would
re-run the SVI interpolation for every path at every step.

TWO CLOCKS, TWO STEP SIZES
--------------------------
`VolSurface.local_vol` returns sqrt((dw/dtau) / D) where tau is dt_vol, the
Business/260 clock.  Its units are therefore VARIANCE PER BUSINESS YEAR, not
per calendar year.  The only thing the Black-Scholes comparison leg
reads is the total variance w, so the scheme is consistent if and only if

    integral of sigma_loc^2 over the simulation  ==  w(y, T)   exactly.

That holds when the diffusion is integrated against d(tau_vol) and fails when
it is integrated against Act/365 steps.  So every path step carries two
increments:

    dt_r  = Act/365       -> drift b * dt_r, and the discount factor
    dt_v  = Business/260  -> variance sigma^2 * dt_v, including the Ito term

Note that the Ito correction -0.5 * sigma^2 belongs with dt_v, not with the
drift: it is a by-product of the diffusion term.  The task statement lumps
both into a single Act/365 dt; that is the one place where it contradicts its
own Step 3 definition of dw/dT = dw/dtau.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .blackscholes import bs_delta_w, bs_price_w
from .conventions import to_date
from .params import validate_stickiness_alpha
from .surface import VolSurface


DEFAULT_TERMINAL_RATIO_BINS = np.array([
    0.0, 0.70, 0.80, 0.90, 0.95, 1.00, 1.05,
    1.10, 1.20, 1.30, 1.50, 2.00, 3.00, np.inf,
])


def _estimator_samples(x: np.ndarray, antithetic: bool) -> np.ndarray:
    """Independent observations used to estimate a Monte Carlo mean.

    With antithetic sampling, the independent observations are pair means,
    not the individual Z and -Z payoffs.
    """
    x = np.asarray(x, dtype=float)
    if not antithetic:
        return x
    half = x.size // 2
    if half == 0:
        return x
    return 0.5 * (x[:half] + x[half:2 * half])


def _mean_stderr(x: np.ndarray, antithetic: bool) -> tuple[float, float]:
    obs = _estimator_samples(x, antithetic)
    mean = float(obs.mean())
    stderr = (float(obs.std(ddof=1) / np.sqrt(obs.size))
              if obs.size > 1 else float("nan"))
    return mean, stderr


def _conditional_mean_stderr(values: np.ndarray, mask: np.ndarray,
                             antithetic: bool) -> tuple[float, float]:
    """Conditional mean and cluster-robust SE, pairing Z with -Z."""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan"), float("nan")
    mean = float(values[mask].mean())
    if not antithetic:
        n = int(mask.sum())
        se = (float(values[mask].std(ddof=1) / np.sqrt(n))
              if n > 1 else float("nan"))
        return mean, se

    half = values.size // 2
    v0, v1 = values[:half], values[half:2 * half]
    m0, m1 = mask[:half], mask[half:2 * half]
    numerator = m0 * v0 + m1 * v1
    denominator = m0.astype(float) + m1.astype(float)
    total_n = float(denominator.sum())
    influence = numerator - mean * denominator
    se = (float(influence.std(ddof=1) * np.sqrt(half) / total_n)
          if half > 1 and total_n > 0 else float("nan"))
    return mean, se


# --------------------------------------------------------------------------- #
# pre-computed local volatility grid
# --------------------------------------------------------------------------- #
@dataclass
class LocalVolGrid:
    dates: list[dt.date]          # business days, INCLUDING the pricing date
    tau_vol: np.ndarray           # dt_vol(pricing_date, date), Business/260
    tau_r: np.ndarray             # dt_r(pricing_date, date), Act/365
    ratios: np.ndarray            # S_t / S0 axis
    sigma: np.ndarray             # shape (n_dates, n_ratios)
    vol_floor: float
    vol_cap: float
    n_undefined: int              # grid points where localvol returned NaN
    n_clipped: int                # finite points the cap/floor actually moved
    n_zero: int                   # points where sigma_loc is exactly zero

    @classmethod
    def build(cls, surface: VolSurface, maturity, n_ratio: int = 1000,
              ratio_max: float = 3.0, ratio_min: float = 1e-3,
              spot_adj: float = 0.0, vol_floor: float = 0.0,
              vol_cap: float = 5.0, local_vol_fn: Callable | None = None,
              alpha: float | None = None) -> "LocalVolGrid":
        """Tabulate sigma_loc on (business day) x (spot ratio).

        The date axis starts at the PRICING DATE.  Dropping it -- which is
        tempting because dt_vol = 0 there and the surface is undefined -- would
        shorten the simulation by one business day and by the weekend in front
        of it, so the paths would no longer span the option's life.  The first
        row is filled with the tau -> 0+ limit, i.e. the next business day's
        local volatility.

        Ratios run over [0, ratio_max]; the first node is clamped to
        `ratio_min` because y = ln(K/F) is undefined at K = 0.

        local_vol_fn
            Optional ``f(T, K) -> sigma_loc`` adapter.  When omitted the
            surface's own method is used with `spot_adj` and `alpha`.

        alpha
            Stickiness override forwarded to `VolSurface.local_vol`; `None`
            uses the per-VolDate `StickinessRatio` from the quotes.  The
            hedging study sweeps this over {0, 0.5, 1, 1.5, 2}.
        """
        dates = [to_date(d) for d in surface.date_axis(maturity, period="1D")]
        if not dates:
            raise ValueError("empty date axis")

        ratios = np.linspace(0.0, ratio_max, n_ratio)
        K = np.maximum(ratios, ratio_min) * surface.market.spot

        if local_vol_fn is None:
            def local_vol_fn(T, K, _s=surface, _a=spot_adj, _al=alpha):
                return _s.local_vol(T, K, spot_adj=_a, alpha=_al)

        raw = np.empty((len(dates), n_ratio))
        for i, T in enumerate(dates):
            # tau = 0 on the pricing date: use the limit from the right.
            T_eval = T if surface.tau_vol(T) > 0 else dates[i + 1]
            raw[i] = np.atleast_1d(np.asarray(local_vol_fn(T_eval, K), dtype=float))

        n_undefined = int(np.count_nonzero(~np.isfinite(raw)))
        filled = _fill_undefined(raw, vol_floor)
        sigma = np.clip(filled, vol_floor, vol_cap)
        n_clipped = int(np.count_nonzero(np.isfinite(raw) & (filled != sigma)))
        n_zero = int(np.count_nonzero(sigma == 0.0))

        return cls(dates=dates,
                   tau_vol=np.array([surface.tau_vol(d) for d in dates]),
                   tau_r=np.array([surface.tau_r(d) for d in dates]),
                   ratios=ratios, sigma=sigma,
                   vol_floor=vol_floor, vol_cap=vol_cap,
                   n_undefined=n_undefined, n_clipped=n_clipped, n_zero=n_zero)

    def sigma_at(self, date_index: int, ratio: np.ndarray) -> np.ndarray:
        """Linear interpolation along the ratio axis at a fixed date index."""
        return np.interp(ratio, self.ratios, self.sigma[date_index],
                         left=self.sigma[date_index, 0],
                         right=self.sigma[date_index, -1])

    def sigma_bilinear(self, date_frac: float, ratio: np.ndarray) -> np.ndarray:
        """Bilinear lookup at a fractional position along the date axis.

        Used by every simulation sub-step.  Holding the date index fixed across
        a business day instead would leave a first-order-in-dtau time error that
        no amount of sub-stepping removes.
        """
        n = len(self.dates)
        x = float(np.clip(date_frac, 0.0, n - 1))
        i0 = int(np.floor(x))
        i1 = min(i0 + 1, n - 1)
        wgt = x - i0
        if wgt == 0.0:
            return self.sigma_at(i0, ratio)
        return (1 - wgt) * self.sigma_at(i0, ratio) + wgt * self.sigma_at(i1, ratio)

    def as_frame(self):
        import pandas as pd
        return pd.DataFrame(self.sigma,
                            index=pd.Index(self.dates, name="date"),
                            columns=pd.Index(self.ratios, name="ratio"))


def _fill_undefined(a: np.ndarray, floor: float) -> np.ndarray:
    """Replace non-finite entries by interpolation along the ratio axis.

    This is a second, purely numerical repair on top of whatever the surface
    already does: where the quotes are calendar-arbitrageable `local_vol`
    returns NaN and the model simply does not exist there.  `LocalVolGrid`
    reports the count as `n_undefined` so a run on the raw quotes cannot
    silently pretend to be a calibrated model.
    """
    out = np.array(a, dtype=float, copy=True)
    for i in range(out.shape[0]):
        row = out[i]
        good = np.isfinite(row)
        if not good.any():
            row[:] = floor
            continue
        idx = np.arange(row.size)
        row[~good] = np.interp(idx[~good], idx[good], row[good])
    return out


# --------------------------------------------------------------------------- #
# simulation engine
# --------------------------------------------------------------------------- #
@dataclass
class MCResult:
    pv: float
    stderr: float
    delta: float | None = None
    delta_stderr: float | None = None
    n_paths: int = 0
    n_steps: int = 0
    extra: dict = field(default_factory=dict)


class LocalVolMC:
    """Risk-neutral Euler-log scheme driven by a pre-tabulated local vol grid."""

    def __init__(self, surface: VolSurface, grid: LocalVolGrid,
                 n_paths: int = 500_000, seed: int = 20260807,
                 antithetic: bool = True, n_substeps: int = 1):
        self.surface = surface
        self.grid = grid
        self.n_paths = int(n_paths)
        self.seed = seed
        self.antithetic = antithetic
        self.n_substeps = max(1, int(n_substeps))

        # the two clocks, see the module docstring
        self.dt_r = np.diff(grid.tau_r)          # Act/365      -> drift, discounting
        self.dt_v = np.diff(grid.tau_vol)        # Business/260 -> variance
        if np.any(self.dt_v <= 0) or np.any(self.dt_r <= 0):
            raise ValueError("the grid date axis must be strictly increasing")
        self.b = surface.market.cost_of_carry

    # ------------------------------------------------------------------ #
    def _check_horizon(self, maturity) -> None:
        """The paths must end exactly on the payoff date.

        `gen_schedule` rolls a non-business maturity backwards, which would
        leave an unsimulated stub between the last grid date and the payoff.
        """
        if self.grid.dates[-1] != to_date(maturity):
            raise ValueError(
                f"grid ends on {self.grid.dates[-1]} but the payoff is on "
                f"{to_date(maturity)}; rebuild the grid for this maturity")

    def total_clocks(self) -> tuple[float, float]:
        """(sum dt_r, sum dt_v) actually simulated.  Used by the regressions."""
        return float(self.dt_r.sum()), float(self.dt_v.sum())

    def _draw(self) -> np.ndarray:
        """Normal draws, shape (n_steps * n_substeps, n_eff)."""
        rng = np.random.default_rng(self.seed)
        n_steps = len(self.dt_r) * self.n_substeps
        if self.antithetic:
            half = self.n_paths // 2
            z = rng.standard_normal((n_steps, half))
            return np.concatenate([z, -z], axis=1)
        return rng.standard_normal((n_steps, self.n_paths))

    def terminal_spots(self, s0: float, z: np.ndarray | None = None,
                       store_paths: bool = False,
                       return_integrated_variance: bool = False):
        """Evolve S from the pricing date to the last grid date.

        The local volatility is re-interpolated in BOTH state and time at every
        sub-step, so refining `n_substeps` refines the whole scheme.
        """
        z = self._draw() if z is None else z
        n_eff = z.shape[1]
        s = np.full(n_eff, float(s0))
        s0_ref = self.surface.market.spot
        m = self.n_substeps
        paths = np.empty((len(self.dt_r) + 1, n_eff)) if store_paths else None
        integrated_variance = (np.zeros(n_eff)
                               if return_integrated_variance else None)
        if store_paths:
            paths[0] = s

        for i in range(len(self.dt_r)):
            hr = self.dt_r[i] / m                 # drift increment, Act/365
            hv = self.dt_v[i] / m                 # variance increment, Bus/260
            rt = np.sqrt(hv)
            for j in range(m):
                sig = self.grid.sigma_bilinear(i + j / m, s / s0_ref)
                if integrated_variance is not None:
                    integrated_variance += sig ** 2 * hv
                s = s * np.exp(self.b * hr - 0.5 * sig ** 2 * hv
                               + sig * rt * z[i * m + j])
            if store_paths:
                paths[i + 1] = s
        if store_paths and return_integrated_variance:
            return s, paths, integrated_variance
        if store_paths:
            return s, paths
        if return_integrated_variance:
            return s, integrated_variance
        return s

    def terminal_spots_gbm(self, s0: float, sigma: float,
                           z: np.ndarray | None = None) -> np.ndarray:
        """Same driving noise, constant volatility.  Used as a control variate.

        Uses the same two clocks as `terminal_spots`, so its exact expectation
        is the Black-Scholes value at forward `s0 * exp(b * tau_r)` and total
        variance `sigma^2 * tau_vol` -- which is what `_pv_cv_exact` prices.
        """
        z = self._draw() if z is None else z
        hv = np.repeat(self.dt_v / self.n_substeps, self.n_substeps)
        drift = self.b * self.dt_r.sum() - 0.5 * sigma ** 2 * self.dt_v.sum()
        diff = sigma * (np.sqrt(hv)[:, None] * z).sum(axis=0)
        return float(s0) * np.exp(drift + diff)

    # ------------------------------------------------------------------ #
    def price_european(self, strike: float, maturity, is_call: bool = True,
                       bump: float | None = None,
                       control_variate: bool = True) -> MCResult:
        """Price a vanilla and, if `bump` is given, a central-difference delta.

        The bumped runs reuse the same normal draws (common random numbers),
        which is what turns a 1% bump into a usable delta at 1e5 paths.

        With `control_variate` a geometric Brownian motion driven by the SAME
        normals, with constant volatility equal to the implied volatility of
        the option, is priced alongside.  Its expectation is known in closed
        form, so subtracting it removes most of the Monte Carlo noise without
        introducing bias.  Raw and adjusted estimates are both reported.
        """
        self._check_horizon(maturity)
        z = self._draw()
        df = self.surface.discount_factor(maturity)
        s0 = self.surface.market.spot
        tau_v = self.surface.tau_vol(maturity)
        tau_r = self.surface.tau_r(maturity)
        sigma_cv = float(self.surface.implied_vol(maturity, strike))

        def _payoff(s_t):
            return np.maximum(s_t - strike, 0.0) if is_call \
                else np.maximum(strike - s_t, 0.0)

        def _pv_lv(spot: float) -> np.ndarray:
            return df * _payoff(self.terminal_spots(spot, z))

        def _pv_cv(spot: float) -> np.ndarray:
            return df * _payoff(self.terminal_spots_gbm(spot, sigma_cv, z))

        def _pv_cv_exact(spot: float) -> float:
            F = spot * np.exp(self.b * tau_r)
            return float(bs_price_w(F, strike, sigma_cv ** 2 * tau_v, df, is_call))

        base = _pv_lv(s0)
        pv_raw, se_raw = _mean_stderr(base, self.antithetic)

        res = MCResult(pv=pv_raw, stderr=se_raw, n_paths=base.size,
                       n_steps=len(self.dt_r) * self.n_substeps)
        res.extra["pv_raw"] = pv_raw
        res.extra["stderr_raw"] = se_raw
        res.extra["sigma_cv"] = sigma_cv
        res.extra["sum_dt_r"] = float(self.dt_r.sum())
        res.extra["sum_dt_v"] = float(self.dt_v.sum())

        beta = 0.0
        if control_variate:
            cv = _pv_cv(s0)
            base_obs = _estimator_samples(base, self.antithetic)
            cv_obs = _estimator_samples(cv, self.antithetic)
            var = float(cv_obs.var(ddof=1))
            beta = (float(np.cov(base_obs, cv_obs, ddof=1)[0, 1] / var)
                    if var > 0 else 0.0)
            adj = base - beta * (cv - _pv_cv_exact(s0))
            res.pv, res.stderr = _mean_stderr(adj, self.antithetic)
            res.extra["beta"] = beta
            res.extra["corr"] = float(np.corrcoef(base_obs, cv_obs)[0, 1])

        if bump is not None:
            def _adjusted(spot: float) -> np.ndarray:
                p = _pv_lv(spot)
                if not control_variate:
                    return p
                return p - beta * (_pv_cv(spot) - _pv_cv_exact(spot))

            up, dn = _adjusted(s0 + bump), _adjusted(s0 - bump)
            diff = (up - dn) / (2.0 * bump)
            res.delta, res.delta_stderr = _mean_stderr(diff, self.antithetic)
            res.extra["pv_up"] = float(up.mean())
            res.extra["pv_dn"] = float(dn.mean())
        return res

    def price_diagnostics(self, strikes: Sequence[float], maturity,
                          ratio_bin_edges: Sequence[float] | None = None
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Price a strike grid and condition integrated variance on terminal spot.

        The implied-volatility Monte Carlo leg uses one constant volatility per
        strike, equal to ImpliedVol(maturity, strike).  The local-volatility leg
        uses one shared set of terminal spots for every strike.  Both legs use
        the same antithetic normal draws.

        The second table reports

            path_integrated_variance
                = sum sigma_loc(t_i, S_i)^2 * delta_tau_vol_i

        conditional on terminal-spot-ratio bins.  Its comparison with implied
        total variance is diagnostic; the two are not identities in a local-vol
        model.
        """
        self._check_horizon(maturity)
        strikes = np.asarray(strikes, dtype=float)
        if strikes.ndim != 1 or strikes.size == 0:
            raise ValueError("strikes must be a non-empty one-dimensional array")

        z = self._draw()
        s0 = self.surface.market.spot
        df = self.surface.discount_factor(maturity)
        tau_v = self.surface.tau_vol(maturity)
        tau_r = self.surface.tau_r(maturity)
        F = self.surface.forward(maturity, s0)
        terminal_lv, integrated_var = self.terminal_spots(
            s0, z, return_integrated_variance=True)
        terminal_implied_w = np.asarray(
            self.surface.implied_total_variance(maturity, terminal_lv),
            dtype=float)

        hv = np.repeat(self.dt_v / self.n_substeps, self.n_substeps)
        brownian = (np.sqrt(hv)[:, None] * z).sum(axis=0)
        rows = []
        for strike in strikes:
            sigma_imp = float(self.surface.implied_vol(maturity, strike))
            w_imp = sigma_imp ** 2 * tau_v
            pv_bs = float(bs_price_w(F, strike, w_imp, df, True))
            terminal_imp = s0 * np.exp(
                self.b * tau_r - 0.5 * w_imp + sigma_imp * brownian)

            payoff_lv = df * np.maximum(terminal_lv - strike, 0.0)
            payoff_imp = df * np.maximum(terminal_imp - strike, 0.0)
            lv_raw, lv_raw_se = _mean_stderr(payoff_lv, self.antithetic)
            imp_mc, imp_mc_se = _mean_stderr(payoff_imp, self.antithetic)

            lv_obs = _estimator_samples(payoff_lv, self.antithetic)
            imp_obs = _estimator_samples(payoff_imp, self.antithetic)
            var_imp = float(imp_obs.var(ddof=1))
            beta = (float(np.cov(lv_obs, imp_obs, ddof=1)[0, 1] / var_imp)
                    if var_imp > 0 else 0.0)
            adjusted = payoff_lv - beta * (payoff_imp - pv_bs)
            lv_mc, lv_mc_se = _mean_stderr(adjusted, self.antithetic)
            paired_diff, paired_diff_se = _mean_stderr(
                payoff_lv - payoff_imp, self.antithetic)

            rows.append({
                "strike": float(strike),
                "level": float(strike / s0),
                "implied_vol": sigma_imp,
                "implied_total_variance": w_imp,
                "bs_pv": pv_bs,
                "mc_implied_pv": imp_mc,
                "mc_implied_stderr": imp_mc_se,
                "mc_implied_minus_bs": imp_mc - pv_bs,
                "mc_implied_rel_error_pct": 100.0 * (imp_mc / pv_bs - 1.0),
                "mc_local_pv_raw": lv_raw,
                "mc_local_stderr_raw": lv_raw_se,
                "control_beta": beta,
                "control_corr": float(np.corrcoef(lv_obs, imp_obs)[0, 1]),
                "mc_local_pv": lv_mc,
                "mc_local_stderr": lv_mc_se,
                "mc_local_minus_bs": lv_mc - pv_bs,
                "mc_local_rel_error_pct": 100.0 * (lv_mc / pv_bs - 1.0),
                "mc_local_zscore_vs_bs": ((lv_mc - pv_bs) / lv_mc_se
                                             if lv_mc_se > 0 else np.nan),
                "paired_raw_local_minus_implied": paired_diff,
                "paired_difference_stderr": paired_diff_se,
                "paired_difference_zscore": (paired_diff / paired_diff_se
                                                if paired_diff_se > 0 else np.nan),
                "n_paths": int(terminal_lv.size),
                "n_steps": int(len(self.dt_r) * self.n_substeps),
            })
        pricing = pd.DataFrame(rows)

        edges = np.asarray(DEFAULT_TERMINAL_RATIO_BINS if ratio_bin_edges is None
                           else ratio_bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("ratio_bin_edges must be strictly increasing")
        ratio = terminal_lv / s0
        bin_rows = []
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = ((ratio >= left) & (ratio < right)
                    if i < edges.size - 2
                    else (ratio >= left) & (ratio <= right))
            count = int(mask.sum())
            if count == 0:
                continue
            mean_iv, iv_se = _conditional_mean_stderr(
                integrated_var, mask, self.antithetic)
            mean_implied_w, implied_w_se = _conditional_mean_stderr(
                terminal_implied_w, mask, self.antithetic)
            mean_spot = float(terminal_lv[mask].mean())
            implied_w_at_mean = float(self.surface.implied_total_variance(
                maturity, mean_spot))
            implied_vol = float(self.surface.implied_vol(maturity, mean_spot))
            iv_values = integrated_var[mask]
            bin_rows.append({
                "bin_left_ratio": float(left),
                "bin_right_ratio": float(right),
                "n_paths": count,
                "path_probability": count / terminal_lv.size,
                "mean_terminal_ratio": float(ratio[mask].mean()),
                "mean_terminal_spot": mean_spot,
                "mean_path_integrated_variance": mean_iv,
                "path_integrated_variance_stderr": iv_se,
                "std_path_integrated_variance": float(iv_values.std(ddof=1))
                if count > 1 else np.nan,
                "p10_path_integrated_variance": float(np.quantile(iv_values, 0.10)),
                "median_path_integrated_variance": float(np.quantile(iv_values, 0.50)),
                "p90_path_integrated_variance": float(np.quantile(iv_values, 0.90)),
                "implied_vol_at_mean_terminal_spot": implied_vol,
                "mean_implied_total_variance_at_terminal_spot": mean_implied_w,
                "mean_implied_total_variance_stderr": implied_w_se,
                "implied_total_variance_at_mean_terminal_spot": implied_w_at_mean,
                "integrated_minus_mean_implied_variance": mean_iv - mean_implied_w,
                "integrated_vs_mean_implied_ratio": (
                    mean_iv / mean_implied_w if mean_implied_w > 0 else np.nan),
            })
        return pricing, pd.DataFrame(bin_rows)

    def bump_delta_diagnostics(self, strikes: Sequence[float], maturity,
                               grid_up: LocalVolGrid,
                               grid_down: LocalVolGrid,
                               bump: float) -> pd.DataFrame:
        """Compare BS, constant-implied-vol MC and bumped local-vol deltas.

        The up/down local-vol grids must be rebuilt before this call with

            spot_adj_up   = log((S0 + bump) / ref_spot)
            spot_adj_down = log((S0 - bump) / ref_spot)

        using the delivered ``localvol(T, K, spot_adj)`` function.  The two
        path sets use the same normal draws.  For every strike, a constant-vol
        GBM driven by those normals supplies both the implied-volatility MC
        delta and a control variate for the local-vol delta.
        """
        self._check_horizon(maturity)
        strikes = np.asarray(strikes, dtype=float)
        if strikes.ndim != 1 or strikes.size == 0:
            raise ValueError("strikes must be a non-empty one-dimensional array")

        s0 = float(self.surface.market.spot)
        bump = float(bump)
        if not (0.0 < bump < s0):
            raise ValueError("bump must be strictly between zero and spot")
        s_up, s_down = s0 + bump, s0 - bump

        for name, shifted in (("up", grid_up), ("down", grid_down)):
            if shifted.dates != self.grid.dates:
                raise ValueError(f"{name} grid date axis differs from the base grid")
            if not np.array_equal(shifted.ratios, self.grid.ratios):
                raise ValueError(f"{name} grid ratio axis differs from the base grid")

        mc_up = LocalVolMC(
            self.surface, grid_up, n_paths=self.n_paths, seed=self.seed,
            antithetic=self.antithetic, n_substeps=self.n_substeps)
        mc_down = LocalVolMC(
            self.surface, grid_down, n_paths=self.n_paths, seed=self.seed,
            antithetic=self.antithetic, n_substeps=self.n_substeps)
        mc_up._check_horizon(maturity)
        mc_down._check_horizon(maturity)

        # One common set of random numbers for all three legs and both bumps.
        z = self._draw()
        terminal_lv_up = mc_up.terminal_spots(s_up, z)
        terminal_lv_down = mc_down.terminal_spots(s_down, z)

        df = self.surface.discount_factor(maturity)
        tau_v = self.surface.tau_vol(maturity)
        tau_r = self.surface.tau_r(maturity)
        carry = float(np.exp(self.b * tau_r))
        F0, F_up, F_down = s0 * carry, s_up * carry, s_down * carry

        hv = np.repeat(self.dt_v / self.n_substeps, self.n_substeps)
        brownian = (np.sqrt(hv)[:, None] * z).sum(axis=0)
        rows = []
        for strike in strikes:
            sigma_imp = float(self.surface.implied_vol(maturity, strike))
            w_imp = sigma_imp ** 2 * tau_v

            # The BS/implied leg keeps the option's base implied volatility
            # fixed under the spot bump for the calibration comparison.
            bs_delta = float(bs_delta_w(
                F0, strike, w_imp, df, carry, is_call=True))
            bs_pv_up = float(bs_price_w(
                F_up, strike, w_imp, df, is_call=True))
            bs_pv_down = float(bs_price_w(
                F_down, strike, w_imp, df, is_call=True))
            bs_delta_bump = (bs_pv_up - bs_pv_down) / (2.0 * bump)

            common_exp = np.exp(
                self.b * tau_r - 0.5 * w_imp + sigma_imp * brownian)
            terminal_imp_up = s_up * common_exp
            terminal_imp_down = s_down * common_exp

            payoff_imp_up = df * np.maximum(terminal_imp_up - strike, 0.0)
            payoff_imp_down = df * np.maximum(terminal_imp_down - strike, 0.0)
            delta_imp_paths = ((payoff_imp_up - payoff_imp_down)
                               / (2.0 * bump))
            delta_imp, delta_imp_se = _mean_stderr(
                delta_imp_paths, self.antithetic)

            payoff_lv_up = df * np.maximum(terminal_lv_up - strike, 0.0)
            payoff_lv_down = df * np.maximum(terminal_lv_down - strike, 0.0)
            delta_lv_paths = ((payoff_lv_up - payoff_lv_down)
                              / (2.0 * bump))
            delta_lv_raw, delta_lv_raw_se = _mean_stderr(
                delta_lv_paths, self.antithetic)

            lv_obs = _estimator_samples(delta_lv_paths, self.antithetic)
            imp_obs = _estimator_samples(delta_imp_paths, self.antithetic)
            var_imp = float(imp_obs.var(ddof=1))
            beta = (float(np.cov(lv_obs, imp_obs, ddof=1)[0, 1] / var_imp)
                    if var_imp > 0 else 0.0)
            adjusted = delta_lv_paths - beta * (
                delta_imp_paths - bs_delta_bump)
            delta_lv, delta_lv_se = _mean_stderr(adjusted, self.antithetic)

            rows.append({
                "strike": float(strike),
                "level": float(strike / s0),
                "implied_vol": sigma_imp,
                "spot_up": s_up,
                "spot_down": s_down,
                "spot_adj_up": float(np.log(s_up / self.surface.ref_spot)),
                "spot_adj_down": float(np.log(s_down / self.surface.ref_spot)),
                "bs_delta": bs_delta,
                "bs_delta_bump": bs_delta_bump,
                "mc_implied_delta": delta_imp,
                "mc_implied_delta_stderr": delta_imp_se,
                "mc_implied_minus_bs_bump": delta_imp - bs_delta_bump,
                "mc_local_delta_raw": delta_lv_raw,
                "mc_local_delta_raw_stderr": delta_lv_raw_se,
                "delta_control_beta": beta,
                "delta_control_corr": float(np.corrcoef(lv_obs, imp_obs)[0, 1]),
                "mc_local_delta": delta_lv,
                "mc_local_delta_stderr": delta_lv_se,
                "mc_local_minus_bs_bump": delta_lv - bs_delta_bump,
                "mc_local_abs_error_vs_bs_bump": abs(delta_lv - bs_delta_bump),
                "mc_local_zscore_vs_bs_bump": (
                    (delta_lv - bs_delta_bump) / delta_lv_se
                    if delta_lv_se > 0 else np.nan),
                "n_paths": int(terminal_lv_up.size),
                "n_steps": int(len(self.dt_r) * self.n_substeps),
                "grid_up_n_undefined": int(grid_up.n_undefined),
                "grid_down_n_undefined": int(grid_down.n_undefined),
            })
        return pd.DataFrame(rows)

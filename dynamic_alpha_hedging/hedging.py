"""Reusable vanilla marks and cached joint Alpha/Beta/Delta MC measurements."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from svi_localvol.blackscholes import bs_price_w, bs_delta_w, bs_vega, bs_gamma_w
from svi_localvol.montecarlo import LocalVolGrid, LocalVolMC
from .config import DynamicAlphaConfig
from .data_loader import date_at_tau
from .step03 import _anchor_at_sticky_strike, _cell_quality


def vanilla_mark(surface, expiry, strike):
    """Raw SVI mark of a call, never an alpha-dependent realised price."""
    tau = surface.tau_vol(expiry)
    if tau <= 0:
        raise ValueError("hedge contract must expire after the holding interval")
    sigma = float(surface.implied_vol(expiry, strike))
    forward, df = surface.forward(expiry), surface.discount_factor(expiry)
    carry = forward / surface.ref_spot
    w = sigma * sigma * tau
    mark = {
        "pv": float(bs_price_w(forward, strike, w, df)),
        "iv": sigma,
        "bs_delta": float(bs_delta_w(forward, strike, w, df, carry)),
        "vega": float(bs_vega(forward, strike, w, df, tau)),
        "gamma": float(bs_gamma_w(forward, strike, w, df, carry)),
    }
    if not np.isfinite(list(mark.values())).all():
        raise ValueError("nonfinite historical option mark or Greeks")
    return mark


def contract_interval(previous, current, tenor, level):
    """One-day P&L holds the SAME strike and calendar expiry at both closes."""
    expiry = date_at_tau(previous, tenor)
    strike = level * previous.ref_spot
    for surface in (previous, current):
        tau = surface.tau_vol(expiry)
        if not surface.taus[0] <= tau <= surface.taus[-1]:
            raise ValueError(f"{surface.market.pricing_date}: contract tenor outside quotes")
    start = vanilla_mark(previous, expiry, strike)
    end = vanilla_mark(current, expiry, strike)
    dt = (current.market.pricing_date - previous.market.pricing_date).days / 365.0
    tau1, tr1 = current.tau_vol(expiry), current.tau_r(expiry)
    # Theta at frozen IV, using both clocks and only previous-close rates.
    frozen_pv = bs_price_w(
        previous.ref_spot * np.exp(previous.market.cost_of_carry * tr1), strike,
        start["iv"] ** 2 * tau1, np.exp(-previous.market.rate * tr1))
    shortened_expiry = date_at_tau(previous, tau1)
    roll_iv = previous.implied_vol(shortened_expiry, strike) - start["iv"]
    return {
        "feature_date": previous.market.pricing_date,
        "label_date": current.market.pricing_date,
        "tenor": tenor, "level": level, "strike": strike, "expiry": expiry,
        "spot": previous.ref_spot, "next_spot": current.ref_spot,
        "dS": current.ref_spot - previous.ref_spot,
        "dlogS": float(np.log(current.ref_spot / previous.ref_spot)),
        "dt_r": dt, "rate": previous.market.rate,
        "income_yield": previous.market.rate - previous.market.cost_of_carry,
        "pv": start["pv"], "next_pv": end["pv"],
        "iv": start["iv"], "next_iv": end["iv"],
        "dV": end["pv"] - start["pv"],
        "bs_delta": start["bs_delta"], "vega": start["vega"],
        "gamma": start["gamma"], "time_pnl": float(frozen_pv-start["pv"]),
        "term_roll_iv": float(roll_iv),
    }


class MCMapStore:
    """One cache entry per date/tenor; strikes and all strategies share paths.

    Cache identity includes quote data, market conventions, numerical settings
    and implementation hashes. A reference inverse never supplies today's delta.
    """

    def __init__(self, config: DynamicAlphaConfig, tenors, levels, cache_dir):
        self.config = config
        self.tenors, self.levels = tuple(tenors), tuple(levels)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        root = Path(__file__).resolve().parent.parent
        self.code_hash = hashlib.sha256(b"".join(
            (root / name).read_bytes() for name in (
                "svi_localvol/montecarlo.py", "svi_localvol/surface.py",
                "svi_localvol/svi.py", "svi_localvol/blackscholes.py",
                "svi_localvol/conventions.py", "dynamic_alpha_hedging/step03.py",
                "dynamic_alpha_hedging/hedging.py"))).hexdigest()

    def get(self, surface):
        pieces = []
        for tenor in self.tenors:
            identity = {
                "market": asdict(surface.market), "quotes": asdict(surface.quotes),
                "config": asdict(self.config), "tenor": tenor,
                "levels": self.levels, "code": self.code_hash,
            }
            digest = hashlib.sha256(json.dumps(
                identity, sort_keys=True, default=str).encode()).hexdigest()
            path = self.cache_dir / f"{surface.market.pricing_date}_{tenor:g}_{digest[:20]}.csv"
            if path.exists():
                measured = pd.read_csv(path)
            else:
                measured = self.measure(surface, tenor)
                temporary = path.with_suffix(".tmp")
                measured.to_csv(temporary, index=False)
                temporary.replace(path)
            pieces.append(measured)
        return pd.concat(pieces, ignore_index=True)

    def measure(self, raw_surface, tenor):
        return self.measure_many(raw_surface, (tenor,))[tenor]

    def measure_many(self, raw_surface, tenors):
        """Share each scalar-Alpha bump pair across this date's requested tenors."""
        c = self.config
        surface = raw_surface.repaired()
        expiries = {tenor: date_at_tau(surface, tenor) for tenor in tenors}
        if not expiries:
            raise ValueError("empty tenor axis")
        last_expiry = max(expiries.values())
        spot = surface.ref_spot
        bump = c.step3_spot_bump_fraction * spot
        args = dict(n_ratio=c.step3_n_ratio, ratio_min=c.step3_ratio_min,
                    ratio_max=c.step3_ratio_max, vol_floor=c.step3_vol_floor,
                    vol_cap=c.step3_vol_cap)
        base = LocalVolGrid.build(surface, last_expiry, **args)
        mc = LocalVolMC(surface, base, n_paths=c.step3_n_paths, seed=c.step3_seed,
                        n_substeps=c.step3_n_substeps, antithetic=c.step3_antithetic)
        rows = {tenor: [] for tenor in expiries}
        for alpha in c.step3_alphas:
            up, down = [LocalVolGrid.build(
                surface, last_expiry, spot_adj=np.log((spot + sign*bump)/spot),
                alpha=alpha, **args) for sign in (1, -1)]
            measurements = mc.bumped_implied_vol_diagnostics_many(
                np.asarray(self.levels)*spot, expiries.values(), up, down, bump)
            for tenor, expiry in expiries.items():
                measured = measurements[expiry].copy()
                measured["calibration_date"] = surface.market.pricing_date
                measured["tenor"], measured["alpha"] = tenor, alpha
                measured["repair_iv_change"] = np.asarray(surface.implied_vol(
                    expiry, np.asarray(self.levels)*spot)) - np.asarray(raw_surface.implied_vol(
                        expiry, np.asarray(self.levels)*spot))
                prefix_up, prefix_down = up.prefix(expiry), down.prefix(expiry)
                measured["grid_undefined_fraction"] = max(
                    prefix_up.n_undefined, prefix_down.n_undefined) / prefix_up.sigma.size
                measured["grid_clipped_fraction"] = max(
                    prefix_up.n_clipped, prefix_down.n_clipped) / prefix_up.sigma.size
                rows[tenor].append(measured)
        results = {}
        for tenor, pieces in rows.items():
            curve = _anchor_at_sticky_strike(pd.concat(pieces, ignore_index=True))
            quality = _cell_quality(curve, c)
            results[tenor] = curve.merge(quality[["tenor", "level", "quality_pass",
                                                "inverse_available", "quality_failures"]],
                                       on=["tenor", "level"], validate="many_to_one")
        return results


def cell_curve(curves, tenor, level):
    return curves[np.isclose(curves.tenor, tenor) & np.isclose(curves.level, level)]


def invert_beta(curve, beta):
    """Return alpha, clipped, fallback; preserve measured shape without projection."""
    if curve.empty or not np.isfinite(beta):
        return 1.0, False, True
    curve = curve.sort_values("alpha")
    b = curve["beta_converter" if "beta_converter" in curve else "beta"].to_numpy(float)
    a = curve.alpha.to_numpy(float)
    if (len(a) < 2 or not np.isfinite(b).all() or not np.all(np.diff(b) < 0)
            or not np.all(np.diff(a) > 0)):
        return 1.0, False, True
    return (float(np.interp(beta, b[::-1], a[::-1])),
            bool(beta < b.min() or beta > b.max()), False)


def delta_at_alpha(curve, alpha, bs_delta):
    """Only finite measured CALL deltas are interpolated; otherwise explicit BS fallback."""
    ordered = curve.sort_values("alpha")
    if (ordered.empty or not np.isfinite(alpha)
            or not np.isfinite(ordered.call_delta).all()
            or alpha < ordered.alpha.min() or alpha > ordered.alpha.max()):
        return float(bs_delta), np.nan, True
    delta = float(np.interp(alpha, ordered.alpha, ordered.call_delta))
    # Conservative linear interpolation of marginal SEs (not independence).
    se = float(np.interp(alpha, ordered.alpha, ordered.call_delta_stderr))
    return delta, se, False

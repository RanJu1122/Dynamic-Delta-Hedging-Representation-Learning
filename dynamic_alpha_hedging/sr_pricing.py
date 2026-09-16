"""Shared-book SR surfaces and multi-expiry MC.

This adapter accepts vector-valued SR; scalar-Alpha precompute uses LocalVolMC.
Both use LocalVolGrid interpolation, and caches track their engine provenance.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json

import numpy as np
import pandas as pd

from svi_localvol.blackscholes import bs_price_w, bs_vega, implied_total_variance
from svi_localvol.montecarlo import (
    LocalVolGrid, _mean_stderr, _control_coefficient, _validate_beta_inversion)
from .bump import SpotBumpPolicy
from .artifacts import file_sha256
from .data_loader import date_at_tau
from .mc_library import NUMERICAL_FIELDS, digest, engine_signature, snapshot_signature
from .precompute import _writer_lock, _csv


SR_KINDS = ("constant_sr", "term_sr", "term_spot_sr")


@dataclass
class AlphaSurface:
    """One close-date policy, frozen throughout both spot bumps.

    Tenor nodes are the existing constant Business/260 research tenors, not
    additional forecasts at raw quoted VolDates. Spatial interpolation holds
    the unshifted y=log(K/F_ref(T)) coordinate fixed across tenor slices.
    """

    surface: object
    tenors: tuple
    levels: tuple
    values: np.ndarray
    kind: str
    anchor_tenor: float = .25

    def __post_init__(self):
        if self.kind not in SR_KINDS:
            raise ValueError(f"unknown SR policy: {self.kind}")
        self.values = np.array(self.values, dtype=float, copy=True)
        if (self.values.shape != (len(self.tenors), len(self.levels))
                or not np.isfinite(self.values).all()
                or np.any((self.values < 0) | (self.values > 2))):
            raise ValueError("Alpha nodes must be a finite tenor/level matrix in [0, 2]")
        for axis in (self.tenors, self.levels):
            if not len(axis) or min(axis) <= 0 or np.any(np.diff(axis) <= 0):
                raise ValueError("SR axes must be positive and strictly increasing")
        self.atm = self._node(self.levels, 1.)
        self.anchor = self._node(self.tenors, self.anchor_tenor)
        self.expiries = [date_at_tau(self.surface, t) for t in self.tenors]
        self.taus = np.array([self.surface.tau_vol(d) for d in self.expiries])
        if np.any(np.diff(self.taus) <= 0):
            raise ValueError("SR tenors round to duplicate business dates")
        self.y_nodes = np.array([np.log(np.asarray(self.levels) * self.surface.ref_spot
                                       / self.surface.forward(d)) for d in self.expiries])
        # dupire use y instead of K or F

    @staticmethod
    def _node(axis, value):
        hits = np.flatnonzero(np.isclose(axis, value, rtol=0, atol=1e-12))
        if len(hits) != 1:
            raise ValueError(f"SR grid requires an exact node at {value}")
        return int(hits[0])

    def __call__(self, T, y):
        y = np.asarray(y, dtype=float)
        tau = self.surface.tau_vol(T)
        if self.kind == "constant_sr":
            return np.full_like(y, self.values[self.anchor, self.atm])
        if self.kind == "term_sr":
            return np.full_like(y, np.interp(tau, self.taus, self.values[:, self.atm]))
        hi = min(int(np.searchsorted(self.taus, tau)), len(self.taus) - 1)
        lo = max(hi - 1, 0)
        weight = (np.clip((tau-self.taus[lo])/(self.taus[hi]-self.taus[lo]), 0, 1)
                  if hi != lo else 0.)
        left = np.interp(y, self.y_nodes[lo], self.values[lo])
        right = np.interp(y, self.y_nodes[hi], self.values[hi])
        return np.clip((1-weight)*left + weight*right, 0., 2.)

    def at_contract(self, T, strike):
        return float(self(T, np.log(strike / self.surface.forward(T))))

    def identity(self):
        if np.ptp(self.values) == 0 or self.kind == "constant_sr":
            return {"constant": float(self.values[self.anchor, self.atm])}
        return {"kind": self.kind, "taus": self.taus.tolist(),
                "y_nodes": self.y_nodes.tolist(),
                "alpha": (self.values[:, self.atm] if self.kind == "term_sr"
                          else self.values).tolist()}


def shared_local_vol(surface, T, K, spot_adj, profile):
    """Mentor's pointwise y_adj rule; no derivatives of alpha are introduced.

    w and its derivatives stay on the reference SVI surface, exactly as in
    VolSurface.local_vol(). Alpha affects only y in the Dupire denominator.
    """
    res = surface.total_variance(T, K, order=2)
    alpha = np.asarray(profile(T, res["y"]), dtype=float)
    if (not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 2))):
        raise ValueError("SR provider returned invalid Alpha")
    alpha = np.broadcast_to(alpha, np.shape(res["y"]))
    w, wy, wyy, wt = (res[k] for k in ("w", "dw_dy", "d2w_dy2", "dw_dtau"))
    y = res["y"] - float(spot_adj)*alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = 1-y/w*wy + .25*(-.25-1/w+y*y/(w*w))*wy*wy + .5*wyy
        return np.sqrt(np.where((denominator > 0) & (wt >= 0) & (w > 0),
                                wt/denominator, np.nan))


def simulate_snapshots(surface, grids, initial_spots, expiries, config):
    """Stream common normals once to the longest expiry; store only expiry states.

    With a common seed, every profile and bump sees the same Brownian path.
    The draw order also matches the legacy engine's antithetic matrix draw.
    """
    grid = next(iter(grids.values()))
    indices = {grid.dates.index(d): d for d in expiries if d in grid.dates}
    if len(indices) != len(set(expiries)) or 0 in indices:
        raise ValueError("every payoff expiry must be a positive exact grid date")
    if any(g.dates != grid.dates or not np.array_equal(g.ratios, grid.ratios)
           for g in grids.values()):
        raise ValueError("bump grids must have identical axes")
    n, substeps = config.step3_n_paths, config.step3_n_substeps
    rng = np.random.default_rng(config.step3_seed)
    spots = {key: np.full(n, float(initial_spots[key])) for key in grids}
    brownian = np.zeros(n)
    snapshots, audit = {}, {key: {"outside": 0, "lookups": 0, "ever": np.zeros(n, bool)}
                            for key in grids}
    for i, (dr, dv) in enumerate(zip(np.diff(grid.tau_r), np.diff(grid.tau_vol))):
        if dr <= 0 or dv <= 0:
            raise ValueError("MC grid clocks must be strictly increasing")
        hr, hv = dr/substeps, dv/substeps
        for j in range(substeps):
            z = rng.standard_normal(n//2 if config.step3_antithetic else n)
            if config.step3_antithetic:
                z = np.concatenate((z, -z))
            brownian += np.sqrt(hv)*z
            for key, g in grids.items():
                ratio = spots[key]/surface.ref_spot
                outside = (ratio < g.ratios[0]) | (ratio > g.ratios[-1])
                audit[key]["outside"] += int(outside.sum())
                audit[key]["lookups"] += n
                audit[key]["ever"] |= outside
                sigma = g.sigma_bilinear(i+j/substeps, ratio)
                spots[key] *= np.exp(surface.market.cost_of_carry*hr - .5*sigma*sigma*hv
                                     + sigma*np.sqrt(hv)*z)
                if not np.isfinite(spots[key]).all():
                    raise ValueError("nonfinite MC path state")
        if i+1 in indices:
            snapshots[indices[i+1]] = ({k: s.copy() for k, s in spots.items()}, brownian.copy())
    metrics = {}
    for key, a in audit.items():
        metrics[f"{key}_outside_lookup_fraction"] = a["outside"]/a["lookups"]
        metrics[f"{key}_outside_path_fraction"] = float(a["ever"].mean())
    return snapshots, metrics


def _controlled_prices(surface, expiry, strike, states, brownian, config):
    """Paired control variates; return call PV/Delta, with IV-beta as an audit."""
    spot, frac = surface.ref_spot, config.step3_spot_bump_fraction
    starts = np.array([spot, spot*(1+frac), spot*(1-frac)])
    tau, tr = surface.tau_vol(expiry), surface.tau_r(expiry)
    df, carry = surface.discount_factor(expiry), np.exp(surface.market.cost_of_carry*tr)
    sigma = float(surface.implied_vol(expiry, strike))
    w = sigma*sigma*tau
    terminal_cv = starts[:, None]*np.exp(surface.market.cost_of_carry*tr-.5*w+sigma*brownian)
    terminal = np.stack([states[k] for k in ("base", "up", "down")])
    candidates = []
    for is_call in (True, False):
        sign = 1 if is_call else -1
        lv = df*np.maximum(sign*(terminal-strike), 0)
        cv = df*np.maximum(sign*(terminal_cv-strike), 0)
        exact = np.asarray(bs_price_w(starts*carry, strike, w, df, is_call))
        def coefficient(a, b):
            return _control_coefficient(a, b, config.step3_antithetic)
        c = coefficient(lv[1]-lv[2], cv[1]-cv[2])
        delta_adjusted = lv[1:]-c*(cv[1:]-exact[1:, None])
        coefficients = np.array([coefficient(x, y) for x, y in zip(lv, cv)])
        adjusted = lv-coefficients[:, None]*(cv-exact[:, None])
        pv, se = np.array([_mean_stderr(x, config.step3_antithetic) for x in adjusted]).T
        delta_pv = np.array([_mean_stderr(x, config.step3_antithetic)[0]
                             for x in delta_adjusted])
        delta, delta_se = _mean_stderr((delta_adjusted[0]-delta_adjusted[1])/(2*spot*frac),
                                      config.step3_antithetic)
        iv, vegas, clipped = [], [], False
        for leg in (1, 2):
            forward = starts[leg]*carry
            lo = df*max(sign*(forward-strike), 0.)
            hi = df*(forward if is_call else strike)
            epsilon = min(1e-12*max(1., spot), (hi-lo)/4)
            price = float(np.clip(pv[leg], lo+epsilon, hi-epsilon))
            clipped |= price != pv[leg]
            variance = float(implied_total_variance(price, forward, strike, df, is_call))
            iv.append(np.sqrt(variance/tau))
            vegas.append(float(bs_vega(forward, strike, variance, df, tau)))
        dlog = np.log(starts[1]/starts[2])
        beta_se = (_mean_stderr(adjusted[1]/vegas[0]-adjusted[2]/vegas[1],
                               config.step3_antithetic)[1]/dlog
                   if min(vegas) > 0 else np.nan)
        if not is_call:  # All exported PVs and deltas are CALL values.
            pv += df*(starts*carry-strike)
            delta_pv += df*(starts[1:]*carry-strike)
            delta += df*carry
        candidates.append({"mc_pv": pv[0], "mc_pv_up": pv[1], "mc_pv_down": pv[2],
                           "mc_pv_stderr": se[0], "delta": delta, "delta_stderr": delta_se,
                           "mc_pv_up_stderr": se[1], "mc_pv_down_stderr": se[2],
                           "mc_delta_pv_up": delta_pv[0], "mc_delta_pv_down": delta_pv[1],
                           "delta_control_beta": c,
                           "price_control_beta_base": coefficients[0],
                           "price_control_beta_up": coefficients[1],
                           "price_control_beta_down": coefficients[2],
                           "beta_model": -(iv[0]-iv[1])/dlog, "beta_model_stderr": beta_se,
                           "price_clipped_for_inversion": bool(clipped),
                           "iv_estimator": "call" if is_call else "put"})
    chosen = min(candidates, key=lambda r: (r["price_clipped_for_inversion"],
               not np.isfinite(r["beta_model_stderr"]),
               r["beta_model_stderr"] if np.isfinite(r["beta_model_stderr"]) else np.inf))
    return _validate_beta_inversion(chosen)


class SharedSRPricer:
    """Content-validated cache of NEW MC results, never a legacy delta reader."""

    def __init__(self, config, cache_dir, *, bump_policy=None):
        self.config, self.cache_dir = config, Path(cache_dir)
        # Legacy shared-SR callers keep uniform bumps unless a policy is supplied.
        self.bump_policy = bump_policy or SpotBumpPolicy(short_business_days=0)
        self.code = digest({"adapter": file_sha256(__file__), "legacy": engine_signature(),
                            "bump": file_sha256(Path(__file__).with_name("bump.py"))})
        self._base_key, self._base = None, None
        self.computed, self.reused = 0, 0

    def get(self, raw_surface, marks, profile):
        contracts = marks[["tenor", "level", "expiry", "strike"]].to_dict("records")
        identity = {"snapshot": snapshot_signature(raw_surface), "profile": profile.identity(),
                    "contracts": contracts, "engine": self.code,
                    "bump_policy": asdict(self.bump_policy),
                    "numerical": {k: getattr(self.config, k) for k in NUMERICAL_FIELDS}}
        key = digest(identity)
        with _writer_lock(self.cache_dir):
            path = self.cache_dir / f"{raw_surface.market.pricing_date}_{key}.csv"
            metadata = path.with_suffix(".json")
            if metadata.exists():
                info = json.loads(metadata.read_text())
                if info["identity"] != json.loads(json.dumps(identity, default=str)) or file_sha256(path) != info["sha256"]:
                    raise ValueError(f"new SR cache checksum/identity mismatch: {path}")
                result = pd.read_csv(path)
                if len(result) != len(marks):
                    raise ValueError("incomplete SR pricing cache")
                self.reused += 1
                return result
            result = self.measure(raw_surface, marks, profile)
            _csv(result, path)
            temporary = metadata.with_suffix(".tmp")
            temporary.write_text(json.dumps({"identity": identity, "sha256": file_sha256(path)},
                                             default=str, indent=2))
            temporary.replace(metadata)
            self.computed += 1
            return result

    def measure(self, raw_surface, marks, profile):
        c, surface = self.config, raw_surface.repaired()
        expiries = sorted(set(marks.expiry))
        args = dict(n_ratio=c.step3_n_ratio, ratio_min=c.step3_ratio_min,
                    ratio_max=c.step3_ratio_max, vol_floor=c.step3_vol_floor, vol_cap=c.step3_vol_cap)
        base_key = digest({"snapshot": snapshot_signature(raw_surface), "expiries": expiries})
        if base_key != self._base_key:
            base_grid = LocalVolGrid.build(surface, expiries[-1], **args)
            snapshots, metrics = simulate_snapshots(surface, {"base": base_grid},
                {"base": surface.ref_spot}, expiries, c)
            metrics.update(base_grid_undefined_fraction=base_grid.n_undefined/base_grid.sigma.size,
                           base_grid_clipped_fraction=base_grid.n_clipped/base_grid.sigma.size)
            self._base_key, self._base = base_key, (snapshots, metrics)
        base_snapshots, base_metrics = self._base
        fractions = {expiry: self.bump_policy.fraction(surface, expiry, c.step3_spot_bump_fraction)
                     for expiry in expiries}
        simulations = {}
        for fraction in sorted(set(fractions.values())):
            group_expiries = [expiry for expiry in expiries if fractions[expiry] == fraction]
            grids, starts = {}, {}
            for name, sign in (("up", 1), ("down", -1)):
                starts[name] = surface.ref_spot*(1+sign*fraction)
                adj = np.log(starts[name]/surface.ref_spot)
                grids[name] = LocalVolGrid.build(surface, group_expiries[-1], **args,
                    local_vol_fn=lambda T, K, shift=adj: shared_local_vol(surface, T, K, shift, profile))
            # The common seed gives identical noise prefixes across bump groups.
            # Short-bump paths stop at their final payoff; never run them out to 2Y.
            snapshots, metrics = simulate_snapshots(surface, grids, starts, group_expiries, c)
            simulations[fraction] = (snapshots, {**base_metrics, **metrics,
                "grid_undefined_fraction": max(g.n_undefined/g.sigma.size for g in grids.values()),
                "grid_clipped_fraction": max(g.n_clipped/g.sigma.size for g in grids.values()),
                "grid_date_rows": len(grids["up"].dates), "grid_ratio_nodes": c.step3_n_ratio,
                "n_paths": c.step3_n_paths,
                "max_n_steps": (len(grids["up"].dates)-1)*c.step3_n_substeps,
                "spot_bump_fraction": fraction,
                "short_expiry_bump_applied": fraction < c.step3_spot_bump_fraction})
        rows = []
        for row in marks.itertuples(index=False):
            fraction = fractions[row.expiry]
            snapshots, audit = simulations[fraction]
            states, brownian = snapshots[row.expiry]
            states = {**states, **base_snapshots[row.expiry][0]}
            result = _controlled_prices(surface, row.expiry, row.strike, states, brownian,
                                        replace(c, step3_spot_bump_fraction=fraction))
            result.update(tenor=row.tenor, level=row.level,
                          alpha_at_contract=profile.at_contract(row.expiry, row.strike),
                          repair_iv_change=float(surface.implied_vol(row.expiry, row.strike)
                                                  - raw_surface.implied_vol(row.expiry, row.strike)))
            rows.append({**result, **audit})
        return pd.DataFrame(rows)

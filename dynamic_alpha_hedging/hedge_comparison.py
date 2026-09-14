"""Small Step 7 controls: direct beta hedges, MC audit and pooled forward maps."""

import json

import numpy as np
import pandas as pd

from svi_localvol.blackscholes import bs_delta_w, bs_vega


def shadow_delta(base_delta, vega, spot, beta):
    """Beta is -dIV/dlogS; vega is dV/dsigma (not per vol percentage point)."""
    return base_delta - vega * beta / spot


def mc_consistency(curve, mark, tau):
    """First-order checks, not convergence certificates; no new MC paths.

    Mid-bump IV approximates the unbumped MC IV. Raw checks use each alpha's
    midpoint Greeks; centered checks use alpha=1 midpoint vega throughout.
    """
    frame = curve.copy()
    required = {"implied_vol_up", "implied_vol_down", "forward_up",
                "forward_down", "beta_model", "beta_converter"}
    if not required.issubset(frame):
        return pd.DataFrame()  # Test providers / old maps without IV diagnostics.
    sigma = (frame.implied_vol_up + frame.implied_vol_down) / 2
    forward = (frame.forward_up + frame.forward_down) / 2
    df = np.exp(-mark.rate * (mark.expiry-mark.feature_date).days / 365)
    w = sigma**2 * tau
    delta = bs_delta_w(forward, mark.strike, w, df, forward/mark.spot)
    vega = bs_vega(forward, mark.strike, w, df, tau)
    anchor = np.flatnonzero(np.isclose(frame.alpha, 1.0))
    if len(anchor) != 1:
        raise ValueError("MC consistency requires one alpha=1 node")
    i = anchor[0]
    mc_one, vega_one = frame.call_delta.iloc[i], np.asarray(vega)[i]
    return pd.DataFrame({
        "feature_date": mark.feature_date, "tenor": mark.tenor,
        "level": mark.level, "alpha": frame.alpha.to_numpy(),
        "beta_raw": frame.beta_model.to_numpy(),
        "beta_converter": frame.beta_converter.to_numpy(),
        "mc_delta": frame.call_delta.to_numpy(),
        "mc_delta_stderr": frame.call_delta_stderr.to_numpy(),
        "mc_midpoint_iv": sigma.to_numpy(),
        "mc_midpoint_vega": np.asarray(vega),
        "raw_chain_residual": np.asarray(frame.call_delta) - shadow_delta(
            delta, vega, mark.spot, frame.beta_model.to_numpy()),
        "centered_adjustment_residual": np.asarray(frame.call_delta) - shadow_delta(
            mc_one, vega_one, mark.spot, frame.beta_converter.to_numpy()),
        "mc_one_minus_bs_delta": mc_one-mark.bs_delta,
        "price_clipped_for_inversion": frame.get("price_clipped_for_inversion", False),
    })


def pooled_converter(tables, train_end):
    """Equal-date mean of forward beta(alpha) nodes, NEVER mean inverse alphas.

    Every sampled date must supply the same axes. Nonfinite means remain NaN
    (no silent omission); nonmonotone pooled curves use the normal BS fallback.
    Across-date std measures state dispersion, not MC standard error.
    """
    raw = pd.concat(tables, ignore_index=True)
    raw["calibration_date"] = pd.to_datetime(raw.calibration_date).dt.date
    dates = sorted(raw.calibration_date.unique())
    if len(dates) < 2 or dates[-1] > train_end:
        raise ValueError("pooled converter requires at least two training-period dates")
    keys = ["tenor", "level", "alpha"]
    if raw.duplicated(["calibration_date", *keys]).any():
        raise ValueError("duplicate pooled converter date/cell/alpha")
    grouped = raw.groupby(keys, sort=True)
    if not grouped.size().eq(len(dates)).all():
        raise ValueError("pooled converter dates must have identical alpha/cell axes")
    result = grouped.beta_converter.agg(
        beta_converter=lambda x: x.mean(skipna=False),
        beta_date_std=lambda x: x.std(ddof=1, skipna=False),
    ).reset_index()
    result["source_dates"] = json.dumps([str(d) for d in dates])
    result["source_date_count"] = len(dates)
    result["calibration_date"] = dates[-1]  # Latest input date, not a synthetic market date.
    result["inverse_available"] = False
    for _, group in result.groupby(["tenor", "level"]):
        b = group.sort_values("alpha").beta_converter.to_numpy(float)
        result.loc[group.index, "inverse_available"] = bool(
            np.isfinite(b).all() and np.all(np.diff(b) < 0))
    return result

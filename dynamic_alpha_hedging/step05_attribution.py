"""Step 5.2: historical option-PV attribution using a stable past elasticity.

This is an attribution diagnostic, not a hedge backtest or a predictor search.
The daily-ratio labels used by Steps 2/4/6 are deliberately left unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import file_sha256
from .data_loader import load_surface_history
from .hedging import contract_interval, invert_beta
from .mc_library import MCLibrary


@dataclass
class AttributionResult:
    elasticities: pd.DataFrame
    option_pnl: pd.DataFrame
    summary: pd.DataFrame
    coverage: pd.DataFrame
    validation: dict
    inputs: dict


def rolling_elasticities(changes, *, window=60, min_observations=20):
    """OLS dIV_surface = intercept - beta*dlogS, through each feature close.

    The window counts observed market transitions, including invalid rows;
    nonconsecutive transitions and nonfinite observations are not fitted.
    Small/zero returns may enter OLS: no division by an individual return.
    """
    if not 3 <= min_observations <= window:
        raise ValueError("attribution requires 3 <= min observations <= window")
    rows = []
    for (tenor, level), group in changes.groupby(["tenor", "level"], sort=True):
        g = group.sort_values("observation_date").copy()
        valid = (g.is_next_business_observation.astype(bool)
                 & np.isfinite(g.dlogS) & np.isfinite(g.dIV_surface))
        x, y = g.dlogS.where(valid), g.dIV_surface.where(valid)
        roll = dict(window=window, min_periods=min_observations)
        n = x.rolling(window, min_periods=1).count()
        sx, sy = x.rolling(**roll).sum(), y.rolling(**roll).sum()
        sxx, sxy = (x*x).rolling(**roll).sum(), (x*y).rolling(**roll).sum()
        denominator = sxx - sx*sx/n
        beta = -(sxy-sx*sy/n)/denominator.where(denominator > 1e-12)
        result = pd.DataFrame({
            "feature_date": g.observation_date, "tenor": tenor, "level": level,
            "beta_regression": beta, "intercept_iv": (sy+beta*sx)/n,
            "regression_nobs": n, "return_sxx_centered": denominator,
            "window_start": g.observation_date.shift(window-1).fillna(g.observation_date.iloc[0]),
            "window_end": g.observation_date,
        })
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def pnl_components(mark, beta):
    """Greeks at the previous close; true dV is a same-K/same-expiry SVI mark.

    The common terms are identical across schemes. The surface response is
    booked once as -vega*beta*dlogS. Alpha=1 uses centered beta=0.
    """
    delta = mark["bs_delta"]*mark["dS"]
    gamma = .5*mark["gamma"]*mark["dS"]**2
    roll = mark["vega"]*mark["term_roll_iv"]
    common = delta+gamma+mark["time_pnl"]+roll
    response = -mark["vega"]*beta*mark["dlogS"]
    return dict(bs_delta_pnl=delta, gamma_pnl=gamma,
                theta_pnl=mark["time_pnl"], term_roll_pnl=roll,
                beta_response_pnl=response, dynamic_delta_group_pnl=delta+response,
                alpha_one_residual=mark["dV"]-common,
                dynamic_alpha_residual=mark["dV"]-common-response)


def _improvement_ci(daily, *, draws=1000, block=10):
    """Paired date-block bootstrap of mean squared normalized PV residuals.

    All nodes in a day remain together. Blocks never cross a calendar gap.
    Output is an exploratory interval conditional on the chosen estimator.
    """
    if len(daily) < 20:
        return np.nan, np.nan
    daily = daily.sort_values("label_date").reset_index(drop=True)
    segments = daily.feature_date.ne(daily.label_date.shift()).cumsum()
    from .step05_diagnostics import block_sample_indices
    selected = block_sample_indices(segments, draws=draws, block=block)
    baseline = daily.baseline_mse.to_numpy()[selected].mean(axis=1)
    dynamic = daily.dynamic_mse.to_numpy()[selected].mean(axis=1)
    valid = baseline > 0
    return tuple(np.quantile(1-np.sqrt(dynamic[valid]/baseline[valid]), [.025, .975]))


def summarize_attribution(frame, train_end):
    rows = []
    valid = frame[frame.attribution_valid].copy()
    for period, period_mask in (
        ("all", np.ones(len(valid), dtype=bool)),
        ("train", valid.label_date <= train_end),
        ("test", valid.label_date > train_end),
    ):
        part = valid.loc[period_mask]
        for quality, quality_mask in (
            ("all_valid", np.ones(len(part), dtype=bool)),
            ("quality_pass_only", part.converter_quality_pass),
        ):
            data = part.loc[quality_mask]
            scopes = [("overall", np.nan, np.nan, data),
                      ("atm_3m", .25, 1., data[np.isclose(data.tenor, .25) & np.isclose(data.level, 1.)]),
                      ("near_atm", np.nan, np.nan, data[data.level.between(.9-1e-12, 1.1+1e-12)])]
            scopes += [("tenor", float(t), np.nan, g) for t, g in data.groupby("tenor")]
            for scope, tenor, level, g in scopes:
                if g.empty:
                    continue
                baseline = g.alpha_one_residual/g.spot
                dynamic = g.dynamic_alpha_residual/g.spot
                direct = g.direct_beta_residual/g.spot
                daily = g[["feature_date", "label_date"]].assign(
                    baseline_mse=baseline**2, dynamic_mse=dynamic**2).groupby(
                    ["feature_date", "label_date"], as_index=False).mean()
                # Point estimate and CI both give equal weight to dates, then nodes.
                b, d = np.sqrt(daily.baseline_mse.mean()), np.sqrt(daily.dynamic_mse.mean())
                low, high = _improvement_ci(daily)
                rows.append(dict(period=period, quality=quality, scope=scope,
                    tenor=tenor, level=level, n_intervals=len(g), n_dates=len(daily),
                    alpha_one_pnl_rmse_spot=b, dynamic_alpha_pnl_rmse_spot=d,
                    rmse_improvement=1-d/b if b > 0 else np.nan,
                    improvement_ci_low=low, improvement_ci_high=high,
                    improvement_positive_95pct=bool(np.isfinite(low) and low > 0),
                    direct_beta_pnl_rmse_spot=float(np.sqrt(pd.DataFrame({
                        "date": g.label_date, "se": direct**2}).groupby("date").se.mean().mean())),
                    clipped_fraction=float(g.alpha_clipped.mean()),
                    quality_pass_fraction=float(g.converter_quality_pass.mean())))
    return pd.DataFrame(rows)


def run_pnl_attribution(changes, factors, config, mc_library, *, window=60,
                        min_observations=20, progress=None):
    """Reprice historical calls and compare mapped dynamic-alpha attribution.

    Strict converter failures remain explicit coverage rows, not hidden alpha=1
    successes. Other numerical-quality failures are retained and separately shown.
    """
    root = Path(mc_library)
    config = MCLibrary.configured(root, config)
    library = MCLibrary(root, config, config.step4_tenors, config.step4_strike_levels)
    history = load_surface_history(config.data_path, config.market_conventions,
        beta_clamp=config.beta_clamp, duplicate_vol_date_policy=config.duplicate_vol_date_policy,
        source_timezone=config.source_timezone, market_timezone=config.market_timezone)
    elasticities = rolling_elasticities(changes, window=window, min_observations=min_observations)
    elasticities["tenor_key"] = elasticities.tenor.round(12)
    elasticities["level_key"] = elasticities.level.round(12)
    estimates = elasticities.set_index(["feature_date", "tenor_key", "level_key"])
    intervals = changes[["previous_date", "observation_date", "is_next_business_observation"]].drop_duplicates().sort_values("observation_date")
    output = []
    for number, pair in enumerate(intervals.itertuples(index=False)):
        previous, current = history[pair.previous_date], history[pair.observation_date]
        for tenor in config.step4_tenors:
            curves = None
            for level in config.step4_strike_levels:
                row = dict(feature_date=pair.previous_date, label_date=pair.observation_date,
                    tenor=tenor, level=level, attribution_valid=False, exclusion_reason="",
                    converter_quality_pass=False, alpha_clipped=False)
                if not pair.is_next_business_observation:
                    row["exclusion_reason"] = "nonconsecutive_business_interval"
                    output.append(row); continue
                key = (pair.previous_date, round(tenor, 12), round(level, 12))
                if key not in estimates.index or not np.isfinite(estimates.loc[key, "beta_regression"]):
                    row["exclusion_reason"] = "insufficient_past_regression"
                    output.append(row); continue
                est = estimates.loc[key]
                row.update(beta_regression=float(est.beta_regression),
                           regression_nobs=int(est.regression_nobs),
                           regression_end=est.window_end)
                try:
                    mark = contract_interval(previous, current, tenor, level)
                except ValueError as error:
                    row["exclusion_reason"] = "unsupported_contract_mark: " + str(error)
                    output.append(row); continue
                row.update(mark)
                # Missing supported shards or fingerprint mismatches must fail loudly.
                if curves is None:
                    curves = library.read(previous, tenor)
                curve = curves[np.isclose(curves.level, level)].sort_values("alpha")
                alpha, clipped, fallback = invert_beta(curve, est.beta_regression)
                row.update(alpha=alpha, alpha_clipped=clipped,
                    converter_quality_pass=bool(curve.quality_pass.all()),
                    converter_quality_failures=str(curve.quality_failures.iloc[0]))
                if fallback:
                    row["exclusion_reason"] = "invalid_beta_alpha_curve"
                    output.append(row); continue
                beta = float(np.interp(alpha, curve.alpha, curve.beta_converter))
                row.update(beta_effective=beta,
                           direct_beta_residual=pnl_components(mark, est.beta_regression)["dynamic_alpha_residual"],
                           **pnl_components(mark, beta))
                row["attribution_valid"] = True
                output.append(row)
        if progress is not None and number % 50 == 0:
            progress(f"Step 5 P&L attribution: {number+1}/{len(intervals)} dates")
    frame = pd.DataFrame(output)
    train_end = factors.loc[factors["sample"].eq("train"), "observation_date"].max()
    summary = summarize_attribution(frame, train_end)
    coverage = frame.assign(reason=frame.exclusion_reason.replace("", "valid")).groupby(
        ["tenor", "reason"], as_index=False).size().rename(columns={"size": "intervals"})
    inputs = {}
    for name, path in (("attribution_svi_parameters", config.data_path),
                       ("attribution_mc_library", root/"library.json"),
                       ("attribution_mc_index", root/"index.json")):
        inputs[name], inputs[name+"_sha256"] = str(path), file_sha256(path)
    primary = summary[(summary.period == "test") & (summary.scope == "overall")
                      & (summary.quality == "all_valid")]
    validation = dict(status="complete", regression_window=window,
        regression_min_observations=min_observations,
        elasticity="past-window OLS dIV_surface = intercept - beta*dlogS; no individual return division",
        regression_cutoff="feature close t; target t+1 excluded",
        regression_intercept_policy="estimated to deconfound slope, not attributed to spot P&L",
        source_beta_policy="separate attribution elasticity; Step2/4 daily-ratio labels unchanged",
        mark_policy="raw SVI European call; same strike and expiry across each interval; no extrapolation",
        alpha_policy="date-t repaired-LV MC centered beta inverse; clip at alpha bounds; invalid curve excluded",
        alpha_one_policy="centered beta=0; common BS delta/gamma/finite theta/term-roll terms",
        common_pnl="BS_delta*dS + 0.5*Gamma*dS^2 + finite_theta + Vega*term_roll_IV",
        response_pnl="-Vega*effective_beta*dlogS, booked once in delta-related attribution",
        residual_policy="actual option dV minus common P&L minus beta response; monetary PV residual",
        score_policy="RMSE of residual/starting spot; equal dates then equal available nodes; paired sample",
        confidence_policy="1000 paired nonoverlapping date-block bootstrap replicates; blocks <=10 within continuous segments, including randomly resampled short segments; exploratory 95% CI",
        train_end=str(train_end), total_intervals=len(frame),
        valid_intervals=int(frame.attribution_valid.sum()),
        valid_dates=int(frame.loc[frame.attribution_valid, "label_date"].nunique()),
        primary_test=primary.iloc[0].to_dict() if len(primary) else {},
        scope_limit="historical SVI marks, not exchange fills; attribution is not trading P&L or a hedge-performance claim")
    return AttributionResult(elasticities.drop(columns=["tenor_key", "level_key"]),
                             frame, summary, coverage, validation, inputs)

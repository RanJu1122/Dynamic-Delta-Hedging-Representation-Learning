"""Step 7: three shared SR policies, hold to expiry and optionally renew each slot."""

from collections import deque
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from svi_localvol.montecarlo import _fill_undefined
from .bump import SpotBumpPolicy
from .artifacts import file_sha256, write_manifest
from .config import STEP4_TENORS, RESEARCH_STRIKE_LEVELS
from .data_loader import date_at_tau, observation_exclusions
from .fixed_book import (contract_schedule, scheduled_expiries, missing_settlements, fixed_marks,
                         interpolate_beta, advance_account)
from .hedging import cell_curve, invert_beta
from .mc_library import MCLibrary
from .precompute import _csv, _writer_lock
from .sr_pricing import AlphaSurface, SharedSRPricer, SR_KINDS, shared_local_vol
from .step06 import LAST_FACTOR_COLUMNS
from .step07 import hedge_error, _paired_improvement_ci
from .step07_shared import (SharedStep7Config, _forecast_nodes, _convert,
                            _used_nodes, _canonical_config)


@dataclass(frozen=True)
class FixedStep7Config(SharedStep7Config):
    book_tenors: tuple = STEP4_TENORS
    book_levels: tuple = RESEARCH_STRIKE_LEVELS
    training_start: str | None = None
    mark_extrapolation: str = "flat_iv"
    renew_expired: bool = True
    flat_spot_alpha_one: bool = True
    raw_only: bool = False  # Compatibility alias for three raw policies plus controls.
    strategy_set: str = "dynamic"
    fixed_baseline: str = "train_best"
    bump_policy: SpotBumpPolicy = field(default_factory=SpotBumpPolicy)

    @property
    def include_controls(self):
        return self.raw_only or self.strategy_set in ("raw_controls", "full")

    @property
    def include_variants(self):
        return not self.raw_only and self.strategy_set == "full"

    @property
    def select_fixed_on_train(self):
        return self.include_controls and self.fixed_baseline == "train_best"

    def __post_init__(self):
        super().__post_init__()
        if self.strategy_set not in ("dynamic", "raw_controls", "full"):
            raise ValueError("strategy set must be dynamic, raw_controls or full")
        if self.fixed_baseline not in ("train_best", "alpha_one"):
            raise ValueError("fixed baseline must be train_best or alpha_one")
        if self.fixed_baseline == "alpha_one" and not self.include_controls:
            raise ValueError("alpha_one baseline requires raw_controls or full strategy set")
        for axis in (self.book_tenors, self.book_levels):
            if not len(axis) or not np.isfinite(axis).all() or min(axis) <= 0 or np.any(np.diff(axis) <= 0):
                raise ValueError("fixed book axes must be finite, positive and increasing")
        if self.mark_extrapolation not in ("flat_iv", "reject"):
            raise ValueError("mark extrapolation must be flat_iv or reject")
        if self.training_start is not None:
            pd.Timestamp(self.training_start).date()


def _warm_dates(inputs, first_test, count):
    daily = inputs.daily[np.isclose(inputs.daily.tenor, .25) & np.isclose(inputs.daily.level, 1.)]
    valid = set(daily.loc[np.isfinite(daily.beta_surface_daily), "observation_date"])
    return [d for d in inputs.history.dates if d < first_test and d in valid][-count:]


def _known_flat_spot(history, date):
    """Use only the current and preceding available close, never the holding-period end."""
    dates = history.dates
    position = bisect_left(dates, date)
    if position == 0:
        return None, False
    previous = dates[position-1]
    return previous, bool(history[date].ref_spot == history[previous].ref_spot)


def _plan(inputs, options, provider):
    dates, cutoff = inputs.history.dates, inputs.forecaster.train_end
    training = [d for d in dates if d <= cutoff]
    testing = [d for d in dates if d >= cutoff]
    if not testing or testing[0] != cutoff:
        raise ValueError("training cutoff must have a market snapshot for the test book inception")
    if options.max_train_dates:
        training = training[-options.max_train_dates-1:]
    if options.max_test_dates:
        testing = testing[:options.max_test_dates+1]
    if (options.select_fixed_on_train and len(training) < 3) or len(testing) < 2:
        raise ValueError("need at least two training intervals and one test interval")
    rejected = []
    if options.select_fixed_on_train:
        # Choose using TRAINING calendar coverage only, never test returns or future test coverage.
        candidates = training[:-2]
        if options.training_start:
            chosen = pd.Timestamp(options.training_start).date()
            if chosen not in candidates:
                raise ValueError("training-start must leave two intervals in the selected training period")
            candidates = [chosen]
        for start in candidates:
            selected = [d for d in training if d >= start]
            expiries = scheduled_expiries(inputs.history, selected, options.book_tenors, options.renew_expired)
            absent = missing_settlements(expiries, selected)
            if not absent:
                training = selected
                break
            rejected.append({"inception": start, "missing_settlements": absent})
        else:
            raise ValueError(f"no training cohort with observed settlement spots: {rejected}")
    cohorts, intervals, marks, rows = {}, {"train": [], "test": []}, {}, []
    samples = (("train", training), ("test", testing)) if options.select_fixed_on_train else (("test", testing),)
    for sample, selected in samples:
        cohort = contract_schedule(inputs.history, selected, options.book_tenors, options.book_levels,
                                   inputs.settings.weights, sample, options.renew_expired)
        cohorts[sample], intervals[sample] = cohort, []
        for date, end in zip(selected, selected[1:]):
            if date >= cohort.expiry.max():
                break
            if sample == "test" and date not in inputs.forecasts.index:
                raise ValueError(f"missing close-t forecast: {date}; do not skip a held-book interval")
            frame = fixed_marks(inputs.history[date], inputs.history[end], cohort,
                                extrapolation=options.mark_extrapolation)
            intervals[sample].append((date, end))
            marks[(sample, date)] = frame
            reference_date, flat_spot = _known_flat_spot(inputs.history, date)
            rows.append(dict(sample=sample, feature_date=date, label_date=end,
                flat_spot_reference_date=reference_date, known_flat_spot=flat_spot,
                n_options=len(frame), n_expired=int(frame.expired.sum()),
                n_opened=int(frame.opened_today.sum()), n_renewed=int(frame.renewed_today.sum()),
                n_alive_end=int((~frame.expired).sum()), business_days=int(frame.business_days.iloc[0]),
                is_gap=bool(frame.is_gap.iloc[0]), mark_extrapolated_cells=int(frame.mark_extrapolated.sum())))
            if sample == "test" and isinstance(provider, MCLibrary):
                for tenor in provider.tenors:
                    key, _ = provider._key(inputs.history[date], tenor)
                    if key not in provider.index:
                        raise FileNotFoundError(f"missing converter shard: {key}")
    plan = pd.DataFrame(rows)
    if options.select_fixed_on_train and len(plan[(plan['sample'] == 'train') & ~plan.is_gap]) < 2:
        raise ValueError("need at least two ordinary training intervals for fixed-alpha selection")
    if isinstance(provider, MCLibrary) and options.include_variants:
        for date in _warm_dates(inputs, intervals["test"][0][0], options.rolling_alpha_observations):
            for tenor in provider.tenors:
                key, _ = provider._key(inputs.history[date], tenor)
                if key not in provider.index:
                    raise FileNotFoundError(f"missing rolling-Alpha warm-up converter shard: {key}")
    return cohorts, intervals, marks, plan, rejected


def _summary(book, nodes, *, baseline_strategy="best_fixed_train"):
    rows = []
    std = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
    rmse = lambda x: float(np.sqrt(np.mean(np.asarray(x)**2))) if len(x) else np.nan
    improvement = lambda x, y: 1-x/y if y > 0 else np.nan
    for name, group in book.groupby("strategy", sort=False):
        group = group.sort_values("feature_date")
        daily, gap = group[~group.is_gap], group[group.is_gap]
        valid = daily[daily.attribution_valid]
        ref = book[(book.strategy == "best_fixed_train") & ~book.is_gap].set_index("label_date")
        one = book[(book.strategy == "fixed_1") & ~book.is_gap].set_index("label_date")
        bs = book[(book.strategy == "bs_delta") & ~book.is_gap].set_index("label_date")
        sig = nodes[nodes.strategy == name]
        alpha_changes = sig.sort_values("feature_date").groupby(["tenor", "level"]).alpha.diff().abs()
        raw = daily.raw_hedge_error.to_numpy()
        def reference_errors(frame, column="raw_hedge_error"):
            return frame.reindex(daily.label_date)[column].to_numpy()
        ref_raw = reference_errors(ref)
        primary_ref = one if baseline_strategy == "fixed_1" else ref
        ci = _paired_improvement_ci(raw, reference_errors(primary_ref),
            segments=group.is_gap.cumsum()[~group.is_gap].to_numpy()) if len(raw) > 1 and not primary_ref.empty else (np.nan, np.nan)
        row = dict(strategy=name, n_test_intervals=len(group), n_daily_intervals=len(daily),
            comparison_baseline=baseline_strategy,
            n_gap_intervals=len(gap), raw_mean_error=float(daily.raw_hedge_error.mean()),
            raw_std_error=std(raw), raw_rmse=rmse(raw), net_std_error=std(daily.net_error),
            net_rmse=rmse(daily.net_error), all_interval_raw_std=std(group.raw_hedge_error),
            raw_pnl_sum=float(group.raw_hedge_error.sum()), net_pnl_sum=float(group.net_error.sum()),
            daily_raw_pnl_sum=float(daily.raw_hedge_error.sum()), gap_raw_pnl_sum=float(gap.raw_hedge_error.sum()),
            gap_squared_error=float((gap.raw_hedge_error**2).sum()),
            final_wealth=float(group.wealth.iloc[-1]), final_cash=float(group.cash_close.iloc[-1]),
            final_stock_position=float(group.stock_position.iloc[-1]),
            final_option_pv=float(group.next_option_pv.iloc[-1]), n_alive_end=int(group.n_alive_end.iloc[-1]),
            expiry_cashflow=float(group.expiry_cashflow.sum()), total_cost=float(group.cost.sum()),
            total_option_premium_paid=float(-group.option_trade_cashflow.sum()),
            total_renewal_premium=float(group.renewal_premium.sum()),
            n_contracts_opened=int(group.n_opened.sum()), n_contracts_renewed=int(group.n_renewed.sum()),
            total_hedge_turnover=float(group.hedge_turnover.sum()),
            total_hedge_notional_turnover=float(group.hedge_notional_turnover.sum()),
            raw_std_improvement_vs_best_fixed=improvement(std(raw), std(ref_raw)),
            raw_std_improvement_vs_alpha_one=improvement(std(raw), std(reference_errors(one))),
            raw_rmse_improvement_vs_alpha_one=improvement(rmse(raw), rmse(reference_errors(one))),
            net_std_improvement_vs_alpha_one=improvement(std(daily.net_error), std(reference_errors(one, "net_error"))),
            net_rmse_improvement_vs_alpha_one=improvement(rmse(daily.net_error), rmse(reference_errors(one, "net_error"))),
            raw_std_improvement_vs_bs=improvement(std(raw), std(reference_errors(bs))),
            improvement_ci_low=ci[0], improvement_ci_high=ci[1],
            mean_alpha_change=float(alpha_changes.mean()),
            inverse_fallback_fraction=float(sig.inverse_fallback.mean()),
            alpha_clipped_fraction=float(sig.alpha_clipped.mean()),
            converter_quality_pass_fraction=float(sig.converter_quality_pass.mean()),
            delta_fallback_fraction=float(group.delta_fallback_fraction.mean()),
            flat_spot_alpha_one_intervals=int(group.flat_spot_alpha_one_fallback.sum()),
            flat_spot_alpha_one_fraction=float(group.flat_spot_alpha_one_fallback.mean()),
            attribution_valid_intervals=len(valid),
            model_attribution_rmse=rmse(valid.attribution_residual),
            forecast_attribution_rmse=rmse(valid.forecast_attribution_residual),
            bs_attribution_rmse=rmse(valid.bs_attribution_residual))
        kind = next((k for k in SR_KINDS if name in (k, k+"_ema")), None)
        if kind and (book.strategy == kind+"_last_observed_factor").any():
            naive = book[(book.strategy == kind+"_last_observed_factor") & ~book.is_gap]
            row["raw_std_improvement_vs_same_policy_persistence"] = improvement(std(raw), std(naive.raw_hedge_error))
        rows.append(row)
    return pd.DataFrame(rows)


def _option_result(frame, priced, policy, measured_one, predicted, inputs, date):
    result = frame.copy()
    if priced is None:
        result["delta"], result["delta_stderr"] = frame.bs_delta, 0.
        result["beta_model"], result["effective_beta"] = 0., 0.
    else:
        if len(priced) != len(frame) or not np.allclose(priced[["tenor", "level"]], frame[["tenor", "level"]]):
            raise ValueError("MC rows do not match fixed contracts")
        for column in priced.columns.difference(["tenor", "level"]):
            result[column] = priced[column].to_numpy()
        result["effective_beta"] = result.beta_model-measured_one
    result["delta_fallback"] = ~np.isfinite(result.delta)
    result.loc[result.delta_fallback, "delta"] = result.loc[result.delta_fallback, "bs_delta"]
    result.loc[result.delta_fallback, "effective_beta"] = 0.
    result["alpha"] = [policy.at_contract(r.expiry, r.strike) for r in frame.itertuples()]
    if predicted is None:
        result["predicted_beta"], result["prediction_extrapolated"] = result.effective_beta, False
    else:
        result["predicted_beta"], result["prediction_extrapolated"] = interpolate_beta(
            inputs.history[date], inputs.config.step4_tenors, inputs.config.step4_strike_levels, predicted, frame)
    result["beta_mapping_gap"] = result.effective_beta-result.predicted_beta
    result["zero_spot_holding_interval"] = result.dS.eq(0)
    result["zero_spot_iv_change"] = (result.next_iv-result.iv).where(result.zero_spot_holding_interval)
    result["zero_spot_pnl_explanation"] = np.where(result.zero_spot_holding_interval,
        "dS=0: stock price PnL and Beta*dlogS are zero; option PnL retains IV/time effects", "")
    result["delta_pnl"] = result.delta*result.dS
    result["hedge_pnl"] = -result.delta_pnl
    result["bs_delta_pnl"] = result.bs_delta*result.dS
    result["gamma_pnl"] = .5*result.gamma*result.dS**2
    result["theta_pnl"] = result.time_pnl
    result["term_roll_pnl"] = result.vega*result.term_roll_iv
    common = result.bs_delta_pnl+result.gamma_pnl+result.theta_pnl+result.term_roll_pnl
    result["bs_attribution_residual"] = result.dV-common
    result["forecast_vega_pnl"] = -result.vega*result.predicted_beta*result.dlogS
    result["model_vega_pnl"] = -result.vega*result.effective_beta*result.dlogS
    result["forecast_attribution_residual"] = result.dV-common-result.forecast_vega_pnl
    result["attribution_residual"] = result.dV-common-result.model_vega_pnl
    result["raw_model_attribution_residual"] = result.dV-common+result.vega*result.beta_model*result.dlogS
    result["anchor_correction_pnl"] = result.raw_model_attribution_residual-result.attribution_residual
    result["mc_delta_attribution_residual"] = result.dV-result.delta_pnl-result.gamma_pnl-result.theta_pnl-result.term_roll_pnl
    result["raw_hedge_error"] = result.dV-result.delta_pnl
    result["carry_hedge_error"] = hedge_error(result.dV, result.delta, result.dS,
        result.pv, result.spot, result.dt_r, result.rate, result.income_yield)
    result["attribution_valid"] &= np.isfinite(result[["attribution_residual", "forecast_attribution_residual",
                                                     "raw_model_attribution_residual"]]).all(axis=1)
    return result


def _book_result(result, previous, *, continues_book=False):
    first = result.iloc[0]
    weighted = lambda col: float((result[col]*result.quantity).sum(skipna=False))
    account = advance_account(previous, pv=weighted("pv"), next_pv=weighted("next_pv"),
        settlement=weighted("expiry_cashflow"), delta=weighted("delta"),
        spot=first.spot, next_spot=first.next_spot, dt=first.dt_r, rate=first.rate,
        income_yield=first.income_yield, cost_bps=first.cost_bps,
        renewal_pv=float((result.pv*result.quantity).where(result.renewed_today, 0.).sum()),
        all_expired=bool(result.expired.all()) and not continues_book)
    columns = ("dV", "delta_pnl", "hedge_pnl", "bs_delta_pnl", "gamma_pnl", "theta_pnl",
        "term_roll_pnl", "forecast_vega_pnl", "model_vega_pnl", "raw_hedge_error", "carry_hedge_error",
        "attribution_residual", "forecast_attribution_residual", "bs_attribution_residual",
        "raw_model_attribution_residual", "anchor_correction_pnl", "mc_delta_attribution_residual")
    return dict(feature_date=first.feature_date, label_date=first.label_date, strategy=first.strategy,
        book_delta=weighted("delta"), option_pv=weighted("pv"), n_options=len(result),
        n_alive_end=int((~result.expired).sum()), n_expired=int(result.expired.sum()),
        n_opened=int(result.opened_today.sum()), n_renewed=int(result.renewed_today.sum()),
        is_gap=bool(first.is_gap), business_days=int(first.business_days),
        zero_spot_holding_interval=bool(first.zero_spot_holding_interval),
        zero_spot_pnl_explanation=first.zero_spot_pnl_explanation,
        mark_extrapolated_cells=int(result.mark_extrapolated.sum()),
        prediction_extrapolated_cells=int(result.prediction_extrapolated.sum()),
        attribution_valid=bool(result.attribution_valid.all()),
        flat_spot_alpha_one_fallback=bool(first.flat_spot_alpha_one_fallback),
        alpha_override_reason=first.alpha_override_reason,
        flat_spot_reference_date=first.flat_spot_reference_date,
        delta_fallback_fraction=float(result.delta_fallback.mean()),
        **{c: weighted(c) for c in columns}, **account)


def _localvol_diagnostics(raw_surface, frame, policy, planned, strategy, date, end,
                          numerical, forced, bump_policy=SpotBumpPolicy(short_business_days=0)):
    """Pointwise audit on the actual fixed contracts used by the raw SR policies."""
    surface = raw_surface.repaired()
    result = frame[["contract_id", "slot_id", "generation", "tenor", "level",
                    "expiry", "strike"]].reset_index(drop=True).copy()
    result.insert(0, "label_date", end)
    result.insert(0, "feature_date", date)
    result.insert(2, "strategy", strategy)
    result.insert(3, "profile_kind", policy.kind)
    result["spot_adj_base"] = 0.0
    result["spot_bump_fraction"] = [bump_policy.fraction(surface, expiry,
        numerical.step3_spot_bump_fraction) for expiry in result.expiry]
    result["spot_adj_up"] = np.log1p(result.spot_bump_fraction)
    result["spot_adj_down"] = np.log1p(-result.spot_bump_fraction)
    result["flat_spot_alpha_one_fallback"] = bool(forced)
    result["alpha"] = np.nan
    result["alpha_before_override"] = np.nan
    result["point_local_vol_base"] = np.nan
    result["point_local_vol_up"] = np.nan
    result["point_local_vol_down"] = np.nan
    result["base_reference_local_vol"] = np.nan
    for bump in ("base", "up", "down"):
        result[f"grid_local_vol_{bump}"] = np.nan
        result[f"grid_row_undefined_{bump}"] = 0
        result[f"grid_row_clipped_{bump}"] = 0
    ratios = np.linspace(0.0, numerical.step3_ratio_max, numerical.step3_n_ratio)
    grid_strikes = np.maximum(ratios, numerical.step3_ratio_min)*surface.ref_spot
    for expiry, index in result.groupby("expiry", sort=False).groups.items():
        loc = np.asarray(list(index), dtype=int)
        bump_fraction = float(result.loc[loc[0], "spot_bump_fraction"])
        strikes = result.loc[loc, "strike"].to_numpy(float)
        contract_ratios = strikes/surface.ref_spot
        y = np.log(strikes/raw_surface.forward(expiry))
        result.loc[loc, "alpha"] = policy(expiry, y)
        result.loc[loc, "alpha_before_override"] = planned(expiry, y)
        for column, shift in (("point_local_vol_base", 0.0),
                              ("point_local_vol_up", np.log1p(bump_fraction)),
                              ("point_local_vol_down", np.log1p(-bump_fraction))):
            result.loc[loc, column] = shared_local_vol(surface, expiry, strikes, shift, policy)
            raw_grid_row = np.asarray(
                shared_local_vol(surface, expiry, grid_strikes, shift, policy), dtype=float)
            undefined = int(np.count_nonzero(~np.isfinite(raw_grid_row)))
            filled = _fill_undefined(raw_grid_row[None, :], numerical.step3_vol_floor)[0]
            sigma = np.clip(filled, numerical.step3_vol_floor, numerical.step3_vol_cap)
            clipped = int(np.count_nonzero(np.isfinite(raw_grid_row) & (filled != sigma)))
            bump = column.removeprefix("point_local_vol_")
            result.loc[loc, f"grid_local_vol_{bump}"] = np.interp(
                contract_ratios, ratios, sigma, left=sigma[0], right=sigma[-1])
            result.loc[loc, f"grid_row_undefined_{bump}"] = undefined
            result.loc[loc, f"grid_row_clipped_{bump}"] = clipped
        result.loc[loc, "base_reference_local_vol"] = surface.local_vol(
            expiry, strikes, spot_adj=0.0, alpha=1.0)
    result["base_reference_abs_diff"] = np.abs(
        result.point_local_vol_base-result.base_reference_local_vol)
    result["up_minus_base"] = result.point_local_vol_up-result.point_local_vol_base
    result["down_minus_base"] = result.point_local_vol_down-result.point_local_vol_base
    result["up_minus_down"] = result.point_local_vol_up-result.point_local_vol_down
    result["all_local_vol_finite"] = np.isfinite(result[["point_local_vol_base",
        "point_local_vol_up", "point_local_vol_down"]]).all(axis=1)
    result["all_grid_local_vol_finite"] = np.isfinite(result[["grid_local_vol_base",
        "grid_local_vol_up", "grid_local_vol_down"]]).all(axis=1)
    result["grid_up_minus_base"] = result.grid_local_vol_up-result.grid_local_vol_base
    result["grid_down_minus_base"] = result.grid_local_vol_down-result.grid_local_vol_base
    result["grid_up_minus_down"] = result.grid_local_vol_up-result.grid_local_vol_down
    return result


def _prepare_localvol_diagnostics(inputs, options, intervals, marks, provider, numerical):
    """Build the raw-policy Local Vol audit without starting path simulation."""
    tenors, levels = inputs.config.step4_tenors, inputs.config.step4_strike_levels
    shape = (len(tenors), len(levels))
    rows = []
    for date, end in intervals["test"]:
        surface = inputs.history[date]
        frame = marks[("test", date)]
        table = provider.get(surface).filter(
            items=["tenor", "level", "alpha", "beta_converter", "quality_pass"])
        forecast = _forecast_nodes(inputs, date)
        raw_nodes = _convert(table, forecast, tenors, levels)
        raw = raw_nodes.alpha.to_numpy().reshape(shape)
        _, flat_spot = _known_flat_spot(inputs.history, date)
        for kind in SR_KINDS:
            planned = AlphaSurface(surface, tenors, levels, raw, kind)
            forced = options.flat_spot_alpha_one and flat_spot
            policy = (AlphaSurface(surface, tenors, levels, np.ones(shape), kind)
                      if forced else planned)
            rows.append(_localvol_diagnostics(
                surface, frame, policy, planned, kind, date, end, numerical, forced, options.bump_policy))
    return pd.concat(rows, ignore_index=True)


def _backtest(inputs, options, intervals, marks, provider, pricer, progress, checkpoint,
              localvol_diagnostics):
    c, tenors, levels = inputs.config, inputs.config.step4_tenors, inputs.config.step4_strike_levels
    shape = (len(tenors), len(levels))
    def profile(date, values, kind="constant_sr"):
        return AlphaSurface(inputs.history[date], tenors, levels,
                            np.full(shape, values) if np.isscalar(values) else values, kind)
    train_rows = []
    for n, (date, end) in enumerate(intervals["train"], 1):
        progress(f"Fixed Step 7 training {n}/{len(intervals['train'])}: {date}")
        frame = marks[("train", date)]
        for alpha in c.step3_alphas:
            priced = pricer.get(inputs.history[date], frame, profile(date, alpha))
            if len(priced) != len(frame) or not np.allclose(priced[["tenor", "level"]], frame[["tenor", "level"]]):
                raise ValueError("training MC rows do not match fixed contracts")
            delta = np.where(np.isfinite(priced.delta), priced.delta, frame.bs_delta)
            train_rows.append(dict(feature_date=date, label_date=end, alpha=alpha,
                raw_error=float(np.sum(frame.quantity*(frame.dV-delta*frame.dS))),
                is_gap=bool(frame.is_gap.iloc[0]), n_options=len(frame),
                n_renewed=int(frame.renewed_today.sum()),
                delta_fallback_cells=int((~np.isfinite(priced.delta)).sum())))
        checkpoint(completed_training_dates=n, active_date=str(date))
    training = pd.DataFrame(train_rows, columns=["feature_date", "label_date", "alpha", "raw_error",
        "is_gap", "n_options", "n_renewed", "delta_fallback_cells"])
    best = 1.  # Also the predetermined EMA initial value when selection is skipped.
    if options.select_fixed_on_train:
        scores = training[~training.is_gap].groupby("alpha").raw_error.std(ddof=1)
        best = float(scores.idxmin())
        checkpoint(training_best_alpha=best, training_fixed_std=scores.to_dict())
    realised = inputs.daily[np.isclose(inputs.daily.tenor, .25) & np.isclose(inputs.daily.level, 1.)]
    realised = realised.set_index("observation_date").beta_surface_daily
    rolling = deque(maxlen=options.rolling_alpha_observations)
    def update_rolling(date, table):
        value, _, fallback = invert_beta(cell_curve(table, .25, 1.), realised.get(date, np.nan))
        if not fallback:
            rolling.append(value)
    if options.include_variants:
        first_test = intervals["test"][0][0]
        for date in _warm_dates(inputs, first_test, options.rolling_alpha_observations):
            update_rolling(date, provider.get(inputs.history[date]))
    smooth, accounts = np.full(shape, best), {}
    option_rows, book_rows, node_rows, audits, quote_rows = [], [], [], [], []
    for n, (date, end) in enumerate(intervals["test"], 1):
        progress(f"Fixed Step 7 test {n}/{len(intervals['test'])}: {date} -> {end}")
        frame, surface = marks[("test", date)], inputs.history[date]
        table = provider.get(surface).filter(items=["tenor", "level", "alpha", "beta_converter", "quality_pass"])
        forecast = _forecast_nodes(inputs, date)
        raw_nodes = _convert(table, forecast, tenors, levels)
        raw = raw_nodes.alpha.to_numpy().reshape(shape)
        strategies = ({f"fixed_{a:g}": (profile(date, a), None, None) for a in c.step3_alphas}
                      if options.include_controls else {})
        if options.include_variants:
            naive = _forecast_nodes(inputs, date, LAST_FACTOR_COLUMNS)
            naive_nodes = _convert(table, naive, tenors, levels)
            decay = 2**(-1/options.half_life) if options.half_life else 0.
            smooth = decay*smooth+(1-decay)*raw  # Observed-close updates; no reset or synthetic gap updates.
            update_rolling(date, table)
            strategies["rolling_alpha_mean"] = (
                profile(date, float(np.mean(rolling)) if rolling else 1.), None, None)
        for kind in SR_KINDS:
            strategies[kind] = (profile(date, raw, kind), raw_nodes, forecast)
            if options.include_variants:
                strategies[kind+"_last_observed_factor"] = (
                    profile(date, naive_nodes.alpha.to_numpy().reshape(shape), kind), naive_nodes, naive)
                if options.half_life:
                    strategies[kind+"_ema"] = (profile(date, smooth, kind), raw_nodes, forecast)
        planned_policies = {name: policy for name, (policy, _, _) in strategies.items()}
        reference_date, flat_spot = _known_flat_spot(inputs.history, date)
        forced = set()
        if options.flat_spot_alpha_one and flat_spot:
            for name, (policy, source, predicted) in list(strategies.items()):
                if source is not None or name == "rolling_alpha_mean":
                    strategies[name] = (profile(date, 1., policy.kind), source, predicted)
                    forced.add(name)
        # One reference MC is necessary for centered-Beta attribution/flat-Spot execution.
        # It is not an extra backtest strategy in dynamic-only mode.
        priced = {"fixed_1": pricer.get(surface, frame, profile(date, 1.))}
        for name, (policy, _, _) in strategies.items():
            # Fixed controls are inserted first; an override uses exactly the same measured Delta.
            if name != "fixed_1":
                priced[name] = priced["fixed_1"] if name in forced else pricer.get(surface, frame, policy)
        measured_one = priced["fixed_1"].beta_model.to_numpy()
        if options.include_controls:
            if options.select_fixed_on_train:
                strategies["best_fixed_train"], priced["best_fixed_train"] = strategies[f"fixed_{best:g}"], priced[f"fixed_{best:g}"]
            strategies["bs_delta"], priced["bs_delta"] = (profile(date, 1.), None, None), None
        else:
            reference = priced["fixed_1"]
            audits.append(dict(feature_date=date, strategy="fixed_1", role="attribution_reference_only",
                **reference.filter(regex="fraction$|^grid_|^n_paths$|^max_n_steps$").max().to_dict(),
                max_delta_stderr=float(reference.delta_stderr.max()),
                iv_inversion_clipped_cells=int(reference.price_clipped_for_inversion.sum())))
        for name, (policy, source, predicted) in strategies.items():
            planned = planned_policies.get(name, policy)
            used = _used_nodes(source, policy.kind) if source is not None else pd.DataFrame([
                dict(tenor=.25, level=1., alpha=policy.values[policy.anchor, policy.atm],
                     predicted_beta=np.nan, alpha_clipped=False, inverse_fallback=False, converter_quality_pass=np.nan)])
            for node in used.to_dict("records"):
                expiry = policy.expiries[AlphaSurface._node(tenors, node["tenor"])]
                node_rows.append({**node, "raw_alpha": node["alpha"],
                    "alpha_before_override": planned.at_contract(expiry, node["level"]*surface.ref_spot),
                    "flat_spot_alpha_one_fallback": name in forced,
                    "alpha_override_reason": "unchanged_observed_spot" if name in forced else "",
                    "alpha": policy.at_contract(expiry, node["level"]*surface.ref_spot),
                    "strategy": name, "feature_date": date, "label_date": end})
            if name in SR_KINDS or name.endswith("_ema"):
                for sl in surface.slices:
                    quote_rows.append(dict(feature_date=date, strategy=name, vol_date=sl.vol_date,
                        tau=sl.tau, atm_alpha=policy.at_contract(sl.vol_date, surface.ref_spot),
                        outside_alpha_tenors=sl.tau < policy.taus[0] or sl.tau > policy.taus[-1]))
            result = _option_result(frame, priced[name], policy, measured_one, predicted, inputs, date)
            result["alpha_before_override"] = [planned.at_contract(r.expiry, r.strike) for r in frame.itertuples()]
            result["flat_spot_alpha_one_fallback"] = name in forced
            result["flat_spot_reference_date"] = reference_date
            result["alpha_override_reason"] = "unchanged_observed_spot" if name in forced else ""
            result["strategy"], result["cost_bps"] = name, inputs.settings.hedge_cost_bps
            result["profile_has_inverse_fallback"] = bool(used.inverse_fallback.any())
            account = _book_result(result, accounts.get(name),
                continues_book=options.renew_expired and n < len(intervals["test"]))
            accounts[name] = account
            book_rows.append(account)
            option_rows.append(result)
            if priced[name] is not None:
                p = priced[name]
                audits.append(dict(feature_date=date, strategy=name,
                    role="backtest_strategy",
                    **p.filter(regex="fraction$|^grid_|^n_paths$|^max_n_steps$").max().to_dict(),
                    max_delta_stderr=float(p.delta_stderr.max()),
                    iv_inversion_clipped_cells=int(p.price_clipped_for_inversion.sum())))
        checkpoint(completed_test_dates=n, active_date=str(date))
    frames = dict(option_pnl=pd.concat(option_rows, ignore_index=True), book_pnl=pd.DataFrame(book_rows),
        alpha_nodes=pd.DataFrame(node_rows), mc_audit=pd.DataFrame(audits), quote_sr=pd.DataFrame(quote_rows),
        localvol_diagnostics=localvol_diagnostics, training_fixed=training)
    frames["summary"] = _summary(frames["book_pnl"], frames["alpha_nodes"],
        baseline_strategy="fixed_1" if options.fixed_baseline == "alpha_one" else "best_fixed_train")
    frames["gap_pnl"] = frames["book_pnl"][frames["book_pnl"].is_gap].copy()
    return frames


def run_fixed_step7(inputs, *, outdir, options=FixedStep7Config(), mc_config=None,
                    prepare_only=False, map_store=None, pricer=None, cache_dir=None, progress=print):
    """Independent output and ledger; the old rolling-book runners remain unchanged."""
    c, settings, outdir = inputs.config, inputs.settings, Path(outdir)
    if settings.book != "full" or settings.converter != "daily":
        raise ValueError("fixed Step 7 requires full book and daily converters")
    if not {0., 1., 2.}.issubset(c.step3_alphas):
        raise ValueError("fixed Step 7 requires fixed alpha 0/1/2 baselines")
    if settings.mc_library is None and map_store is None:
        raise ValueError("fixed Step 7 requires a converter library")
    AlphaSurface._node(c.step4_tenors, .25)
    AlphaSurface._node(c.step4_strike_levels, 1.)
    cache = Path(cache_dir) if cache_dir else outdir / "fixed_mc_cache"
    if cache.resolve() == outdir.resolve():
        raise ValueError("MC cache must be separate from result files")
    if settings.mc_library is not None:
        library = Path(settings.mc_library).resolve()
        if any(p.resolve() == library or library in p.resolve().parents for p in (cache, outdir)):
            raise ValueError("new outputs must be outside the read-only converter library")
    provider = map_store or MCLibrary(settings.mc_library, c, c.step4_tenors, c.step4_strike_levels)
    numerical = mc_config or c
    pricer = pricer or SharedSRPricer(numerical, cache, bump_policy=options.bump_policy)
    if isinstance(pricer, SharedSRPricer) and (pricer.config != numerical
                                               or pricer.bump_policy != options.bump_policy):
        raise ValueError("supplied pricer must match the run's MC settings and bump policy")
    run_config = _canonical_config(dict(research=asdict(c), backtest=asdict(settings),
                                       fixed_book=asdict(options), new_mc=asdict(numerical)))
    sources = dict(inputs.sources)
    for name in ("step07_fixed_book.py", "fixed_book.py", "sr_pricing.py", "step07_shared.py", "bump.py"):
        path = Path(__file__).with_name(name)
        sources[name] = str(path.resolve())
        sources[name+"_sha256"] = file_sha256(path)
    stage = "dynamic_alpha_step07_fixed_book"
    with _writer_lock(outdir):
        manifest = outdir / "manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if (old["stage"] != stage or _canonical_config(old["config"]) != run_config
                    or old["inputs"] != json.loads(json.dumps(sources, default=str))):
                raise ValueError("output contains a different backtest/configuration/source; choose a new directory")
        progress("Fixed Step 7: validating fixed contracts, settlement coverage and converters")
        cohorts, intervals, marks, plan, rejected = _plan(inputs, options, provider)
        _csv(plan, outdir / "plan.csv")
        for sample, cohort in cohorts.items():
            _csv(cohort, outdir / f"{sample}_contracts.csv")
        inputs.forecasts.to_csv(outdir / "factor_forecasts.csv")
        joblib.dump(inputs.forecaster, outdir / "forecaster.joblib")
        localvol_diagnostics = _prepare_localvol_diagnostics(
            inputs, options, intervals, marks, provider, numerical)
        _csv(localvol_diagnostics, outdir / "localvol_diagnostics.csv")
        validation = dict(status="prepared_only", mc_run=False, train_end=inputs.forecaster.train_end,
            training_intervals=len(intervals["train"]), test_intervals=len(intervals["test"]),
            rejected_training_inceptions=rejected, excluded_observation_dates=observation_exclusions(),
            n_initial_options=int(cohorts["test"].generation.eq(0).sum()),
            n_test_contract_generations=len(cohorts["test"]),
            signal_nodes=len(c.step4_tenors)*len(c.step4_strike_levels),
            signal_tenors=c.step4_tenors, signal_levels=c.step4_strike_levels,
            book_tenors=options.book_tenors, book_levels=options.book_levels,
            signal_book_axes_match=(tuple(c.step4_tenors) == tuple(options.book_tenors)
                                    and tuple(c.step4_strike_levels) == tuple(options.book_levels)),
            training_inception=cohorts["train"].inception.iloc[0] if "train" in cohorts else None,
            test_inception=cohorts["test"].inception.iloc[0],
            diagnostic_date_limits=bool(options.max_train_dates or options.max_test_dates),
            gap_policy="same contracts and last hedge held across missing dates; no synthetic rehedges",
            end_policy="mark open positions, no forced liquidation; natural expiry cash settlement",
            renewal_policy=("on expiry close, original tenor and level times current Spot; preserve slot quantity"
                            if options.renew_expired else "no renewal"),
            terminal_renewal_policy="terminal close is mark/settlement only; no new holding interval opened",
            zero_spot_policy="retain market observations and all hedge PnL; realised Beta ratio stays undefined",
            flat_spot_alpha_policy=("adaptive strategies use fixed-1 Delta after an unchanged observed close; "
                "fixed/BS controls unchanged; latent EMA continues, execution override is not smoothed"
                if options.flat_spot_alpha_one else "no Alpha override on unchanged observed Spot"),
            mark_policy=options.mark_extrapolation, beta_workflow="daily_only_v1",
            delta_source="new MC at actual fixed K and calendar expiry; converter supplies Alpha only",
            alpha_policy="constant 3M ATM / term ATM / term and log(K/F); flat node boundaries",
            strategy_mode=("raw, persistence and optional EMA SR policies plus controls" if options.include_variants
                else "three raw SR policies plus fixed/BS controls" if options.include_controls
                else "only constant_sr, term_sr, term_spot_sr"),
            internal_reference="fixed_1 MC for centered Beta and flat-Spot execution; not a strategy unless controls enabled",
            bump_policy=asdict(options.bump_policy),
            localvol_diagnostics=("raw and LocalVolGrid-sanitized base/up/down values at every held contract; "
                                  "base is Alpha-invariant, bumps use the executed dynamic Alpha"),
            smoothing=("EMA per observed close on stable signal nodes; no reset at gaps" if options.include_variants and options.half_life
                       else "disabled by selected strategy set"),
            comparison_baseline=("fixed_1" if options.fixed_baseline == "alpha_one"
                                 else "best_fixed_train" if options.include_controls else None),
            smoothing_initialization=("training-selected fixed Alpha" if options.select_fixed_on_train
                                      else "predetermined Alpha=1"),
            training_selection=("same renewal policy as test; earliest settlement-covered start; daily raw std"
                                if options.select_fixed_on_train else "skipped: predetermined Alpha=1 baseline; all strategies evaluated on test only"
                                if options.fixed_baseline == "alpha_one" else "skipped: no fixed-control backtests requested"),
            headline_metric="std(book dV - net book delta*dS) on one-business-day intervals",
            all_interval_pnl="includes gaps, expiry cash and terminal marks; no end trades",
            attribution_policy="BS delta + gamma + finite theta + term roll + Vega beta; expiry IV undefined",
            alpha_one_policy="center measured beta for attribution only; never change delta or PnL",
            quality_policy="audit retained; inverse fallback Alpha=1, nonfinite delta fallback BS",
            planned_training_profiles=len(intervals["train"])*len(c.step3_alphas),
            planned_test_profiles=len(intervals["test"])*(len(c.step3_alphas)+7+(3 if options.half_life else 0)
                if options.include_variants else len(c.step3_alphas)+len(SR_KINDS)
                if options.include_controls else len(SR_KINDS)+1))
        def checkpoint(**updates):
            validation.update(updates)
            write_manifest(manifest, stage=stage, config=run_config, inputs=sources, validation=validation)
        checkpoint()
        if prepare_only:
            return dict(plan=plan, localvol_diagnostics=localvol_diagnostics,
                        validation=validation, **{f"{k}_contracts": v for k, v in cohorts.items()})
        checkpoint(status="running", mc_run=True)
        try:
            frames = _backtest(inputs, options, intervals, marks, provider, pricer, progress, checkpoint,
                               localvol_diagnostics)
            for name, frame in frames.items():
                _csv(frame, outdir / f"{name}.csv")
        except BaseException as exc:
            checkpoint(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
            raise
        checkpoint(status="complete", computed_profiles=pricer.computed, reused_profiles=pricer.reused)
        return frames

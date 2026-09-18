"""Step 7: all-date ex-ante hedging, alpha smoothing, attribution and net book P&L.

Positions are a prescribed rolling book of European calls. Every overnight
interval holds fixed K, expiry and quantity. Roll trades are financed through
the cash account. Historical marks are SVI marks, not exchange executions.
"""

from collections import deque
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from svi_localvol.conventions import nb_biz_days
from .artifacts import file_sha256, read_manifest, write_manifest
from .config import DynamicAlphaConfig
from .factors import schema_for, copy_metadata
from .data_loader import load_surface_history, observation_exclusions
from .hedging import (MCMapStore, cell_curve, contract_interval, invert_beta,
                      delta_at_alpha)
from .hedge_comparison import shadow_delta, mc_consistency, pooled_converter
from .step06 import (FACTOR_COLUMNS, LAST_FACTOR_COLUMNS, fit_factor_forecaster, load_step6_inputs,
                     _ordered_loadings, _canonical_axes, catboost_parameters_from_step6,
                     hgb_parameters_from_step6)


@dataclass(frozen=True)
class Step7Config:
    book: str = "atm"
    factor_count: int = 1
    converter: str = "daily"
    alpha_half_life: float = 10.0
    hedge_cost_bps: float = 0.0
    weights: str = "contracts"
    reference_inverse: Path | None = None
    converter_dates: int = 10
    mc_library: Path | None = None
    model: str = "hist_gradient_boosting"
    model_params: dict = field(default_factory=dict)

    def __post_init__(self):
        from .step06 import factor_model
        factor_model(self.model, self.model_params)  # Reject unknown controls before work.
        if self.book not in ("atm", "near_atm", "full"):
            raise ValueError("book must be atm, near_atm or full")
        if self.factor_count not in (1, 2, 3):
            raise ValueError("factor_count must be 1, 2 or 3")
        if self.converter not in ("daily", "fixed", "pooled"):
            raise ValueError("converter must be daily, fixed or pooled")
        if self.converter_dates < 2:
            raise ValueError("converter_dates must be at least 2")
        if self.converter == "fixed" and self.reference_inverse is None:
            raise ValueError("fixed converter requires reference_inverse")
        if self.converter != "fixed" and self.reference_inverse is not None:
            raise ValueError("reference_inverse is only used by the fixed converter")
        if (not np.isfinite(self.alpha_half_life) or self.alpha_half_life < 0
                or not np.isfinite(self.hedge_cost_bps) or self.hedge_cost_bps < 0):
            raise ValueError("half life and hedge costs must be finite and nonnegative")
        if self.weights not in ("contracts", "equal_vega"):
            raise ValueError("weights must be contracts or equal_vega")

    def axes(self, config):
        if self.book == "atm":
            return (config.step4_anchor_tenor,), (config.step4_anchor_level,)
        if self.book == "near_atm":
            return (0.25, 0.5, 1.0), (0.9, 1.0)
        return config.step4_tenors, config.step4_strike_levels


@dataclass
class Step7Inputs:
    config: DynamicAlphaConfig
    settings: Step7Config
    history: object
    forecaster: object
    forecasts: pd.DataFrame
    loadings: pd.DataFrame
    daily: pd.DataFrame
    reference: pd.DataFrame | None
    sources: dict


def prepare_step7(config=DynamicAlphaConfig(), settings=Step7Config(), *,
                  input_root=Path("output/dynamic_alpha")):
    """Validate provenance and fit the original Step 6 model, without running MC."""
    root = Path(input_root)
    if settings.mc_library is not None:
        from .mc_library import MCLibrary
        config = MCLibrary.configured(settings.mc_library, config)
    manifest = read_manifest(root / "step06/manifest.json",
                             expected_stage="dynamic_alpha_step06")
    if manifest["validation"].get("beta_workflow") != "daily_only_v1":
        raise ValueError("Step 6 uses an obsolete beta workflow; rebuild Steps 2, 4, 5, 6 for daily-only")
    if settings.model == "catboost":
        settings = replace(settings, model_params=catboost_parameters_from_step6(
            manifest["validation"], settings.model_params))
    elif settings.model == "hist_gradient_boosting":
        settings = replace(settings, model_params=hgb_parameters_from_step6(
            manifest["validation"], settings.model_params))
    # Validate the entire recorded chain, not only the final CSVs.
    sources = {}
    for step in ("step01", "step02", "step04", "step05", "step06"):
        path = root / step / "manifest.json"
        data = read_manifest(path)
        if step in ("step02", "step05", "step06") and data["validation"].get("beta_workflow") != "daily_only_v1":
            raise ValueError(f"obsolete {step} beta workflow; rebuild daily-only upstream artifacts")
        for key, digest in data["inputs"].items():
            if key.endswith("_sha256"):
                source = Path(data["inputs"][key[:-7]])
                if file_sha256(source) != digest:
                    raise ValueError(f"stale {step} input: {source}")
        sources[step + "_manifest"] = str(path)
        sources[step + "_manifest_sha256"] = file_sha256(path)
    stored = manifest["config"]
    method = stored.get("step4_factor_method", "atm_anchored")
    if config.step4_factor_method != method:
        raise ValueError("Step 7 factor method differs from Step 6")
    for step in ("step04", "step05"):
        upstream = read_manifest(root / step / "manifest.json")
        if upstream["config"].get("step4_factor_method", "atm_anchored") != method:
            raise ValueError(f"{step} factor method mismatch; rebuild downstream")
        if upstream["validation"].get("factor_basis_id") != manifest["validation"].get("factor_basis_id"):
            raise ValueError(f"{step} factor basis mismatch; rebuild downstream")

    for key in ("beta_min_abs_dlogS", "step4_train_fraction", "rate", "dividend",
                "repo", "holidays", "step4_anchor_tenor", "step4_anchor_level",
                "step4_tenors", "step4_strike_levels", "tenors", "strike_levels"):
        current = json.loads(json.dumps(asdict(config)[key], default=str))
        if current != stored[key]:
            raise ValueError(f"Step 7 {key} differs from the fitted Step 6 config")
    step1_manifest = read_manifest(root / "step01/manifest.json")
    if step1_manifest["validation"].get("excluded_observation_dates") != observation_exclusions():
        raise ValueError("Observation exclusions changed: rebuild Steps 1, 2, 4, 5, 6 before Step 7")
    source_data = step1_manifest["inputs"]
    if file_sha256(config.data_path) != source_data["svi_parameters_sha256"]:
        raise ValueError("Step 7 source data differs from Step 1")
    panel, loadings, daily = load_step6_inputs(
        factor_state_panel_path=root / "step05/factor_state_panel.csv",
        factor_loadings_path=root / "step04/factor_loadings.csv",
        daily_beta_path=root / "step02/beta_daily.csv")
    forecaster, features = fit_factor_forecaster(
        panel, loadings, config, model_name=settings.model,
        model_parameters=settings.model_params)
    features = features[features.observation_date >= forecaster.train_end]
    forecasts = forecaster.predict(features).set_index("observation_date")
    naive = forecaster.naive_predict(features).set_index("observation_date")
    schema = schema_for(loadings)
    for factor, last in zip(schema.scores, schema.last):
        forecasts[last] = naive[factor]
    forecasts["naive_uses_close_t_factors"] = features.set_index("observation_date")[list(schema.scores)].notna().all(axis=1)
    copy_metadata(loadings, forecasts)
    ordered = _ordered_loadings(loadings, config)
    tenors, levels = settings.axes(config)
    for tenor in tenors:
        for level in levels:
            if (tenor, level) not in ordered.index:
                raise ValueError(f"book cell {(tenor, level)} not in fitted loadings")
    history = load_surface_history(config.data_path, config.market_conventions,
                                   beta_clamp=config.beta_clamp,
                                   duplicate_vol_date_policy=config.duplicate_vol_date_policy,
                                   source_timezone=config.source_timezone,
                                   market_timezone=config.market_timezone)
    reference = None
    if settings.converter == "fixed":
        reference = pd.read_csv(settings.reference_inverse)
        reference["calibration_date"] = pd.to_datetime(reference.calibration_date).dt.date
        dates = reference.calibration_date.unique()
        if len(dates) != 1 or dates[0] > forecaster.train_end:
            raise ValueError("fixed inverse must have exactly one training-period date")
        if "source_dates" in reference:
            source_dates = [pd.Timestamp(d).date() for value in reference.source_dates.unique()
                            for d in json.loads(value)]
            if not source_dates or max(source_dates) != dates[0]:
                raise ValueError("pooled reference latest date differs from source dates")
        sources["reference_inverse"] = str(settings.reference_inverse)
        sources["reference_inverse_sha256"] = file_sha256(settings.reference_inverse)
    sources["svi_parameters"] = str(config.data_path)
    sources["svi_parameters_sha256"] = file_sha256(config.data_path)
    if settings.mc_library is not None:
        from .mc_library import MCLibrary
        MCLibrary(settings.mc_library, config, tenors, levels)  # Validate pricing/axes now.
        for name in ("library", "index"):
            path = Path(settings.mc_library) / f"{name}.json"
            sources[f"mc_{name}"] = str(path)
            sources[f"mc_{name}_sha256"] = file_sha256(path)
    return Step7Inputs(config, settings, history, forecaster, forecasts, ordered,
                       _canonical_axes(daily, config),
                       reference, sources)


def _intervals(inputs):
    """Missing observations are data coverage exclusions, never return-threshold filters."""
    history, config = inputs.history, inputs.config
    result, excluded = [], []
    for previous, current in zip(history.dates, history.dates[1:]):
        if nb_biz_days(previous, current, config.holidays) != 1:
            excluded.append({"previous": previous, "current": current,
                             "reason": "nonconsecutive source observations"})
        else:
            result.append((previous, current))
    return result, excluded


def _marks(inputs, previous, current):
    tenors, levels = inputs.settings.axes(inputs.config)
    frame = pd.DataFrame([contract_interval(inputs.history[previous],
                                           inputs.history[current], tenor, level)
                          for tenor in tenors for level in levels])
    if inputs.settings.weights == "equal_vega":
        if (frame.vega <= 1e-8).any():
            raise ValueError("equal-vega book has negligible vega; use contracts weights")
        frame["quantity"] = 100.0 / (len(frame) * frame.vega)
    else:
        frame["quantity"] = 1.0
    return frame


def hedge_error(dv, delta, ds, pv, spot, dt, rate, income_yield):
    """Excess P&L of long option, short delta and financing, before trading costs."""
    return (dv - delta * ds + (delta * spot - pv) * np.expm1(rate * dt)
            - delta * spot * np.expm1(income_yield * dt))


def cash_account(*, wealth, pv, next_pv, delta, spot, next_spot, dt, rate,
                 income_yield, previous_delta, cost_bps, liquidate=False):
    """Self-financing roll/rehedge with cost on NET stock trades, including final exit."""
    units = abs(delta - previous_delta)
    cost_open = units * spot * cost_bps / 10000.0
    cost_close = abs(delta) * next_spot * cost_bps / 10000.0 if liquidate else 0.0
    cash_open = wealth - pv + delta * spot - cost_open
    cash_close = (cash_open * np.exp(rate * dt)
                  - delta * spot * np.expm1(income_yield * dt) - cost_close)
    wealth_end = next_pv - delta * next_spot + cash_close
    return {
        "cash_open": cash_open, "cash_close": cash_close,
        "wealth": wealth_end, "net_error": wealth_end-wealth*np.exp(rate*dt),
        "cost": cost_open+cost_close,
        "hedge_turnover": units + (abs(delta) if liquidate else 0.0),
        "hedge_notional_turnover": units*spot + (
            abs(delta)*next_spot if liquidate else 0.0),
    }


def _known_series(frame, value, cells):
    return {cell: (cell_curve(frame, *cell).set_index("observation_date")[value]
                    .sort_index()) for cell in cells}


def run_step7(inputs, *, outdir, map_store=None, cache_dir=None, progress=print):
    """Run training fixed-alpha selection and all-date test books on one engine."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config, settings = inputs.config, inputs.settings
    if not {0., 1., 2.}.issubset(config.step3_alphas):
        raise ValueError("Step 7 requires the document's fixed alpha 0/1/2 baselines")
    tenors, levels = settings.axes(config)
    cells = [(t, m) for t in tenors for m in levels]
    if settings.mc_library is not None:
        if cache_dir is not None:
            raise ValueError("choose mc_library or legacy mc_cache, not both")
        from .mc_library import MCLibrary
        provider = map_store or MCLibrary(settings.mc_library, config, tenors, levels)
    else:
        provider = map_store or MCMapStore(config, tenors, levels, cache_dir or outdir / "mc_cache")
    pairs, excluded = _intervals(inputs)
    training = [(p, d) for p, d in pairs if d <= inputs.forecaster.train_end]
    testing = [(p, d) for p, d in pairs if d > inputs.forecaster.train_end]
    if not training or not testing:
        raise ValueError("Step 7 requires training and test holding intervals")
    if any(p not in inputs.forecasts.index for p, _ in testing):
        raise ValueError("missing close-t forecast: generate signals on every market date")
    schema = schema_for(inputs.loadings)
    required_forecasts = {*schema.scores, *schema.last, "naive_uses_close_t_factors"}
    missing = required_forecasts.difference(inputs.forecasts.columns)
    if missing:
        raise ValueError(f"missing daily-only model/persistence forecast columns: {sorted(missing)}")
    realised = _known_series(inputs.daily, "beta_surface_daily", cells)
    alpha_history = {cell: deque(maxlen=20) for cell in cells}
    fixed_errors = {a: [] for a in config.step3_alphas}
    train_delta_fallbacks = {a: 0 for a in config.step3_alphas}
    map_rows, train_exclusions = [], []
    reference = inputs.reference
    if settings.converter == "pooled":
        # Select by training chronology only, never by test outcomes or MC fit.
        supported_dates = []
        for previous, current in training:
            try:
                _marks(inputs, previous, current)
            except ValueError:
                continue
            supported_dates.append(previous)
        indices = np.unique(np.linspace(0, len(supported_dates)-1,
                                        min(settings.converter_dates, len(supported_dates)),
                                        dtype=int))
        selected = [supported_dates[i] for i in indices]
        if len(selected) < 2:
            raise ValueError("too few supported training dates for pooled converter")
        sampled = []
        for date in selected:
            progress(f"Step 7 pooled converter: {date} ({len(selected)} training dates)")
            sampled.append(provider.get(inputs.history[date]))
        reference = pooled_converter(sampled, inputs.forecaster.train_end)

    def table_at(date):
        table = provider.get(inputs.history[date])
        table["feature_date"] = date
        return table

    def converter(table, cell):
        return cell_curve(reference if reference is not None else table,
                          *cell)

    def update_alpha_history(table, date):
        for cell in cells:
            # Only today's realised ratio; no repetition of a stale observation.
            beta = realised[cell].get(date, np.nan)
            alpha, _, fallback = invert_beta(converter(table, cell), beta)
            if not fallback:
                alpha_history[cell].append(alpha)

    for number, (previous, current) in enumerate(training, 1):
        if number > 1 and training[number-2][1] != previous:
            for queue in alpha_history.values():
                queue.clear()
        try:
            marks = _marks(inputs, previous, current)
        except ValueError as exc:
            train_exclusions.append({"date": current, "reason": str(exc)})
            continue
        progress(f"Step 7 training {number}/{len(training)}: {previous}")
        table = table_at(previous)
        update_alpha_history(table, previous)
        for alpha in config.step3_alphas:
            error = 0.0
            for row in marks.itertuples():
                delta, _, fallback = delta_at_alpha(cell_curve(table, row.tenor, row.level),
                                                    alpha, row.bs_delta)
                train_delta_fallbacks[alpha] += int(fallback)
                error += row.quantity * hedge_error(
                    row.dV, delta, row.dS, row.pv, row.spot, row.dt_r,
                    row.rate, row.income_yield)
            fixed_errors[alpha].append(error)
    if min(map(len, fixed_errors.values())) < 2:
        raise ValueError("too few supported training intervals to select a fixed alpha")
    best_alpha = min(fixed_errors, key=lambda a: np.std(fixed_errors[a], ddof=1))
    progress(f"Step 7 training-best fixed alpha: {best_alpha:g}")
    signals, option_rows, book_rows, audits = [], [], [], []
    smoothed = {cell: best_alpha for cell in cells}
    books = {}
    segment = 0
    for number, (previous, current) in enumerate(testing, 1):
        segment_start = number == 1 or testing[number-2][1] != previous
        segment_end = number == len(testing) or testing[number][0] != current
        if segment_start:
            segment += 1
            if number > 1 or training[-1][1] != previous:
                smoothed = {cell: best_alpha for cell in cells}
                for queue in alpha_history.values():
                    queue.clear()
        progress(f"Step 7 test {number}/{len(testing)}: {previous} -> {current}")
        marks = _marks(inputs, previous, current)  # Incomplete test books fail explicitly.
        table = table_at(previous)
        map_rows.append(table)
        update_alpha_history(table, previous)
        z = inputs.forecasts.loc[previous, list(schema.scores)].to_numpy(float)
        naive_z = inputs.forecasts.loc[previous, list(schema.last)].to_numpy(float)
        day_rows = []
        for row in marks.itertuples(index=False):
            cell = row.tenor, row.level
            loading = inputs.loadings.loc[cell]
            beta = float(loading.factor_intercept + z[:settings.factor_count] @
                         loading[list(schema.loadings)].to_numpy(float)[:settings.factor_count])
            if not np.isfinite(beta):
                raise ValueError(f"nonfinite close-t beta forecast: {previous}, {cell}")
            curve, inverse = cell_curve(table, *cell), converter(table, cell)
            if "implied_vol_up" in curve:
                audits.append(mc_consistency(
                    curve, row, inputs.history[previous].tau_vol(row.expiry)))
            mc_one, mc_one_se, mc_one_fallback = delta_at_alpha(curve, 1., row.bs_delta)
            raw, clipped, fallback = invert_beta(inverse, beta)
            decay = 0.0 if settings.alpha_half_life == 0 else 2**(-1/settings.alpha_half_life)
            smoothed[cell] = decay*smoothed[cell] + (1-decay)*raw
            naive_beta = float(loading.factor_intercept + naive_z[:settings.factor_count] @
                               loading[list(schema.loadings)].to_numpy(float)[:settings.factor_count])
            if not np.isfinite(naive_beta):
                raise ValueError(f"nonfinite persistence forecast: {previous}, {cell}")
            naive_alpha, naive_clipped, naive_fallback = invert_beta(inverse, naive_beta)
            mean_alpha = float(np.mean(alpha_history[cell])) if alpha_history[cell] else 1.0
            strategies = {f"fixed_{a:g}": (a, False, False) for a in config.step3_alphas}
            strategies.update({
                "best_fixed_train": (best_alpha, False, False),
                "bs_delta": (1.0, False, False),
                "last_observed_factor": (naive_alpha, naive_clipped, naive_fallback),
                "rolling_alpha_mean": (mean_alpha, False, not alpha_history[cell]),
                "dynamic_raw": (raw, clipped, fallback),
                "dynamic_ema": (smoothed[cell], clipped, fallback),
                "shadow_bs": (np.nan, False, False),
                "shadow_mc_base": (np.nan, False, False),
            })
            for strategy, (alpha, alpha_clipped, inverse_fallback) in strategies.items():
                direct = strategy.startswith("shadow_")
                raw_model_beta = np.nan
                if direct:
                    base = row.bs_delta if strategy == "shadow_bs" else mc_one
                    delta = shadow_delta(base, row.vega, row.spot, beta)
                    delta_se = 0. if strategy == "shadow_bs" else mc_one_se
                    delta_fallback = strategy == "shadow_mc_base" and mc_one_fallback
                    effective_beta = beta  # No inverse, clipping or alpha smoothing.
                elif strategy == "bs_delta":
                    delta, delta_se, delta_fallback = row.bs_delta, 0.0, False
                    effective_beta = 0.0
                else:
                    delta, delta_se, delta_fallback = delta_at_alpha(curve, alpha, row.bs_delta)
                    ordered = curve.sort_values("alpha")
                    effective_beta = (float(np.interp(alpha, ordered.alpha, ordered.beta_converter))
                                      if len(ordered) and np.isfinite(ordered.beta_converter).all()
                                      else np.nan)
                    if delta_fallback:
                        effective_beta = 0.0
                    elif "beta_model" in ordered and np.isfinite(ordered.beta_model).all():
                        raw_model_beta = float(np.interp(alpha, ordered.alpha, ordered.beta_model))
                # BS-based attribution prevents double-counting shadow delta.
                common_pnl = (row.bs_delta*row.dS + 0.5*row.gamma*row.dS**2
                              + row.time_pnl + row.vega*row.term_roll_iv)
                attribution = row.dV-common_pnl+row.vega*effective_beta*row.dlogS
                forecast_beta = naive_beta if strategy == "last_observed_factor" else beta
                signal = {
                    "feature_date": previous, "label_date": current,
                    "segment": segment,
                    "tenor": row.tenor, "level": row.level, "strategy": strategy,
                    "predicted_beta": forecast_beta, "raw_predicted_alpha": (np.nan if direct else naive_alpha if strategy == "last_observed_factor" else raw),
                    "naive_beta": naive_beta,
                    "naive_uses_close_t_factors": bool(inputs.forecasts.loc[previous, "naive_uses_close_t_factors"]),
                    "alpha": alpha,
                    **dict(zip(schema.scores, naive_z if strategy == "last_observed_factor" else z)),
                    "converter_date": (None if direct or strategy == "bs_delta" else
                                       previous if reference is None else
                                       reference.calibration_date.iloc[0]),
                    "effective_beta": effective_beta, "alpha_clipped": alpha_clipped,
                    "raw_model_beta": raw_model_beta,
                    "beta_mapping_gap": effective_beta-forecast_beta,
                    "inverse_fallback": inverse_fallback, "delta_fallback": delta_fallback,
                }
                signals.append(signal)
                result = {**row._asdict(), **signal, "delta": delta,
                          "delta_stderr": delta_se,
                          "raw_hedge_error": row.dV-delta*row.dS,
                          "carry_hedge_error": hedge_error(
                              row.dV, delta, row.dS, row.pv, row.spot, row.dt_r,
                              row.rate, row.income_yield),
                          "attribution_residual": attribution,
                          "raw_model_attribution_residual": (
                              row.dV-common_pnl+row.vega*raw_model_beta*row.dlogS),
                          "anchor_correction_pnl": (
                              row.vega*(raw_model_beta-effective_beta)*row.dlogS),
                          "forecast_attribution_residual": (
                              row.dV-common_pnl+row.vega*forecast_beta*row.dlogS)}
                option_rows.append(result)
                day_rows.append(result)
        day = pd.DataFrame(day_rows)
        for strategy, group in day.groupby("strategy", sort=False):
            first = group.iloc[0]
            old = books.get(strategy, {"wealth": 0., "offset": 0.})
            if segment_start:
                # Independent complete-data segments, not an invented overnight gap trade.
                state = {"wealth": 0., "delta": 0.,
                         "offset": old["offset"]+old["wealth"]}
            else:
                state = old
            weighted = lambda column: float((group[column]*group.quantity).sum(skipna=False))
            delta = weighted("delta")
            account = cash_account(
                wealth=state["wealth"], pv=weighted("pv"), next_pv=weighted("next_pv"),
                delta=delta, spot=first.spot, next_spot=first.next_spot,
                dt=first.dt_r, rate=first.rate, income_yield=first.income_yield,
                previous_delta=state["delta"], cost_bps=settings.hedge_cost_bps,
                liquidate=segment_end)
            book_rows.append({
                "feature_date": previous, "label_date": current, "strategy": strategy,
                "segment": segment, "segment_start": segment_start, "segment_end": segment_end,
                "book_delta": delta, "option_pv": weighted("pv"),
                "next_option_pv": weighted("next_pv"),
                "raw_hedge_error": weighted("raw_hedge_error"),
                "gross_error": weighted("carry_hedge_error"),
                "attribution_residual": weighted("attribution_residual"),
                "raw_model_attribution_residual": weighted("raw_model_attribution_residual"),
                "anchor_correction_pnl": weighted("anchor_correction_pnl"),
                "forecast_attribution_residual": weighted("forecast_attribution_residual"),
                "n_options": len(group), **account,
                "segment_wealth": account["wealth"],
                "wealth": state["offset"]+account["wealth"]})
            books[strategy] = {"wealth": account["wealth"], "delta": delta,
                               "offset": state["offset"]}
    frames = {
        "signals": pd.DataFrame(signals), "mc_map": pd.concat(map_rows, ignore_index=True),
        "option_pnl": pd.DataFrame(option_rows), "book_pnl": pd.DataFrame(book_rows)}
    if audits:
        frames["mc_consistency"] = pd.concat(audits, ignore_index=True)
    if reference is not None:
        frames["converter"] = reference
    frames["summary"] = summarize_step7(frames["book_pnl"], frames["signals"])
    frames["cell_summary"] = pd.DataFrame([{
        "strategy": strategy, "tenor": tenor, "level": level,
        "n_test_dates": len(group),
        "gross_std_error": float(group.carry_hedge_error.std(ddof=1)),
        "gross_rmse": float(np.sqrt(np.mean(group.carry_hedge_error**2))),
        "attribution_rmse": float(np.sqrt(np.mean(group.attribution_residual.to_numpy(float)**2))),
    } for (strategy, tenor, level), group in frames["option_pnl"].groupby(
        ["strategy", "tenor", "level"])])
    for name, frame in frames.items():
        frame.to_csv(outdir / f"{name}.csv", index=False)
    joblib.dump(inputs.forecaster, outdir / "forecaster.joblib")
    inputs.forecasts.to_csv(outdir / "factor_forecasts.csv")
    validation = {
        "status": "complete", "train_end": inputs.forecaster.train_end,
        "test_start": testing[0][1], "test_end": testing[-1][1],
        "test_intervals": len(testing), "training_intervals": len(fixed_errors[best_alpha]),
        "training_best_alpha": best_alpha,
        "training_fixed_std": {str(a): float(np.std(e, ddof=1)) for a,e in fixed_errors.items()},
        "training_delta_fallback_cells": train_delta_fallbacks,
        "data_exclusions": excluded, "training_contract_exclusions": train_exclusions,
        "marking": "raw SVI historical marks; European calls; no exchange bid/ask",
        "book_policy": "daily rolling book; fixed K/T/quantity within each interval",
        "alpha_initialization": "training-best fixed alpha",
        "test_return_filter": False, "option_transaction_cost": "not modelled",
        "excluded_observation_dates": observation_exclusions(),
        "test_segments": segment,
        "gap_policy": "independent segments; close at last valid date, reopen after gap; no gap P&L/carry",
        "wealth_policy": "sum of closed segment wealth plus current segment wealth; no gap interest",
        "mc_access": "read_only_library" if settings.mc_library is not None else "legacy_cache_or_compute",
        "attribution_beta_policy": "primary=centered effective beta; raw_model_beta audit in parallel",
        "anchor_identity": "raw_model_attribution_residual = attribution_residual + anchor_correction_pnl",
        "anchor_delta_policy": "MC delta and actual hedge P&L are never re-centered",
        "predictor_model": settings.model, "predictor_parameters": settings.model_params,
        "beta_workflow": "daily_only_v1",
        "prediction_benchmark": "last_observed_factor; same factor count and converter as dynamic_raw",
        "delta_policy": "daily MC interpolation plus shadow BS/MC-base controls; audited fallback",
        "shadow_policy": "base_delta - raw_SVI_vega * predicted_beta / spot; no alpha clipping/EMA",
        "reference_policy": settings.converter,
        "converter_source_dates": (json.loads(reference.source_dates.iloc[0])
            if reference is not None and "source_dates" in reference else
            [str(reference.calibration_date.iloc[0])] if reference is not None else []),
        "pooled_policy": "equal mean of forward curves on evenly spaced supported training dates",
        "numerical_convergence": "not certified by completion; inspect MC SE and rerun sensitivity",
    }
    write_manifest(outdir / "manifest.json", stage="dynamic_alpha_step07",
                   config={"research": asdict(config), "backtest": asdict(settings)},
                   inputs=inputs.sources, validation=validation)
    return frames


def _paired_improvement_ci(values, baseline, *, statistic="std", segments=None):
    """Paired circular 10-observation block bootstrap; descriptive research CI."""
    values, baseline = np.asarray(values), np.asarray(baseline)
    n = len(values)
    if n < 20 or not np.isfinite(values).all() or not np.isfinite(baseline).all():
        return np.nan, np.nan
    rng = np.random.default_rng(20260807)
    groups = [np.arange(n)] if segments is None else [
        np.flatnonzero(np.asarray(segments) == value) for value in pd.unique(segments)]
    pieces = []
    for group in groups:
        length = len(group)
        starts = rng.integers(length, size=(1000, (length+9)//10))
        local = ((starts[:, :, None]+np.arange(10)) % length).reshape(1000, -1)[:, :length]
        pieces.append(group[local])
    indices = np.concatenate(pieces, axis=1)
    score = (lambda x: x.std(axis=1, ddof=1)) if statistic == "std" else (
        lambda x: np.sqrt(np.mean(x*x, axis=1)))
    denominator = score(baseline[indices])
    valid = denominator > 0
    if not valid.any():
        return np.nan, np.nan
    improvement = 1-score(values[indices])[valid]/denominator[valid]
    return tuple(np.quantile(improvement, [.025, .975]))


def summarize_step7(book, signals):
    baseline = book[book.strategy.eq("best_fixed_train")].set_index("label_date")
    baseline_std = baseline.net_error.std(ddof=1)
    naive = book[book.strategy.eq("last_observed_factor")].set_index("label_date")
    naive_std = naive.net_error.std(ddof=1)
    fixed_one = book[book.strategy.eq("fixed_1")]
    fixed_one_std = fixed_one.net_error.std(ddof=1)
    fixed_one_rmse = float(np.sqrt(np.mean(fixed_one.net_error**2)))
    fixed_one_attribution = float(np.sqrt(np.mean(
        fixed_one.attribution_residual.to_numpy(float)**2)))
    rows = []
    for strategy, group in book.groupby("strategy", sort=False):
        group = group.sort_values("label_date")
        values = group.net_error.to_numpy(float)
        paired_fixed = baseline.loc[group.label_date, "net_error"].to_numpy(float)
        segments = group["segment"].to_numpy()
        std_ci = _paired_improvement_ci(values, paired_fixed, segments=segments)
        naive_values = naive.loc[group.label_date, "net_error"].to_numpy(float)
        naive_ci = _paired_improvement_ci(values, naive_values, segments=segments)
        paired_attribution = fixed_one.set_index("label_date").loc[
            group.label_date, "attribution_residual"].to_numpy(float)
        attr_ci = _paired_improvement_ci(group.attribution_residual.to_numpy(float),
                                         paired_attribution, statistic="rmse", segments=segments)
        attr_rmse = float(np.sqrt(np.mean(group.attribution_residual.to_numpy(float)**2)))
        sig = signals[signals.strategy.eq(strategy)]
        alpha_changes = sig.sort_values("feature_date").groupby(["segment", "tenor", "level"]).alpha.diff().abs()
        rows.append({
            "scope": "book", "strategy": strategy, "n_test_dates": len(group),
            "mean_error": float(values.mean()), "std_error": float(values.std(ddof=1)),
            "rmse": float(np.sqrt(np.mean(values**2))), "mae": float(np.mean(abs(values))),
            "absolute_error_q95": float(np.quantile(abs(values), .95)),
            "absolute_error_q99": float(np.quantile(abs(values), .99)),
            "std_improvement_vs_best_fixed": (1-values.std(ddof=1)/baseline_std
                                              if baseline_std > 0 else np.nan),
            "std_improvement_ci_low": std_ci[0], "std_improvement_ci_high": std_ci[1],
            "std_improvement_vs_last_observed_factor": (1-values.std(ddof=1)/naive_std if naive_std > 0 else np.nan),
            "std_improvement_vs_last_factor_ci_low": naive_ci[0],
            "std_improvement_vs_last_factor_ci_high": naive_ci[1],
            "std_improvement_vs_alpha_one": (1-values.std(ddof=1)/fixed_one_std
                                             if fixed_one_std > 0 else np.nan),
            "rmse_improvement_vs_alpha_one": (1-np.sqrt(np.mean(values**2))/fixed_one_rmse
                                              if fixed_one_rmse > 0 else np.nan),
            "gross_std_error": float(group.gross_error.std(ddof=1)),
            "total_cost": float(group.cost.sum()), "final_wealth": float(group.wealth.iloc[-1]),
            "mean_alpha_change": float(alpha_changes.mean()),
            "total_hedge_turnover": float(group.hedge_turnover.sum()),
            "alpha_clipped_fraction": float(sig.alpha_clipped.mean()),
            "inverse_fallback_fraction": float(sig.inverse_fallback.mean()),
            "delta_fallback_fraction": float(sig.delta_fallback.mean()),
            "attribution_rmse": attr_rmse,
            "raw_model_attribution_rmse": float(np.sqrt(np.mean(
                group.raw_model_attribution_residual.to_numpy(float)**2))),
            "anchor_correction_pnl_sum": float(group.anchor_correction_pnl.sum(min_count=1)),
            "attribution_valid_dates": int(np.isfinite(group.attribution_residual).sum()),
            "attribution_improvement_vs_alpha_one": (
                1-attr_rmse/fixed_one_attribution
                if fixed_one_attribution > 0 else np.nan),
            "attribution_improvement_ci_low": attr_ci[0],
            "attribution_improvement_ci_high": attr_ci[1],
            "forecast_attribution_rmse": float(np.sqrt(np.mean(
                group.forecast_attribution_residual.to_numpy(float)**2))),
        })
    return pd.DataFrame(rows)

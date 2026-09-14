"""Independent Step 7: one SR policy per date/strategy for the whole option book."""

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from .artifacts import file_sha256, write_manifest
from .data_loader import observation_exclusions
from .hedging import cell_curve, invert_beta
from .mc_library import MCLibrary
from .precompute import _csv, _writer_lock
from .sr_pricing import AlphaSurface, SharedSRPricer, SR_KINDS
from .step06 import FACTOR_COLUMNS, LAST_FACTOR_COLUMNS
from .step07 import _intervals, _marks, cash_account, hedge_error, _paired_improvement_ci


@dataclass(frozen=True)
class SharedStep7Config:
    # Raw tests always run; EMA variants are additional smoothing comparisons.
    half_life: float = 10.
    rolling_alpha_observations: int = 20
    max_train_dates: int | None = None
    max_test_dates: int | None = None

    def __post_init__(self):
        if not np.isfinite(self.half_life) or self.half_life < 0:
            raise ValueError("half-life must be finite and nonnegative")
        if self.rolling_alpha_observations < 1:
            raise ValueError("rolling Alpha window must be positive")
        for value in (self.max_train_dates, self.max_test_dates):
            if value is not None and value < 1:
                raise ValueError("diagnostic date limits must be positive")


def _forecast_nodes(inputs, date, columns=FACTOR_COLUMNS):
    z = inputs.forecasts.loc[date, list(columns)].to_numpy(float)
    count = inputs.settings.factor_count
    columns = ["atm_beta_loading", "shape_loading_1", "shape_loading_2"][:count]
    values = (inputs.loadings.factor_intercept.to_numpy(float)
              + inputs.loadings[columns].to_numpy(float) @ z[:count])
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite close-t factor forecast at {date}")
    return pd.Series(values, index=inputs.loadings.index)


def _convert(table, beta, tenors, levels):
    rows = []
    for tenor in tenors:
        for level in levels:
            curve = cell_curve(table, tenor, level)
            value = float(beta.loc[(tenor, level)])
            alpha, clipped, fallback = invert_beta(curve, value)
            rows.append({"tenor": tenor, "level": level, "predicted_beta": value,
                         "alpha": alpha, "alpha_clipped": clipped, "inverse_fallback": fallback,
                         "converter_quality_pass": bool(curve.quality_pass.all()) if len(curve) else False})
    return pd.DataFrame(rows)


def _used_nodes(nodes, kind):
    if kind == "constant_sr":
        return nodes[np.isclose(nodes.tenor, .25) & np.isclose(nodes.level, 1.)]
    if kind == "term_sr":
        return nodes[np.isclose(nodes.level, 1.)]
    return nodes


def _canonical_config(value):
    value = json.loads(json.dumps(value, default=str))
    for section, keys in (("research", ("data_path",)), ("new_mc", ("data_path",)),
                          ("backtest", ("mc_library", "reference_inverse"))):
        for key in keys:
            if value[section].get(key) is not None:
                value[section][key] = str(Path(value[section][key]).resolve())
    return value


def _plan(inputs, options, provider):
    pairs, excluded = _intervals(inputs)
    training = [(p, d) for p, d in pairs if d <= inputs.forecaster.train_end]
    testing = [(p, d) for p, d in pairs if d > inputs.forecaster.train_end]
    if options.max_train_dates:
        training = training[-options.max_train_dates:]
    if options.max_test_dates:
        testing = testing[:options.max_test_dates]
    marks, rows, supported_training = {}, [], []
    for sample, intervals in (("train", training), ("test", testing)):
        for previous, current in intervals:
            if sample == "test" and previous not in inputs.forecasts.index:
                raise ValueError(f"missing close-t forecast: {previous}")
            try:
                marks[previous] = _marks(inputs, previous, current)
            except ValueError as exc:
                if sample == "test":
                    raise
                rows.append(dict(feature_date=previous, label_date=current, sample=sample,
                                 supported=False, reason=str(exc)))
                continue
            if sample == "train":
                supported_training.append((previous, current))
            # Validate coverage without reading any legacy Delta or starting MC.
            if isinstance(provider, MCLibrary):
                for tenor in provider.tenors:
                    key, _ = provider._key(inputs.history[previous], tenor)
                    if key not in provider.index:
                        raise FileNotFoundError(f"missing converter shard: {key}")
            rows.append(dict(feature_date=previous, label_date=current, sample=sample,
                             supported=True, reason=""))
    if len(supported_training) < 2 or not testing:
        raise ValueError("need at least two supported training intervals and one test interval")
    return supported_training, testing, marks, pd.DataFrame(rows), excluded


def _summary(book, nodes):
    rows = []
    baseline = book[book.strategy.eq("best_fixed_train")].set_index("label_date")
    fixed_one = book[book.strategy.eq("fixed_1")].set_index("label_date")
    for strategy, group in book.groupby("strategy", sort=False):
        group = group.sort_values("label_date")
        raw, net = group.raw_hedge_error.to_numpy(), group.net_error.to_numpy()
        ref = baseline.loc[group.label_date].raw_hedge_error.to_numpy()
        one = fixed_one.loc[group.label_date].raw_hedge_error.to_numpy()
        ci = _paired_improvement_ci(raw, ref, segments=group.segment.to_numpy())
        sig = nodes[nodes.strategy.eq(strategy)]
        changes = sig.sort_values("feature_date").groupby(["segment", "tenor", "level"]).alpha.diff().abs()
        std = lambda x: float(np.std(x, ddof=1))
        rmse = lambda x: float(np.sqrt(np.mean(np.asarray(x)**2)))
        improvement = lambda x, y: 1-x/y if y > 0 else np.nan
        record = {"strategy": strategy, "n_test_dates": len(group),
                  "raw_mean_error": float(raw.mean()), "raw_std_error": std(raw),
                  "raw_rmse": rmse(raw), "net_mean_error": float(net.mean()),
                  "net_std_error": std(net), "net_rmse": rmse(net),
                  "raw_std_improvement_vs_best_fixed": improvement(std(raw), std(ref)),
                  "raw_std_improvement_vs_alpha_one": improvement(std(raw), std(one)),
                  "improvement_ci_low": ci[0], "improvement_ci_high": ci[1],
                  "raw_pnl_sum": float(raw.sum()), "net_pnl_sum": float(net.sum()),
                  "final_wealth": float(group.wealth.iloc[-1]), "total_cost": float(group.cost.sum()),
                  "total_hedge_turnover": float(group.hedge_turnover.sum()),
                  "total_hedge_notional_turnover": float(group.hedge_notional_turnover.sum()),
                  "mean_alpha_change": float(changes.mean()),
                  "alpha_clipped_fraction": float(sig.alpha_clipped.mean()),
                  "inverse_fallback_fraction": float(sig.inverse_fallback.mean()),
                  "converter_quality_pass_fraction": float(sig.converter_quality_pass.mean()),
                  "delta_fallback_fraction": float(group.delta_fallback_fraction.mean()),
                  "forecast_attribution_rmse": rmse(group.forecast_attribution_residual),
                  "model_attribution_rmse": rmse(group.attribution_residual),
                  "raw_model_attribution_rmse": rmse(group.raw_model_attribution_residual),
                  "attribution_valid_dates": int(np.isfinite(group.attribution_residual).sum()),
                  "bs_attribution_rmse": rmse(group.bs_attribution_residual)}
        kind = next((k for k in SR_KINDS if strategy in (k, k+"_ema")), None)
        if kind:
            naive = book[book.strategy.eq(kind+"_last_observed_factor")].raw_hedge_error
            record["raw_std_improvement_vs_same_policy_persistence"] = improvement(std(raw), std(naive))
        record["model_attribution_improvement_vs_alpha_one"] = improvement(
            record["model_attribution_rmse"], rmse(fixed_one.attribution_residual))
        rows.append(record)
    return pd.DataFrame(rows)


def run_shared_step7(inputs, *, outdir, options=SharedStep7Config(), mc_config=None,
                     prepare_only=False, map_store=None, pricer=None, cache_dir=None, progress=print):
    """Fit upstream separately via prepare_step7; all hedges here use new MC."""
    outdir = Path(outdir)
    settings, config = inputs.settings, inputs.config
    if settings.converter != "daily" or (settings.mc_library is None and map_store is None):
        raise ValueError("shared Step 7 requires the existing daily converter library")
    if not {0., 1., 2.}.issubset(config.step3_alphas):
        raise ValueError("shared Step 7 requires fixed alpha 0/1/2 controls")
    tenors, levels = config.step4_tenors, config.step4_strike_levels
    AlphaSurface._node(tenors, .25)
    AlphaSurface._node(levels, 1.)
    provider = map_store or MCLibrary(settings.mc_library, config, tenors, levels)
    numerical = mc_config or config
    if cache_dir is not None and Path(cache_dir).resolve() == outdir.resolve():
        raise ValueError("MC cache must be a separate directory from the result files")
    if settings.mc_library is not None:
        library_root = Path(settings.mc_library).resolve()
        for directory in (outdir, Path(cache_dir) if cache_dir else outdir / "sr_mc_cache"):
            if directory.resolve() == library_root or library_root in directory.resolve().parents:
                raise ValueError("new outputs/cache must be outside the read-only converter library")
    pricer = pricer or SharedSRPricer(numerical, cache_dir or outdir / "sr_mc_cache")
    run_config = _canonical_config({"research": asdict(config), "backtest": asdict(settings),
                                   "shared_sr": asdict(options), "new_mc": asdict(numerical)})
    stage = "dynamic_alpha_step07_shared_sr"
    with _writer_lock(outdir):
        prior = outdir / "manifest.json"
        if prior.exists():
            old = json.loads(prior.read_text())
            if old["stage"] != stage or _canonical_config(old["config"]) != run_config:
                raise ValueError("output contains a different backtest/configuration; choose a new directory")
        progress("Shared Step 7: validating book intervals and converter coverage")
        training, testing, marks, plan, excluded = _plan(inputs, options, provider)
        _csv(plan, outdir / "plan.csv")
        inputs.forecasts.to_csv(outdir / "factor_forecasts.csv")
        joblib.dump(inputs.forecaster, outdir / "forecaster.joblib")
        validation = {"status": "prepared_only", "mc_run": False,
            "train_end": inputs.forecaster.train_end, "training_intervals": len(training),
            "test_intervals": len(testing), "data_exclusions": excluded,
            "excluded_observation_dates": observation_exclusions(),
            "diagnostic_date_limits": bool(options.max_train_dates or options.max_test_dates),
            "delta_source": "new shared-surface MC; legacy library supplies beta-alpha only",
            "alpha_nodes": "constant Business/260 tenors; interpolate to all simulation dates",
            "spatial_interpolation": "linear in unshifted log(K/F_ref); flat outside node range",
            "bump_policy": "freeze forecast Alpha policy during up/down; Alpha multiplied once",
            "headline_metric": "std(book dV - net book delta * dS)",
            "marking": "raw SVI European calls; fixed K/expiry/quantity per overnight interval",
            "pnl_policy": "long option / short stock; cash financing, net stock costs, gap segmentation",
            "attribution_policy": "BS delta + gamma + finite-step theta + term roll + vega*dIV; no double counting",
            "alpha_one_policy": "center measured beta for attribution only; never alter MC delta or PnL",
            "smoothing": "raw always; additional node EMA variants" if options.half_life else "raw only by explicit setting",
            "converter_scope": "scalar-alpha maps are node initialization, not joint calibration",
            "quality_policy": "audit failures retained; inverse fallback alpha=1; nonfinite delta fallback BS",
            "planned_training_profiles": len(training)*len(config.step3_alphas),
            "planned_test_profiles": len(testing)*(len(config.step3_alphas)+7+(3 if options.half_life else 0))}
        sources = {**inputs.sources, "shared_step7_code": str(Path(__file__).resolve()),
                   "shared_step7_code_sha256": file_sha256(__file__),
                   "sr_pricing_code": str(Path(__file__).with_name("sr_pricing.py").resolve()),
                   "sr_pricing_code_sha256": file_sha256(Path(__file__).with_name("sr_pricing.py"))}
        def save_status():
            write_manifest(outdir / "manifest.json", stage=stage, config=run_config,
                           inputs=sources, validation=validation)
        save_status()
        if prepare_only:
            return {"plan": plan, "validation": validation}
        validation.update(status="running", mc_run=True)
        save_status()
        try:
            result = _backtest(inputs, options, training, testing, marks, provider, pricer,
                               outdir, validation, save_status, progress)
        except BaseException as exc:
            validation.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                              error=f"{type(exc).__name__}: {exc}")
            save_status()
            raise
        validation.update(status="complete", computed_profiles=pricer.computed,
                          reused_profiles=pricer.reused)
        save_status()
        return result


def _backtest(inputs, options, training, testing, marks, provider, pricer,
              outdir, validation, save_status, progress):
    c, s = inputs.config, inputs.settings
    tenors, levels = c.step4_tenors, c.step4_strike_levels
    shape = (len(tenors), len(levels))
    def profile(date, values, kind="constant_sr"):
        return AlphaSurface(inputs.history[date], tenors, levels,
                            np.full(shape, values) if np.isscalar(values) else values, kind)
    training_rows = []
    for n, (date, end) in enumerate(training, 1):
        progress(f"Shared Step 7 training {n}/{len(training)}: {date}")
        frame = marks[date]
        for alpha in c.step3_alphas:
            priced = pricer.get(inputs.history[date], frame, profile(date, alpha))
            delta = np.where(np.isfinite(priced.delta), priced.delta, frame.bs_delta)
            training_rows.append({"feature_date": date, "label_date": end, "alpha": alpha,
                "raw_error": float(np.sum(frame.quantity*(frame.dV-delta*frame.dS))),
                "delta_fallback_cells": int((~np.isfinite(priced.delta)).sum())})
        validation.update(completed_training_dates=n, active_date=str(date))
        save_status()
    training_frame = pd.DataFrame(training_rows)
    scores = training_frame.groupby("alpha").raw_error.std(ddof=1)
    best_alpha = float(scores.idxmin())
    validation.update(training_best_alpha=best_alpha, training_fixed_std=scores.to_dict())
    _csv(training_frame, outdir / "training_fixed.csv")
    progress(f"Shared Step 7 training-best fixed alpha: {best_alpha:g}")
    daily = inputs.daily[np.isclose(inputs.daily.tenor, .25) & np.isclose(inputs.daily.level, 1.)]
    realised = daily.set_index("observation_date").beta_surface_daily
    rolling = deque(maxlen=options.rolling_alpha_observations)
    def update_rolling(date, table):
        value, _, fallback = invert_beta(cell_curve(table, .25, 1.), realised.get(date, np.nan))
        if not fallback:
            rolling.append(value)
    # Warm-up uses only past realised observations in the contiguous training tail.
    warm = []
    for pair in reversed(training):
        if warm and pair[1] != warm[-1][0]:
            break
        warm.append(pair)
        if len(warm) >= options.rolling_alpha_observations:
            break
    if training[-1][1] == testing[0][0]:
        for date, _ in reversed(warm):
            update_rolling(date, provider.get(inputs.history[date]))
    smooth = np.full(shape, best_alpha)
    books, node_rows, option_rows, book_rows, audits, quote_rows = {}, [], [], [], [], []
    segment = 0
    for n, (date, end) in enumerate(testing, 1):
        start = n == 1 or testing[n-2][1] != date
        finish = n == len(testing) or testing[n][0] != end
        if start:
            segment += 1
            smooth = np.full(shape, best_alpha)
            if n > 1 or training[-1][1] != date:
                rolling.clear()
        progress(f"Shared Step 7 test {n}/{len(testing)}: {date} -> {end}")
        frame, surface = marks[date], inputs.history[date]
        # Drop all legacy prices/deltas immediately; only converter/audit fields enter signals.
        table = provider.get(surface).filter(items=["tenor", "level", "alpha", "beta_converter", "quality_pass"])
        predicted, naive = _forecast_nodes(inputs, date), _forecast_nodes(inputs, date, LAST_FACTOR_COLUMNS)
        raw_nodes, naive_nodes = (_convert(table, b, tenors, levels) for b in (predicted, naive))
        raw = raw_nodes.alpha.to_numpy().reshape(shape)
        decay = 2**(-1/options.half_life) if options.half_life else 0.
        smooth = decay*smooth + (1-decay)*raw
        update_rolling(date, table)
        strategies = {f"fixed_{a:g}": (profile(date, a), None, None) for a in c.step3_alphas}
        strategies["rolling_alpha_mean"] = (profile(date, float(np.mean(rolling)) if rolling else 1.), None, None)
        for kind in SR_KINDS:
            strategies[kind] = (profile(date, raw, kind), raw_nodes, predicted)
            strategies[kind+"_last_observed_factor"] = (
                profile(date, naive_nodes.alpha.to_numpy().reshape(shape), kind), naive_nodes, naive)
            if options.half_life:
                strategies[kind+"_ema"] = (profile(date, smooth, kind), raw_nodes, predicted)
        priced_by_strategy = {name: pricer.get(surface, frame, policy)
                              for name, (policy, _, _) in strategies.items()}
        measured_one = priced_by_strategy["fixed_1"].beta_model.to_numpy()
        strategies["best_fixed_train"] = strategies[f"fixed_{best_alpha:g}"]
        priced_by_strategy["best_fixed_train"] = priced_by_strategy[f"fixed_{best_alpha:g}"]
        strategies["bs_delta"] = (profile(date, 1.), None, None)
        priced_by_strategy["bs_delta"] = None
        for name, (policy, source_nodes, forecast) in strategies.items():
            used = _used_nodes(source_nodes, policy.kind) if source_nodes is not None else pd.DataFrame()
            if source_nodes is None:
                used = pd.DataFrame([dict(tenor=.25, level=1., alpha=policy.values[policy.anchor, policy.atm],
                    predicted_beta=np.nan, alpha_clipped=False, inverse_fallback=False,
                    converter_quality_pass=np.nan)])
            for node in used.to_dict("records"):
                expiry = policy.expiries[AlphaSurface._node(tenors, node["tenor"])]
                node_rows.append({**node, "raw_alpha": node["alpha"],
                    "alpha": policy.at_contract(expiry, node["level"]*surface.ref_spot),
                    "strategy": name, "feature_date": date, "label_date": end, "segment": segment})
            if name in SR_KINDS or name.endswith("_ema"):
                for sl in surface.slices:
                    quote_rows.append(dict(feature_date=date, strategy=name, vol_date=sl.vol_date,
                        tau=sl.tau, atm_alpha=policy.at_contract(sl.vol_date, surface.ref_spot),
                        outside_alpha_tenors=sl.tau < policy.taus[0] or sl.tau > policy.taus[-1]))
            priced = priced_by_strategy[name]
            result = frame.copy()
            if priced is None:
                result["delta"], result["delta_stderr"] = frame.bs_delta, 0.
                result["beta_model"] = 0.
                result["effective_beta"] = 0.
            else:
                if not np.allclose(priced[["tenor", "level"]], frame[["tenor", "level"]]):
                    raise ValueError("new MC rows do not match book contracts")
                for column in priced.columns.difference(["tenor", "level"]):
                    result[column] = priced[column].to_numpy()
                result["effective_beta"] = result.beta_model - measured_one
                audits.append({"feature_date": date, "strategy": name,
                    **priced.filter(regex="fraction$|^grid_|^n_paths$|^max_n_steps$").max().to_dict(),
                    "max_delta_stderr": float(priced.delta_stderr.max()),
                    "iv_inversion_clipped_cells": int(priced.price_clipped_for_inversion.sum())})
            result["delta_fallback"] = ~np.isfinite(result.delta)
            result.loc[result.delta_fallback, "delta"] = result.loc[result.delta_fallback, "bs_delta"]
            result.loc[result.delta_fallback, "effective_beta"] = 0.
            result["strategy"], result["segment"] = name, segment
            result["alpha"] = [policy.at_contract(r.expiry, r.strike) for r in frame.itertuples()]
            result["profile_has_inverse_fallback"] = bool(used.inverse_fallback.any())
            result["predicted_beta"] = ([forecast.loc[(r.tenor, r.level)] for r in frame.itertuples()]
                                        if forecast is not None else result.effective_beta)
            result["beta_mapping_gap"] = result.effective_beta-result.predicted_beta
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
            result["mc_delta_attribution_residual"] = (result.dV-result.delta_pnl-result.gamma_pnl
                                                       -result.theta_pnl-result.term_roll_pnl)
            result["raw_hedge_error"] = result.dV-result.delta_pnl
            result["carry_hedge_error"] = hedge_error(result.dV, result.delta, result.dS,
                result.pv, result.spot, result.dt_r, result.rate, result.income_yield)
            weighted = lambda col: float((result[col]*result.quantity).sum(skipna=False))
            previous = books.get(name, {"wealth": 0., "offset": 0., "delta": 0.})
            if start:
                previous = {"wealth": 0., "offset": previous["offset"]+previous["wealth"], "delta": 0.}
            first = result.iloc[0]
            account = cash_account(wealth=previous["wealth"], pv=weighted("pv"), next_pv=weighted("next_pv"),
                delta=weighted("delta"), spot=first.spot, next_spot=first.next_spot, dt=first.dt_r,
                rate=first.rate, income_yield=first.income_yield, previous_delta=previous["delta"],
                cost_bps=s.hedge_cost_bps, liquidate=finish)
            book_rows.append({"feature_date": date, "label_date": end, "strategy": name,
                "segment": segment, "segment_start": start, "segment_end": finish,
                "book_delta": weighted("delta"), "option_pv": weighted("pv"),
                "next_option_pv": weighted("next_pv"), "n_options": len(result),
                "delta_fallback_fraction": float(result.delta_fallback.mean()),
                **{col: weighted(col) for col in ("dV", "delta_pnl", "hedge_pnl", "bs_delta_pnl",
                    "gamma_pnl", "theta_pnl", "term_roll_pnl", "forecast_vega_pnl", "model_vega_pnl",
                    "raw_hedge_error", "carry_hedge_error", "attribution_residual", "bs_attribution_residual",
                    "forecast_attribution_residual", "raw_model_attribution_residual", "anchor_correction_pnl")},
                **account, "segment_wealth": account["wealth"], "wealth": previous["offset"]+account["wealth"]})
            books[name] = {"wealth": account["wealth"], "offset": previous["offset"], "delta": weighted("delta")}
            option_rows.append(result)
        validation.update(completed_test_dates=n, active_date=str(date), test_segments=segment)
        save_status()
    frames = {"option_pnl": pd.concat(option_rows, ignore_index=True), "book_pnl": pd.DataFrame(book_rows),
              "alpha_nodes": pd.DataFrame(node_rows), "mc_audit": pd.DataFrame(audits),
              "quote_sr": pd.DataFrame(quote_rows), "training_fixed": training_frame}
    frames["summary"] = _summary(frames["book_pnl"], frames["alpha_nodes"])
    for name, frame in frames.items():
        _csv(frame, outdir / f"{name}.csv")
    return frames

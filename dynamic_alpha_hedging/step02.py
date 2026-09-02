"""Dynamic Alpha Step 2: empirical beta on the rolling surface grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .artifacts import file_sha256, write_manifest
from .config import DynamicAlphaConfig


REQUIRED_CHANGE_COLUMNS = {
    "observation_date", "previous_date", "tenor", "level", "dlogS",
    "iv_previous", "iv_current", "dIV_grid", "smile_crossing_iv",
    "dIV_surface", "is_next_business_observation", "previous_atm_iv",
}


@dataclass
class Step2Result:
    """Raw grid beta and the primary skew-adjusted surface beta."""

    config: DynamicAlphaConfig
    daily: pd.DataFrame
    rolling: pd.DataFrame
    summary: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
    rolling_sensitivity: pd.DataFrame
    rolling_sensitivity_summary: pd.DataFrame
    reasonableness: pd.DataFrame
    regime_checks: pd.DataFrame
    term_structure: pd.DataFrame
    validation: dict[str, object]


def load_step1_changes(path: str | Path) -> pd.DataFrame:
    changes = pd.read_csv(path)
    date_columns = (
        "observation_date", "previous_date", "previous_actual_expiry",
        "current_actual_expiry",
    )
    for column in date_columns:
        if column in changes:
            changes[column] = pd.to_datetime(changes[column]).dt.date
    return changes


def _validate_input(changes: pd.DataFrame) -> None:
    missing = REQUIRED_CHANGE_COLUMNS.difference(changes.columns)
    if missing:
        raise ValueError(f"Step 1 grid-change file misses {sorted(missing)}")
    duplicates = changes.duplicated(["observation_date", "tenor", "level"])
    if duplicates.any():
        raise ValueError(
            f"Step 1 changes contain {int(duplicates.sum())} duplicate keys")
    if not np.allclose(
            changes["dIV_grid"],
            changes["iv_current"] - changes["iv_previous"],
            equal_nan=True):
        raise ValueError("dIV_grid is inconsistent with the two grid IVs")
    if not np.allclose(
            changes["dIV_surface"],
            changes["dIV_grid"] - changes["smile_crossing_iv"],
            equal_nan=True):
        raise ValueError("dIV_surface is inconsistent with the skew decomposition")


def _daily_beta(changes: pd.DataFrame, threshold: float,
                require_consecutive: bool) -> pd.DataFrame:
    daily = changes.copy()
    usable = (daily["dlogS"].abs() >= threshold) & daily["dlogS"].notna()
    if require_consecutive:
        usable &= daily["is_next_business_observation"].astype(bool)
    grid_usable = usable & daily["dIV_grid"].notna()
    surface_usable = usable & daily["dIV_surface"].notna()
    daily["beta_grid_raw_daily"] = np.where(
        grid_usable, -daily["dIV_grid"] / daily["dlogS"], np.nan)
    daily["beta_surface_daily"] = np.where(
        surface_usable, -daily["dIV_surface"] / daily["dlogS"], np.nan)
    daily["daily_ratio_usable"] = surface_usable
    return daily


def _fit_beta(sample: pd.DataFrame, y: str) -> dict[str, float]:
    fit = stats.linregress(sample["dlogS"], sample[y])
    return {
        "beta": float(-fit.slope),
        "intercept": float(fit.intercept),
        "r_squared": float(fit.rvalue ** 2),
        "slope_stderr": float(fit.stderr),
    }


def _rolling_beta(changes: pd.DataFrame, window: int,
                  min_obs: int, require_consecutive: bool) -> pd.DataFrame:
    records: list[dict] = []
    keys = ["tenor", "level"]
    ordered = changes.sort_values([*keys, "observation_date"])

    for (tenor, level), group in ordered.groupby(keys, sort=True):
        group = group.reset_index(drop=True)
        eligible = group[["dlogS", "dIV_grid", "dIV_surface"]].notna().all(axis=1)
        if require_consecutive:
            eligible &= group["is_next_business_observation"].astype(bool)
        valid_so_far: list[int] = []
        for end in range(len(group)):
            if bool(eligible.iloc[end]):
                valid_so_far.append(end)
            positions = valid_so_far[-window:]
            sample = group.iloc[positions][
                ["dlogS", "dIV_grid", "dIV_surface"]]
            row = group.iloc[end]
            record = {
                "observation_date": row["observation_date"],
                "tenor": float(tenor),
                "level": float(level),
                "window_start_date": (group.iloc[positions[0]]["observation_date"]
                                      if positions else np.nan),
                "window": int(window),
                "nobs": int(len(sample)),
                "current_observation_usable": bool(eligible.iloc[end]),
                "beta_grid_raw_rolling": np.nan,
                "grid_intercept": np.nan,
                "grid_r_squared": np.nan,
                "grid_slope_stderr": np.nan,
                "beta_surface_rolling": np.nan,
                "surface_intercept": np.nan,
                "surface_r_squared": np.nan,
                "surface_slope_stderr": np.nan,
            }
            if (bool(eligible.iloc[end]) and len(sample) >= min_obs
                    and sample["dlogS"].nunique() >= 2):
                grid = _fit_beta(sample, "dIV_grid")
                surface = _fit_beta(sample, "dIV_surface")
                record.update({
                    "beta_grid_raw_rolling": grid["beta"],
                    "grid_intercept": grid["intercept"],
                    "grid_r_squared": grid["r_squared"],
                    "grid_slope_stderr": grid["slope_stderr"],
                    "beta_surface_rolling": surface["beta"],
                    "surface_intercept": surface["intercept"],
                    "surface_r_squared": surface["r_squared"],
                    "surface_slope_stderr": surface["slope_stderr"],
                })
            records.append(record)
    return pd.DataFrame(records)


def _summary(daily: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    daily_summary = (daily.groupby(["tenor", "level"])
                     [["beta_grid_raw_daily", "beta_surface_daily"]]
                     .agg(["count", "mean", "std"]))
    daily_summary.columns = ["_".join(column) for column in daily_summary.columns]
    rolling_summary = (rolling.groupby(["tenor", "level"])
                       [["beta_grid_raw_rolling", "beta_surface_rolling"]]
                       .agg(["count", "mean", "std"]))
    rolling_summary.columns = [
        "_".join(column) for column in rolling_summary.columns]
    return daily_summary.reset_index().merge(
        rolling_summary.reset_index(), on=["tenor", "level"], how="outer")


def _distribution_rows(frame: pd.DataFrame, value_column: str, *,
                       estimator: str, extra: dict | None = None) -> pd.DataFrame:
    """Distribution diagnostics by surface cell for one beta estimate."""
    rows: list[dict] = []
    extra = {} if extra is None else extra
    for (tenor, level), group in frame.groupby(["tenor", "level"]):
        values = group[value_column].replace([np.inf, -np.inf], np.nan).dropna()
        row = {
            "estimator": estimator,
            "tenor": float(tenor),
            "level": float(level),
            "count": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "q01": float(values.quantile(0.01)),
            "q05": float(values.quantile(0.05)),
            "q95": float(values.quantile(0.95)),
            "q99": float(values.quantile(0.99)),
            "max": float(values.max()),
            "positive_fraction": float((values > 0).mean()),
        }
        row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def _threshold_sensitivity(changes: pd.DataFrame,
                           config: DynamicAlphaConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    consecutive = changes["is_next_business_observation"].astype(bool)
    for threshold in config.beta_threshold_sensitivity:
        usable = (consecutive & changes["dlogS"].notna()
                  & changes["dIV_surface"].notna()
                  & (changes["dlogS"].abs() >= threshold))
        sample = changes.loc[usable, ["tenor", "level"]].copy()
        sample["beta_surface_daily"] = (
            -changes.loc[usable, "dIV_surface"]
            / changes.loc[usable, "dlogS"])
        rows.append(_distribution_rows(
            sample, "beta_surface_daily", estimator="daily_ratio",
            extra={"threshold": float(threshold)}))
    return pd.concat(rows, ignore_index=True)


def _weighted_fit(x: np.ndarray, y: np.ndarray,
                  weights: np.ndarray) -> dict[str, float] | None:
    sw = float(weights.sum())
    if sw <= 0:
        return None
    xbar = float(np.dot(weights, x) / sw)
    ybar = float(np.dot(weights, y) / sw)
    centered_x = x - xbar
    centered_y = y - ybar
    sxx = float(np.dot(weights, centered_x * centered_x))
    if sxx <= 0:
        return None
    slope = float(np.dot(weights, centered_x * centered_y) / sxx)
    intercept = ybar - slope * xbar
    residual = y - (intercept + slope * x)
    sse = float(np.dot(weights, residual * residual))
    sst = float(np.dot(weights, centered_y * centered_y))
    r_squared = 1.0 - sse / sst if sst > 0 else np.nan
    effective_n = sw * sw / float(np.dot(weights, weights))
    return {
        "beta_surface": -slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "effective_n": effective_n,
    }


def _rolling_sensitivity(changes: pd.DataFrame,
                         config: DynamicAlphaConfig) -> pd.DataFrame:
    """Rolling surface-beta paths for the document's smoothing choices."""
    settings = {(int(window), "equal")
                for window in config.beta_window_sensitivity}
    settings.update((int(config.beta_window), weighting)
                    for weighting in config.beta_weight_sensitivity)
    records: list[dict] = []
    ordered = changes.sort_values(["tenor", "level", "observation_date"])

    for (tenor, level), group in ordered.groupby(["tenor", "level"], sort=True):
        group = group.reset_index(drop=True)
        x_all = group["dlogS"].to_numpy(dtype=float)
        y_all = group["dIV_surface"].to_numpy(dtype=float)
        consecutive = group["is_next_business_observation"].to_numpy(dtype=bool)
        dates = group["observation_date"].to_numpy()
        for window, weighting in sorted(settings):
            min_obs = min(config.beta_min_obs, window)
            eligible = (consecutive & np.isfinite(x_all) & np.isfinite(y_all))
            valid_so_far: list[int] = []
            for end in range(len(group)):
                if eligible[end]:
                    valid_so_far.append(end)
                if not eligible[end]:
                    continue
                positions = np.asarray(valid_so_far[-window:], dtype=int)
                if len(positions) < min_obs:
                    continue
                x = x_all[positions]
                y = y_all[positions]
                if weighting == "equal":
                    weights = np.ones(len(positions), dtype=float)
                elif weighting == "abs_dlogS":
                    weights = np.abs(x)
                else:
                    age = end - positions
                    weights = np.power(
                        0.5, age / config.beta_time_decay_half_life)
                fit = _weighted_fit(x, y, weights)
                if fit is None:
                    continue
                records.append({
                    "observation_date": dates[end],
                    "tenor": float(tenor),
                    "level": float(level),
                    "window": int(window),
                    "weighting": weighting,
                    "nobs": int(len(positions)),
                    **fit,
                })
    return pd.DataFrame(records)


def _rolling_sensitivity_summary(sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["tenor", "level", "window", "weighting"]
    for key, group in sensitivity.groupby(keys):
        values = group["beta_surface"].dropna()
        rows.append({
            **dict(zip(keys, key)),
            "count": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "positive_fraction": float((values > 0).mean()),
            "median_r_squared": float(group["r_squared"].median()),
            "median_effective_n": float(group["effective_n"].median()),
        })
    return pd.DataFrame(rows)


def _regime_checks(changes: pd.DataFrame,
                   config: DynamicAlphaConfig) -> pd.DataFrame:
    """Down/up and prior-ATM-IV high/low conditional beta checks."""
    base = changes[
        changes["is_next_business_observation"].astype(bool)
        & changes[["dlogS", "dIV_surface"]].notna().all(axis=1)].copy()
    base["spot_direction"] = np.where(base["dlogS"] < 0, "down", "up")
    median_iv = base.groupby("tenor")["previous_atm_iv"].transform("median")
    base["vol_regime"] = np.where(
        base["previous_atm_iv"] >= median_iv, "high", "low")
    rows: list[dict] = []

    for regime_type in ("spot_direction", "vol_regime"):
        for (regime, tenor, level), group in base.groupby(
                [regime_type, "tenor", "level"]):
            regression = group[["dlogS", "dIV_surface"]].dropna()
            fit_values = {
                "beta_regression": np.nan,
                "intercept": np.nan,
                "r_squared": np.nan,
                "slope_stderr": np.nan,
            }
            if (len(regression) >= config.beta_min_obs
                    and regression["dlogS"].nunique() >= 2):
                fit = _fit_beta(regression, "dIV_surface")
                fit_values = {
                    "beta_regression": fit["beta"],
                    "intercept": fit["intercept"],
                    "r_squared": fit["r_squared"],
                    "slope_stderr": fit["slope_stderr"],
                }
            ratio_sample = group[
                group["dlogS"].abs() >= config.beta_min_abs_dlogS]
            ratios = (-ratio_sample["dIV_surface"]
                      / ratio_sample["dlogS"]).replace(
                          [np.inf, -np.inf], np.nan).dropna()
            rows.append({
                "regime_type": regime_type,
                "regime": regime,
                "tenor": float(tenor),
                "level": float(level),
                "regression_nobs": int(len(regression)),
                "daily_ratio_count": int(len(ratios)),
                "daily_ratio_mean": float(ratios.mean()),
                "daily_ratio_median": float(ratios.median()),
                "daily_positive_fraction": float((ratios > 0).mean()),
                **fit_values,
            })
    return pd.DataFrame(rows)


def _term_structure(rolling: pd.DataFrame) -> pd.DataFrame:
    atm = rolling[np.isclose(rolling["level"], 1.0)]
    rows: list[dict] = []
    for tenor, group in atm.groupby("tenor"):
        values = group["beta_surface_rolling"].dropna()
        rows.append({
            "tenor": float(tenor),
            "count": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "positive_fraction": float((values > 0).mean()),
            "median_r_squared": float(group["surface_r_squared"].median()),
            "median_slope_stderr": float(group["surface_slope_stderr"].median()),
        })
    return pd.DataFrame(rows)


def run_step2(changes: pd.DataFrame,
              config: DynamicAlphaConfig = DynamicAlphaConfig()) -> Step2Result:
    """Estimate diagnostic grid beta and primary surface-motion beta."""
    _validate_input(changes)
    daily = _daily_beta(
        changes, config.beta_min_abs_dlogS,
        config.beta_require_consecutive_business_days)
    rolling = _rolling_beta(
        changes, config.beta_window, config.beta_min_obs,
        config.beta_require_consecutive_business_days)
    summary = _summary(daily, rolling)
    threshold_sensitivity = _threshold_sensitivity(changes, config)
    rolling_sensitivity = _rolling_sensitivity(changes, config)
    rolling_sensitivity_summary = _rolling_sensitivity_summary(
        rolling_sensitivity)
    reasonableness = pd.concat([
        _distribution_rows(
            daily, "beta_surface_daily", estimator="daily_ratio"),
        _distribution_rows(
            rolling, "beta_surface_rolling", estimator="rolling_ols"),
    ], ignore_index=True)
    regime_checks = _regime_checks(changes, config)
    term_structure = _term_structure(rolling)
    validation = {
        "input_rows": int(len(changes)),
        "input_transitions": int(changes["observation_date"].nunique()),
        "daily_beta_rows": int(daily["beta_surface_daily"].notna().sum()),
        "rolling_beta_rows": int(
            rolling["beta_surface_rolling"].notna().sum()),
        "daily_threshold": float(config.beta_min_abs_dlogS),
        "require_consecutive_business_days": bool(
            config.beta_require_consecutive_business_days),
        "excluded_nonconsecutive_rows": int((
            ~changes["is_next_business_observation"].astype(bool)).sum()),
        "rolling_window": int(config.beta_window),
        "rolling_min_obs": int(config.beta_min_obs),
        "primary_beta": "beta_surface",
        "input_definition": (
            "same tenor and strike level across daily grids; previous-surface "
            "smile traversal removed"),
        "alpha_one_sanity_target": "beta_surface approximately zero",
    }
    return Step2Result(
        config, daily, rolling, summary, threshold_sensitivity,
        rolling_sensitivity, rolling_sensitivity_summary, reasonableness,
        regime_checks, term_structure, validation)


def save_step2(result: Step2Result, *, step1_changes_path: str | Path,
               outdir: str | Path = "output/dynamic_alpha/step02") -> Path:
    """Write canonical Step 2 artefacts and upstream hashes."""
    source = Path(step1_changes_path)
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(target / "beta_daily.csv", index=False)
    result.rolling.to_csv(target / "beta_rolling.csv", index=False)
    result.summary.to_csv(target / "summary.csv", index=False)
    result.threshold_sensitivity.to_csv(
        target / "beta_threshold_sensitivity.csv", index=False)
    result.rolling_sensitivity.to_csv(
        target / "beta_rolling_sensitivity.csv", index=False)
    result.rolling_sensitivity_summary.to_csv(
        target / "beta_rolling_sensitivity_summary.csv", index=False)
    result.reasonableness.to_csv(
        target / "beta_reasonableness.csv", index=False)
    result.regime_checks.to_csv(
        target / "beta_regime_checks.csv", index=False)
    result.term_structure.to_csv(
        target / "beta_term_structure.csv", index=False)
    inputs = {
        "step01_grid_changes": str(source),
        "step01_grid_changes_sha256": file_sha256(source),
    }
    step1_manifest = source.parent / "manifest.json"
    if step1_manifest.exists():
        inputs["step01_manifest"] = str(step1_manifest)
        inputs["step01_manifest_sha256"] = file_sha256(step1_manifest)
    return write_manifest(
        target / "manifest.json",
        stage="dynamic_alpha_step02",
        config=result.config,
        inputs=inputs,
        validation=result.validation,
    )

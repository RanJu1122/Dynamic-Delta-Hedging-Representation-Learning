"""Dynamic Alpha Step 2: empirical beta on the rolling surface grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
    summary: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
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
    usable = (daily["dlogS"].abs() >= threshold) & np.isfinite(daily["dlogS"]) & daily["dlogS"].ne(0)
    if require_consecutive:
        usable &= daily["is_next_business_observation"].astype(bool)
    grid_usable = usable & daily["dIV_grid"].notna()
    surface_usable = usable & daily["dIV_surface"].notna()
    # Divide only where identified; zero-return days retain dIV, never fabricate Beta=0.
    for column, numerator, mask in (("beta_grid_raw_daily", "dIV_grid", grid_usable),
                                     ("beta_surface_daily", "dIV_surface", surface_usable)):
        daily[column] = np.divide(-daily[numerator].to_numpy(float),
            daily["dlogS"].to_numpy(float), out=np.full(len(daily), np.nan), where=mask.to_numpy())
    daily["zero_spot_return"] = daily.dlogS.eq(0)
    daily["beta_unavailable_reason"] = np.select(
        [daily.zero_spot_return, ~np.isfinite(daily.dlogS),
         ~daily.is_next_business_observation.astype(bool) & require_consecutive,
         daily.dlogS.abs().lt(threshold), ~np.isfinite(daily.dIV_surface)],
        ["zero_spot_return", "nonfinite_return", "nonconsecutive_observations",
         "below_return_threshold", "unavailable_surface_change"], default="")
    daily["daily_ratio_usable"] = surface_usable
    return daily


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
        usable = (consecutive & np.isfinite(changes["dlogS"]) & changes["dlogS"].ne(0)
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


def _summary(daily):
    result = daily.groupby(["tenor", "level"])[["beta_grid_raw_daily", "beta_surface_daily"]].agg(["count", "mean", "std"])
    result.columns = ["_".join(column) for column in result.columns]
    return result.reset_index()


def _regime_checks(daily):
    base = daily[daily.daily_ratio_usable].copy()
    base["spot_direction"] = np.where(base.dlogS < 0, "down", "up")
    median_iv = base.groupby("tenor").previous_atm_iv.transform("median")
    base["vol_regime"] = np.where(base.previous_atm_iv >= median_iv, "high", "low")
    rows = []
    for regime_type in ("spot_direction", "vol_regime"):
        for (regime, tenor, level), group in base.groupby([regime_type, "tenor", "level"]):
            beta = group.beta_surface_daily
            rows.append(dict(regime_type=regime_type, regime=regime, tenor=tenor, level=level,
                             daily_ratio_count=len(beta), daily_ratio_mean=beta.mean(),
                             daily_ratio_median=beta.median(), daily_positive_fraction=(beta > 0).mean()))
    return pd.DataFrame(rows)


def _term_structure(daily):
    atm = daily[np.isclose(daily.level, 1.)]
    return _distribution_rows(atm, "beta_surface_daily", estimator="daily_ratio")


def run_step2(changes, config=DynamicAlphaConfig()):
    """Daily-ratio estimator only; no regression-beta feature or benchmark."""
    from .grid import validate_cells
    _validate_input(changes)
    validate_cells(changes, config.tenors, config.strike_levels, source="Step 1 changes")
    daily = _daily_beta(changes, config.beta_min_abs_dlogS, config.beta_require_consecutive_business_days)
    validation = {
        "beta_workflow": "daily_only_v1",
        "input_rows": len(changes), "input_transitions": changes.observation_date.nunique(),
        "daily_beta_rows": int(daily.beta_surface_daily.notna().sum()),
        "zero_spot_return_rows": int(daily.zero_spot_return.sum()),
        "zero_spot_nonzero_iv_change_rows": int((daily.zero_spot_return &
            daily.dIV_surface.notna() & daily.dIV_surface.ne(0)).sum()),
        "zero_spot_policy": "keep snapshot and IV change; ratio Beta undefined; retain hedge intervals",
        "daily_threshold": config.beta_min_abs_dlogS,
        "require_consecutive_business_days": config.beta_require_consecutive_business_days,
        "excluded_nonconsecutive_rows": int((~changes.is_next_business_observation.astype(bool)).sum()),
        "primary_beta": "beta_surface", "estimator": "daily_ratio",
        "input_definition": "same tenor and strike level across daily grids; previous-surface smile traversal removed",
        "alpha_one_sanity_target": "beta_surface approximately zero",
    }
    return Step2Result(config, daily, _summary(daily), _threshold_sensitivity(changes, config),
                       _distribution_rows(daily, "beta_surface_daily", estimator="daily_ratio"),
                       _regime_checks(daily), _term_structure(daily), validation)


def save_step2(result: Step2Result, *, step1_changes_path: str | Path,
               outdir: str | Path = "output/dynamic_alpha/step02") -> Path:
    """Write canonical Step 2 artefacts and upstream hashes."""
    source = Path(step1_changes_path)
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    for obsolete in ("beta_rolling.csv", "beta_rolling_sensitivity.csv", "beta_rolling_sensitivity_summary.csv"):
        (target / obsolete).unlink(missing_ok=True)
    result.daily.to_csv(target / "beta_daily.csv", index=False)
    result.summary.to_csv(target / "summary.csv", index=False)
    result.threshold_sensitivity.to_csv(
        target / "beta_threshold_sensitivity.csv", index=False)
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

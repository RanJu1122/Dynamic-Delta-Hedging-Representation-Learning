"""Dynamic Alpha Step 1: daily IV grid and surface-motion decomposition.

The research coordinate is a rolling ``(tenor, K / spot)`` grid.  Therefore a
cell-to-cell daily IV change contains both genuine surface motion and the
mechanical traversal of the previous day's smile as ``K = level * spot`` moves.
Step 1 records both and removes the latter with a no-look-ahead counterfactual
on the previous surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from svi_localvol.conventions import nb_biz_days

from .artifacts import file_sha256, write_manifest
from .config import DynamicAlphaConfig
from .data_loader import (ImpliedVolPanel, SurfaceHistory, implied_vol_panel,
                          load_surface_history, raw_quote_frame)


@dataclass
class Step1Result:
    """Canonical Step 1 output."""

    config: DynamicAlphaConfig
    iv_state: ImpliedVolPanel
    grid_changes: pd.DataFrame
    skipped_observations: pd.DataFrame
    summary: pd.DataFrame
    validation: dict[str, object]


def _grid_changes(panel: ImpliedVolPanel, history: SurfaceHistory,
                  config: DynamicAlphaConfig) -> pd.DataFrame:
    """Decompose adjacent changes on the fixed ``(tenor, level)`` grid.

    For a cell whose strike changes from ``K0`` to ``K1``:

    ``dIV_grid = IV_t(tau, K1) - IV_t-1(tau, K0)``

    ``smile_crossing = IV_t-1(tau, K1) - IV_t-1(tau, K0)``

    ``dIV_surface = dIV_grid - smile_crossing``

    The counterfactual uses only the previous surface.  It is the finite-move
    equivalent of subtracting ``(dIV/dlogK) * dlogS`` and makes a sticky-strike
    surface produce ``dIV_surface ~= 0`` without defining the research panel by
    a fixed option contract.
    """
    rows: list[dict] = []
    levels = np.asarray(panel.levels, dtype=float)
    atm_indices = np.flatnonzero(np.isclose(levels, 1.0))
    atm_index = int(atm_indices[0]) if len(atm_indices) else None

    for current_index in range(1, len(panel.dates)):
        previous_index = current_index - 1
        previous_date = panel.dates[previous_index]
        observation_date = panel.dates[current_index]
        previous_surface = history[previous_date]
        previous_spot = float(previous_surface.ref_spot)
        current_spot = float(history[observation_date].ref_spot)
        dlog_spot = float(np.log(current_spot / previous_spot))
        calendar_days = (observation_date - previous_date).days
        business_days = nb_biz_days(
            previous_date, observation_date, config.holidays)

        for tenor_index, tenor in enumerate(panel.expiries):
            previous_expiry = panel.vol_dates[previous_index, tenor_index]
            current_expiry = panel.vol_dates[current_index, tenor_index]
            previous_strikes = panel.strikes[previous_index, tenor_index]
            current_strikes = panel.strikes[current_index, tenor_index]
            iv_previous = panel.iv[previous_index, tenor_index]
            iv_current = panel.iv[current_index, tenor_index]
            previous_atm_iv = (float(iv_previous[atm_index])
                               if atm_index is not None else np.nan)
            current_atm_iv = (float(iv_current[atm_index])
                              if atm_index is not None else np.nan)

            supported = np.isfinite(iv_previous) & np.isfinite(iv_current)
            smile_slopes = np.full(levels.shape, np.nan)
            counterfactual_iv = np.full(levels.shape, np.nan)
            if supported.any():
                query_strikes = current_strikes[supported]
                counterfactual_iv[supported] = np.asarray(
                    previous_surface.implied_vol(previous_expiry, query_strikes),
                    dtype=float,
                )
                smile_slopes[supported] = np.asarray(
                    previous_surface.implied_vol_log_strike_slope(
                        previous_expiry, previous_strikes[supported]),
                    dtype=float,
                )

            for level_index, level in enumerate(levels):
                old_iv = float(iv_previous[level_index])
                new_iv = float(iv_current[level_index])
                old_strike = float(previous_strikes[level_index])
                new_strike = float(current_strikes[level_index])
                dlog_strike = float(np.log(new_strike / old_strike))
                grid_change = new_iv - old_iv
                crossing = float(counterfactual_iv[level_index] - old_iv)
                surface_change = grid_change - crossing
                rows.append({
                    "observation_date": observation_date,
                    "previous_date": previous_date,
                    "calendar_days": int(calendar_days),
                    "business_days": int(business_days),
                    "is_next_business_observation": bool(business_days == 1),
                    "tenor": float(tenor),
                    "level": float(level),
                    "previous_actual_expiry": previous_expiry,
                    "current_actual_expiry": current_expiry,
                    "previous_spot": previous_spot,
                    "current_spot": current_spot,
                    "previous_strike": old_strike,
                    "current_strike": new_strike,
                    "dlogS": dlog_spot,
                    "dlogK": dlog_strike,
                    "iv_previous": old_iv,
                    "iv_current": new_iv,
                    "previous_atm_iv": previous_atm_iv,
                    "current_atm_iv": current_atm_iv,
                    "dIV_grid": grid_change,
                    "smile_slope_logK_previous": float(
                        smile_slopes[level_index]),
                    "iv_previous_at_current_strike": float(
                        counterfactual_iv[level_index]),
                    "smile_crossing_iv": crossing,
                    "dIV_surface": surface_change,
                })

    return pd.DataFrame(rows)


def _summary(changes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (tenor, level), group in changes.groupby(["tenor", "level"]):
        for variable in ("dIV_grid", "smile_crossing_iv", "dIV_surface"):
            valid = group[[variable, "dlogS"]].dropna()
            values = valid[variable]
            rows.append({
                "variable": variable,
                "tenor": float(tenor),
                "level": float(level),
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "corr_dlogS": (float(valid[variable].corr(valid["dlogS"]))
                                if len(valid) >= 2 else np.nan),
            })
    return pd.DataFrame(rows)


def _validate(result: Step1Result) -> dict[str, object]:
    changes = result.grid_changes
    valid = changes[["dIV_grid", "smile_crossing_iv", "dIV_surface",
                     "dlogS"]].notna().all(axis=1)
    previous_finite = changes["previous_strike"].notna()
    current_finite = changes["current_strike"].notna()
    log_finite = changes["dlogK"].notna()
    previous_anchor = np.allclose(
        changes.loc[previous_finite, "previous_strike"],
        (changes.loc[previous_finite, "level"]
         * changes.loc[previous_finite, "previous_spot"]))
    current_anchor = np.allclose(
        changes.loc[current_finite, "current_strike"],
        (changes.loc[current_finite, "level"]
         * changes.loc[current_finite, "current_spot"]))
    log_move = np.allclose(
        changes.loc[log_finite, "dlogK"],
        changes.loc[log_finite, "dlogS"])
    decomposition = np.allclose(
        changes.loc[valid, "dIV_surface"],
        changes.loc[valid, "dIV_grid"]
        - changes.loc[valid, "smile_crossing_iv"],
    )
    return {
        "n_state_rows": int(np.prod(result.iv_state.shape)),
        "n_skipped_observations": int(len(result.skipped_observations)),
        "n_transitions": int(changes["observation_date"].nunique()),
        "n_grid_change_rows": int(len(changes)),
        "n_valid_grid_change_rows": int(valid.sum()),
        "n_nonconsecutive_business_transitions": int(
            (~changes[["observation_date", "is_next_business_observation"]]
             .drop_duplicates()["is_next_business_observation"]).sum()),
        "rolling_spot_anchor_pass": bool(previous_anchor and current_anchor),
        "dlogK_equals_dlogS_pass": bool(log_move),
        "surface_decomposition_pass": bool(decomposition),
        "skew_control_information_set": "previous surface only",
        "date_policy": (
            "Asia/Shanghai timestamps convert to America/New_York; "
            "date-only keys pass through unchanged; weekends rejected"),
    }


def run_step1(config: DynamicAlphaConfig = DynamicAlphaConfig(), *,
              history: SurfaceHistory | None = None) -> Step1Result:
    """Build the sole Step 1 grid and its no-look-ahead decomposition."""
    if history is None:
        history = load_surface_history(
            config.data_path,
            config.market_conventions,
            beta_clamp=config.beta_clamp,
            duplicate_vol_date_policy=config.duplicate_vol_date_policy,
            source_timezone=config.source_timezone,
            market_timezone=config.market_timezone,
        )
    if len(history) < 2:
        raise ValueError("Step 1 requires at least two calibrated surfaces")
    if not config.allow_skipped_surfaces and not history.skipped.empty:
        raise ValueError(
            f"Step 1 refuses a history that skipped {len(history.skipped)} "
            "observation(s). Resolve preflight blockers before generating "
            "official research output.")
    weekend_dates = [date for date in history.dates if date.weekday() >= 5]
    if config.require_weekday_observations and weekend_dates:
        raise ValueError(
            f"Step 1 expects Monday-Friday market dates; found "
            f"{len(weekend_dates)} weekend observation(s). Run preflight and "
            "replace the input file before research output is generated.")

    state = implied_vol_panel(
        history,
        levels=config.strike_levels,
        expiry_axis=config.expiry_axis,
        tenors=config.tenors,
        level_anchor=config.level_anchor,
        extrapolation=config.extrapolation,
    )
    changes = _grid_changes(state, history, config)
    result = Step1Result(
        config, state, changes, history.skipped.copy(), _summary(changes), {})
    result.validation = _validate(result)
    return result


def save_step1(result: Step1Result,
               outdir: str | Path = "output/dynamic_alpha/step01") -> Path:
    """Write canonical Step 1 artefacts and their provenance manifest."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    raw_quotes = raw_quote_frame(
        result.config.data_path,
        source_timezone=result.config.source_timezone,
        market_timezone=result.config.market_timezone,
        holidays=result.config.holidays,
    )
    raw_quotes.to_csv(target / "raw_svi_quotes.csv", index=False)
    result.iv_state.to_frame().to_csv(target / "iv_state.csv", index=False)
    result.grid_changes.to_csv(target / "grid_changes.csv", index=False)
    result.skipped_observations.to_csv(
        target / "skipped_observations.csv", index=False)
    result.summary.to_csv(target / "summary.csv", index=False)
    result.iv_state.coverage_frame().to_csv(
        target / "tenor_coverage.csv", index=False)
    result.validation.update({
        "n_raw_quote_rows": int(len(raw_quotes)),
        "n_raw_observations": int(raw_quotes["source_key"].nunique()),
        "raw_observations_with_time": int(
            raw_quotes.loc[raw_quotes["source_has_time"], "source_key"].nunique()),
    })
    return write_manifest(
        target / "manifest.json",
        stage="dynamic_alpha_step01",
        config=result.config,
        inputs={
            "svi_parameters": str(result.config.data_path),
            "svi_parameters_sha256": file_sha256(result.config.data_path),
        },
        validation=result.validation,
    )

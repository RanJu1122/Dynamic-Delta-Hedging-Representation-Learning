"""Dynamic Alpha Step 5: next-period beta predictability diagnostics.

The primary target is the next observed day's realised ``beta_surface_daily``.
The latest known daily beta at close t is used as a persistence
feature and benchmark; no regression-beta input is used.  Every predictive row uses state available through
close t.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import file_sha256, write_manifest
from .config import DynamicAlphaConfig


FACTOR_COLUMNS = ("atm_beta_factor", "shape_score_1", "shape_score_2")
DIAGNOSTIC_FACTOR_COLUMNS = FACTOR_COLUMNS
STATE_FEATURES = (
    "dlogS", "atm_iv_3m", "atm_iv_change_1d", "atm_iv_change_5d",
    "smile_slope_3m", "term_slope_1y_minus_3m", "realized_vol_20d",
    "recent_return_5d", "recent_return_20d", "vol_of_vol_20d",
)
TRAIN_FRACTION = 0.75
PREDICTION_FEATURES = ("last_daily_beta", *STATE_FEATURES)
PREDICTION_MODELS = (
    "sticky_strike", "training_mean", "last_observed_beta", "ridge_state",
    "hist_gradient_boosting",
)


@dataclass
class Step5Result:
    """Daily state panel and the diagnostics that decide whether Step 6 runs."""

    config: DynamicAlphaConfig
    factor_state_panel: pd.DataFrame
    factor_acf: pd.DataFrame
    spot_regression: pd.DataFrame
    state_correlations: pd.DataFrame
    daily_beta_predictions: pd.DataFrame
    daily_beta_model_summary: pd.DataFrame
    prediction_splits: pd.DataFrame
    attribution_baseline: pd.DataFrame
    validation: dict[str, object]


def _read_csv(path: str | Path, date_columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in date_columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column]).dt.date
    return frame


def load_step5_inputs(*, factor_scores_path: str | Path,
                      factor_loadings_path: str | Path,
                      iv_state_path: str | Path,
                      grid_changes_path: str | Path,
                      daily_beta_path: str | Path,
                      ) -> tuple[pd.DataFrame, pd.DataFrame,
                                 pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factors = _read_csv(factor_scores_path, ("observation_date",))
    loadings = _read_csv(factor_loadings_path, ())
    iv_state = _read_csv(iv_state_path, ("date", "actual_expiry"))
    changes = _read_csv(
        grid_changes_path,
        ("observation_date", "previous_date", "previous_actual_expiry",
         "current_actual_expiry"))
    daily_beta = _read_csv(
        daily_beta_path, ("observation_date", "previous_date"))
    return factors, loadings, iv_state, changes, daily_beta


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    mapped = values.astype(str).str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError("boolean column contains values other than true/false")
    return mapped.astype(bool)


def _one_daily_market_row(changes: pd.DataFrame) -> pd.DataFrame:
    required = {
        "observation_date", "previous_date", "is_next_business_observation",
        "current_spot", "dlogS",
    }
    missing = required.difference(changes.columns)
    if missing:
        raise ValueError(f"Step 1 changes miss {sorted(missing)}")
    check_columns = [
        "previous_date", "is_next_business_observation", "current_spot", "dlogS"]
    for column in check_columns:
        counts = changes.groupby("observation_date")[column].nunique(dropna=False)
        if (counts > 1).any():
            raise ValueError(f"Step 1 {column} is not unique within a date")
    daily = (changes.sort_values("observation_date")
             .groupby("observation_date", as_index=False)[check_columns].first())
    daily["is_next_business_observation"] = _as_bool(
        daily["is_next_business_observation"])
    daily = daily.rename(columns={"dlogS": "dlogS_raw"})
    daily["dlogS"] = daily["dlogS_raw"].where(
        daily["is_next_business_observation"])
    return daily


def _iv_series(iv_state: pd.DataFrame, tenor: float,
               level: float, name: str) -> pd.DataFrame:
    required = {"date", "expiry", "level", "implied_vol"}
    missing = required.difference(iv_state.columns)
    if missing:
        raise ValueError(f"Step 1 IV state misses {sorted(missing)}")
    selected = iv_state[
        np.isclose(iv_state["expiry"].astype(float), tenor, atol=1e-12)
        & np.isclose(iv_state["level"].astype(float), level, atol=1e-12)
    ][["date", "implied_vol"]].copy()
    if selected["date"].duplicated().any():
        raise ValueError(f"duplicate IV state rows for tenor={tenor}, level={level}")
    if selected.empty:
        raise ValueError(f"no IV state for tenor={tenor}, level={level}")
    return selected.rename(columns={"implied_vol": name})


def _all_consecutive(flag: pd.Series, window: int) -> pd.Series:
    return flag.astype(float).rolling(window, min_periods=window).sum().eq(window)


def _build_factor_state_panel(factors: pd.DataFrame, iv_state: pd.DataFrame,
                              changes: pd.DataFrame) -> pd.DataFrame:
    required_factors = {"observation_date", *DIAGNOSTIC_FACTOR_COLUMNS}
    missing = required_factors.difference(factors.columns)
    if missing:
        raise ValueError(f"Step 4 factor scores miss {sorted(missing)}")
    if factors["observation_date"].duplicated().any():
        raise ValueError("Step 4 factor scores contain duplicate dates")

    panel = _one_daily_market_row(changes)
    iv_parts = [
        _iv_series(iv_state, 0.25, 1.0, "atm_iv_3m"),
        _iv_series(iv_state, 0.25, 0.9, "iv_3m_level_09"),
        _iv_series(iv_state, 0.25, 1.1, "iv_3m_level_11"),
        _iv_series(iv_state, 1.0, 1.0, "atm_iv_1y"),
    ]
    for part in iv_parts:
        panel = panel.merge(
            part, left_on="observation_date", right_on="date",
            how="left", validate="one_to_one").drop(columns="date")

    panel = panel.sort_values("observation_date").reset_index(drop=True)
    panel["atm_iv_change_1d"] = panel["atm_iv_3m"].diff().where(
        panel["is_next_business_observation"])
    panel["smile_slope_3m"] = (
        (panel["iv_3m_level_11"] - panel["iv_3m_level_09"])
        / (np.log(1.1) - np.log(0.9)))
    panel["term_slope_1y_minus_3m"] = (
        panel["atm_iv_1y"] - panel["atm_iv_3m"])

    for window in (5, 20):
        valid_window = _all_consecutive(
            panel["is_next_business_observation"], window)
        panel[f"recent_return_{window}d"] = (
            panel["dlogS"].rolling(window, min_periods=window).sum()
            .where(valid_window))
    valid_5 = _all_consecutive(panel["is_next_business_observation"], 5)
    valid_20 = _all_consecutive(panel["is_next_business_observation"], 20)
    panel["atm_iv_change_5d"] = (
        panel["atm_iv_change_1d"].rolling(5, min_periods=5).sum()
        .where(valid_5))
    panel["realized_vol_20d"] = (
        panel["dlogS"].rolling(20, min_periods=20).std(ddof=1)
        * np.sqrt(260.0)).where(valid_20)
    panel["vol_of_vol_20d"] = (
        panel["atm_iv_change_1d"].rolling(20, min_periods=20).std(ddof=1)
        * np.sqrt(260.0)).where(valid_20)

    panel = panel.merge(
        factors, on="observation_date", how="left", validate="one_to_one")
    panel["factor_available"] = panel[list(FACTOR_COLUMNS)].notna().all(axis=1)
    panel["segment"] = (~panel["is_next_business_observation"]).cumsum()
    next_is_observation = (panel["previous_date"].shift(-1).eq(
        panel["observation_date"])
        & panel["is_next_business_observation"].shift(-1, fill_value=False))
    for factor in FACTOR_COLUMNS:
        # These are explicit Step 6 labels.  The shift is allowed only when the
        # following row really is the next market observation.
        panel[f"next_{factor}"] = panel[factor].shift(-1).where(
            next_is_observation)
    return panel


def _paired_correlation(x: pd.Series, y: pd.Series) -> tuple[int, float, float]:
    sample = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 3 or sample["x"].nunique() < 2 \
            or sample["y"].nunique() < 2:
        return len(sample), np.nan, np.nan
    pearson = float(sample["x"].corr(sample["y"]))
    spearman = float(sample["x"].rank().corr(sample["y"].rank()))
    return len(sample), pearson, spearman


def _simple_regression(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    sample = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 3 or sample["x"].nunique() < 2:
        return {
            "nobs": len(sample), "intercept": np.nan, "slope": np.nan,
            "r_squared": np.nan, "rmse": np.nan, "slope_stderr": np.nan,
        }
    design = np.column_stack([np.ones(len(sample)), sample["x"].to_numpy()])
    target = sample["y"].to_numpy()
    coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coef
    residual = target - fitted
    sse = float(residual @ residual)
    centered = target - target.mean()
    sst = float(centered @ centered)
    sigma2 = sse / (len(sample) - design.shape[1])
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    return {
        "nobs": len(sample),
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "r_squared": float(1.0 - sse / sst) if sst > 0.0 else np.nan,
        "rmse": float(np.sqrt(sse / len(sample))),
        "slope_stderr": float(np.sqrt(covariance[1, 1])),
    }


def _spot_regressions(panel: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    with_residuals = panel.copy()
    for target in DIAGNOSTIC_FACTOR_COLUMNS:
        masks = {
            "all": panel["dlogS"].notna(),
            "up": panel["dlogS"] > 0.0,
            "down": panel["dlogS"] < 0.0,
        }
        for sample_name, mask in masks.items():
            result = _simple_regression(
                panel.loc[mask, "dlogS"], panel.loc[mask, target])
            rows.append({"factor": target, "sample": sample_name, **result})
        overall = rows[-3]
        with_residuals[f"{target}_spot_residual"] = (
            panel[target]
            - overall["intercept"] - overall["slope"] * panel["dlogS"])
    return pd.DataFrame(rows), with_residuals


def _factor_acf(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for target in DIAGNOSTIC_FACTOR_COLUMNS:
        series_map = {
            "level": panel[target],
            "first_difference": panel[target].diff(),
            "spot_residual": panel[f"{target}_spot_residual"],
        }
        for transform, series in series_map.items():
            for lag in range(1, 11):
                n, correlation, _ = _paired_correlation(series, series.shift(lag))
                rows.append({
                    "factor": target,
                    "transform": transform,
                    "lag": lag,
                    "n_pairs": n,
                    "autocorrelation": correlation,
                })
    return pd.DataFrame(rows)


def _next_observation_mask(panel: pd.DataFrame) -> pd.Series:
    next_previous_date = panel["previous_date"].shift(-1)
    mask = next_previous_date.eq(panel["observation_date"])
    if "is_next_business_observation" in panel:
        mask &= _as_bool(panel["is_next_business_observation"]).shift(-1, fill_value=False)
    return mask


def _state_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    next_is_observation = _next_observation_mask(panel)
    for target in FACTOR_COLUMNS:
        for feature in STATE_FEATURES:
            n, pearson, spearman = _paired_correlation(
                panel[feature], panel[target])
            rows.append({
                "factor": target, "feature": feature,
                "relationship": "same_close", "nobs": n,
                "pearson": pearson, "spearman": spearman,
            })
            future = panel[target].shift(-1).where(next_is_observation)
            n, pearson, spearman = _paired_correlation(panel[feature], future)
            rows.append({
                "factor": target, "feature": feature,
                "relationship": "close_t_to_close_t_plus_1", "nobs": n,
                "pearson": pearson, "spearman": spearman,
            })
    return pd.DataFrame(rows)


def _canonical_cell_columns(frame: pd.DataFrame,
                            config: DynamicAlphaConfig) -> pd.DataFrame:
    """Map harmless CSV float round-off onto the configured research grid."""
    result = frame.copy()
    for column, expected in (("tenor", config.tenors),
                             ("level", config.strike_levels)):
        grid = np.asarray(expected, dtype=float)

        def canonical(value: float) -> float:
            distances = np.abs(grid - float(value))
            nearest = int(np.argmin(distances))
            if distances[nearest] > 1e-12:
                raise ValueError(f"unexpected {column}={value!r}")
            return float(grid[nearest])

        result[column] = result[column].map(canonical)
    return result


def _daily_prediction_sample(panel: pd.DataFrame, daily_beta: pd.DataFrame,
                             config: DynamicAlphaConfig) -> pd.DataFrame:
    """Join a t+1 realised daily-beta label to information known at close t."""
    daily_required = {
        "observation_date", "previous_date", "tenor", "level", "dlogS",
        "dIV_surface", "beta_surface_daily", "daily_ratio_usable",
        "is_next_business_observation",
    }
    missing = daily_required.difference(daily_beta.columns)
    if missing:
        raise ValueError(f"Step 2 daily beta misses {sorted(missing)}")
    daily = _canonical_cell_columns(daily_beta, config)
    daily["daily_ratio_usable"] = _as_bool(daily["daily_ratio_usable"])
    daily["is_next_business_observation"] = _as_bool(
        daily["is_next_business_observation"])
    retained = (
        daily["tenor"].isin(config.step4_tenors)
        & daily["level"].isin(config.step4_strike_levels)
        & daily["daily_ratio_usable"]
        & daily["is_next_business_observation"]
    )
    labels = daily.loc[retained, [
        "previous_date", "observation_date", "tenor", "level", "dlogS",
        "dIV_surface", "beta_surface_daily",
    ]].copy()
    labels = labels.rename(columns={
        "previous_date": "feature_date",
        "observation_date": "label_date",
        "dlogS": "label_dlogS",
        "dIV_surface": "label_dIV_surface",
        "beta_surface_daily": "target_beta_daily",
    })
    labels = labels.replace([np.inf, -np.inf], np.nan).dropna()
    if labels.duplicated(["label_date", "tenor", "level"]).any():
        raise ValueError("Step 2 daily beta has duplicate date/cell labels")
    expected = -labels["label_dIV_surface"] / labels["label_dlogS"]
    if not np.allclose(
            labels["target_beta_daily"], expected, rtol=1e-10, atol=1e-12):
        raise ValueError("beta_surface_daily is inconsistent with -dIV/dlogS")

    known_state = panel[["observation_date", *STATE_FEATURES]].rename(
        columns={"observation_date": "feature_date"})
    known = daily.sort_values(["tenor", "level", "observation_date"]).copy()
    # Persistence uses close-t daily beta; unavailable small-return days may use
    # the last valid observation in the same segment, never cross a data gap.
    known["segment"] = known.groupby(["tenor", "level"])["is_next_business_observation"].transform(
        lambda flag: (~flag).cumsum())
    keys = ["tenor", "level", "segment"]
    known["last_daily_beta"] = known.groupby(keys).beta_surface_daily.ffill()
    known["last_daily_beta_date"] = known.observation_date.where(known.beta_surface_daily.notna())
    known["last_daily_beta_date"] = known.groupby(keys).last_daily_beta_date.ffill()
    known = known[["observation_date", "tenor", "level", "last_daily_beta", "last_daily_beta_date"]].rename(
        columns={"observation_date": "feature_date"})
    sample = labels.merge(known_state, on="feature_date", how="left", validate="many_to_one")
    sample = sample.merge(known, on=["feature_date", "tenor", "level"], how="left", validate="one_to_one")
    return (sample.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["target_beta_daily"])
            .sort_values(["tenor", "level", "label_date"]).reset_index(drop=True))


def _fit_daily_beta_models(sample: pd.DataFrame
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit per-cell models on an ordered split and return test predictions."""
    rows: list[pd.DataFrame] = []
    split_rows: list[dict] = []
    for (tenor, level), group in sample.groupby(["tenor", "level"], sort=True):
        group = group.sort_values("label_date").reset_index(drop=True)
        if len(group) < 40:
            continue
        split = int(np.floor(TRAIN_FRACTION * len(group)))
        split = min(max(split, 30), len(group) - 10)
        train, test = group.iloc[:split], group.iloc[split:]
        x_train = train[list(PREDICTION_FEATURES)].to_numpy(dtype=float)
        x_test = test[list(PREDICTION_FEATURES)].to_numpy(dtype=float)
        y_train = train["target_beta_daily"].to_numpy(dtype=float)

        n_splits = min(3, max(2, len(train) // 60))
        ridge_pipeline = make_pipeline(
            SimpleImputer(
                strategy="median", add_indicator=True,
                keep_empty_features=True),
            StandardScaler(), Ridge(),
        )
        ridge = GridSearchCV(
            ridge_pipeline,
            param_grid={"ridge__alpha": np.logspace(-2, 2, 7)},
            cv=TimeSeriesSplit(n_splits=n_splits),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        nonlinear = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=7,
            min_samples_leaf=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=20260807,
        )
        ridge.fit(x_train, y_train)
        nonlinear.fit(x_train, y_train)
        predictions = {
            "sticky_strike": np.zeros(len(test)),
            "training_mean": np.full(len(test), y_train.mean()),
            "last_observed_beta": test["last_daily_beta"].fillna(y_train.mean()).to_numpy(dtype=float),
            "ridge_state": ridge.predict(x_test),
            "hist_gradient_boosting": nonlinear.predict(x_test),
        }
        ridge_alpha = float(ridge.best_params_["ridge__alpha"])
        common = test[[
            "feature_date", "label_date", "tenor", "level",
            "target_beta_daily", "label_dlogS", "label_dIV_surface",
            "last_daily_beta", "last_daily_beta_date",
        ]]
        for model, predicted in predictions.items():
            output = common.copy()
            output["model"] = model
            output["predicted_beta"] = predicted
            output["predicted_dIV_surface"] = (
                -predicted * output["label_dlogS"].to_numpy(dtype=float))
            output["beta_error"] = (
                output["target_beta_daily"] - output["predicted_beta"])
            output["dIV_error"] = (
                output["label_dIV_surface"]
                - output["predicted_dIV_surface"])
            rows.append(output)
        split_rows.append({
            "tenor": float(tenor), "level": float(level),
            "n_total": len(group), "n_train": len(train), "n_test": len(test),
            "train_start": train["label_date"].iloc[0],
            "train_end": train["label_date"].iloc[-1],
            "test_start": test["label_date"].iloc[0],
            "test_end": test["label_date"].iloc[-1],
            "ridge_alpha": ridge_alpha,
        })
    if not rows:
        raise ValueError("too few complete daily-beta observations for modelling")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(split_rows)


def _daily_prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise direct beta error and the corresponding one-day dIV error."""
    rows: list[dict] = []

    def summarize(group: pd.DataFrame, scope: str,
                  tenor: float | None, level: float | None) -> None:
        def improvement(value: float, benchmark: float) -> float:
            if benchmark > 1e-15:
                return 1.0 - value / benchmark
            return 0.0 if value <= 1e-15 else np.nan

        pivot_beta = group.pivot_table(
            index=["label_date", "tenor", "level"], columns="model",
            values="predicted_beta")
        pivot_div = group.pivot_table(
            index=["label_date", "tenor", "level"], columns="model",
            values="predicted_dIV_surface")
        actual = (group.drop_duplicates(["label_date", "tenor", "level"])
                  .set_index(["label_date", "tenor", "level"])
                  .sort_index())
        pivot_beta = pivot_beta.reindex(actual.index)
        pivot_div = pivot_div.reindex(actual.index)
        y_beta = actual["target_beta_daily"].to_numpy(dtype=float)
        y_div = actual["label_dIV_surface"].to_numpy(dtype=float)
        mean_sse = float(np.sum(
            (y_beta - pivot_beta["training_mean"].to_numpy()) ** 2))
        naive_div_rmse = float(np.sqrt(np.mean(
            (y_div - pivot_div["last_observed_beta"].to_numpy()) ** 2)))
        sticky_div_rmse = float(np.sqrt(np.mean(
            (y_div - pivot_div["sticky_strike"].to_numpy()) ** 2)))
        for model in PREDICTION_MODELS:
            beta_hat = pivot_beta[model].to_numpy(dtype=float)
            div_hat = pivot_div[model].to_numpy(dtype=float)
            beta_error = y_beta - beta_hat
            div_error = y_div - div_hat
            correlation = (float(np.corrcoef(y_beta, beta_hat)[0, 1])
                           if np.std(beta_hat) > 0.0 else np.nan)
            beta_sse = float(beta_error @ beta_error)
            div_rmse = float(np.sqrt(np.mean(div_error ** 2)))
            rows.append({
                "scope": scope, "tenor": tenor, "level": level,
                "model": model, "n_test": len(actual),
                "beta_rmse": float(np.sqrt(np.mean(beta_error ** 2))),
                "beta_mae": float(np.mean(np.abs(beta_error))),
                "beta_correlation": correlation,
                "beta_oos_r2_vs_training_mean": (
                    1.0 - beta_sse / mean_sse if mean_sse > 0.0 else np.nan),
                "dIV_rmse": div_rmse,
                "dIV_mae": float(np.mean(np.abs(div_error))),
                "dIV_rmse_reduction_vs_sticky_strike": improvement(
                    div_rmse, sticky_div_rmse),
                "dIV_rmse_improvement_vs_last_observed_beta": improvement(
                    div_rmse, naive_div_rmse),
            })

    summarize(predictions, "overall", None, None)
    for tenor, group in predictions.groupby("tenor", sort=True):
        summarize(group, "tenor", float(tenor), None)
    for (tenor, level), group in predictions.groupby(
            ["tenor", "level"], sort=True):
        summarize(group, "cell", float(tenor), float(level))
    return pd.DataFrame(rows)


def _attribution_baseline(factors: pd.DataFrame, loadings: pd.DataFrame,
                          changes: pd.DataFrame) -> pd.DataFrame:
    required_loadings = {
        "tenor", "level", "factor_intercept", "atm_beta_loading",
        "shape_loading_1", "shape_loading_2",
    }
    missing = required_loadings.difference(loadings.columns)
    if missing:
        raise ValueError(f"Step 4 factor loadings miss {sorted(missing)}")
    required_changes = {
        "observation_date", "previous_date", "tenor", "level", "dlogS",
        "dIV_surface", "is_next_business_observation",
    }
    missing = required_changes.difference(changes.columns)
    if missing:
        raise ValueError(f"Step 1 changes miss {sorted(missing)}")

    prior = factors[["observation_date", *FACTOR_COLUMNS]].rename(columns={
        "observation_date": "previous_date",
        **{factor: f"previous_{factor}" for factor in FACTOR_COLUMNS},
    })
    data = changes.copy()
    data["is_next_business_observation"] = _as_bool(
        data["is_next_business_observation"])
    data["tenor_key"] = data["tenor"].astype(float).round(12)
    data["level_key"] = data["level"].astype(float).round(12)
    loading_keys = loadings.copy()
    loading_keys["tenor_key"] = loading_keys["tenor"].astype(float).round(12)
    loading_keys["level_key"] = loading_keys["level"].astype(float).round(12)
    loading_columns = [
        "tenor_key", "level_key", "factor_intercept", "atm_beta_loading",
        "shape_loading_1", "shape_loading_2",
    ]
    data = data.merge(prior, on="previous_date", how="inner", validate="many_to_one")
    data = data.merge(
        loading_keys[loading_columns], on=["tenor_key", "level_key"],
        how="inner", validate="many_to_one")
    data = data[
        data["is_next_business_observation"]
        & data[["dIV_surface", "dlogS"]].notna().all(axis=1)].copy()
    data["beta_hat_previous"] = (
        data["factor_intercept"]
        + data["previous_atm_beta_factor"] * data["atm_beta_loading"]
        + data["previous_shape_score_1"] * data["shape_loading_1"]
        + data["previous_shape_score_2"] * data["shape_loading_2"])
    data["sticky_strike_residual"] = data["dIV_surface"]
    data["factor_beta_residual"] = (
        data["dIV_surface"] + data["beta_hat_previous"] * data["dlogS"])

    def summarize(sample: pd.DataFrame, scope: str,
                  tenor: float | None, level: float | None) -> dict:
        sticky = sample["sticky_strike_residual"].to_numpy(dtype=float)
        dynamic = sample["factor_beta_residual"].to_numpy(dtype=float)
        sticky_rmse = float(np.sqrt(np.mean(sticky ** 2)))
        dynamic_rmse = float(np.sqrt(np.mean(dynamic ** 2)))
        sticky_mae = float(np.mean(np.abs(sticky)))
        dynamic_mae = float(np.mean(np.abs(dynamic)))
        return {
            "scope": scope, "tenor": tenor, "level": level,
            "nobs": len(sample),
            "sticky_strike_rmse": sticky_rmse,
            "factor_beta_rmse": dynamic_rmse,
            "rmse_reduction": 1.0 - dynamic_rmse / sticky_rmse,
            "sticky_strike_mae": sticky_mae,
            "factor_beta_mae": dynamic_mae,
            "mae_reduction": 1.0 - dynamic_mae / sticky_mae,
        }

    rows = [summarize(data, "overall", None, None)]
    for tenor, group in data.groupby("tenor", sort=True):
        rows.append(summarize(group, "tenor", float(tenor), None))
    for (tenor, level), group in data.groupby(["tenor", "level"], sort=True):
        rows.append(summarize(group, "cell", float(tenor), float(level)))
    return pd.DataFrame(rows)


def run_step5(factors: pd.DataFrame, loadings: pd.DataFrame,
              iv_state: pd.DataFrame, changes: pd.DataFrame,
              daily_beta: pd.DataFrame,
              config: DynamicAlphaConfig = DynamicAlphaConfig()) -> Step5Result:
    """Test whether close-t state predicts the next realised daily beta."""
    panel = _build_factor_state_panel(factors, iv_state, changes)
    spot_regression, panel = _spot_regressions(panel)
    factor_acf = _factor_acf(panel)
    state_correlations = _state_correlations(panel)
    prediction_sample = _daily_prediction_sample(
        panel, daily_beta, config)
    predictions, prediction_splits = _fit_daily_beta_models(prediction_sample)
    model_summary = _daily_prediction_summary(predictions)
    attribution = _attribution_baseline(factors, loadings, changes)

    primary = model_summary[model_summary["scope"].eq("overall")].set_index(
        "model")
    primary_model = "hist_gradient_boosting"
    primary_div_rmse = float(primary.loc[primary_model, "dIV_rmse"])
    naive_div_rmse = float(primary.loc["last_observed_beta", "dIV_rmse"])
    primary_beta_r2 = float(
        primary.loc[primary_model, "beta_oos_r2_vs_training_mean"])
    overall_attribution = attribution[attribution["scope"].eq("overall")].iloc[0]
    acf1 = factor_acf[
        factor_acf["factor"].eq("atm_beta_factor")
        & factor_acf["transform"].eq("level")
        & factor_acf["lag"].eq(1)].iloc[0]
    validation = {
        "calendar_date_count": len(panel),
        "factor_date_count": int(panel["factor_available"].sum()),
        "next_factor_label_count": int(
            panel[[f"next_{factor}" for factor in FACTOR_COLUMNS]]
            .notna().all(axis=1).sum()),
        "factor_targets_for_step6": tuple(
            f"next_{factor}" for factor in FACTOR_COLUMNS),
        "state_feature_count": len(STATE_FEATURES),
        "prediction_feature_count": len(PREDICTION_FEATURES),
        "prediction_target": "next-observation beta_surface_daily",
        "daily_beta_definition": "-dIV_surface / dlogS",
        "daily_beta_min_abs_dlogS": float(config.beta_min_abs_dlogS),
        "daily_beta_label_date_count": int(
            prediction_sample["label_date"].nunique()),
        "daily_beta_complete_row_count": len(prediction_sample),
        "modelled_cell_count": len(prediction_splits),
        "prediction_horizon": "close t to next observed business close",
        "train_fraction": TRAIN_FRACTION,
        "feature_cutoff": "all predictive features available through close t",
        "label_leakage_columns_in_feature_set": [],
        "prediction_features": PREDICTION_FEATURES,
        "beta_workflow": "daily_only_v1",
        "persistence_policy": "close-t daily beta; within-segment last valid value; training mean if unavailable",
        "complex_model": "sklearn HistGradientBoostingRegressor",
        "missing_state_policy": (
            "Ridge uses train-only median imputation plus missing indicators; "
            "HistGradientBoosting uses native missing-value splits"),
        "atm_beta_factor_acf_lag1": float(acf1["autocorrelation"]),
        "primary_state_model": primary_model,
        "primary_state_beta_oos_r2_vs_training_mean": primary_beta_r2,
        "primary_state_dIV_rmse": primary_div_rmse,
        "last_observed_beta_dIV_rmse": naive_div_rmse,
        "primary_state_dIV_rmse_improvement_vs_last_observed_beta": (
            1.0 - primary_div_rmse / naive_div_rmse if naive_div_rmse > 0 else np.nan),
        "ex_ante_predictability_gate": bool(
            primary_beta_r2 > 0.0 and primary_div_rmse < naive_div_rmse),
        "step6_recommendation": (
            "promote a state model only if it predicts next-day realised beta "
            "and beats the last-observed daily-beta benchmark out of sample"),
        "attribution_beta_timing": (
            "previous-close factor beta multiplied by current realized dlogS"),
        "overall_attribution_rmse_reduction": float(
            overall_attribution["rmse_reduction"]),
        "ex_post_surface_attribution_value": bool(
            overall_attribution["rmse_reduction"] > 0.0),
        "factor_model_role": (
            "Step 4 daily-beta loadings are fitted on its chronological "
            "training sample; current per-cell forecasts remain the Step 6 "
            "benchmark for later 1/2/3-factor forecast comparisons"),
    }
    return Step5Result(
        config, panel, factor_acf, spot_regression, state_correlations,
        predictions, model_summary, prediction_splits, attribution, validation)


def _save_plot(result: Step5Result, target: Path) -> list[str]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/svi-localvol-mpl")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    available = result.factor_state_panel[
        result.factor_state_panel["factor_available"]]
    acf = result.factor_acf[
        result.factor_acf["transform"].eq("level")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    dates = pd.to_datetime(available["observation_date"])
    axes[0, 0].plot(dates, available["atm_beta_factor"], label="ATM beta")
    axes[0, 0].plot(dates, available["shape_score_1"], alpha=0.65,
                    label="shape 1")
    axes[0, 0].plot(dates, available["shape_score_2"], alpha=0.65,
                    label="shape 2")
    axes[0, 0].set_title("Step 4 factors")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].scatter(
        available["dlogS"], available["atm_beta_factor"], s=10, alpha=0.5)
    axes[0, 1].set(
        xlabel="same-close dlogS", ylabel="3M ATM beta factor",
        title="Contemporaneous spot relationship")

    for factor in FACTOR_COLUMNS:
        selected = acf[acf["factor"].eq(factor)]
        axes[1, 0].plot(selected["lag"], selected["autocorrelation"],
                        marker="o", label=factor)
    axes[1, 0].set(
        xlabel="lag", ylabel="autocorrelation", title="Factor level ACF")
    axes[1, 0].legend(fontsize=8)

    primary = result.daily_beta_model_summary[
        result.daily_beta_model_summary["scope"].eq("overall")]
    axes[1, 1].bar(primary["model"], primary["dIV_rmse"])
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].set(
        ylabel="OOS dIV RMSE", title="Next-day daily-beta models")
    fig.tight_layout()
    name = "factor_diagnostics.png"
    fig.savefig(target / name, dpi=160)
    plt.close(fig)
    return [name]


def save_step5(result: Step5Result, *, factor_scores_path: str | Path,
               factor_loadings_path: str | Path, iv_state_path: str | Path,
               grid_changes_path: str | Path, daily_beta_path: str | Path,
               outdir: str | Path = "output/dynamic_alpha/step05") -> Path:
    """Write the Step 5 gate and provenance for every upstream artefact."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    result.factor_state_panel.to_csv(
        target / "factor_state_panel.csv", index=False)
    result.factor_acf.to_csv(target / "factor_acf.csv", index=False)
    result.spot_regression.to_csv(
        target / "contemporaneous_spot_regression.csv", index=False)
    result.state_correlations.to_csv(
        target / "state_correlations.csv", index=False)
    result.daily_beta_predictions.to_csv(
        target / "daily_beta_predictions.csv", index=False)
    result.daily_beta_model_summary.to_csv(
        target / "daily_beta_model_summary.csv", index=False)
    result.prediction_splits.to_csv(
        target / "prediction_splits.csv", index=False)
    result.attribution_baseline.to_csv(
        target / "attribution_baseline.csv", index=False)
    # Remove the superseded file whose label was next-day rolling PCA factors.
    (target / "predictability_baselines.csv").unlink(missing_ok=True)
    (target / "factor_diagnostics.png").unlink(missing_ok=True)
    result.validation["plot_files"] = _save_plot(result, target)

    input_paths = {
        "step04_factor_scores": Path(factor_scores_path),
        "step04_factor_loadings": Path(factor_loadings_path),
        "step01_iv_state": Path(iv_state_path),
        "step01_grid_changes": Path(grid_changes_path),
        "step02_daily_beta": Path(daily_beta_path),
    }
    inputs: dict[str, str] = {}
    for name, path in input_paths.items():
        inputs[name] = str(path)
        inputs[f"{name}_sha256"] = file_sha256(path)
    for stage, directory in (
            ("step04", Path(factor_scores_path).parent),
            ("step02", Path(daily_beta_path).parent),
            ("step01", Path(iv_state_path).parent)):
        manifest = directory / "manifest.json"
        if manifest.exists():
            inputs[f"{stage}_manifest"] = str(manifest)
            inputs[f"{stage}_manifest_sha256"] = file_sha256(manifest)
    return write_manifest(
        target / "manifest.json", stage="dynamic_alpha_step05",
        config=result.config, inputs=inputs, validation=result.validation)

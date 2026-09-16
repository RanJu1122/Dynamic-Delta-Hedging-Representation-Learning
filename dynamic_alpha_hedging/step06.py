"""Dynamic Alpha Step 6: forecast next-observation daily-beta factors.

``g(state_t)`` is a supervised mapping from information known at close ``t``
to one of the Step 4 factor scores at the next market observation.  The module
fits the three factors separately, reconstructs nested one/two/three-factor
beta surfaces, and evaluates them without using next-day information as a
feature.  Alpha conversion deliberately remains a Step 7 responsibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import file_sha256, write_manifest
from .config import DynamicAlphaConfig
from .grid import validate_cells
from .factors import schema_for, validate_factor_pair, copy_metadata, metadata_value
from .step05 import FACTOR_COLUMNS, STATE_FEATURES


LAST_FACTOR_COLUMNS = tuple(f"last_{name}" for name in FACTOR_COLUMNS)
PREDICTION_FEATURES = (*STATE_FEATURES, *LAST_FACTOR_COLUMNS)
FACTOR_MODELS = (
    "training_mean", "last_observed_factor",
    "ridge_state", "hist_gradient_boosting",
)
PRIMARY_MODEL = "hist_gradient_boosting"


@dataclass
class Step6Result:
    """Factor forecasts and nested full-surface evaluation."""

    config: DynamicAlphaConfig
    factor_predictions: pd.DataFrame
    factor_model_summary: pd.DataFrame
    surface_model_summary: pd.DataFrame
    feature_importance: pd.DataFrame
    validation: dict[str, object]


def _read_csv(path: str | Path, date_columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in date_columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column]).dt.date
    return frame


def load_step6_inputs(*, factor_state_panel_path: str | Path,
                      factor_loadings_path: str | Path,
                      daily_beta_path: str | Path,
                      ) -> tuple[pd.DataFrame, pd.DataFrame,
                                 pd.DataFrame]:
    panel = _read_csv(
        factor_state_panel_path, ("observation_date", "previous_date"))
    loadings = _read_csv(factor_loadings_path, ())
    daily = _read_csv(
        daily_beta_path, ("observation_date", "previous_date"))
    return panel, loadings, daily


def _canonical_axes(frame: pd.DataFrame,
                    config: DynamicAlphaConfig) -> pd.DataFrame:
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


def _expected_cells(config: DynamicAlphaConfig) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [config.step4_tenors, config.step4_strike_levels],
        names=["tenor", "level"])


def _ordered_loadings(loadings: pd.DataFrame, config: DynamicAlphaConfig
                      ) -> pd.DataFrame:
    schema = schema_for(loadings)
    if schema.method != config.step4_factor_method:
        raise ValueError("factor method mismatch between loadings and config")
    required = {"tenor", "level", "mean_beta_train", "factor_intercept", *schema.loadings}
    missing = required.difference(loadings.columns)
    if missing:
        raise ValueError(f"Step 4 factor loadings miss {sorted(missing)}")
    validate_cells(loadings, config.step4_tenors, config.step4_strike_levels,
                   source="Step 4 loadings")
    values = _canonical_axes(loadings, config)
    if values.duplicated(["tenor", "level"]).any():
        raise ValueError("Step 4 loadings contain duplicate cells")
    ordered = values.set_index(["tenor", "level"]).reindex(
        _expected_cells(config))
    if ordered[list(required - {"tenor", "level"})].isna().any().any():
        raise ValueError("Step 4 loadings do not cover the configured surface")
    return ordered


def factor_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Close-t inputs on ALL dates; availability of future labels is irrelevant."""
    values = panel.sort_values("observation_date").reset_index(drop=True).copy()
    if values["observation_date"].duplicated().any():
        raise ValueError("duplicate factor-state observation dates")
    if "is_next_business_observation" in values:
        from .step05 import _as_bool
        values["segment"] = (~_as_bool(values["is_next_business_observation"])).cumsum()
    schema = schema_for(panel)
    for factor, last in zip(schema.scores, schema.last):
        values[last] = (values.groupby("segment")[factor].ffill()
                        if "segment" in values else values[factor].ffill())
    return values.replace([np.inf, -np.inf], np.nan)


def primary_factor_model(**parameters) -> HistGradientBoostingRegressor:
    """One shared definition for Step 6 evaluation and Step 7 deployment."""
    defaults = dict(loss="absolute_error", learning_rate=0.05, max_iter=200,
                    max_leaf_nodes=7, min_samples_leaf=15, l2_regularization=1.0,
                    early_stopping=False, random_state=20260807)
    defaults.update(parameters)
    return HistGradientBoostingRegressor(**defaults)


def factor_model(name="hist_gradient_boosting", parameters=None):
    """Explicit model choices; fit/imputation always use training rows only."""
    from sklearn.dummy import DummyRegressor
    parameters = dict(parameters or {})
    allowed = {
        "hist_gradient_boosting": {"loss", "learning_rate", "max_iter",
            "max_leaf_nodes", "min_samples_leaf", "l2_regularization",
            "max_depth", "random_state"},
        "ridge": {"alpha"}, "training_mean": set(),
    }
    if name not in allowed or set(parameters) - allowed[name]:
        raise ValueError(f"unsupported factor model or parameters: {name}, {sorted(parameters)}")
    if name == "hist_gradient_boosting":
        return primary_factor_model(**parameters)
    estimator = Ridge(**parameters) if name == "ridge" else DummyRegressor(strategy="mean")
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                      keep_empty_features=True), StandardScaler(), estimator)


@dataclass
class FactorForecaster:
    models: dict
    train_end: object
    features: tuple[str, ...] = PREDICTION_FEATURES
    training_means: dict = field(default_factory=dict)
    factor_method: str = "atm_anchored"

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return close-t signals; do not require a realised t+1 label."""
        result = features[["observation_date"]].copy()
        x = features[list(self.features)].to_numpy(dtype=float)
        for name, model in self.models.items():
            result[name] = model.predict(x)
        return result


    def naive_predict(self, features):
        """Persistence benchmark; same close-t cutoff and declared missing-value policy."""
        result = features[["observation_date"]].copy()
        schema = schema_for(self.factor_method)
        for factor, last in zip(schema.scores, schema.last):
            result[factor] = features[last].fillna(self.training_means[factor])
        return result


def fit_factor_forecaster(panel: pd.DataFrame, loadings: pd.DataFrame,
                          config: DynamicAlphaConfig, *,
                          model_name="hist_gradient_boosting", model_parameters=None
                          ) -> tuple[FactorForecaster, pd.DataFrame]:
    schema = validate_factor_pair(panel, loadings, config)
    prediction_features = (*STATE_FEATURES, *schema.last)
    _ordered_loadings(loadings, config)
    sample = _prediction_panel(panel)
    train = sample[sample["label_sample"].eq("train")]
    models = {}
    for factor in schema.scores:
        model = factor_model(model_name, model_parameters)
        model.fit(train[list(prediction_features)].to_numpy(float),
                  train[f"next_{factor}"].to_numpy(float))
        models[factor] = model
    means = {factor: float(train[f"next_{factor}"].mean()) for factor in schema.scores}
    return (FactorForecaster(models, train["label_date"].max(), features=prediction_features, training_means=means, factor_method=schema.method),
            factor_features(panel))


def _prediction_panel(panel: pd.DataFrame) -> pd.DataFrame:
    schema = schema_for(panel)
    next_factors = tuple(f"next_{name}" for name in schema.scores)
    required = {
        "observation_date", "previous_date", "sample", "dlogS",
        *STATE_FEATURES, *schema.scores, *next_factors,
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Step 5 factor-state panel misses {sorted(missing)}")
    values = factor_features(panel)
    next_is_observation = values["previous_date"].shift(-1).eq(
        values["observation_date"])
    if "is_next_business_observation" in values:
        from .step05 import _as_bool
        next_is_observation &= _as_bool(values["is_next_business_observation"]).shift(
            -1, fill_value=False)
    values["label_date"] = values["observation_date"].shift(-1).where(
        next_is_observation)
    values["label_sample"] = values["sample"].shift(-1).where(
        next_is_observation)
    values["label_dlogS"] = values["dlogS"].shift(-1).where(
        next_is_observation)

    for factor, target, last in zip(
            schema.scores, next_factors, schema.last):
        expected = values[factor].shift(-1).where(next_is_observation)
        check = values[target].notna() | expected.notna()
        if not np.allclose(
                values.loc[check, target], expected.loc[check],
                equal_nan=True, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Step 5 {target} is not the next observation")
        # factor_features has already forward-filled within each valid segment.

    usable = values[list(next_factors)].notna().all(axis=1)
    usable &= values["label_sample"].isin(["train", "test"])
    result = values.loc[usable].replace([np.inf, -np.inf], np.nan).copy()
    if not (result["label_sample"].eq("train").any()
            and result["label_sample"].eq("test").any()):
        raise ValueError("Step 6 requires both train and test factor labels")
    return result


def _fit_models(sample: pd.DataFrame
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    schema = schema_for(sample)
    prediction_features = (*STATE_FEATURES, *schema.last)
    train = sample[sample["label_sample"].eq("train")]
    test = sample[sample["label_sample"].eq("test")]
    x_train = train[list(prediction_features)].to_numpy(dtype=float)
    x_test = test[list(prediction_features)].to_numpy(dtype=float)
    prediction_rows: list[pd.DataFrame] = []
    importance_rows: list[dict] = []
    ridge_alphas: dict[str, float] = {}

    for factor, last_column in zip(schema.scores, schema.last):
        target = f"next_{factor}"
        y_train = train[target].to_numpy(dtype=float)
        y_test = test[target].to_numpy(dtype=float)
        train_mean = float(y_train.mean())

        ridge_pipeline = make_pipeline(
            SimpleImputer(
                strategy="median", add_indicator=True,
                keep_empty_features=True),
            StandardScaler(), Ridge(),
        )
        ridge = GridSearchCV(
            ridge_pipeline,
            param_grid={"ridge__alpha": np.logspace(-2, 2, 7)},
            cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        nonlinear = primary_factor_model()
        ridge.fit(x_train, y_train)
        nonlinear.fit(x_train, y_train)
        ridge_alphas[factor] = float(ridge.best_params_["ridge__alpha"])

        last_observed = test[last_column].fillna(train_mean).to_numpy(float)
        predictions = {
            "training_mean": np.full(len(test), train_mean),
            "last_observed_factor": last_observed,
            "ridge_state": ridge.predict(x_test),
            "hist_gradient_boosting": nonlinear.predict(x_test),
        }
        common = test[[
            "observation_date", "label_date", "label_dlogS"]].rename(
                columns={"observation_date": "feature_date"})
        for model, predicted in predictions.items():
            output = common.copy()
            output["naive_uses_close_t_factor"] = test[factor].notna().to_numpy()
            output["naive_training_mean_fallback"] = test[last_column].isna().to_numpy()
            output["factor"] = factor
            output["model"] = model
            output["actual_factor"] = y_test
            output["predicted_factor"] = predicted
            output["error"] = y_test - predicted
            prediction_rows.append(output)

        importance = permutation_importance(
            nonlinear, x_test, y_test, scoring="neg_mean_squared_error",
            n_repeats=10, random_state=20260807)
        for feature, mean, std in zip(
                prediction_features, importance.importances_mean,
                importance.importances_std):
            importance_rows.append({
                "factor": factor,
                "model": PRIMARY_MODEL,
                "feature": feature,
                "importance_mean": float(mean),
                "importance_std": float(std),
            })

    return (pd.concat(prediction_rows, ignore_index=True),
            pd.DataFrame(importance_rows), ridge_alphas)


def _factor_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for factor, factor_data in predictions.groupby("factor", sort=False):
        mean_data = factor_data[factor_data["model"].eq("training_mean")]
        mean_sse = float(np.sum(mean_data["error"].to_numpy() ** 2))
        naive = factor_data[factor_data["model"].eq("last_observed_factor")]
        naive_sse = float(np.sum(naive["error"].to_numpy() ** 2))
        naive_rmse = float(np.sqrt(naive_sse / len(naive)))
        for model, data in factor_data.groupby("model", sort=False):
            actual = data["actual_factor"].to_numpy(dtype=float)
            predicted = data["predicted_factor"].to_numpy(dtype=float)
            error = actual - predicted
            correlation = (float(np.corrcoef(actual, predicted)[0, 1])
                           if np.std(predicted) > 0.0 else np.nan)
            sse = float(error @ error)
            rows.append({
                "factor": factor,
                "model": model,
                "n_test": len(data),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "oos_r_squared_vs_last_observed_factor": (1-float(error @ error)/naive_sse if naive_sse > 0 else np.nan),
                "rmse_improvement_vs_last_observed_factor": (1-float(np.sqrt(np.mean(error**2)))/naive_rmse if naive_rmse > 0 else np.nan),
                "naive_exact_close_fraction": float(data.naive_uses_close_t_factor.mean()),
                "naive_training_mean_fallback_fraction": float(data.naive_training_mean_fallback.mean()),
                "mae": float(np.mean(np.abs(error))),
                "correlation": correlation,
                "oos_r_squared_vs_training_mean": (
                    1.0 - sse / mean_sse if mean_sse > 0.0 else np.nan),
            })
    return pd.DataFrame(rows)


def _actual_surface(daily: pd.DataFrame, dates: pd.Index,
                    config: DynamicAlphaConfig) -> pd.DataFrame:
    required = {"observation_date", "tenor", "level", "beta_surface_daily"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Step 2 daily beta misses {sorted(missing)}")
    values = _canonical_axes(daily, config)
    matrix = values.pivot(
        index="observation_date", columns=["tenor", "level"],
        values="beta_surface_daily").reindex(
            index=dates, columns=_expected_cells(config))
    if matrix.isna().any().any():
        raise ValueError("Step 6 test dates lack complete actual beta surfaces")
    return matrix


def _surface_summary(factor_predictions: pd.DataFrame,
                     loadings: pd.DataFrame, daily: pd.DataFrame,
                     config: DynamicAlphaConfig) -> pd.DataFrame:
    schema = schema_for(loadings)
    one_model = factor_predictions[
        factor_predictions["model"].eq(FACTOR_MODELS[0])]
    date_map = (one_model[["feature_date", "label_date", "label_dlogS"]]
                .drop_duplicates().sort_values("label_date"))
    label_dates = pd.Index(date_map["label_date"])
    feature_dates = pd.Index(date_map["feature_date"])
    actual_frame = _actual_surface(daily, label_dates, config)
    actual = actual_frame.to_numpy(dtype=float)
    label_dlogS = date_map["label_dlogS"].to_numpy(dtype=float)

    intercept = loadings["factor_intercept"].to_numpy(dtype=float)
    means = loadings["mean_beta_train"].to_numpy(dtype=float)
    factor_loadings = loadings[list(schema.loadings)].to_numpy(dtype=float).T
    surfaces: dict[tuple[str, int], np.ndarray] = {
        ("sticky_strike", 0): np.zeros_like(actual),
        ("training_mean_surface", 0): np.broadcast_to(means, actual.shape),
    }
    for model in FACTOR_MODELS:
        selected = factor_predictions[factor_predictions["model"].eq(model)]
        scores = selected.pivot(
            index="label_date", columns="factor",
            values="predicted_factor").reindex(
                index=label_dates, columns=schema.scores)
        if scores.isna().any().any():
            raise ValueError(f"incomplete factor predictions for {model}")
        score_values = scores.to_numpy(dtype=float)
        for count in (1, 2, 3):
            surfaces[(f"factor_{model}", count)] = (
                intercept
                + score_values[:, :count] @ factor_loadings[:count])

    cells = actual_frame.columns
    anchor_index = int(cells.get_loc(
        (config.step4_anchor_tenor, config.step4_anchor_level)))
    scopes: list[tuple[str, float | None, np.ndarray]] = [
        ("overall", None, np.arange(len(cells))),
        ("atm_anchor" if schema.method == "atm_anchored" else "atm_reference", float(config.step4_anchor_tenor),
         np.asarray([anchor_index])),
    ]
    tenor_values = cells.get_level_values("tenor").to_numpy(dtype=float)
    for tenor in config.step4_tenors:
        scopes.append(("tenor", float(tenor), np.flatnonzero(
            np.isclose(tenor_values, tenor, atol=1e-12))))

    rows: list[dict] = []
    for scope, tenor, indices in scopes:
        actual_scope = actual[:, indices]
        mean_scope = np.broadcast_to(means[indices], actual_scope.shape)
        actual_div = -actual_scope * label_dlogS[:, None]
        sticky_rmse = float(np.sqrt(np.mean(actual_div ** 2)))
        baseline_sse = float(np.sum((actual_scope - mean_scope) ** 2))
        for (model, count), predicted in surfaces.items():
            predicted_scope = predicted[:, indices]
            baseline_count = count or 3
            naive_scope = surfaces[("factor_last_observed_factor", baseline_count)][:, indices]
            naive_div = -naive_scope * label_dlogS[:, None]
            naive_rmse = float(np.sqrt(np.mean((actual_div-naive_div)**2)))
            beta_error = actual_scope - predicted_scope
            predicted_div = -predicted_scope * label_dlogS[:, None]
            div_error = actual_div - predicted_div
            beta_sse = float(np.sum(beta_error ** 2))
            flat_actual = actual_scope.ravel()
            flat_predicted = predicted_scope.ravel()
            correlation = (
                float(np.corrcoef(flat_actual, flat_predicted)[0, 1])
                if np.std(flat_predicted) > 0.0 else np.nan)
            div_rmse = float(np.sqrt(np.mean(div_error ** 2)))
            rows.append({
                "scope": scope,
                "tenor": tenor,
                "model": model,
                "factor_count": count,
                "persistence_comparison_factor_count": baseline_count,
                "n_test_dates": len(label_dates),
                "n_beta_observations": beta_error.size,
                "beta_rmse": float(np.sqrt(np.mean(beta_error ** 2))),
                "beta_mae": float(np.mean(np.abs(beta_error))),
                "beta_correlation": correlation,
                "beta_oos_r_squared_vs_training_mean_surface": (
                    1.0 - beta_sse / baseline_sse
                    if baseline_sse > 0.0 else np.nan),
                "dIV_rmse": div_rmse,
                "dIV_mae": float(np.mean(np.abs(div_error))),
                "dIV_rmse_improvement_vs_last_factor": (
                    1.0 - div_rmse / naive_rmse
                    if naive_rmse > 0.0 else np.nan),
                "dIV_rmse_reduction_vs_sticky_strike": (
                    1.0 - div_rmse / sticky_rmse
                    if sticky_rmse > 0.0 else np.nan),
            })
    return pd.DataFrame(rows)


def run_step6(factor_state_panel: pd.DataFrame,
              factor_loadings: pd.DataFrame,
              daily_beta: pd.DataFrame,
              config: DynamicAlphaConfig = DynamicAlphaConfig()) -> Step6Result:
    """Forecast the next three factors and compare nested beta surfaces."""
    schema = validate_factor_pair(factor_state_panel, factor_loadings, config)
    prediction_features = (*STATE_FEATURES, *schema.last)
    loadings = _ordered_loadings(factor_loadings, config)
    sample = _prediction_panel(factor_state_panel)
    predictions, importance, ridge_alphas = _fit_models(sample)
    factor_summary = _factor_summary(predictions)
    surface_summary = _surface_summary(
        predictions, loadings, daily_beta, config)

    primary_factors = factor_summary[
        factor_summary["model"].eq(PRIMARY_MODEL)].set_index("factor")
    primary_surfaces = surface_summary[
        surface_summary["scope"].eq("overall")
        & surface_summary["model"].eq(f"factor_{PRIMARY_MODEL}")]
    primary_surfaces = primary_surfaces.set_index("factor_count")
    validation = {
        "factor_method": schema.method,
        "factor_basis_id": metadata_value(loadings, "factor_basis_id"),
        "prediction_definition": "g(state_t) -> factor(t+1)",
        "beta_workflow": "daily_only_v1",
        "primary_benchmark": "last_observed_factor",
        "persistence_policy": "close-t daily factor; last valid within segment if missing; training mean if none",
        "factor_targets": tuple(f"next_{name}" for name in schema.scores),
        "train_label_count": int(sample["label_sample"].eq("train").sum()),
        "test_label_count": int(sample["label_sample"].eq("test").sum()),
        "test_start_date": str(sample.loc[
            sample["label_sample"].eq("test"), "label_date"].min()),
        "test_end_date": str(sample.loc[
            sample["label_sample"].eq("test"), "label_date"].max()),
        "prediction_features": prediction_features,
        "feature_count": len(prediction_features),
        "feature_cutoff": "close t; label-date variables are evaluation only",
        "label_leakage_columns_in_feature_set": [],
        "split_policy": "reuse Step 4 train/test label assignment; no shuffle",
        "primary_model": PRIMARY_MODEL,
        "ridge_alphas": ridge_alphas,
        "factor_oos_r_squared": {f: float(primary_factors.loc[f, "oos_r_squared_vs_training_mean"])
                                 for f in schema.scores},
        "factor_correlation": {f: float(primary_factors.loc[f, "correlation"])
                               for f in schema.scores},
        "one_factor_dIV_improvement_vs_last_factor": float(primary_surfaces.loc[
            1, "dIV_rmse_improvement_vs_last_factor"]),
        "two_factor_dIV_improvement_vs_last_factor": float(primary_surfaces.loc[
            2, "dIV_rmse_improvement_vs_last_factor"]),
        "three_factor_dIV_improvement_vs_last_factor": float(primary_surfaces.loc[
            3, "dIV_rmse_improvement_vs_last_factor"]),
        "model_selection_policy": (
            "report predefined 1/2/3-factor variants; Step 7 retains an "
            "extra factor only when it improves OOS surface/dIV and hedging"),
        "alpha_used": False,
        "step7_note": (
            "Beta-to-alpha inversion and delta backtesting remain downstream"),
    }
    if schema.method == "atm_anchored":
        validation.update(
            atm_factor_oos_r_squared=validation["factor_oos_r_squared"]["atm_beta_factor"],
            atm_factor_correlation=validation["factor_correlation"]["atm_beta_factor"],
            shape_1_oos_r_squared=validation["factor_oos_r_squared"]["shape_score_1"],
            shape_2_oos_r_squared=validation["factor_oos_r_squared"]["shape_score_2"])
    atm_scope = "atm_anchor" if schema.method == "atm_anchored" else "atm_reference"
    atm_results = surface_summary[surface_summary.scope.eq(atm_scope)
                                  & surface_summary.model.eq(f"factor_{PRIMARY_MODEL}")]
    validation["atm_beta_surface_oos_r_squared_by_factor_count"] = {
        str(int(row.factor_count)): float(row.beta_oos_r_squared_vs_training_mean_surface)
        for row in atm_results.itertuples()}
    for frame in (predictions, factor_summary, surface_summary, importance):
        copy_metadata(loadings, frame)
    return Step6Result(
        config, predictions, factor_summary, surface_summary,
        importance, validation)


def _save_plots(result: Step6Result, target: Path) -> list[str]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/svi-localvol-mpl")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    files: list[str] = []
    primary = result.factor_predictions[
        result.factor_predictions["model"].eq(PRIMARY_MODEL)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, factor in zip(axes, schema_for(result.config.step4_factor_method).scores):
        data = primary[primary["factor"].eq(factor)]
        dates = pd.to_datetime(data["label_date"])
        ax.plot(dates, data["actual_factor"], label="actual", linewidth=1)
        ax.plot(dates, data["predicted_factor"], label="predicted",
                linewidth=1)
        ax.set_ylabel(factor)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("label date")
    fig.suptitle("Step 6 out-of-sample factor forecasts")
    fig.tight_layout()
    name = "factor_predictions.png"
    fig.savefig(target / name, dpi=160)
    plt.close(fig)
    files.append(name)

    factor_skill = result.factor_model_summary[
        result.factor_model_summary["model"].eq(PRIMARY_MODEL)]
    surface_skill = result.surface_model_summary[
        result.surface_model_summary["scope"].eq("overall")
        & result.surface_model_summary["model"].eq(
            f"factor_{PRIMARY_MODEL}")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(factor_skill["factor"],
                factor_skill["oos_r_squared_vs_training_mean"])
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(ylabel="OOS R-squared", title="Factor forecast skill")
    axes[1].bar(surface_skill["factor_count"],
                surface_skill["dIV_rmse_improvement_vs_last_factor"])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(xticks=(1, 2, 3), xlabel="factor count",
                ylabel="dIV RMSE improvement",
                title="Nested surface forecasts")
    fig.tight_layout()
    name = "forecast_skill.png"
    fig.savefig(target / name, dpi=160)
    plt.close(fig)
    files.append(name)
    return files


def save_step6(result: Step6Result, *, factor_state_panel_path: str | Path,
               factor_loadings_path: str | Path,
               daily_beta_path: str | Path,
               outdir: str | Path = "output/dynamic_alpha/step06") -> Path:
    """Save compact Step 6 forecast outputs with upstream hashes."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    result.factor_predictions.to_csv(
        target / "factor_predictions.csv", index=False)
    result.factor_model_summary.to_csv(
        target / "factor_model_summary.csv", index=False)
    result.surface_model_summary.to_csv(
        target / "surface_model_summary.csv", index=False)
    result.feature_importance.to_csv(
        target / "feature_importance.csv", index=False)
    for plot in ("factor_predictions.png", "forecast_skill.png"):
        (target / plot).unlink(missing_ok=True)
    result.validation["plot_files"] = _save_plots(result, target)

    paths = {
        "step05_factor_state_panel": Path(factor_state_panel_path),
        "step04_factor_loadings": Path(factor_loadings_path),
        "step02_daily_beta": Path(daily_beta_path),
    }
    inputs: dict[str, str] = {}
    for name, path in paths.items():
        inputs[name] = str(path)
        inputs[f"{name}_sha256"] = file_sha256(path)
    for stage, directory in (
            ("step05", Path(factor_state_panel_path).parent),
            ("step04", Path(factor_loadings_path).parent),
            ("step02", Path(daily_beta_path).parent)):
        manifest = directory / "manifest.json"
        if manifest.exists():
            inputs[f"{stage}_manifest"] = str(manifest)
            inputs[f"{stage}_manifest_sha256"] = file_sha256(manifest)
    return write_manifest(
        target / "manifest.json", stage="dynamic_alpha_step06",
        config=result.config, inputs=inputs, validation=result.validation)

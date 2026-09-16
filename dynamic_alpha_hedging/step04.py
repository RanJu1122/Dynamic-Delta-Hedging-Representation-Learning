"""Train-only daily-beta decomposition: ATM anchor or ordinary centered PCA.

Both methods retain three factors, fit only chronological training dates and
project later observations onto the frozen basis. PCA is not standardized.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import file_sha256, write_manifest
from .config import DynamicAlphaConfig
from .factors import schema_for, basis_id


BETA_COLUMN = "beta_surface_daily"
N_SHAPE_COMPONENTS = 2
N_FACTORS = 1 + N_SHAPE_COMPONENTS
REQUIRED_COLUMNS = {"observation_date", "tenor", "level", BETA_COLUMN}


@dataclass
class Step4Result:
    """A fitted three-factor model and its reconstruction diagnostics."""

    config: DynamicAlphaConfig
    explained_variance: pd.DataFrame
    loadings: pd.DataFrame
    scores: pd.DataFrame
    reconstruction_by_cell: pd.DataFrame
    date_coverage: pd.DataFrame
    validation: dict[str, object]
    surface_examples: pd.DataFrame


def load_step2_beta(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "observation_date" in frame:
        frame["observation_date"] = pd.to_datetime(
            frame["observation_date"]).dt.date
    return frame


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    mapped = values.astype(str).str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError("boolean column contains values other than true/false")
    return mapped.astype(bool)


def _canonical_axis(values: pd.Series, expected: tuple[float, ...],
                    name: str) -> pd.Series:
    """Map harmless CSV float round-off back to configured grid values."""
    grid = np.asarray(expected, dtype=float)

    def canonical(value: float) -> float:
        distances = np.abs(grid - float(value))
        nearest = int(np.argmin(distances))
        if distances[nearest] > 1e-12:
            raise ValueError(
                f"Step 2 beta contains unexpected {name}={value!r}")
        return float(grid[nearest])

    return values.map(canonical)


def _surface_matrix(beta: pd.DataFrame, config: DynamicAlphaConfig
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED_COLUMNS.difference(beta.columns)
    if missing:
        raise ValueError(f"Step 2 daily beta file misses {sorted(missing)}")
    beta = beta.copy()
    beta["tenor"] = _canonical_axis(beta["tenor"], config.tenors, "tenor")
    beta["level"] = _canonical_axis(
        beta["level"], config.strike_levels, "level")
    if beta.duplicated(["observation_date", "tenor", "level"]).any():
        raise ValueError("Step 2 beta contains duplicate date/tenor/level keys")

    # A finite, usable daily ratio is the formal Step 4 target.
    valid = pd.Series(True, index=beta.index)
    for flag in ("daily_ratio_usable", "is_next_business_observation"):
        if flag in beta:
            valid &= _as_bool(beta[flag])
    valid &= np.isfinite(beta[BETA_COLUMN])
    beta.loc[~valid, BETA_COLUMN] = np.nan

    configured = pd.MultiIndex.from_product(
        [config.tenors, config.strike_levels], names=["tenor", "level"])
    expected = pd.MultiIndex.from_product(
        [config.step4_tenors, config.step4_strike_levels],
        names=["tenor", "level"])
    observed = pd.MultiIndex.from_frame(
        beta[["tenor", "level"]].drop_duplicates()).sort_values()
    unexpected = observed.difference(configured)
    missing_cells = expected.difference(observed)
    if len(unexpected) or len(missing_cells):
        raise ValueError(
            "Step 2 beta axes differ from the configured surface: "
            f"missing={list(missing_cells)}, unexpected={list(unexpected)}")

    matrix = beta.pivot(
        index="observation_date", columns=["tenor", "level"],
        values=BETA_COLUMN).reindex(columns=expected).sort_index()
    coverage = pd.DataFrame({
        "observation_date": matrix.index,
        "available_beta_cells": matrix.notna().sum(axis=1).to_numpy(),
    })
    coverage["required_beta_cells"] = len(expected)
    coverage["complete_surface"] = (
        coverage["available_beta_cells"] == coverage["required_beta_cells"])
    complete = matrix.dropna(axis=0, how="any")
    if complete.shape[1] < N_FACTORS:
        raise ValueError(
            f"Three-factor decomposition needs at least {N_FACTORS} beta cells; "
            f"found {complete.shape[1]}")
    if len(complete) < N_FACTORS + 1:
        raise ValueError(
            f"Step 4 needs at least {N_FACTORS + 1} complete beta surfaces; "
            f"found {len(complete)}")
    return complete, coverage


def _orient_components(components: np.ndarray,
                       scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resolve PCA's arbitrary signs by making its largest loading positive."""
    components = components.copy()
    scores = scores.copy()
    for k in range(components.shape[0]):
        anchor = int(np.argmax(np.abs(components[k])))
        if components[k, anchor] < 0.0:
            components[k] *= -1.0
            scores[:, k] *= -1.0
    return components, scores


def _r_squared(actual: np.ndarray, fitted: np.ndarray,
               baseline: np.ndarray) -> float:
    sse = float(np.sum((actual - fitted) ** 2))
    sst = float(np.sum((actual - baseline) ** 2))
    return float(1.0 - sse / sst) if sst > 0.0 else np.nan


def _cell_r_squared(actual: np.ndarray, fitted: np.ndarray,
                    baseline: np.ndarray) -> np.ndarray:
    sse = np.sum((actual - fitted) ** 2, axis=0)
    sst = np.sum((actual - baseline) ** 2, axis=0)
    result = np.full_like(sst, np.nan, dtype=float)
    np.divide(sse, sst, out=result, where=sst > 0.0)
    return 1.0 - result


def run_step4(beta: pd.DataFrame,
              config: DynamicAlphaConfig = DynamicAlphaConfig()) -> Step4Result:
    """Fit the configured basis, then evaluate nested 1/2/3-factor reconstructions."""
    schema = schema_for(config.step4_factor_method)
    matrix, date_coverage = _surface_matrix(beta, config)
    values = matrix.to_numpy(dtype=float)
    n_dates = len(values)
    split = int(np.floor(config.step4_train_fraction * n_dates))
    split = min(max(split, N_FACTORS + 1), n_dates - 1)
    train = values[:split]
    test = values[split:]
    train_mean = train.mean(axis=0)

    anchor = (config.step4_anchor_tenor, config.step4_anchor_level)
    try:
        anchor_index = int(matrix.columns.get_loc(anchor))
    except KeyError as exc:
        raise ValueError(f"Step 4 anchor {anchor} is absent from axes") from exc

    atm_beta = values[:, anchor_index]
    if schema.method == "atm_anchored":
        # Factor 1 is observed directly.  Cell regressions use training dates only.
        design_train = np.column_stack([np.ones(split), atm_beta[:split]])
        coefficients, _, _, _ = np.linalg.lstsq(design_train, train, rcond=None)
        factor_intercept = coefficients[0]
        atm_loading = coefficients[1]
        factor_intercept[anchor_index] = 0.0
        atm_loading[anchor_index] = 1.0

        one_factor = factor_intercept + atm_beta[:, None] * atm_loading
        residual = values - one_factor
        residual[:, anchor_index] = 0.0
        _, singular_values, vt = np.linalg.svd(
            residual[:split], full_matrices=False)
        shape_loadings = vt[:N_SHAPE_COMPONENTS]
        shape_scores = residual @ shape_loadings.T
        shape_loadings, shape_scores = _orient_components(
            shape_loadings, shape_scores)

        components = np.vstack([atm_loading, shape_loadings])
        factor_scores = np.column_stack([atm_beta, shape_scores])
        factor_types = ("observed_anchor", "residual_pca", "residual_pca")
        reported_singular_values = (np.nan, singular_values[0], singular_values[1])
    else:
        factor_intercept = train_mean.copy()
        centered = values - train_mean
        _, singular_values, vt = np.linalg.svd(centered[:split], full_matrices=False)
        components = vt[:N_FACTORS]
        factor_scores = centered @ components.T
        components, factor_scores = _orient_components(components, factor_scores)
        factor_types = ("pca",) * N_FACTORS
        reported_singular_values = singular_values[:N_FACTORS]

    reconstructed = np.broadcast_to(factor_intercept, values.shape).copy()
    reconstructions = []
    for k in range(N_FACTORS):
        reconstructed = reconstructed + factor_scores[:, [k]] * components[[k], :]
        reconstructions.append(reconstructed)

    train_baseline = np.broadcast_to(train_mean, train.shape)
    test_baseline = np.broadcast_to(train_mean, test.shape)
    train_r2 = [
        _r_squared(train, fitted[:split], train_baseline)
        for fitted in reconstructions]
    test_r2 = [
        _r_squared(test, fitted[split:], test_baseline)
        for fitted in reconstructions]
    full_baseline = np.broadcast_to(train_mean, values.shape)
    full_r2 = [
        _r_squared(values, fitted, full_baseline)
        for fitted in reconstructions]

    factor_names = schema.scores
    explained = pd.DataFrame({
        "factor_number": np.arange(1, N_FACTORS + 1),
        "factor": factor_names,
        "factor_type": factor_types,
        "pca_singular_value": reported_singular_values,
        "incremental_train_explained_variance_ratio": np.diff(
            np.r_[0.0, train_r2]),
        "cumulative_train_explained_variance_ratio": train_r2,
        "incremental_test_reconstruction_r_squared": np.diff(
            np.r_[0.0, test_r2]),
        "cumulative_test_reconstruction_r_squared": test_r2,
        "cumulative_full_reconstruction_r_squared": full_r2,
    })

    axes = matrix.columns.to_frame(index=False)
    loadings = axes.assign(mean_beta_train=train_mean, factor_intercept=factor_intercept)
    scores = pd.DataFrame({
        "observation_date": matrix.index,
        "sample": np.where(np.arange(n_dates) < split, "train", "test"),
        "atm_beta_observed": atm_beta,
    })
    for k, (score, loading) in enumerate(zip(schema.scores, schema.loadings)):
        loadings[loading] = components[k]
        scores[score] = factor_scores[:, k]
    fitted_id = basis_id(schema.method, axes.to_numpy().tolist(), matrix.index[:split],
                         train, factor_intercept, components)
    for frame in (scores, loadings, explained):
        frame["factor_method"] = schema.method
        frame["factor_basis_id"] = fitted_id
    if schema.method == "atm_anchored":
        explained["residual_pca_singular_value"] = reported_singular_values
    for n_factors, fitted in enumerate(reconstructions, start=1):
        scores[f"reconstruction_rmse_{n_factors}factor"] = np.sqrt(
            np.mean((values - fitted) ** 2, axis=1))

    reconstruction_by_cell = axes.assign(
        actual_beta_std_train=train.std(axis=0, ddof=1))
    for n_factors, fitted in enumerate(reconstructions, start=1):
        reconstruction_by_cell[
            f"train_residual_rmse_{n_factors}factor"] = np.sqrt(
                np.mean((train - fitted[:split]) ** 2, axis=0))
        reconstruction_by_cell[
            f"train_reconstruction_r_squared_{n_factors}factor"] = (
                _cell_r_squared(train, fitted[:split], train_baseline))
        reconstruction_by_cell[
            f"test_residual_rmse_{n_factors}factor"] = np.sqrt(
                np.mean((test - fitted[split:]) ** 2, axis=0))
        reconstruction_by_cell[
            f"test_reconstruction_r_squared_{n_factors}factor"] = (
                _cell_r_squared(test, fitted[split:], test_baseline))

    retained_tenors = tuple(float(x) for x in config.step4_tenors)
    retained_levels = tuple(float(x) for x in config.step4_strike_levels)
    excluded_tenors = tuple(
        float(x) for x in config.tenors if x not in retained_tenors)
    excluded_levels = tuple(
        float(x) for x in config.strike_levels if x not in retained_levels)
    validation = {
        "factor_method": schema.method,
        "factor_basis_id": fitted_id,
        "factor_columns": schema.scores,
        "loading_columns": schema.loadings,
        "input_beta_column": BETA_COLUMN,
        "input_date_count": int(beta["observation_date"].nunique()),
        "complete_surface_date_count": int(n_dates),
        "excluded_incomplete_date_count": int(
            beta["observation_date"].nunique() - n_dates),
        "train_date_count": int(split),
        "test_date_count": int(n_dates - split),
        "train_fraction": float(config.step4_train_fraction),
        "train_start_date": str(matrix.index[0]),
        "train_end_date": str(matrix.index[split - 1]),
        "test_start_date": str(matrix.index[split]),
        "test_end_date": str(matrix.index[-1]),
        "loadings_fit_sample": "chronological training dates only",
        "test_surface_role": "projection onto frozen training loadings only",
        "source_tenor_count": len(config.tenors),
        "tenor_count": len(retained_tenors),
        "retained_tenors": retained_tenors,
        "excluded_tenors": excluded_tenors,
        "source_level_count": len(config.strike_levels),
        "level_count": len(retained_levels),
        "retained_levels": retained_levels,
        "excluded_levels": excluded_levels,
        "surface_cell_count": int(matrix.shape[1]),
        "full_surface_input": bool(
            matrix.shape[1] == len(retained_tenors) * len(retained_levels)),
        "n_factors": N_FACTORS,
        "n_observed_anchor_factors": int(schema.method == "atm_anchored"),
        "n_residual_pca_factors": N_SHAPE_COMPONENTS if schema.method == "atm_anchored" else 0,
        "n_ordinary_pca_factors": N_FACTORS if schema.method == "pca" else 0,
        "centered": True,
        "variance_standardized": False,
        "one_factor_train_explained_variance_ratio": float(train_r2[0]),
        "two_factor_train_explained_variance_ratio": float(train_r2[1]),
        "three_factor_train_explained_variance_ratio": float(train_r2[2]),
        "one_factor_test_reconstruction_r_squared": float(test_r2[0]),
        "two_factor_test_reconstruction_r_squared": float(test_r2[1]),
        "three_factor_test_reconstruction_r_squared": float(test_r2[2]),
        "three_factor_train_85pct_pass": bool(train_r2[2] >= 0.85),
        "anchor_tenor": float(config.step4_anchor_tenor),
        "anchor_level": float(config.step4_anchor_level),
        "anchor_normalization_required": schema.method == "atm_anchored",
        "anchor_normalization_pass": (bool(
            np.isclose(components[0, anchor_index], 1.0)
            and np.allclose(components[1:, anchor_index], 0.0)
            and np.isclose(factor_intercept[anchor_index], 0.0))
            if schema.method == "atm_anchored" else None),
        "anchor_beta_reconstruction_rmse": float(np.sqrt(np.mean(
            (reconstructions[-1][:, anchor_index] - atm_beta) ** 2))),
        "model": "beta_hat = factor_intercept + " + " + ".join(
            f"{score} * {loading}" for score, loading in zip(schema.scores, schema.loadings)),
        "factor_parameterization": (
            "observed ATM beta plus two train-residual PCs" if schema.method == "atm_anchored"
            else "three ordinary PCs of the train-centered beta surface; PC1 is not ATM beta"),
        "missing_data_policy": (
            "use configured Step 4 axes and usable daily ratios, then require "
            "a complete retained surface; no imputation"),
    }
    if schema.method == "atm_anchored":
        validation.update(anchor_atm_loading=float(components[0, anchor_index]),
                          anchor_shape_loading_1=float(components[1, anchor_index]),
                          anchor_shape_loading_2=float(components[2, anchor_index]))
    # Fixed, evenly spaced training dates; never choose examples by test performance.
    example_indices = np.unique(np.linspace(0, split - 1, min(5, split), dtype=int))
    examples = pd.concat([
        axes.assign(observation_date=matrix.index[i], actual_beta=values[i],
                    factor_intercept=factor_intercept,
                    reconstructed_beta_1factor=reconstructions[0][i],
                    reconstructed_beta_2factor=reconstructions[1][i],
                    reconstructed_beta=reconstructions[-1][i],
                    atm_beta_observed=atm_beta[i]) for i in example_indices], ignore_index=True)
    return Step4Result(
        config, explained, loadings, scores, reconstruction_by_cell,
        date_coverage, validation, examples)


def _save_plots(result: Step4Result, target: Path) -> list[str]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/svi-localvol-mpl")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    schema = schema_for(result.config.step4_factor_method)
    files: list[str] = []
    shown = result.explained_variance
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(shown["factor_number"],
           shown["incremental_train_explained_variance_ratio"])
    ax.plot(shown["factor_number"],
            shown["cumulative_train_explained_variance_ratio"], marker="o",
            label="train")
    ax.plot(shown["factor_number"],
            shown["cumulative_test_reconstruction_r_squared"], marker="o",
            label="test projection")
    ax.axhline(0.85, color="grey", linestyle="--", linewidth=1)
    ax.set(xticks=(1, 2, 3), xlabel="number of retained factors",
           ylabel="explained variance / reconstruction R-squared",
           title=f"Daily-beta reconstruction ({schema.method})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = "explained_variance.png"
    fig.savefig(target / name, dpi=160)
    plt.close(fig)
    files.append(name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    columns = schema.loadings
    titles = schema.labels
    for ax, column, title in zip(axes, columns, titles):
        table = result.loadings.pivot(
            index="tenor", columns="level", values=column)
        image = ax.imshow(table, aspect="auto", origin="lower", cmap="coolwarm")
        ax.set_xticks(range(len(table.columns)), labels=[
            f"{x:g}" for x in table.columns], rotation=45)
        ax.set_yticks(range(len(table.index)), labels=[
            f"{12 * x:g}M" for x in table.index])
        ax.set(xlabel="strike level", title=f"{title} loading")
        fig.colorbar(image, ax=ax, shrink=0.8)
    axes[0].set_ylabel("constant tenor")
    fig.tight_layout()
    name = "factor_loadings.png"
    fig.savefig(target / name, dpi=160)
    plt.close(fig)
    files.append(name)

    dates = pd.to_datetime(result.scores["observation_date"])
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, column, label in zip(
            axes,
            schema.scores, schema.labels):
        ax.plot(dates, result.scores[column])
        ax.set_ylabel(label)
    axes[-1].set_xlabel("observation date")
    fig.suptitle("Daily-beta surface factor scores")
    fig.tight_layout()
    name = "factor_scores.png"
    fig.savefig(target / name, dpi=160)
    plt.close(fig)
    files.append(name)
    tenors = result.surface_examples.tenor.unique()
    fig, axes = plt.subplots(2, len(tenors), figsize=(3 * len(tenors), 6), squeeze=False)
    for j, tenor in enumerate(tenors):
        subset = result.surface_examples[result.surface_examples.tenor.eq(tenor)]
        for date, group in subset.groupby("observation_date"):
            group = group.sort_values("level")
            axes[0, j].plot(group.level, group.actual_beta, label=str(date))
            atm = float(group.atm_beta_observed.iloc[0])
            if abs(atm) > 1e-8:
                axes[1, j].plot(group.level, group.actual_beta / atm)
        axes[0, j].set_title(f"{12 * tenor:g}M")
        axes[1, j].set_xlabel("K / spot")
    axes[0, 0].set_ylabel("Observed beta")
    axes[1, 0].set_ylabel("Beta / observed reference ATM beta")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Fixed training-date cross sections (near-zero ATM ratios omitted)")
    fig.tight_layout()
    name = "surface_cross_sections.png"
    fig.savefig(target / name, dpi=160)
    plt.close(fig)
    files.append(name)
    return files


def save_step4(result: Step4Result, *, step2_beta_path: str | Path,
               outdir: str | Path = "output/dynamic_alpha/step04") -> Path:
    """Save the daily-beta factor model and its upstream provenance."""
    source = Path(step2_beta_path)
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    result.explained_variance.to_csv(
        target / "explained_variance.csv", index=False)
    result.loadings.to_csv(target / "factor_loadings.csv", index=False)
    result.scores.to_csv(target / "factor_scores.csv", index=False)
    result.reconstruction_by_cell.to_csv(
        target / "reconstruction_by_cell.csv", index=False)
    result.date_coverage.to_csv(target / "date_coverage.csv", index=False)
    result.surface_examples.to_csv(target / "surface_examples.csv", index=False)
    # Remove superseded rolling-PCA artefacts when replacing an old run.
    (target / "pca_loadings.csv").unlink(missing_ok=True)
    (target / "pc_loadings.png").unlink(missing_ok=True)
    for plot in ("explained_variance.png", "factor_loadings.png",
                 "factor_scores.png"):
        (target / plot).unlink(missing_ok=True)
    result.validation["plot_files"] = _save_plots(result, target)

    from .step04_report import save_readable_report
    result.validation["plot_files"] += save_readable_report(result, target, source)
    result.validation["readable_report"] = "READ_ME_FIRST_CN.html"

    inputs = {
        "step02_beta_daily": str(source),
        "step02_beta_daily_sha256": file_sha256(source),
    }
    step2_manifest = source.parent / "manifest.json"
    if step2_manifest.exists():
        inputs["step02_manifest"] = str(step2_manifest)
        inputs["step02_manifest_sha256"] = file_sha256(step2_manifest)
    return write_manifest(
        target / "manifest.json", stage="dynamic_alpha_step04",
        config=result.config, inputs=inputs, validation=result.validation)

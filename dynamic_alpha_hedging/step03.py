"""Dynamic Alpha Step 3: measure and invert the model beta(alpha) map.

The strike is fixed at ``level * calibration refSpot`` while spot is bumped up
and down.  For every alpha, two alpha-specific local-vol grids reprice the same
option.  Inverting the prices to Black implied vols gives

    beta(alpha) = -(IV_up - IV_down) / log(S_up / S_down).

This is the fixed-strike surface response estimated in Step 2.  The converter
keeps the raw Monte-Carlo shape: it only subtracts beta(alpha=1), so no
monotonic regression can manufacture an invertible relationship.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from svi_localvol.montecarlo import LocalVolGrid, LocalVolMC
from svi_localvol.surface import VolSurface

from .artifacts import file_sha256, write_manifest
from .config import DynamicAlphaConfig
from .data_loader import SurfaceHistory, date_at_tau, load_surface_history


@dataclass
class Step3Result:
    """The measured curve, accepted inverse knots and cell-level audit."""

    config: DynamicAlphaConfig
    calibration_date: object
    selected_quotes: pd.DataFrame
    curve: pd.DataFrame
    inverse: pd.DataFrame
    quality: pd.DataFrame
    validation: dict[str, object]


def _supports_tenors(surface: VolSurface, tenors: tuple[float, ...]) -> bool:
    return bool(min(tenors) >= surface.taus[0]
                and max(tenors) <= surface.taus[-1])


def _select_surface(history: SurfaceHistory,
                    config: DynamicAlphaConfig) -> tuple[object, VolSurface]:
    requested = config.step3_calibration_date
    if requested is not None:
        if requested not in history.surfaces:
            raise ValueError(
                f"Step 3 calibration date {requested} has no usable surface")
        surface = history[requested]
        if not _supports_tenors(surface, config.tenors):
            raise ValueError(
                f"Step 3 calibration date {requested} does not cover every "
                "configured tenor without extrapolation")
        return requested, surface

    candidates = [date for date in history.dates
                  if _supports_tenors(history[date], config.tenors)]
    if not candidates:
        raise ValueError("no usable surface covers every configured Step 3 tenor")

    # One fixed, representative SVI state: choose the medoid of the configured
    # IV panel rather than a potentially exceptional first/last observation.
    levels = np.asarray(config.strike_levels, dtype=float)
    vectors = []
    for date in candidates:
        surface = history[date]
        vector = []
        for tenor in config.tenors:
            maturity = date_at_tau(surface, tenor)
            vector.extend(np.asarray(surface.implied_vol(
                maturity, levels * surface.ref_spot), dtype=float))
        vectors.append(vector)
    matrix = np.asarray(vectors, dtype=float)
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    scale = np.where(mad > 1e-8, 1.4826 * mad, matrix.std(axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    distance = np.mean(((matrix - median) / scale) ** 2, axis=1)
    selected = candidates[int(np.argmin(distance))]
    return selected, history[selected]


def _quote_frame(surface: VolSurface) -> pd.DataFrame:
    return pd.DataFrame([{
        "calibration_date": surface.market.pricing_date,
        "ref_spot": surface.ref_spot,
        "VolDate": quote.vol_date,
        "ATMVol": quote.atm_vol,
        "Skew": quote.skew,
        "Putwing": quote.putwing,
        "Callwing": quote.callwing,
        "Kurt": quote.kurt,
        "StickinessRatio": quote.alpha,
    } for quote in surface.quotes.quotes])


def _anchor_at_sticky_strike(raw: pd.DataFrame) -> pd.DataFrame:
    """Subtract raw beta(1) within each cell; never reshape the MC curve."""
    groups = []
    for _, group in raw.groupby(["tenor", "level"], sort=True):
        group = group.sort_values("alpha").copy()
        anchor_rows = group[np.isclose(group["alpha"], 1.0)]
        if len(anchor_rows) != 1:
            raise ValueError("each Step 3 cell must contain exactly one alpha=1")
        anchor = float(anchor_rows["beta_model"].iloc[0])
        group["beta_alpha_one_raw"] = anchor
        group["beta_converter"] = group["beta_model"] - anchor
        groups.append(group)
    return pd.concat(groups, ignore_index=True)


def _cell_quality(curve: pd.DataFrame,
                  config: DynamicAlphaConfig) -> pd.DataFrame:
    """Audit numerical quality without suppressing measured curve output."""
    rows = []
    for (tenor, level), group in curve.groupby(["tenor", "level"], sort=True):
        ordered = group.sort_values("alpha")
        beta = ordered["beta_model"].to_numpy(dtype=float)
        stderr = ordered["beta_model_stderr"].to_numpy(dtype=float)
        alpha_one = ordered[np.isclose(ordered["alpha"], 1.0)].iloc[0]
        span = float(beta[0] - beta[-1])
        span_stderr = float(np.hypot(stderr[0], stderr[-1]))
        span_z = (span / span_stderr if span_stderr > 0.0
                  else (np.inf if span > 0.0 else 0.0))
        max_stderr = float(np.max(stderr))
        max_undefined = float(ordered["grid_undefined_fraction"].max())
        max_clipped = float(ordered["grid_clipped_fraction"].max())

        checks = {
            "raw_beta_strictly_decreasing": bool(np.all(np.diff(beta) < 0.0)),
            "alpha_one_abs_pass": bool(
                abs(float(alpha_one["beta_model"]))
                <= config.step3_alpha_one_abs_tolerance),
            "stderr_pass": bool(np.isfinite(stderr).all()
                                and max_stderr <= config.step3_max_beta_stderr),
            "span_pass": bool(span >= config.step3_min_beta_span
                              and span_z >= config.step3_min_span_z),
            "price_inversion_pass": bool(
                not ordered["price_clipped_for_inversion"].any()),
            "grid_quality_pass": bool(
                max_undefined <= config.step3_max_grid_undefined_fraction
                and max_clipped <= config.step3_max_grid_clipped_fraction),
        }
        failed = [name for name, passed in checks.items() if not passed]
        inverse_available = checks["raw_beta_strictly_decreasing"]
        rows.append({
            "calibration_date": ordered["calibration_date"].iloc[0],
            "tenor": float(tenor),
            "level": float(level),
            "beta_alpha_0": float(beta[0]),
            "beta_alpha_1": float(alpha_one["beta_model"]),
            "beta_alpha_2": float(beta[-1]),
            "beta_span": span,
            "beta_span_stderr": span_stderr,
            "beta_span_z": span_z,
            "max_beta_stderr": max_stderr,
            "max_grid_undefined_fraction": max_undefined,
            "max_grid_clipped_fraction": max_clipped,
            **checks,
            "quality_pass": not failed,
            # This is a mathematical requirement, not a quality threshold:
            # a non-monotone sampled curve has no unique piecewise-linear
            # beta -> alpha inverse.  Its raw curve is still retained.
            "inverse_available": inverse_available,
            "quality_failures": ";".join(failed),
        })
    return pd.DataFrame(rows)


def _inverse_knots(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tenor, level), group in curve.groupby(["tenor", "level"], sort=True):
        if not bool(group["inverse_available"].iloc[0]):
            continue
        for row in group.sort_values("beta_converter").itertuples():
            rows.append({
                "calibration_date": row.calibration_date,
                "tenor": float(tenor),
                "level": float(level),
                "beta": float(row.beta_converter),
                "alpha": float(row.alpha),
            })
    return pd.DataFrame(rows, columns=[
        "calibration_date", "tenor", "level", "beta", "alpha"])


def alpha_from_beta(inverse: pd.DataFrame, *, tenor: float, level: float,
                    beta, clip: bool = False):
    """Linearly invert one accepted cell; extrapolation is opt-in."""
    cell = inverse[
        np.isclose(inverse["tenor"], tenor)
        & np.isclose(inverse["level"], level)].sort_values("beta")
    if cell.empty:
        raise KeyError(f"no accepted inverse curve for tenor={tenor}, level={level}")
    x = cell["beta"].to_numpy(dtype=float)
    y = cell["alpha"].to_numpy(dtype=float)
    target = np.asarray(beta, dtype=float)
    if not clip and ((target < x[0]).any() or (target > x[-1]).any()):
        raise ValueError(
            f"beta lies outside measured range [{x[0]:.6g}, {x[-1]:.6g}]")
    result = np.interp(target, x, y, left=y[0], right=y[-1])
    return float(result) if target.ndim == 0 else result


def run_step3(config: DynamicAlphaConfig = DynamicAlphaConfig(), *,
              history: SurfaceHistory | None = None) -> Step3Result:
    """Measure beta(alpha) on one fixed, fully covered SVI surface."""
    if history is None:
        history = load_surface_history(
            config.data_path, config.market_conventions,
            beta_clamp=config.beta_clamp,
            duplicate_vol_date_policy=config.duplicate_vol_date_policy,
            source_timezone=config.source_timezone,
            market_timezone=config.market_timezone,
        )
    calibration_date, raw_surface = _select_surface(history, config)
    surface = raw_surface.repaired()
    spot = float(surface.ref_spot)
    bump = config.step3_spot_bump_fraction * spot
    spot_up, spot_down = spot + bump, spot - bump
    dlog_spot = float(np.log(spot_up / spot_down))
    strikes = np.asarray(config.strike_levels, dtype=float) * spot
    records = []

    for tenor in config.tenors:
        maturity = date_at_tau(surface, tenor)
        base_grid = LocalVolGrid.build(
            surface, maturity, n_ratio=config.step3_n_ratio,
            ratio_min=config.step3_ratio_min,
            ratio_max=config.step3_ratio_max,
            vol_floor=config.step3_vol_floor, vol_cap=config.step3_vol_cap)
        mc = LocalVolMC(
            surface, base_grid, n_paths=config.step3_n_paths,
            seed=config.step3_seed, antithetic=config.step3_antithetic,
            n_substeps=config.step3_n_substeps)

        for alpha in config.step3_alphas:
            grids = []
            for bumped_spot in (spot_up, spot_down):
                grids.append(LocalVolGrid.build(
                    surface, maturity, n_ratio=config.step3_n_ratio,
                    ratio_min=config.step3_ratio_min,
                    ratio_max=config.step3_ratio_max,
                    spot_adj=float(np.log(bumped_spot / spot)), alpha=alpha,
                    vol_floor=config.step3_vol_floor,
                    vol_cap=config.step3_vol_cap))
            grid_up, grid_down = grids
            measured = mc.bumped_implied_vol_diagnostics(
                strikes, maturity, grid_up=grid_up, grid_down=grid_down,
                bump=bump)
            measured.insert(0, "calibration_date", calibration_date)
            measured.insert(1, "tenor", float(tenor))
            measured.insert(2, "actual_expiry", maturity)
            measured.insert(3, "alpha", float(alpha))
            measured["dlogS_bump"] = dlog_spot
            grid_size = float(grid_up.sigma.size)
            measured["grid_undefined_fraction"] = np.maximum(
                measured["grid_up_n_undefined"] / grid_size,
                measured["grid_down_n_undefined"] / grid_size)
            measured["grid_clipped_fraction"] = np.maximum(
                measured["grid_up_n_clipped"] / grid_size,
                measured["grid_down_n_clipped"] / grid_size)
            records.append(measured)

    curve = _anchor_at_sticky_strike(pd.concat(records, ignore_index=True))
    quality = _cell_quality(curve, config)
    curve = curve.merge(
        quality[["tenor", "level", "quality_pass", "inverse_available",
                 "quality_failures"]],
        on=["tenor", "level"], how="left", validate="many_to_one")
    inverse = _inverse_knots(curve)
    alpha_one = curve[np.isclose(curve["alpha"], 1.0)]
    validation = {
        "calibration_date": calibration_date,
        "surface_policy": "calendar-repaired representative SVI surface",
        "selection_policy": (
            "explicit configured date" if config.step3_calibration_date
            else "robust IV-grid medoid among fully covered surfaces"),
        "n_tenors": len(config.tenors),
        "n_levels": len(config.strike_levels),
        "n_alphas": len(config.step3_alphas),
        "n_curve_rows": len(curve),
        "fixed_strike_pass": bool(np.allclose(
            curve["strike"], curve["level"] * spot)),
        "alpha_one_abs_pass_count": int(quality["alpha_one_abs_pass"].sum()),
        "alpha_one_total_count": len(quality),
        "converter_alpha_one_zero_pass": bool(np.allclose(
            alpha_one["beta_converter"], 0.0)),
        "raw_monotone_cell_count": int(
            quality["raw_beta_strictly_decreasing"].sum()),
        "quality_pass_cell_count": int(quality["quality_pass"].sum()),
        "inverse_available_cell_count": int(
            quality["inverse_available"].sum()),
        "converter_total_cell_count": len(quality),
        "price_inversion_clip_count": int(
            curve["price_clipped_for_inversion"].sum()),
        "max_beta_stderr": float(curve["beta_model_stderr"].max()),
        "max_grid_undefined_fraction": float(
            curve["grid_undefined_fraction"].max()),
        "max_grid_clipped_fraction": float(
            curve["grid_clipped_fraction"].max()),
        "primary_beta": "beta_model",
        "converter_beta": "raw beta minus raw beta(alpha=1)",
        "postprocessing": "constant alpha=1 anchor only; no PAVA/projection",
        "definition": (
            "fixed strike; alpha-specific local-vol MC spot bump; prices "
            "inverted to implied vol; beta=-dIV/dlogS"),
    }
    return Step3Result(
        config, calibration_date, _quote_frame(raw_surface), curve, inverse,
        quality, validation)


def _save_plots(result: Step3Result, target: Path) -> list[str]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/svi-localvol-mpl")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    # Quality is displayed in the audit CSV; plots always show measured data.
    plotted = result.curve
    files = []
    atm = plotted[np.isclose(plotted["level"], 1.0)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for tenor, group in atm.groupby("tenor"):
        ax.plot(group["alpha"], group["beta_converter"], marker="o",
                label=f"{12 * tenor:g}M")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set(xlabel="alpha", ylabel="model beta",
           title="ATM beta(alpha) curves")
    if not atm.empty:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    name = "beta_alpha_atm.png"
    fig.savefig(target / name, dpi=160)
    plt.close(fig)
    files.append(name)

    representative = min(result.config.tenors, key=lambda x: abs(x - 0.25))
    slice_ = plotted[np.isclose(plotted["tenor"], representative)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for level, group in slice_.groupby("level"):
        ax.plot(group["alpha"], group["beta_converter"], marker="o",
                label=f"K/S={level:g}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set(xlabel="alpha", ylabel="model beta",
           title=f"{12 * representative:g}M beta(alpha) curves")
    if not slice_.empty:
        ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    name = "beta_alpha_3m_smile.png"
    fig.savefig(target / name, dpi=160)
    plt.close(fig)
    files.append(name)
    return files


_CURVE_COLUMNS = [
    "calibration_date", "tenor", "actual_expiry", "level", "strike",
    "alpha", "option_type", "spot_up", "spot_down", "dlogS_bump",
    "base_implied_vol", "implied_vol_up", "implied_vol_down", "dIV_model",
    "beta_model", "beta_model_stderr", "beta_alpha_one_raw",
    "beta_converter", "pv_up", "pv_down", "pv_up_stderr", "pv_down_stderr",
    "price_clipped_for_inversion", "grid_undefined_fraction",
    "grid_clipped_fraction", "n_paths", "n_steps", "quality_pass",
    "inverse_available", "quality_failures",
]


def save_step3(result: Step3Result,
               outdir: str | Path = "output/dynamic_alpha/step03") -> Path:
    """Write only the production converter, its audit and provenance."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    result.selected_quotes.to_csv(target / "selected_svi_quotes.csv", index=False)
    result.curve[_CURVE_COLUMNS].to_csv(
        target / "beta_alpha_curve.csv", index=False)
    result.quality.to_csv(target / "cell_quality.csv", index=False)
    result.inverse.to_csv(target / "alpha_beta_inverse.csv", index=False)
    plots = _save_plots(result, target)
    result.validation["plot_files"] = plots
    return write_manifest(
        target / "manifest.json", stage="dynamic_alpha_step03",
        config=result.config,
        inputs={
            "svi_parameters": str(result.config.data_path),
            "svi_parameters_sha256": file_sha256(result.config.data_path),
        },
        validation=result.validation)

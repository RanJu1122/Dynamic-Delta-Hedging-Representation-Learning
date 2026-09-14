"""Model-independent historical MC preparation and fixed-date diagnostic plots."""

from pathlib import Path
from contextlib import contextmanager
import fcntl

import numpy as np
import pandas as pd

from .artifacts import file_sha256, write_manifest
from .data_loader import (load_surface_history, date_at_tau, observation_exclusions,
                          raw_quote_frame)
from .mc_library import MCLibrary, axis
from .step03 import _cell_quality


def middle_dates(dates, count=3):
    """Consecutive central available snapshots; never selected by hedge P&L/quality."""
    if count not in (3, 5):
        raise ValueError("plot date count must be 3 or 5")
    count = min(count, len(dates))
    start = (len(dates)-count)//2
    return list(dates[start:start+count])


def _plots(frame, directory, date):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    output = []
    # A four-dimensional relation needs slices: show both ATM term and 3M smile.
    for name, subset, y in (
        ("atm", frame[np.isclose(frame.level, 1.)], "tenor"),
        ("3m", frame[np.isclose(frame.tenor, .25)], "level"),
    ):
        if subset.empty:
            continue
        fig = plt.figure(figsize=(13, 5))
        for n, column in enumerate(("beta_model", "beta_converter"), 1):
            pivot = subset.pivot(index=y, columns="alpha", values=column).sort_index()
            xgrid, ygrid = np.meshgrid(pivot.columns.to_numpy(float), pivot.index.to_numpy(float))
            ax = fig.add_subplot(1, 2, n, projection="3d")
            if min(pivot.shape) >= 2:
                ax.plot_surface(xgrid, ygrid, pivot.to_numpy(float), cmap="viridis", alpha=.8)
            else:
                ax.scatter(xgrid.ravel(), ygrid.ravel(), pivot.to_numpy(float).ravel())
            anchor = pivot.columns[np.isclose(pivot.columns, 1.)][0]
            ax.plot(np.ones(len(pivot)), pivot.index, pivot[anchor], color="red", label="alpha=1")
            ax.plot(np.ones(len(pivot)), pivot.index, np.zeros(len(pivot)), "k--", label="beta=0")
            ax.set(xlabel="alpha", ylabel=y, zlabel="beta", title=f"{date} {name}: {column}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = directory / f"{date}_{name}_beta_alpha.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        output.append(str(path))
    tenors = sorted(frame.tenor.unique())
    fig = plt.figure(figsize=(12, 4 * len(tenors)))
    for i, tenor in enumerate(tenors):
        subset = frame[np.isclose(frame.tenor, tenor)]
        for j, column in enumerate(("beta_model", "beta_converter")):
            pivot = subset.pivot(index="level", columns="alpha", values=column).sort_index()
            x, y = np.meshgrid(pivot.columns.to_numpy(float), pivot.index.to_numpy(float))
            ax = fig.add_subplot(len(tenors), 2, 2*i+j+1, projection="3d")
            if min(pivot.shape) >= 2:
                ax.plot_surface(x, y, pivot.to_numpy(float), cmap="viridis", alpha=.8)
            else:
                ax.scatter(x.ravel(), y.ravel(), pivot.to_numpy(float).ravel())
            anchor = pivot.columns[np.isclose(pivot.columns, 1.)][0]
            ax.plot(np.ones(len(pivot)), pivot.index, pivot[anchor], "r-", label="alpha=1 raw/centered")
            ax.plot(np.ones(len(pivot)), pivot.index, np.zeros(len(pivot)), "k--", label="beta=0")
            ax.set(xlabel="alpha", ylabel="K/refSpot", zlabel="beta",
                   title=f"{date} tenor={tenor:g}: {column}")
            ax.legend(fontsize=7)
    fig.tight_layout()
    path = directory / f"{date}_all_tenors_beta_alpha.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    output.append(str(path))
    _csv(frame, directory / f"{date}_beta_alpha.csv")
    return output


def _csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


@contextmanager
def _writer_lock(target):
    """One nonblocking writer per library; process exit releases the lock."""
    target.mkdir(parents=True, exist_ok=True)
    with (target / ".precompute.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another precompute is writing {target}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_precompute(config, *, outdir, tenors, levels, plot_count=3, plan_only=False,
                   progress=print):
    """Build all valid date/tenor shards; quality failures are audited, not hidden.

    Unsupported tenor cells are listed, never extrapolated. Runtime pricing
    errors stop the run; completed shards survive and resume on the next call.
    """
    tenors, levels = axis(tenors, "tenors"), axis(levels, "levels")
    if plot_count not in (3, 5):
        raise ValueError("plot_count must be 3 or 5")
    if not plan_only:
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise ImportError('Install plotting first: python -m pip install -e ".[plots]"') from exc
    history = load_surface_history(
        config.data_path, config.market_conventions, beta_clamp=config.beta_clamp,
        duplicate_vol_date_policy=config.duplicate_vol_date_policy,
        source_timezone=config.source_timezone, market_timezone=config.market_timezone)
    if not history.dates:
        raise ValueError("no usable observations")
    if config.require_weekday_observations and any(d.weekday() >= 5 for d in history.dates):
        raise ValueError("weekend source observations: resolve date convention before MC")
    target = Path(outdir)
    selected = middle_dates(history.dates, plot_count)
    coverage = []
    for date in history.dates:
        surface = history[date]
        for tenor in tenors:
            expiry = date_at_tau(surface, tenor)
            supported = (surface.taus[0] <= tenor <= surface.taus[-1]
                         and surface.taus[0] <= surface.tau_vol(expiry) <= surface.taus[-1])
            coverage.append({"date": date, "tenor": tenor, "actual_expiry": expiry,
                             "supported": supported,
                             "reason": "" if supported else "outside quoted tenor range"})
    coverage = pd.DataFrame(coverage)
    work = coverage[coverage.supported]
    if work.empty:
        raise ValueError("no supported date/tenor jobs; choose tenors within quoted coverage")
    validation = {
        "status": "planned", "mc_run": False,
        "observation_count": len(history.dates), "date_tenor_jobs": len(work),
        "bump_pairs_before_cache": len(work)*len(config.step3_alphas),
        "unsupported_date_tenor_cells": int((~coverage.supported).sum()),
        "excluded_observation_dates": observation_exclusions(),
        "plot_dates": selected, "requested_plot_count": plot_count, "plots": [],
        "plot_policy": "middle valid snapshots; all-tenor alpha/level slices plus ATM term slice",
        "causality": "each shard uses its own dated quotes only; no beta labels/models",
        "beta_policy": "raw and alpha-one-centered beta retained; raw MC delta unchanged",
        "quality_policy": "audit only; completion is not a convergence certificate",
        "completed_jobs": 0, "reused_jobs": 0, "computed_jobs": 0,
    }
    inputs = {"svi_parameters": str(config.data_path),
              "svi_parameters_sha256": file_sha256(config.data_path)}

    def manifest(name):
        return write_manifest(target / name, stage="dynamic_alpha_mc_precompute",
                              config={"numerical_and_market": config.__dict__,
                                      "tenors": tenors, "levels": levels},
                              inputs=inputs, validation=validation)

    with _writer_lock(target):
        # Validate an existing library BEFORE touching its reports or manifests.
        library = None
        if (target / "library.json").exists():
            library = MCLibrary(target, config, tenors, levels, writable=True)
        if not plan_only and library is None:
            library = MCLibrary(target, config, tenors, levels, writable=True)
        _csv(coverage, target / "coverage.csv")
        _csv(history.skipped, target / "excluded_observations.csv")
        _csv(raw_quote_frame(config.data_path, source_timezone=config.source_timezone,
                            market_timezone=config.market_timezone, holidays=config.holidays),
             target / "raw_svi_quotes.csv")
        _csv(pd.DataFrame({"date": selected}), target / "plot_dates.csv")
        manifest("plan.json")
        progress(f"MC plan: {len(history.dates)} dates, {len(work)} supported date/tenor jobs, "
                 f"{validation['bump_pairs_before_cache']} bump pairs before cache")
        if plan_only:
            return validation
        validation.update(status="running", mc_run=True)
        manifest("manifest.json")
        audits, pictures, chosen = [], [], {}
        active_job = None
        try:
            for n, row in enumerate(work.itertuples(index=False), 1):
                active_job = {"date": row.date, "tenor": row.tenor}
                progress(f"Precompute {n}/{len(work)}: {row.date}, tenor={row.tenor:g}")
                key, _ = library._key(history[row.date], row.tenor)
                reused = key in library.index
                frame = library.ensure(history[row.date], row.tenor)
                audits.append(_cell_quality(frame, config))
                validation["completed_jobs"] += 1
                validation["reused_jobs" if reused else "computed_jobs"] += 1
                manifest("manifest.json")
                if row.date in selected:
                    chosen.setdefault(row.date, []).append(frame)
            active_job = {"phase": "quality_reports_and_plots"}
            quality = pd.concat(audits, ignore_index=True)
            _csv(quality, target / "quality.csv")
            _csv(quality[["calibration_date", "tenor", "level", "beta_alpha_1", "alpha_one_abs_pass",
                          "max_beta_stderr", "quality_pass", "quality_failures"]],
                 target / "alpha_one_audit.csv")
            for date, pieces in chosen.items():
                pictures.extend(_plots(pd.concat(pieces, ignore_index=True), target / "plots", date))
            validation.update(status="complete", plots=pictures,
                              plot_dates_without_supported_tenors=[d for d in selected if d not in chosen],
                              quality_pass_cells=int(quality.quality_pass.sum()), quality_cells=len(quality))
            manifest("manifest.json")
        except BaseException as exc:
            validation.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                              failed_job=active_job, error=f"{type(exc).__name__}: {exc}")
            manifest("manifest.json")
            raise
    return validation

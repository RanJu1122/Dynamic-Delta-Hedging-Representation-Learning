"""Preflight boundary for the dynamic-alpha hedging study.

This module intentionally does not estimate beta, fit a model or run a hedge.
It makes the research inputs and conventions explicit, audits the pickle, and
checks that the existing SVI pipeline can calibrate every observation under a
declared repair policy.  Later stages enter through this module and
``data_loader.py`` and use only the document's alpha convention.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from svi_localvol.conventions import is_biz_day, to_date
from svi_localvol.params import ALPHA_STICKY_STRIKE, validate_stickiness_alpha

from .config import DEFAULT_DATA_PATH, DynamicAlphaConfig
from .data_loader import (REQUIRED_QUOTE_FIELDS, SurfaceHistory,
                          load_quote_file, load_surface_history)


@dataclass(frozen=True)
class QuoteAudit:
    n_records: int
    first_date: dt.date
    last_date: dt.date
    sha256: str
    quote_count_values: tuple[int, ...]
    alpha_values: tuple[float, ...]
    non_business_dates: tuple[dt.date, ...]
    unsorted_vol_date_records: tuple[dt.date, ...]
    malformed: pd.DataFrame = field(repr=False)


@dataclass
class PreflightReport:
    config: DynamicAlphaConfig
    audit: QuoteAudit
    strict_history: SurfaceHistory | None
    prepared_history: SurfaceHistory | None
    tenor_coverage: pd.DataFrame
    blockers: list[str]
    warnings: list[str]

    @property
    def ready_for_step1(self) -> bool:
        return not self.blockers

    def format(self) -> str:
        a = self.audit
        status = "READY" if self.ready_for_step1 else "BLOCKED"
        lines = [
            f"dynamic-alpha preflight: {status}",
            f"data: {a.n_records} observations, {a.first_date} -> {a.last_date}",
            f"sha256: {a.sha256}",
            f"quote counts per day: {list(a.quote_count_values)}",
            f"StickinessRatio values: {list(a.alpha_values)}",
            "alpha convention: 0=sticky local vol, 1=sticky strike, "
            "2=sticky moneyness",
            f"strikes: {self.config.strike_levels[0]:.2f} .. "
            f"{self.config.strike_levels[-1]:.2f} (K = level * refSpot)",
            f"tenors: {list(self.config.tenors)} Business/260 years; "
            f"extrapolation={self.config.extrapolation}",
            f"rates: r={self.config.rate:g}, q={self.config.dividend:g}, "
            f"repo={self.config.repo:g}; calendar={self.config.holiday_calendar_name}",
            f"stored non-business observation dates: {len(a.non_business_dates)}",
            f"records whose VolDate arrays needed aligned sorting: "
            f"{len(a.unsorted_vol_date_records)}",
        ]
        if self.strict_history is not None and self.prepared_history is not None:
            lines.append(
                f"surface calibration: strict={len(self.strict_history)}/"
                f"{a.n_records}, configured={len(self.prepared_history)}/"
                f"{a.n_records} (SVI-JW beta_clamp={self.config.beta_clamp:g})")
        if not self.tenor_coverage.empty:
            lines.append("tenor nodes outside daily quote ranges "
                         "(kept as NaN in Step 1):")
            for row in self.tenor_coverage.itertuples(index=False):
                lines.append(f"  {row.tenor:g}y: {row.n_extrapolated}/"
                             f"{row.n_dates}")
        if self.blockers:
            lines.append("blockers:")
            lines.extend(f"  - {x}" for x in self.blockers)
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - {x}" for x in self.warnings)
        return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_quote_file(config: DynamicAlphaConfig) -> QuoteAudit:
    """Schema and calendar audit; no SVI calibration and no research output."""
    records = load_quote_file(
        config.data_path,
        observation_date_shift=config.observation_date_shift,
        roll_observation_dates=config.roll_observation_dates,
        holidays=config.holidays,
    )
    malformed: list[dict] = []
    quote_counts: set[int] = set()
    alpha_values: set[float] = set()
    unsorted: list[dt.date] = []

    for date, record in sorted(records.items()):
        missing = [name for name in REQUIRED_QUOTE_FIELDS if name not in record]
        if missing:
            malformed.append({"date": date, "issue": "missing_fields",
                              "detail": ",".join(missing)})
            continue
        lengths = {name: len(record[name]) for name in REQUIRED_QUOTE_FIELDS
                   if name not in ("Spot",)}
        if len(set(lengths.values())) != 1:
            malformed.append({"date": date, "issue": "length_mismatch",
                              "detail": repr(lengths)})
            continue
        n = lengths["VolDate"]
        quote_counts.add(n)
        vol_dates = [to_date(x) for x in record["VolDate"]]
        if vol_dates != sorted(vol_dates):
            unsorted.append(date)
        if len(set(vol_dates)) != n:
            malformed.append({"date": date, "issue": "duplicate_vol_date",
                              "detail": str(n - len(set(vol_dates)))})
        if any(expiry <= date for expiry in vol_dates):
            malformed.append({"date": date, "issue": "expired_vol_date",
                              "detail": "VolDate must be after observation"})
        numeric = [float(record["Spot"])]
        for name in ("ATMVol", "Skew", "Putwing", "Callwing", "Kurt",
                     "StickinessRatio"):
            numeric.extend(float(x) for x in record[name])
        if not np.isfinite(numeric).all():
            malformed.append({"date": date, "issue": "non_finite",
                              "detail": "numeric field contains NaN/inf"})
        if float(record["Spot"]) <= 0:
            malformed.append({"date": date, "issue": "non_positive_spot",
                              "detail": str(record["Spot"])})
        for alpha in record["StickinessRatio"]:
            value = validate_stickiness_alpha(alpha)
            alpha_values.add(value)

    dates = sorted(records)
    if not dates:
        raise ValueError("quote file is empty")
    non_business = tuple(d for d in dates if not is_biz_day(d, config.holidays))
    return QuoteAudit(
        n_records=len(records), first_date=dates[0], last_date=dates[-1],
        sha256=_sha256(config.data_path),
        quote_count_values=tuple(sorted(quote_counts)),
        alpha_values=tuple(sorted(alpha_values)),
        non_business_dates=non_business,
        unsorted_vol_date_records=tuple(unsorted),
        malformed=pd.DataFrame(malformed, columns=["date", "issue", "detail"]),
    )


def _tenor_coverage(history: SurfaceHistory,
                    tenors: Sequence[float]) -> pd.DataFrame:
    rows = []
    for tenor in tenors:
        outside = sum(float(tenor) < surface.taus[0]
                      or float(tenor) > surface.taus[-1]
                      for surface in history.surfaces.values())
        rows.append({"tenor": float(tenor), "n_dates": len(history),
                     "n_extrapolated": int(outside),
                     "fraction_extrapolated": outside / len(history)})
    return pd.DataFrame(rows)


def run_preflight(config: DynamicAlphaConfig = DynamicAlphaConfig()) -> PreflightReport:
    """Audit all preconditions without creating Step 1 IV/dIV outputs."""
    blockers: list[str] = []
    warnings: list[str] = []
    strict: SurfaceHistory | None = None
    prepared: SurfaceHistory | None = None
    coverage = pd.DataFrame(columns=["tenor", "n_dates", "n_extrapolated",
                                     "fraction_extrapolated"])

    try:
        audit = audit_quote_file(config)
    except Exception as exc:
        # Keep the CLI diagnostic useful when a requested date transform itself
        # is invalid (notably when it creates duplicate observation dates).
        empty = QuoteAudit(0, dt.date.min, dt.date.min, "", (), (), (), (),
                           pd.DataFrame(columns=["date", "issue", "detail"]))
        return PreflightReport(config, empty, None, None, coverage,
                               [f"quote audit failed: {type(exc).__name__}: {exc}"], [])

    if not audit.malformed.empty:
        blockers.append(f"{len(audit.malformed)} malformed quote record(s)")
    if config.require_business_observation_dates and audit.non_business_dates:
        blockers.append(
            f"{len(audit.non_business_dates)} observation keys are not business "
            "dates under the declared calendar; confirm source timestamp semantics "
            "before computing dlogS/dIV")
    if audit.alpha_values != (ALPHA_STICKY_STRIKE,):
        warnings.append("input StickinessRatio is not uniformly alpha=1; inspect "
                        "whether the file already contains a dynamic policy")
    if audit.unsorted_vol_date_records:
        warnings.append(
            f"{len(audit.unsorted_vol_date_records)} records store expiries out of "
            "order; VolQuoteSet sorts aligned quote objects before calibration")
    warnings.append("r/q/repo are not present in svi_data.pkl; the configured "
                    "Task-7 values are assumptions and must be confirmed")
    warnings.append("the bundled calendar contains scheduled holidays only; append "
                    "exchange one-off closures before final research")

    try:
        strict = load_surface_history(
            config.data_path, config.market_conventions, beta_clamp=0.0,
            observation_date_shift=config.observation_date_shift,
            roll_observation_dates=config.roll_observation_dates)
        prepared = load_surface_history(
            config.data_path, config.market_conventions,
            beta_clamp=config.beta_clamp,
            observation_date_shift=config.observation_date_shift,
            roll_observation_dates=config.roll_observation_dates)
        coverage = _tenor_coverage(prepared, config.tenors)
        if len(prepared) != audit.n_records:
            blockers.append(
                f"configured SVI repair builds {len(prepared)}/{audit.n_records} "
                "daily surfaces; inspect prepared_history.skipped")
        if len(strict) != audit.n_records:
            warnings.append(
                f"{audit.n_records - len(strict)} day(s) violate the exact SVI-JW "
                f"convexity transform; configured beta_clamp={config.beta_clamp:g} "
                "makes this repair explicit")
        if int(coverage["n_extrapolated"].sum()) > 0:
            warnings.append("constant-tenor nodes outside a day's quoted expiry "
                            "range will be NaN, never flat-filled")
    except Exception as exc:
        blockers.append(f"surface calibration failed: {type(exc).__name__}: {exc}")

    return PreflightReport(config, audit, strict, prepared, coverage,
                           blockers, warnings)


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Audit readiness for dynamic-alpha research Step 1")
    parser.add_argument("data", nargs="?", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--observation-date-shift", type=int, default=0)
    parser.add_argument("--roll-observation-dates", action="store_true")
    parser.add_argument("--allow-non-business-observation-dates",
                        action="store_true")
    args = parser.parse_args()
    config = DynamicAlphaConfig(
        data_path=args.data,
        observation_date_shift=args.observation_date_shift,
        roll_observation_dates=args.roll_observation_dates,
        require_business_observation_dates=(
            not args.allow_non_business_observation_dates),
    )
    report = run_preflight(config)
    print(report.format())
    raise SystemExit(0 if report.ready_for_step1 else 2)


if __name__ == "__main__":
    cli()

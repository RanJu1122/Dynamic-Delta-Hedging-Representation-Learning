"""Calibration Step 2: implied-volatility surface and ATM validation."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from svi_localvol.surface import VolSurface


def run(surface: VolSurface, maturity, levels) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("CALIBRATION STEP 2  --  ImpliedVol(T, K)")
    print("=" * 78)
    dates = surface.date_axis(maturity, "1D")
    implied = surface.implied_vol_matrix(dates, levels)
    print(f"date axis: {len(dates)} business days, {dates[0]} -> {dates[-1]}")
    print(f"matrix shape: {implied.shape} (date x K/S0)")
    print(pd.concat([implied.head(3), implied.tail(3)]).to_string(
        float_format=lambda x: f"{x:.4f}"))
    print("\nATM checks on quoted VolDates:")
    for slice_, quote in zip(surface.slices, surface.quotes.quotes):
        forward = surface.forward(slice_.vol_date)
        got = surface.implied_vol(slice_.vol_date, forward)
        print(f"  {slice_.vol_date} got={got:.10f} target={quote.atm_vol:.10f} "
              f"err={got - quote.atm_vol:+.2e}")
    return implied


def validate(surface: VolSurface) -> pd.DataFrame:
    checks = surface.self_check()
    return checks[["VolDate", "err_impliedvol_atm", "pass"]].copy()


def save(result: pd.DataFrame, checks: pd.DataFrame, outdir: str | Path) -> None:
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    result.to_csv(target / "step02_implied_vol_matrix.csv")
    checks.to_csv(target / "step02_validation.csv", index=False)

"""Calibration Step 3: Dupire local-volatility surface and arbitrage flags."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from svi_localvol.surface import VolSurface


def run(surface: VolSurface, maturity, levels) -> dict:
    print("\n" + "=" * 78)
    print("CALIBRATION STEP 3  --  localvol(T, K, spot_adj)")
    print("=" * 78)
    report = surface.arbitrage_report()
    print("butterfly diagnostics:")
    print(report["butterfly"].drop(columns=["y_grid_lo", "y_grid_hi"]).to_string(
        index=False, float_format=lambda x: f"{x: .4f}"))
    print("\ncalendar diagnostics:")
    print(report["calendar"].drop(columns=["y_grid_lo", "y_grid_hi"]).to_string(
        index=False, float_format=lambda x: f"{x: .2e}"))

    dates = surface.date_axis(maturity, "1D")
    local = surface.local_vol_matrix(dates, levels, spot_adj=0.0)
    shifted = surface.local_vol_matrix(dates, levels, spot_adj=0.02)
    flags = surface.local_vol_arbitrage_map(dates, levels)
    n_bad = int((flags != "ok").to_numpy().sum())
    print(f"\nlocal-vol matrix {local.shape}; {n_bad}/{flags.size} points flagged")
    print(pd.concat([local.head(3), local.tail(3)]).to_string(
        float_format=lambda x: f"{x:.4f}", na_rep="nan"))
    return {"local_vol": local, "local_vol_shift": shifted, "flags": flags,
            "report": report, "dates": dates}


def validate(result: dict) -> dict:
    flags = result["flags"]
    return {"n_points": int(flags.size),
            "n_arbitrage_flagged": int((flags != "ok").to_numpy().sum())}


def save(result: dict, checks: dict, outdir: str | Path) -> None:
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    result["local_vol"].to_csv(target / "step03_local_vol_matrix.csv")
    result["local_vol_shift"].to_csv(
        target / "step03_local_vol_spot_adj_002.csv")
    result["flags"].to_csv(target / "step03_arbitrage_flags.csv")
    result["report"]["butterfly"].to_csv(
        target / "step03_butterfly.csv", index=False)
    result["report"]["calendar"].to_csv(
        target / "step03_calendar.csv", index=False)
    pd.Series(checks).to_json(target / "step03_validation.json", indent=2)

"""Calibration Step 1: SVI-JW to SVI-raw and round-trip validation."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from svi_localvol.surface import VolSurface


def run(surface: VolSurface) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("CALIBRATION STEP 1  --  SVI-JW  ->  SVI-raw")
    print("=" * 78)
    table = surface.slice_table()
    cols = ["VolDate", "dt_vol", "dt_r", "forward", "a", "b", "rho", "m", "sigma"]
    print(table[cols].to_string(index=False,
                                float_format=lambda x: f"{x:11.6f}"))
    checks = surface.self_check()
    print("\nself-checks (max abs error per slice):")
    print(checks.to_string(index=False, float_format=lambda x: f"{x: .2e}"))
    print("  " + ("all round-trip checks pass" if checks["pass"].all()
                    else "!! at least one round-trip check failed"))
    return table


def validate(surface: VolSurface) -> pd.DataFrame:
    return surface.self_check()


def save(result: pd.DataFrame, checks: pd.DataFrame, outdir: str | Path) -> None:
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    result.to_csv(target / "step01_raw_parameters.csv", index=False)
    checks.to_csv(target / "step01_validation.csv", index=False)

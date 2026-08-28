"""Orchestrate the four independently runnable calibration validation steps."""

from __future__ import annotations

from pathlib import Path

from . import plots
from .config import DEFAULT_STRIKE_LEVELS, TEST_MARKET, TEST_VOL_PARAMS
from .step01_svi_conversion import (run as run_step01, save as save_step01,
                                    validate as validate_step01)
from .step02_implied_surface import (run as run_step02, save as save_step02,
                                     validate as validate_step02)
from .step03_localvol_surface import (run as run_step03, save as save_step03,
                                      validate as validate_step03)
from .step04_mc_validation import (run as run_step04, save as save_step04,
                                   validate as validate_step04)
from .task_api import (ImpliedVol, build_surface, gen_schedule, localvol,
                       param_convert)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "pricing_calibration"


def main(n_paths: int = 100_000, outdir: Path = OUTPUT_DIR) -> dict:
    """Run Calibration Steps 1-4 and persist their acceptance artefacts."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    surface = build_surface(TEST_MARKET, TEST_VOL_PARAMS)
    levels = DEFAULT_STRIKE_LEVELS
    maturity = surface.slices[-1].vol_date
    strike = surface.market.spot

    print("=" * 78)
    print("Pricing SVI/local-vol calibration validation")
    print("=" * 78)
    step01 = run_step01(surface)
    step02 = run_step02(surface, maturity, levels)
    step03 = run_step03(surface, maturity, levels)
    repaired = surface.repaired()
    step04 = run_step04(repaired, maturity, strike, n_paths,
                        raw_surface=surface, comparison_levels=levels)

    save_step01(step01, validate_step01(surface), outdir)
    save_step02(step02, validate_step02(surface), outdir)
    save_step03(step03, validate_step03(step03), outdir)
    save_step04(step04, validate_step04(step04), outdir)

    try:
        repaired_local = repaired.local_vol_matrix(step03["dates"], levels)
        plots.plot_surfaces(step02, step03["local_vol"], repaired_local,
                            outdir / "surfaces.png")
        plots.plot_smile_slices(surface, surface.quotes.vol_dates, levels,
                                outdir / "smiles.png")
        plots.plot_step4_pricing_errors(
            step04["pricing_errors"], outdir / "step04_pricing_errors.png")
        plots.plot_terminal_variance_bins(
            step04["terminal_variance_bins"],
            outdir / "step04_terminal_variance_bins.png")
    except Exception as exc:  # noqa: BLE001
        print(f"(plots skipped: {exc})")

    print(f"\nwritten to {outdir}")
    return {"surface": surface, "implied_vol": step02,
            **step03, **step04}


__all__ = [
    "ImpliedVol", "build_surface", "gen_schedule", "localvol", "main",
    "param_convert",
]

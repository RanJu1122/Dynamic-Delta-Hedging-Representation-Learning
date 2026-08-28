"""Calibration Step 4: validate local-vol prices and deltas against BS."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from svi_localvol.blackscholes import bs_price_w
from svi_localvol.montecarlo import LocalVolGrid, LocalVolMC
from svi_localvol.surface import VolSurface

from .config import DEFAULT_STRIKE_LEVELS


def run(surface: VolSurface, maturity, strike: float, n_paths: int,
        n_substeps: int = 16, raw_surface: VolSurface | None = None,
        comparison_levels=None) -> dict:
    """Run the document's MC acceptance check on a repaired surface."""
    print("\n" + "=" * 78)
    print("CALIBRATION STEP 4  --  Local-vol Monte Carlo vs Black-Scholes")
    print("=" * 78)

    if raw_surface is not None:
        raw_grid = LocalVolGrid.build(raw_surface, maturity,
                                      n_ratio=1000, ratio_max=3.0)
        print(f"raw grid undefined points: {raw_grid.n_undefined:,}/"
              f"{raw_grid.sigma.size:,}")

    spot = float(surface.market.spot)
    bump = 0.01 * spot
    spot_up, spot_down = spot + bump, spot - bump
    shift_up = float(np.log(spot_up / surface.ref_spot))
    shift_down = float(np.log(spot_down / surface.ref_spot))

    grid = LocalVolGrid.build(surface, maturity, n_ratio=1000, ratio_max=3.0)
    grid_up = LocalVolGrid.build(surface, maturity, n_ratio=1000, ratio_max=3.0,
                                 spot_adj=shift_up)
    grid_down = LocalVolGrid.build(surface, maturity, n_ratio=1000,
                                   ratio_max=3.0, spot_adj=shift_down)
    mc = LocalVolMC(surface, grid, n_paths=n_paths, seed=20260807,
                    antithetic=True, n_substeps=n_substeps)
    result = mc.price_european(strike, maturity, is_call=True)

    levels = np.asarray(DEFAULT_STRIKE_LEVELS if comparison_levels is None
                        else comparison_levels, dtype=float)
    strikes = levels * spot
    pricing, variance_bins = mc.price_diagnostics(strikes, maturity)
    deltas = mc.bump_delta_diagnostics(
        strikes, maturity, grid_up=grid_up, grid_down=grid_down, bump=bump)

    tau_vol = surface.tau_vol(maturity)
    forward = surface.forward(maturity, spot)
    discount = surface.discount_factor(maturity)
    sigma_imp = float(surface.implied_vol(maturity, strike))
    pv_bs = float(bs_price_w(forward, strike, sigma_imp ** 2 * tau_vol,
                             discount, True))
    atm = deltas.iloc[int(np.argmin(np.abs(deltas["strike"] - strike)))]
    delta_bs = float(atm["bs_delta"])
    delta_local = float(atm["mc_local_delta"])

    print(f"paths={result.n_paths:,}; steps={result.n_steps}; "
          f"sum_dt_r={mc.total_clocks()[0]:.6f}; "
          f"sum_dt_vol={mc.total_clocks()[1]:.6f}")
    print(f"ATM PV: BS={pv_bs:.8f}, MC={result.pv:.8f}, "
          f"relative error={(result.pv / pv_bs - 1) * 100:+.4f}%")
    print(f"ATM delta: BS={delta_bs:.6f}, local-vol MC={delta_local:.6f}, "
          f"stderr={float(atm['mc_local_delta_stderr']):.6f}")

    return {
        "grid": grid,
        "grid_up": grid_up,
        "grid_down": grid_down,
        "mc": result,
        "pv_bs": pv_bs,
        "delta_bs": delta_bs,
        "delta_local": delta_local,
        "sigma_imp": sigma_imp,
        "pricing_errors": pricing,
        "delta_comparison": deltas,
        "terminal_variance_bins": variance_bins,
    }


def validate(result: dict) -> dict:
    pv_bs = float(result["pv_bs"])
    mc_result = result["mc"]
    deltas = result["delta_comparison"]
    atm = deltas.iloc[int(np.argmin(np.abs(deltas["level"] - 1.0)))]
    pv_relative_error = float(mc_result.pv / pv_bs - 1.0)
    delta_error = float(atm["mc_local_minus_bs_bump"])
    delta_stderr = float(atm["mc_local_delta_stderr"])
    return {
        "pv_relative_error": pv_relative_error,
        "pv_pass_1pct": abs(pv_relative_error) < 0.01,
        "delta_error": delta_error,
        "delta_stderr": delta_stderr,
        "delta_within_2_stderr": abs(delta_error) <= 2 * delta_stderr,
    }


def save(result: dict, checks: dict, outdir: str | Path) -> None:
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    result["grid"].as_frame().to_csv(target / "step04_localvol_grid.csv")
    result["pricing_errors"].to_csv(
        target / "step04_pricing_errors.csv", index=False)
    result["delta_comparison"].to_csv(
        target / "step04_delta_comparison.csv", index=False)
    result["terminal_variance_bins"].to_csv(
        target / "step04_terminal_variance_bins.csv", index=False)
    pd.Series(checks).to_json(target / "step04_validation.json", indent=2)

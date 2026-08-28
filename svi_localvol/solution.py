"""Pricing Practice 3 -- SVI-based local volatility calibration.

Entry point.  Exposes the function names required by the task statement

    param_convert(...)
    gen_schedule(...)
    ImpliedVol(T, K)
    localvol(T, K, spot_adj)

and a `main()` that runs Step 1 -> Step 4 on the supplied test parameters and
writes every intermediate result to ./output.

Run:
    python solution.py                 # full run, 100k paths
    python solution.py --fast          # 20k paths, for a quick smoke test
"""

from __future__ import annotations

import argparse
import datetime as dt
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from .blackscholes import bs_price_w
from .conventions import gen_schedule as _gen_schedule
from .deltas import (LOCAL_VOL_LIKE, STICKY_MONEYNESS,
                     STICKY_STRIKE, delta_finite_difference,
                     delta_range, smile_greeks)
from .montecarlo import LocalVolGrid, LocalVolMC
from .params import (DEFAULT_STRIKE_LEVELS, TEST_MARKET,
                     TEST_VOL_PARAMS, MarketData, VolQuoteSet)
from .surface import VolSurface
from .svi import jw_to_raw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# --------------------------------------------------------------------------- #
# module-level surface backing the fixed-signature functions
# --------------------------------------------------------------------------- #
_SURFACE: VolSurface | None = None


def build_surface(market: MarketData = TEST_MARKET,
                  vol_params: dict = TEST_VOL_PARAMS) -> VolSurface:
    """Build (and register) the surface used by ImpliedVol / localvol."""
    global _SURFACE
    _SURFACE = VolSurface(market, VolQuoteSet.from_dict(vol_params))
    return _SURFACE
# the surface info store at surface.slices and the shape looks like:
# surface
# └── slices
#     ├── Slice(T1, tau1, SVIRaw(a1,b1,rho1,m1,sigma1))
#     ├── Slice(T2, tau2, SVIRaw(a2,b2,rho2,m2,sigma2))
#     ├── ...
#     └── Slice(T8, tau8, SVIRaw(a8,b8,rho8,m8,sigma8))


def _surface() -> VolSurface:
    return _SURFACE if _SURFACE is not None else build_surface()


def set_surface(surface: VolSurface) -> VolSurface:
    """Point the module-level ImpliedVol / localvol at `surface`."""
    global _SURFACE
    _SURFACE = surface
    return surface


@contextmanager
def using_surface(surface: VolSurface):
    """Temporarily rebind the fixed-signature functions to another surface.

    Step 3 must run on the quotes as they are, so the module default is the
    UNREPAIRED surface and `localvol` returns NaN inside the calendar spread
    arbitrage.  Step 4 needs a surface on which the model exists everywhere.
    Rather than bypassing the delivered functions, Step 4 rebinds them for the
    duration of the grid build, so the local vol table really is produced by
    `localvol(date, K = ratio * S0, spot_adj = 0)` as the task asks.
    """
    global _SURFACE
    previous = _SURFACE
    _SURFACE = surface
    try:
        yield surface
    finally:
        _SURFACE = previous


# --------------------------------------------------------------------------- #
# required signatures
# --------------------------------------------------------------------------- #
def param_convert(atm_vol, skew, putwing, callwing, kurt, tau):
    """SVI-JW -> SVI-raw for one expiry.  Returns (a, b, rho, m, sigma)."""
    return jw_to_raw(atm_var=atm_vol ** 2, skew=skew, putwing=putwing,
                     callwing=callwing, min_imp_var=kurt ** 2,
                     tau=tau).as_tuple()


def gen_schedule(start, end, period="1D", bizconv="Following", hol=None):
    """Business-day schedule between start and end, inclusive."""
    return _gen_schedule(start, end, period=period, bizconv=bizconv, hol=hol)


def ImpliedVol(T, K):                                    # noqa: N802
    """Black-Scholes implied volatility.  T: date; K: scalar or array (K = level * S0)."""
    return _surface().implied_vol(T, K)


def localvol(T, K, spot_adj: float = 0.0, alpha: float | None = None):
    """Dupire-Gatheral local volatility, same shape as K.

    The task-mandated signature is ``localvol(T, K, spot_adj)``; `alpha` is an
    optional keyword that overrides the per-VolDate `StickinessRatio`.  The
    shift applied is ``y_adj = y - alpha * spot_adj``.
    """
    return _surface().local_vol(T, K, spot_adj=spot_adj, alpha=alpha)


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def step1_parameters(surface: VolSurface) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("STEP 1  --  SVI-JW  ->  SVI-raw")
    print("=" * 78)
    tbl = surface.slice_table()
    cols = ["VolDate", "dt_vol", "dt_r", "forward", "a", "b", "rho", "m", "sigma"]
    print(tbl[cols].to_string(index=False,
                              float_format=lambda x: f"{x:11.6f}"))

    chk = surface.self_check()
    print("\nself-checks (max abs error per slice):")
    print(chk.to_string(index=False,
                        float_format=lambda x: f"{x: .2e}"))
    if not chk["pass"].all():
        print("  !! at least one round-trip check failed")
    else:
        print("  all round-trip checks pass "
              "(w(0)=atmvar*tau, ImpliedVol(VolDate,F)=ATMVol, JW round-trip)")
    return tbl


def step2_implied(surface: VolSurface, maturity, levels) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("STEP 2  --  ImpliedVol(T, K) and the implied volatility matrix")
    print("=" * 78)
    dates = surface.date_axis(maturity, "1D")
    iv = surface.implied_vol_matrix(dates, levels)
    print(f"date axis: {len(dates)} business days, "
          f"{dates[0]} -> {dates[-1]}")
    print(f"matrix shape: {iv.shape}  (rows = dates, cols = level = K / S0)")
    print("\nfirst / last rows:")
    show = pd.concat([iv.head(3), iv.tail(3)])
    print(show.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nATM check on the VolDates -- ImpliedVol(VolDate, F) vs ATMVol:")
    for s, q in zip(surface.slices, surface.quotes.quotes):
        F = surface.forward(s.vol_date)
        got = surface.implied_vol(s.vol_date, F)
        print(f"  {s.vol_date}  F={F:.6f}  got={got:.10f}  "
              f"target={q.atm_vol:.10f}  err={got - q.atm_vol:+.2e}")
    return iv


def step3_localvol(surface: VolSurface, maturity, levels) -> dict:
    print("\n" + "=" * 78)
    print("STEP 3  --  localvol(T, K, spot_adj), Dupire-Gatheral")
    print("=" * 78)

    rep = surface.arbitrage_report()
    print("butterfly per slice (D = g > 0 required):")
    print(rep["butterfly"].drop(columns=["y_grid_lo", "y_grid_hi"]).to_string(
        index=False, float_format=lambda x: f"{x: .4f}"))
    cal = rep["calendar"]
    y_lo, y_hi = cal["y_grid_lo"].iloc[0], cal["y_grid_hi"].iloc[0]
    print("\ncalendar spread between adjacent slices "
          "(crossedness > 0 means w decreases in T somewhere)")
    print(f"measured on y = ln(K/F) in [{y_lo:+.2f}, {y_hi:+.2f}]:")
    print(cal.drop(columns=["y_grid_lo", "y_grid_hi"]).to_string(
        index=False, float_format=lambda x: f"{x: .2e}"))
    if cal["truncated"].any():
        print("  ! 'truncated' = the violation was still live at the grid edge,")
        print("    so 'crossedness' is a lower bound on that row.")
    if cal[["unbounded_call_wing", "unbounded_put_wing"]].to_numpy().any():
        print("  ! 'unbounded_*_wing' = the near slice has the steeper "
              "asymptotic wing slope b(1 +/- rho),")
        print("    so w_lo - w_hi grows linearly for all large |y|: the "
              "crossedness is analytically unbounded")
        print("    and NO finite grid can measure it.  Only the traded band "
              "is economically meaningful.")

    dates = surface.date_axis(maturity, "1D")
    lv = surface.local_vol_matrix(dates, levels, spot_adj=0.0)
    flags = surface.local_vol_arbitrage_map(dates, levels)

    n_bad = int((flags != "ok").to_numpy().sum())
    print(f"\nlocal vol matrix {lv.shape}; "
          f"{n_bad} of {flags.size} grid points are arbitrage-broken "
          f"({100 * n_bad / flags.size:.1f}%)")
    if n_bad:
        broken = (flags != "ok")
        by_col = broken.sum(axis=0)
        print("  broken points per level:")
        print("   " + "  ".join(f"{c:.2f}:{int(v)}"
                                for c, v in by_col.items() if v))
        first_bad = flags[broken.any(axis=1)].index[0]
        print(f"  first affected date: {first_bad}")
        print("  -> NaN is returned there on purpose; see arbitrage_flags.csv")

    print("\nlocal vol (spot_adj = 0), first / last rows:")
    print(pd.concat([lv.head(3), lv.tail(3)]).to_string(
        float_format=lambda x: f"{x:.4f}", na_rep="   nan"))

    lv_shift = surface.local_vol_matrix(dates, levels, spot_adj=0.02)
    mid = dates[len(dates) // 2]
    cmp = pd.DataFrame({"spot_adj=0.00": lv.loc[mid],
                        "spot_adj=0.02": lv_shift.loc[mid]})
    cmp["diff"] = cmp["spot_adj=0.02"] - cmp["spot_adj=0.00"]
    print(f"\nstickiness shift on the slice {mid}:")
    print(cmp.to_string(float_format=lambda x: f"{x:+.4f}", na_rep="   nan"))

    return {"local_vol": lv, "local_vol_shift": lv_shift, "flags": flags,
            "report": rep, "dates": dates}


def step4_montecarlo(surface: VolSurface, maturity, strike, n_paths: int,
                     n_substeps: int = 16, raw_surface: VolSurface | None = None,
                     comparison_levels=None) -> dict:
    print("\n" + "=" * 78)
    print("STEP 4  --  Monte Carlo under local vol vs Black-Scholes")
    print("=" * 78)
    print("run on the CALENDAR-REPAIRED surface.  For reference, the same grid")
    print("built on the quotes as they stand:")
    if raw_surface is not None:
        raw_grid = LocalVolGrid.build(raw_surface, maturity,
                                      n_ratio=1000, ratio_max=3.0)
        n = raw_grid.sigma.size
        print(f"  raw quotes : {raw_grid.n_undefined:,} of {n:,} grid points "
              f"({100 * raw_grid.n_undefined / n:.1f}%) have NO local vol "
              "(dw/dT < 0).")

    # Build every table through the DELIVERED localvol().  The base table uses
    # spot_adj=0 for PV.  The up/down tables implement the mentor's delta scope:
    # each bumped spot gets its own log(shifted_spot / ref_spot) adjustment.
    h = 0.01 * surface.market.spot
    spot_up = surface.market.spot + h
    spot_down = surface.market.spot - h
    spot_adj_up = float(np.log(spot_up / surface.ref_spot))
    spot_adj_down = float(np.log(spot_down / surface.ref_spot))
    with using_surface(surface):
        grid = LocalVolGrid.build(
            surface, maturity, n_ratio=1000, ratio_max=3.0,
            local_vol_fn=lambda T, K: localvol(T, K, 0.0))
        grid_up = LocalVolGrid.build(
            surface, maturity, n_ratio=1000, ratio_max=3.0,
            local_vol_fn=lambda T, K: localvol(T, K, spot_adj_up))
        grid_down = LocalVolGrid.build(
            surface, maturity, n_ratio=1000, ratio_max=3.0,
            local_vol_fn=lambda T, K: localvol(T, K, spot_adj_down))

    n = grid.sigma.size
    print(f"  repaired   : {grid.n_undefined:,} of {n:,} undefined, "
          f"{grid.n_clipped:,} clipped to [{grid.vol_floor}, {grid.vol_cap}], "
          f"{grid.n_zero:,} ({100 * grid.n_zero / n:.1f}%) are exactly zero")
    print(f"  delta up   : spot_adj={spot_adj_up:+.8f}, "
          f"{grid_up.n_undefined:,} undefined grid points")
    print(f"  delta down : spot_adj={spot_adj_down:+.8f}, "
          f"{grid_down.n_undefined:,} undefined grid points")
    print(f"\nlocal vol grid: {grid.sigma.shape} (dates x spot ratios), "
          f"date axis {grid.dates[0]} -> {grid.dates[-1]}")

    mc = LocalVolMC(surface, grid, n_paths=n_paths, seed=20260807,
                    antithetic=True, n_substeps=n_substeps)
    sum_r, sum_v = mc.total_clocks()
    print(f"clocks: sum dt_r = {sum_r:.6f} (target tau_r = "
          f"{surface.tau_r(maturity):.6f}), "
          f"sum dt_v = {sum_v:.6f} (target tau_vol = "
          f"{surface.tau_vol(maturity):.6f})")
    res = mc.price_european(strike, maturity, is_call=True, bump=None)
    levels_diag = np.asarray(DEFAULT_STRIKE_LEVELS if comparison_levels is None
                             else comparison_levels, dtype=float)
    pricing_errors, variance_bins = mc.step4_diagnostics(
        levels_diag * surface.market.spot, maturity)
    delta_comparison = mc.step4_delta_diagnostics(
        levels_diag * surface.market.spot, maturity,
        grid_up=grid_up, grid_down=grid_down, bump=h)

    tau_v = surface.tau_vol(maturity)
    F = surface.forward(maturity, surface.market.spot)
    df = surface.discount_factor(maturity)
    sigma_imp = float(surface.implied_vol(maturity, strike))
    pv_bs = float(bs_price_w(F, strike, sigma_imp ** 2 * tau_v, df, True))
    atm_delta = delta_comparison.iloc[
        int(np.argmin(np.abs(delta_comparison["strike"].to_numpy() - strike)))]
    delta_bs = float(atm_delta["bs_delta"])
    delta_local = float(atm_delta["mc_local_delta"])

    print(f"\nvanilla call  K = {strike}  T = {maturity}")
    print(f"  paths {res.n_paths:,}, {res.n_steps} steps "
          f"({n_substeps} sub-steps per business day), antithetic + CRN")
    print(f"  control variate: GBM at sigma = {res.extra['sigma_cv']:.6f}, "
          f"corr {res.extra.get('corr', float('nan')):.4f}")

    rel = res.pv / pv_bs - 1.0
    z_score = (res.pv - pv_bs) / res.stderr if res.stderr > 0 else float("nan")
    print("\n  -- PV comparison --")
    print(f"    implied vol           {sigma_imp:.8f}")
    print(f"    BS PV                 {pv_bs:.8f}")
    print(f"    MC PV (raw)           {res.extra['pv_raw']:.8f}  "
          f"(stderr {res.extra['stderr_raw']:.2e})")
    print(f"    MC PV (control var)   {res.pv:.8f}  (stderr {res.stderr:.2e}"
          f" = {100 * res.stderr / pv_bs:.3f} % of PV)")
    print(f"    relative error        {rel * 100:+.4f} %"
          f"   -> {'PASS' if abs(rel) < 0.01 else 'FAIL'} (<1%)")
    print(f"    error / stderr        {z_score:+.2f}")

    print("\n  -- mentor-scope delta comparisons --")
    print("    up/down local-vol grids are rebuilt with spot_adj = "
          "log(shifted_spot / ref_spot)")
    delta_cols = ["level", "bs_delta", "bs_delta_bump",
                  "mc_implied_delta", "mc_local_delta",
                  "mc_local_abs_error_vs_bs_bump",
                  "mc_local_delta_stderr"]
    print(delta_comparison[delta_cols].to_string(
        index=False, float_format=lambda x: f"{x:.6f}"))

    cols = ["level", "implied_vol", "bs_pv", "mc_implied_pv",
            "mc_local_pv", "mc_local_rel_error_pct"]
    print("\n  -- strike-grid PV comparison --")
    print(pricing_errors[cols].to_string(
        index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n  terminal-spot variance bins:")
    print(variance_bins[["bin_left_ratio", "bin_right_ratio", "n_paths",
                         "mean_path_integrated_variance",
                         "mean_implied_total_variance_at_terminal_spot"]].to_string(
        index=False, float_format=lambda x: f"{x:.6f}"))

    return {"grid": grid, "grid_up": grid_up, "grid_down": grid_down,
            "mc": res, "pv_bs": pv_bs, "delta_bs": delta_bs,
            "delta_local": delta_local, "sigma_imp": sigma_imp,
            "pricing_errors": pricing_errors,
            "delta_comparison": delta_comparison,
            "terminal_variance_bins": variance_bins}


def step5_delta_range(surface: VolSurface, maturity, strike,
                      mc_delta: float | None = None) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("STEP 5 (extension)  --  delta is a range, not a number")
    print("=" * 78)

    R_grid = [LOCAL_VOL_LIKE, -0.5, STICKY_STRIKE, 0.5, STICKY_MONEYNESS]
    rows = []
    for R in R_grid:
        g = smile_greeks(surface, maturity, strike,
                         spot=surface.market.spot, stickiness=R)
        fd = float(np.atleast_1d(
            delta_finite_difference(surface, maturity, strike,
                                    surface.market.spot, R))[0])
        rows.append({
            "stickiness_R": R,
            "label": {-1.0: "local-vol-like", 0.0: "sticky strike",
                      1.0: "sticky moneyness"}.get(R, ""),
            "delta": float(np.atleast_1d(g["delta"])[0]),
            "delta_fd_check": fd,
            "vega_sigma": float(np.atleast_1d(g["vega_sigma"])[0]),
        })
    tbl = pd.DataFrame(rows)
    print(f"\nvanilla call  K = {strike}  T = {maturity}")
    print(tbl.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    span = tbl["delta"].max() - tbl["delta"].min()
    print(f"\ndelta spans {tbl['delta'].min():.4f} .. {tbl['delta'].max():.4f}"
          f"  (width {span:.4f}, i.e. {span * 100:.1f} delta points)")
    print("  closed form vs bump-and-reprice values are shown above.")
    if mc_delta is not None:
        print(f"  the Step 4 Monte Carlo delta ({mc_delta:.4f}) sits at the "
              "local-vol end of this range.")
    return tbl


def step6_backtest(surface: VolSurface, maturity, strike) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("STEP 6 (extension)  --  hedging backtest scaffold")
    print("=" * 78)

    R_TRUE = 0.5
    states = bt.simulate_states_from_local_vol(surface, maturity, seed=11,
                                               stickiness_truth=R_TRUE)
    opt = bt.OptionSpec(strike=strike, maturity=maturity, is_call=True)
    engine = bt.HedgeBacktester(states, opt)

    R_grid = np.round(np.arange(-1.0, 1.51, 0.25), 3)
    best, tbl = engine.optimal_stickiness(R_grid, objective="pnl_mad")
    print(f"synthetic path: {len(states)} business days "
          f"({states[0].date} -> {states[-1].date})")
    print(f"the spot follows the calibrated local vol model; the surface is")
    print(f"rolled each day with a TRUE stickiness ratio of {R_TRUE:+.2f}.")
    print("\nhedging P&L by assumed stickiness ratio:")
    print(tbl.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print(f"\nR* minimising the mean absolute daily P&L: {best:+.2f}"
          f"   (true value {R_TRUE:+.2f})")
    print("\nby regime (realised vol terciles):")
    print(engine.sweep_by_regime(R_grid, objective="pnl_mad").to_string(index=False))
    return tbl


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(n_paths: int = 100_000, run_backtest: bool = True,
         outdir: Path = OUTPUT_DIR) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    surface = build_surface(TEST_MARKET, TEST_VOL_PARAMS)

    levels = DEFAULT_STRIKE_LEVELS
    maturity = surface.slices[-1].vol_date          # furthest VolDate
    strike = surface.market.spot                    # K_call = S0

    print("=" * 78)
    print("Pricing Practice 3 -- SVI based local volatility calibration")
    print("=" * 78)
    m = surface.market
    print(f"pricing_date {m.pricing_date}   S0 {m.spot}   refSpot {surface.ref_spot}")
    print(f"rate {m.rate}   dividend {m.dividend}   repo {m.repo}   "
          f"cost of carry b {m.cost_of_carry}")
    print(f"strike levels {levels[0]:.2f} .. {levels[-1]:.2f} "
          f"({len(levels)} columns)")

    params = step1_parameters(surface)
    iv = step2_implied(surface, maturity, levels)
    #looks like this:
    # dt.date          0.80   0.85   0.90  ...  1.20   1.25   1.30
    # 2026-08-10    ...    ...    ...
    # 2026-08-11    ...    ...    ...
    # ...
    # 2027-04-16    ...    ...    ...

    lv = step3_localvol(surface, maturity, levels)

    # Step 4 uses the calendar-repaired surface; Step 3 retains the raw quotes.
    surface_mc = surface.repaired()
    mc = step4_montecarlo(surface_mc, maturity, strike, n_paths,
                          raw_surface=surface, comparison_levels=levels)
    dr = step5_delta_range(surface_mc, maturity, strike, mc_delta=mc["mc"].delta)
    bt_tbl = step6_backtest(surface_mc, maturity, strike) if run_backtest else None

    params.to_csv(outdir / "step1_raw_parameters.csv", index=False)
    surface.self_check().to_csv(outdir / "step1_self_check.csv", index=False)
    iv.to_csv(outdir / "step2_implied_vol_matrix.csv")
    lv["local_vol"].to_csv(outdir / "step3_local_vol_matrix.csv")
    lv["local_vol_shift"].to_csv(outdir / "step3_local_vol_spot_adj_002.csv")
    lv["flags"].to_csv(outdir / "step3_arbitrage_flags.csv")
    lv["report"]["butterfly"].to_csv(outdir / "step3_butterfly.csv", index=False)
    lv["report"]["calendar"].to_csv(outdir / "step3_calendar.csv", index=False)
    mc["grid"].as_frame().to_csv(outdir / "step4_localvol_grid.csv")
    mc["pricing_errors"].to_csv(outdir / "step4_pricing_errors.csv", index=False)
    mc["delta_comparison"].to_csv(
        outdir / "step4_delta_comparison.csv", index=False)
    mc["terminal_variance_bins"].to_csv(
        outdir / "step4_terminal_variance_bins.csv", index=False)
    dr.to_csv(outdir / "step5_delta_range.csv", index=False)
    if bt_tbl is not None:
        bt_tbl.to_csv(outdir / "step6_backtest_sweep.csv", index=False)

    try:
        from . import plots
        lv_rep = surface_mc.local_vol_matrix(lv["dates"], levels)
        plots.plot_surfaces(iv, lv["local_vol"], lv_rep,
                            outdir / "surfaces.png")
        plots.plot_smile_slices(surface, [s.vol_date for s in surface.slices],
                                levels, outdir / "smiles.png")
        plots.plot_delta_range(surface_mc, maturity,
                               np.arange(0.80, 1.301, 0.025),
                               outdir / "delta_range.png")
        plots.plot_step4_pricing_errors(
            mc["pricing_errors"], outdir / "step4_pricing_errors.png")
        plots.plot_terminal_variance_bins(
            mc["terminal_variance_bins"],
            outdir / "step4_terminal_variance_bins.png")
    except Exception as exc:                                   # noqa: BLE001
        print(f"(plots skipped: {exc})")

    print("\n" + "=" * 78)
    print(f"written to {outdir}")
    for p in sorted(outdir.iterdir()):
        print(f"  {p.name}")
    print("=" * 78)

    return {"surface": surface, "implied_vol": iv, **lv, **mc,
            "delta_range": dr}


def cli() -> None:
    """Command-line entry point shared by ``python solution.py`` and ``-m``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="20k Monte Carlo paths instead of 100k")
    ap.add_argument("--no-backtest", action="store_true")
    args = ap.parse_args()
    main(n_paths=20_000 if args.fast else 100_000,
         run_backtest=not args.no_backtest)


if __name__ == "__main__":
    cli()

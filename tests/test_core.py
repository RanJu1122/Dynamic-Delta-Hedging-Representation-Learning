"""Unit and numerical-regression tests for the reusable core library."""

from __future__ import annotations

import datetime as dt
import numpy as np

from pricing_svi_localvol_calibration.config import (
    DEFAULT_STRIKE_LEVELS, TEST_MARKET, TEST_VOL_PARAMS)
from svi_localvol.blackscholes import (bs_delta_w, bs_price_w,
                                       implied_total_variance)
from svi_localvol.conventions import (dt_r, dt_vol, gen_schedule, is_biz_day,
                                      nb_biz_days,
                                      us_equity_scheduled_holidays)
from svi_localvol.montecarlo import LocalVolGrid, LocalVolMC
from svi_localvol.params import (ALPHA_STICKY_LOCAL_VOL,
                                 ALPHA_STICKY_MONEYNESS,
                                 ALPHA_STICKY_STRIKE, VolQuoteSet,
                                 validate_stickiness_alpha)
from svi_localvol.surface import VolSurface
from svi_localvol.svi import g_function, jw_to_raw, raw_to_jw

SURF = VolSurface(TEST_MARKET, VolQuoteSet.from_dict(TEST_VOL_PARAMS))
SURF_REP = SURF.repaired()
MATURITY = SURF.slices[-1].vol_date


# --------------------------------------------------------------------------- #
# conventions
# --------------------------------------------------------------------------- #
def test_two_clocks_are_independent():
    """A weekend adds calendar time but no volatility time."""
    fri, mon = dt.date(2026, 8, 7), dt.date(2026, 8, 10)
    assert nb_biz_days(fri, mon) == 1
    assert abs(dt_r(fri, mon) - 3 / 365) < 1e-15
    assert abs(dt_vol(fri, mon) - 1 / 260) < 1e-15


def test_schedule_is_business_days_only():
    dates = gen_schedule(dt.date(2026, 8, 7), dt.date(2026, 8, 31), "1D")
    assert all(d.weekday() < 5 for d in dates)
    assert dates[0] == dt.date(2026, 8, 7)
    assert dates[-1] == dt.date(2026, 8, 31)
    assert all(b > a for a, b in zip(dates, dates[1:]))


def test_holidays_are_excluded():
    hol = [dt.date(2026, 8, 12)]
    dates = gen_schedule(dt.date(2026, 8, 10), dt.date(2026, 8, 14), "1D", hol=hol)
    assert dt.date(2026, 8, 12) not in dates


def test_us_equity_calendar_excludes_fixed_and_movable_holidays():
    holidays = us_equity_scheduled_holidays(2025, 2026)
    assert not is_biz_day(dt.date(2025, 7, 4), holidays)
    assert not is_biz_day(dt.date(2025, 4, 18), holidays)  # Good Friday
    assert not is_biz_day(dt.date(2025, 11, 27), holidays)


# --------------------------------------------------------------------------- #
# Step 1
# --------------------------------------------------------------------------- #
def test_jw_raw_roundtrip():
    """The five quotes must come back exactly through raw_to_jw.

    `raw_to_jw` returns Gatheral's total-variance skew psi_t; the quoted Skew
    field is the volatility skew d(sigma)/dk = psi_t / sqrt(tau).
    """
    for s, q in zip(SURF.slices, SURF.quotes.quotes):
        back = raw_to_jw(s.raw, s.tau)
        assert abs(back["atm_vol"] - q.atm_vol) < 1e-12
        assert abs(back["skew"] / np.sqrt(s.tau) - q.skew) < 1e-12
        assert abs(back["putwing"] - q.putwing) < 1e-12
        assert abs(back["callwing"] - q.callwing) < 1e-12
        assert abs(back["kurt"] - q.kurt) < 1e-12


def test_quoted_skew_is_the_volatility_skew():
    """The `Skew` field is d(sigma_BS)/dk at k = 0, not Gatheral's psi_t.

    This pins down the sqrt(omg * tau) in the task statement, which is CORRECT
    and not a typo: psi_t = Skew * sqrt(tau).  The check differentiates the
    calibrated smile numerically, so it cannot be satisfied by a compensating
    error in raw_to_jw.
    """
    h = 1e-6
    for s, q in zip(SURF.slices, SURF.quotes.quotes):
        up = np.sqrt(float(s.raw.w(h)) / s.tau)
        dn = np.sqrt(float(s.raw.w(-h)) / s.tau)
        assert abs((up - dn) / (2 * h) - q.skew) < 1e-6
        # and psi_t, what raw_to_jw returns, is sqrt(tau) times that
        assert abs(raw_to_jw(s.raw, s.tau)["skew"]
                   - q.skew * np.sqrt(s.tau)) < 1e-12


def test_dropping_the_sqrt_tau_breaks_the_skew():
    """Guard against reverting to beta = rho - 2 psi sqrt(omg) / b."""
    q, tau = SURF.quotes.quotes[0], SURF.slices[0].tau
    wrong = jw_to_raw(atm_var=q.atm_var, skew=q.skew / np.sqrt(tau),
                      putwing=q.putwing, callwing=q.callwing,
                      min_imp_var=q.min_imp_var, tau=tau)
    h = 1e-6
    got = (np.sqrt(float(wrong.w(h)) / tau)
           - np.sqrt(float(wrong.w(-h)) / tau)) / (2 * h)
    assert abs(got - q.skew) > 0.1        # badly wrong, as documented


def test_w_at_zero_is_atm_total_variance():
    for s, q in zip(SURF.slices, SURF.quotes.quotes):
        assert abs(float(s.raw.w(0.0)) - q.atm_var * s.tau) < 1e-14


def test_analytic_derivatives_match_finite_difference():
    y = np.linspace(-0.6, 0.4, 41)
    h = 1e-6
    for s in SURF.slices:
        fd1 = (s.raw.w(y + h) - s.raw.w(y - h)) / (2 * h)
        fd2 = (s.raw.w(y + h) + s.raw.w(y - h) - 2 * s.raw.w(y)) / h ** 2
        assert np.max(np.abs(fd1 - s.raw.dw_dy(y))) < 1e-6
        assert np.max(np.abs(fd2 - s.raw.d2w_dy2(y))) < 1e-3


# --------------------------------------------------------------------------- #
# Step 2
# --------------------------------------------------------------------------- #
def test_atm_implied_vol_on_voldates():
    for s, q in zip(SURF.slices, SURF.quotes.quotes):
        F = SURF.forward(s.vol_date)
        assert abs(SURF.implied_vol(s.vol_date, F) - q.atm_vol) < 1e-12


def test_w_at_a_voldate_is_exactly_the_slice():
    """On a quoted expiry the interpolation must return the slice untouched."""
    K = np.array([0.85, 1.0, 1.15])
    for s in SURF.slices:
        y = SURF.log_moneyness(s.vol_date, K)
        got = SURF.implied_total_variance(s.vol_date, K)
        assert np.max(np.abs(got - s.raw.w(y))) < 1e-14


def test_repaired_surface_has_no_calendar_arbitrage():
    """dw/dT >= 0 everywhere on the repaired surface, by construction."""
    dates = SURF_REP.date_axis(MATURITY, "1D")[::7]
    K = np.linspace(0.6, 1.8, 61)
    for T in dates:
        if SURF_REP.tau_vol(T) <= 0:
            continue
        res = SURF_REP.total_variance(T, K, order=1)
        assert np.all(np.asarray(res["dw_dtau"]) >= -1e-14)


def test_scalar_and_vector_agree():
    T = dt.date(2026, 12, 1)
    vec = SURF.implied_vol(T, np.array([0.9, 1.0, 1.1]))
    for i, k in enumerate([0.9, 1.0, 1.1]):
        assert abs(SURF.implied_vol(T, k) - vec[i]) < 1e-14


def test_black_scholes_inversion_roundtrip():
    T = dt.date(2027, 1, 15)
    K = np.array([0.85, 1.0, 1.15])
    F, df = SURF.forward(T), SURF.discount_factor(T)
    w = SURF.implied_total_variance(T, K)
    px = bs_price_w(F, K, w, df, True)
    assert np.max(np.abs(implied_total_variance(px, F, K, df, True) - w)) < 1e-8


# --------------------------------------------------------------------------- #
# Step 3
# --------------------------------------------------------------------------- #
def test_dupire_denominator_equals_gatheral_g():
    """D in the local vol formula IS equation (2.1) of the paper."""
    y = np.linspace(-0.5, 0.3, 33)
    for s in SURF.slices:
        w, dw, d2w = s.raw.w(y), s.raw.dw_dy(y), s.raw.d2w_dy2(y)
        g = g_function(w, dw, d2w, y)
        D = (1 - (y / w) * dw
             + 0.25 * (-0.25 - 1 / w + y ** 2 / w ** 2) * dw ** 2
             + 0.5 * d2w)
        assert np.max(np.abs(g - D)) < 1e-12


def test_local_vol_nan_exactly_where_arbitrage_is():
    dates = SURF.date_axis(MATURITY, "1D")[::20]
    K = DEFAULT_STRIKE_LEVELS
    for T in dates:
        if SURF.tau_vol(T) <= 0:
            continue
        lv, diag = SURF.local_vol(T, K, return_diagnostics=True)
        bad = diag["calendar_arb"] | diag["butterfly_arb"]
        assert np.all(np.isnan(lv[bad]))
        assert np.all(np.isfinite(lv[~bad]))


def test_calendar_repair_removes_all_nan():
    dates = SURF_REP.date_axis(MATURITY, "1D")
    lv = SURF_REP.local_vol_matrix(dates, DEFAULT_STRIKE_LEVELS)
    assert np.isfinite(lv.to_numpy()).all()


def test_repair_never_lowers_total_variance():
    T = dt.date(2026, 11, 20)
    K = np.linspace(0.6, 1.6, 51)
    assert np.all(SURF_REP.implied_total_variance(T, K)
                  >= SURF.implied_total_variance(T, K) - 1e-14)


def test_spot_adj_shifts_the_smile():
    T = dt.date(2026, 12, 11)
    K = np.array([0.9, 1.0, 1.1])
    base = SURF_REP.local_vol(T, K, spot_adj=0.0)
    up = SURF_REP.local_vol(T, K, spot_adj=0.02)
    assert np.any(np.abs(up - base) > 1e-6)


# --------------------------------------------------------------------------- #
# stickiness parameter alpha  (dynamic-hedging study convention)
# --------------------------------------------------------------------------- #
def test_alpha_boundary_rejects_negative_values():
    assert validate_stickiness_alpha(0.0) == 0.0
    assert validate_stickiness_alpha(2.0) == 2.0
    try:
        validate_stickiness_alpha(-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative stickiness alpha must be rejected")


def test_alpha_comes_from_the_quotes_by_default():
    """StickinessRatio must actually reach the Dupire denominator."""
    T, K = dt.date(2026, 12, 11), np.array([0.9, 1.0, 1.1])
    assert SURF_REP.alpha_at(T) == ALPHA_STICKY_STRIKE     # the shipped data
    explicit = SURF_REP.local_vol(T, K, spot_adj=0.02,
                                  alpha=ALPHA_STICKY_STRIKE)
    implicit = SURF_REP.local_vol(T, K, spot_adj=0.02)
    assert np.allclose(explicit, implicit, equal_nan=True)


def test_alpha_zero_ignores_spot_adj():
    """alpha = 0 is sticky local vol: sigma_loc(S, t) does not move at all."""
    T, K = dt.date(2026, 12, 11), np.array([0.85, 1.0, 1.15])
    base = SURF_REP.local_vol(T, K, spot_adj=0.0)
    for adj in (-0.05, 0.05):
        shifted = SURF_REP.local_vol(T, K, spot_adj=adj,
                                     alpha=ALPHA_STICKY_LOCAL_VOL)
        assert np.allclose(base, shifted, equal_nan=True)


def test_alpha_scales_the_shift_linearly():
    """alpha * spot_adj is the only way alpha enters."""
    T, K = dt.date(2026, 12, 11), np.array([0.9, 1.0, 1.1])
    a = SURF_REP.local_vol(T, K, spot_adj=0.02, alpha=2.0)
    b = SURF_REP.local_vol(T, K, spot_adj=0.04, alpha=1.0)
    assert np.allclose(a, b, equal_nan=True)


def test_alpha_is_interpolated_between_voldates():
    quotes = VolQuoteSet.from_dict(
        {**TEST_VOL_PARAMS,
         "StickinessRatio": [0.0, 2.0] + [1.0] * 6})
    surf = VolSurface(TEST_MARKET, quotes)
    lo, hi = surf.slices[0], surf.slices[1]
    assert surf.alpha_at(lo.vol_date) == 0.0
    assert surf.alpha_at(hi.vol_date) == 2.0
    mid = surf.alpha_at(dt.date(2026, 10, 2))               # between the two
    assert 0.0 < mid < 2.0


# --------------------------------------------------------------------------- #
# Step 4
# --------------------------------------------------------------------------- #
def test_mc_grid_starts_on_the_pricing_date():
    """Dropping the pricing date would shorten every path by a business day."""
    grid = LocalVolGrid.build(SURF_REP, MATURITY)
    assert grid.dates[0] == SURF_REP.market.pricing_date
    assert grid.dates[-1] == MATURITY
    assert np.isfinite(grid.sigma).all()


def test_mc_clocks_span_exactly_the_option_life():
    """Deterministic guard on both time axes.

    sum(dt_r) must be the Act/365 life used by the forward and the discount
    factor; sum(dt_v) must be the Business/260 life that scales total variance.
    This single assertion catches a dropped first day and a mixed-up clock.
    """
    grid = LocalVolGrid.build(SURF_REP, MATURITY)
    mc = LocalVolMC(SURF_REP, grid, n_paths=2)
    sum_r, sum_v = mc.total_clocks()
    assert abs(sum_r - SURF_REP.tau_r(MATURITY)) < 1e-12
    assert abs(sum_v - SURF_REP.tau_vol(MATURITY)) < 1e-12


def test_scheme_accumulates_exactly_the_quoted_total_variance():
    """The noise-free core of the Step 4 check.

    With a constant local vol and antithetic draws the sum of the Brownian
    increments is exactly zero, so

        mean(log S_T) = log S0 + b * tau_r - 0.5 * sigma^2 * tau_vol

    holds to machine precision.  Any error in the two step sizes, in the Ito
    term's clock, or in the horizon shows up here with no Monte Carlo noise to
    hide behind -- unlike a 1% gate on a PV whose stderr is 0.3%.
    """
    grid = LocalVolGrid.build(SURF_REP, MATURITY)
    sigma = 0.20
    grid.sigma[:] = sigma
    mc = LocalVolMC(SURF_REP, grid, n_paths=20_000, seed=3, antithetic=True)
    log_s = np.log(mc.terminal_spots(1.0))

    b = SURF_REP.market.cost_of_carry
    expect = b * SURF_REP.tau_r(MATURITY) - 0.5 * sigma ** 2 * SURF_REP.tau_vol(MATURITY)
    assert abs(float(log_s.mean()) - expect) < 1e-12

    # and the realised variance is sigma^2 * tau_vol, not sigma^2 * tau_r
    var = float(log_s.var(ddof=1))
    assert abs(var / (sigma ** 2 * SURF_REP.tau_vol(MATURITY)) - 1) < 0.03


def test_path_integrated_variance_uses_the_volatility_clock():
    grid = LocalVolGrid.build(SURF_REP, MATURITY)
    sigma = 0.20
    grid.sigma[:] = sigma
    mc = LocalVolMC(SURF_REP, grid, n_paths=2_000, seed=31,
                    antithetic=True, n_substeps=3)
    _, integrated = mc.terminal_spots(
        SURF_REP.market.spot, return_integrated_variance=True)
    target = sigma ** 2 * SURF_REP.tau_vol(MATURITY)
    assert np.max(np.abs(integrated - target)) < 1e-14


def test_price_diagnostic_tables_are_complete():
    grid = LocalVolGrid.build(SURF_REP, MATURITY)
    mc = LocalVolMC(SURF_REP, grid, n_paths=20_000, seed=37,
                    antithetic=True, n_substeps=2)
    pricing, bins = mc.price_diagnostics([0.9, 1.0, 1.1], MATURITY)

    assert pricing.shape[0] == 3
    assert {"bs_pv", "mc_implied_pv", "mc_local_pv",
            "paired_difference_stderr"}.issubset(pricing.columns)
    assert np.all(np.abs(pricing["mc_implied_pv"] - pricing["bs_pv"])
                  < 4 * pricing["mc_implied_stderr"])

    assert abs(float(bins["path_probability"].sum()) - 1.0) < 1e-14
    assert int(bins["n_paths"].sum()) == 20_000
    assert {"mean_path_integrated_variance",
            "mean_implied_total_variance_at_terminal_spot",
            "integrated_minus_mean_implied_variance"}.issubset(bins.columns)
    assert np.isfinite(bins["mean_path_integrated_variance"]).all()


def test_bump_delta_table_is_complete():
    """The delta check must rebuild its up/down local-vol grids."""
    h = 0.01 * SURF_REP.market.spot
    base = LocalVolGrid.build(SURF_REP, MATURITY, n_ratio=300)
    up = LocalVolGrid.build(
        SURF_REP, MATURITY, n_ratio=300,
        spot_adj=np.log((SURF_REP.market.spot + h) / SURF_REP.ref_spot))
    down = LocalVolGrid.build(
        SURF_REP, MATURITY, n_ratio=300,
        spot_adj=np.log((SURF_REP.market.spot - h) / SURF_REP.ref_spot))
    assert np.max(np.abs(up.sigma - down.sigma)) > 1e-6

    mc = LocalVolMC(SURF_REP, base, n_paths=20_000, seed=37,
                    antithetic=True, n_substeps=2)
    table = mc.bump_delta_diagnostics(
        [0.9, 1.0, 1.1], MATURITY, up, down, bump=h)

    required = {
        "bs_delta", "bs_delta_bump", "mc_implied_delta",
        "mc_implied_delta_stderr", "mc_local_delta",
        "mc_local_delta_stderr", "mc_local_abs_error_vs_bs_bump",
        "spot_adj_up", "spot_adj_down",
    }
    assert table.shape[0] == 3
    assert required.issubset(table.columns)
    assert np.allclose(table["spot_adj_up"], np.log(1.01))
    assert np.allclose(table["spot_adj_down"], np.log(0.99))
    assert np.all(np.abs(table["mc_implied_delta"] - table["bs_delta_bump"])
                  < 4 * table["mc_implied_delta_stderr"])
    assert np.isfinite(table["mc_local_delta"]).all()


def test_mc_engine_reproduces_black_scholes_when_smile_is_off():
    """The decisive engine regression.

    Overwrite the local vol table with a constant.  The scheme must then match
    Black-Scholes for BOTH price and delta.  Any residual gap in the smiled run
    is therefore a property of the local volatility model, not of the code.

    The total variance is sigma^2 * tau_VOL, because that is the measure the
    diffusion is integrated against; the forward and the discount factor still
    use tau_r.
    """
    grid = LocalVolGrid.build(SURF_REP, MATURITY)
    sigma = 0.20
    grid.sigma[:] = sigma
    mc = LocalVolMC(SURF_REP, grid, n_paths=200_000, seed=1)
    res = mc.price_european(1.0, MATURITY, True, bump=0.01,
                            control_variate=False)

    carry = np.exp(SURF_REP.market.cost_of_carry * SURF_REP.tau_r(MATURITY))
    F, df = carry, SURF_REP.discount_factor(MATURITY)
    w = sigma ** 2 * SURF_REP.tau_vol(MATURITY)
    pv_bs = float(bs_price_w(F, 1.0, w, df, True))
    d_bs = float(bs_delta_w(F, 1.0, w, df, carry, True))

    assert abs(res.pv / pv_bs - 1) < 0.01
    assert abs(res.delta / d_bs - 1) < 0.01


def test_mc_price_matches_black_scholes_with_the_smile():
    """Step 4 proper.  Asserted against the standard error, not just 1%."""
    grid = LocalVolGrid.build(SURF_REP, MATURITY)
    mc = LocalVolMC(SURF_REP, grid, n_paths=100_000, seed=20260807, n_substeps=8)
    res = mc.price_european(1.0, MATURITY, True)

    sigma = float(SURF_REP.implied_vol(MATURITY, 1.0))
    F = SURF_REP.forward(MATURITY, 1.0)
    pv_bs = float(bs_price_w(F, 1.0, sigma ** 2 * SURF_REP.tau_vol(MATURITY),
                             SURF_REP.discount_factor(MATURITY), True))
    assert abs(res.pv / pv_bs - 1) < 0.01
    assert abs(res.pv - pv_bs) < 4 * res.stderr


def test_mc_is_a_martingale_under_the_forward_measure():
    """E[S_T] must be the forward, which pins the drift clock to Act/365."""
    grid = LocalVolGrid.build(SURF_REP, MATURITY)
    mc = LocalVolMC(SURF_REP, grid, n_paths=200_000, seed=5, n_substeps=4)
    s_t = mc.terminal_spots(SURF_REP.market.spot)
    half = s_t.size // 2
    pair_means = 0.5 * (s_t[:half] + s_t[half:])
    se = float(pair_means.std(ddof=1) / np.sqrt(pair_means.size))
    assert abs(float(pair_means.mean()) - SURF_REP.forward(
        MATURITY, SURF_REP.market.spot)) < 4 * se


def test_substep_refinement_has_converged():
    """The Euler bias must be gone by the production sub-step count.

    Two runs at different `n_substeps` use different seeds.  Their difference
    is judged against the combined standard error rather than a flat
    percentage tolerance.
    """
    grid = LocalVolGrid.build(SURF_REP, MATURITY)
    out = []
    for m in (4, 16):
        mc = LocalVolMC(SURF_REP, grid, n_paths=100_000,
                        seed=17 + m, n_substeps=m)
        r = mc.price_european(1.0, MATURITY, True)
        out.append((r.pv, r.stderr))
    gap = abs(out[1][0] - out[0][0])
    combined = float(np.hypot(out[0][1], out[1][1]))
    assert gap < 4 * combined, f"gap {gap:.2e} vs 4 sigma {4 * combined:.2e}"

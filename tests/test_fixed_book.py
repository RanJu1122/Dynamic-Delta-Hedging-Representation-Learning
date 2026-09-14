"""Fixed-contract identity, cash settlement, gap carry and new SR integration."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from dynamic_alpha_hedging.cli import _config
from dynamic_alpha_hedging.fixed_book import (make_cohort, fixed_marks, interpolate_beta, advance_account,
                                             contract_schedule)
from dynamic_alpha_hedging.step07_fixed_book import FixedStep7Config, run_fixed_step7
from dynamic_alpha_hedging.step06 import FACTOR_COLUMNS, LAST_FACTOR_COLUMNS
from dynamic_alpha_hedging.sr_pricing import AlphaSurface, SharedSRPricer, SR_KINDS
from svi_localvol.surface import VolSurface
from svi_localvol.params import VolQuoteSet
from pricing_svi_localvol_calibration.config import TEST_MARKET, TEST_VOL_PARAMS
from tests.test_shared_step07 import _fixture, TENORS, LEVELS
from tests.test_precompute import raises


def _surface(date, spot=1.):
    return VolSurface(replace(TEST_MARKET, pricing_date=pd.Timestamp(date).date(), spot=spot),
                      VolQuoteSet.from_dict({**TEST_VOL_PARAMS, "Spot": spot}))


def _fixed_fixture(removed=None):
    inputs, _, maps, pricer = _fixture()
    dates = [d.date() for d in pd.bdate_range("2026-08-03", periods=9)]
    inputs.history.clear()
    for i, d in enumerate(dates):
        if d != removed:
            inputs.history[d] = _surface(d, 1.+.01*np.sin(i))
    inputs.forecasts = pd.DataFrame({c: np.linspace(.1, .2, 6) for c in FACTOR_COLUMNS}, index=dates[2:-1])
    for c in LAST_FACTOR_COLUMNS:
        inputs.forecasts[c] = .1
    options = FixedStep7Config(book_tenors=(2/260, 5/260), book_levels=LEVELS,
                               renew_expired=False, strategy_set="full")
    return inputs, maps, pricer, options


def test_fixed_contracts_age_and_settle_without_reissuing_strikes():
    start, next_day = _surface("2026-08-03"), _surface("2026-08-04", 1.03)
    expiry = _surface("2026-08-05", 1.05)
    cohort = make_cohort(start, (2/260, .25), LEVELS, "contracts", "test")
    day1 = fixed_marks(start, next_day, cohort)
    day2 = fixed_marks(next_day, expiry, cohort)
    assert (day1.strike == day2.strike).all() and (day1.expiry == day2.expiry).all()
    assert (day1.quantity == day2.quantity).all()
    assert (day2.tenor < day1.tenor).all() and (day2.level < day1.level).all()
    terminal = day2[day2.expired]
    assert len(terminal) == 3 and terminal.next_pv.eq(0).all()
    assert terminal.next_iv.isna().all() and not terminal.attribution_valid.any()
    assert np.allclose(terminal.dV, terminal.expiry_cashflow-terminal.pv)
    day3 = fixed_marks(expiry, _surface("2026-08-06", 1.06), cohort)
    assert len(day3) == 3 and not day3.contract_id.isin(terminal.contract_id).any()
    assert day1.mark_extrapolated.any()
    with raises(ValueError, "outside quoted"):
        fixed_marks(start, next_day, cohort, extrapolation="reject")


def test_cash_ledger_no_final_liquidation_and_natural_expiry_transfer():
    # Zero-rate identity, including initial cost and open mark-to-market holdings.
    row = advance_account(None, pv=10., next_pv=12., settlement=0., delta=.5,
        spot=100., next_spot=101., dt=1/365, rate=0., income_yield=0., cost_bps=10.)
    assert np.isclose(row["wealth"], 2.-.5-.05)
    assert row["stock_position"] == -.5 and row["natural_close_units"] == 0.
    assert row["hedge_turnover"] == .5
    # Final settlement receives 15 exactly once, then closes the exhausted book hedge.
    end = advance_account(row, pv=12., next_pv=0., settlement=15., delta=.6,
        spot=101., next_spot=102., dt=1/365, rate=0., income_yield=0., cost_bps=10., all_expired=True)
    assert np.isclose(end["wealth_change"], 3.-.6-.1*101/1000-.6*102/1000)
    assert end["stock_position"] == 0. and end["next_option_pv"] == 0.
    assert end["option_trade_cashflow"] == 0.
    assert np.isclose(end["wealth"], end["cash_close"])


def test_beta_interpolation_uses_current_contract_coordinates_and_does_not_clip():
    surface = _surface("2026-08-03")
    cohort = make_cohort(surface, TENORS, LEVELS, "contracts", "test")
    frame = fixed_marks(surface, _surface("2026-08-04"), cohort)
    axis = pd.MultiIndex.from_product([TENORS, LEVELS])
    beta = pd.Series([-5., -3., -1., 3., 5., 7.], index=axis)
    interpolated, outside = interpolate_beta(surface, TENORS, LEVELS, beta, frame)
    assert np.allclose(interpolated, beta.to_numpy())
    assert not outside.any()
    later = _surface("2026-08-05", 1.02)
    aged = fixed_marks(later, _surface("2026-08-06", 1.02), cohort)
    changed, flags = interpolate_beta(later, TENORS, LEVELS, beta, aged)
    assert not np.allclose(changed, interpolated)
    assert flags.any()


def test_fixed_runner_preserves_book_ledger_and_expiry_attribution(tmp_path):
    inputs, maps, pricer, options = _fixed_fixture()
    result = run_fixed_step7(inputs, outdir=tmp_path, options=options, map_store=maps,
                            pricer=pricer, progress=lambda _: None)
    options_frame, book = result["option_pnl"], result["book_pnl"]
    identity = options_frame.groupby("contract_id")[["strike", "expiry", "quantity"]].nunique()
    assert identity.eq(1).all().all()
    assert len(book.strategy.unique()) == 17
    for name, group in book.groupby("strategy"):
        group = group.sort_values("feature_date")
        assert group.n_options.tolist() == [6, 6, 3, 3, 3]
        assert group.n_alive_end.iloc[-1] == 0
        assert group.option_trade_cashflow.iloc[1:].eq(0).all()
        assert group.natural_close_units.iloc[:-1].eq(0).all()
        assert group.stock_position.iloc[-1] == 0.
        assert np.isclose(group.wealth_change.sum(), group.wealth.iloc[-1])
        spots = np.array([inputs.history[d].ref_spot for d in group.label_date])
        assert np.allclose(group.wealth, group.next_option_pv+group.cash_close+group.stock_position*spots)
    assert np.allclose(book.raw_hedge_error, book.dV+book.hedge_pnl)
    assert np.allclose(book.net_error, book.carry_hedge_error-book.cost_open*
                       np.exp(inputs.config.rate*(pd.to_datetime(book.label_date)-pd.to_datetime(book.feature_date)).dt.days/365)
                       -book.cost_natural_close)
    expiry_rows = options_frame[options_frame.expired]
    assert expiry_rows.next_pv.eq(0).all() and not expiry_rows.attribution_valid.any()
    regular = options_frame[options_frame.attribution_valid]
    assert np.allclose(regular.dV, regular.bs_delta_pnl+regular.gamma_pnl+regular.theta_pnl+
                       regular.term_roll_pnl+regular.model_vega_pnl+regular.attribution_residual)
    assert (tmp_path / "test_contracts.csv").exists()
    assert not options_frame.delta_fallback.any()  # Quality failures remain audit-only.
    localvol = result["localvol_diagnostics"]
    assert set(localvol.strategy) == {"constant_sr", "term_sr", "term_spot_sr"}
    assert localvol.all_local_vol_finite.all()
    assert localvol.all_grid_local_vol_finite.all()
    assert localvol[["grid_local_vol_base", "grid_local_vol_up",
                     "grid_local_vol_down"]].le(inputs.config.step3_vol_cap).all().all()
    assert localvol.base_reference_abs_diff.max() < 1e-12
    keys = ["feature_date", "contract_id"]
    assert localvol.groupby(keys).point_local_vol_base.nunique().eq(1).all()


def test_fixed_raw_only_keeps_three_raw_sr_policies_and_controls(tmp_path):
    inputs, maps, pricer, options = _fixed_fixture()
    result = run_fixed_step7(inputs, outdir=tmp_path, options=replace(options, raw_only=True),
                            map_store=maps, pricer=pricer, progress=lambda _: None)
    strategies = set(result["book_pnl"].strategy)
    assert {"constant_sr", "term_sr", "term_spot_sr", "best_fixed_train", "bs_delta"} <= strategies
    assert "rolling_alpha_mean" not in strategies
    assert not any(name.endswith("_ema") or name.endswith("_last_observed_factor")
                   for name in strategies)
    assert set(result["localvol_diagnostics"].strategy) == set(SR_KINDS)


def test_data_end_keeps_positions_and_gap_is_included_without_ema_reset(tmp_path):
    removed = pd.Timestamp("2026-08-06").date()
    inputs, maps, pricer, options = _fixed_fixture(removed)
    options = replace(options, max_test_dates=2)
    result = run_fixed_step7(inputs, outdir=tmp_path, options=options,
                            map_store=maps, pricer=pricer, progress=lambda _: None)
    group = result["book_pnl"].query("strategy == 'term_sr_ema'").sort_values("feature_date")
    assert group.is_gap.tolist() == [True, False]
    assert group.business_days.tolist() == [2, 1]
    assert group.natural_close_units.eq(0).all()
    assert group.stock_position.iloc[-1] != 0. and group.n_alive_end.iloc[-1] == 3
    assert group.option_trade_cashflow.iloc[-1] == 0.
    assert np.isclose(group.hedge_turnover.iloc[-1], abs(group.book_delta.iloc[-1]-group.book_delta.iloc[0]))
    nodes = result["alpha_nodes"].query("strategy == 'term_sr_ema'")
    a = nodes[np.isclose(nodes.tenor, .25)].alpha.to_numpy()
    raw = result["alpha_nodes"].query("strategy == 'term_sr'")
    raw = raw[np.isclose(raw.tenor, .25)].alpha.to_numpy()
    decay = 2**(-1/options.half_life)
    assert np.isclose(a[1], decay*a[0]+(1-decay)*raw[1])
    summary = result["summary"].set_index("strategy").loc["term_sr_ema"]
    assert np.isclose(summary.raw_pnl_sum, group.raw_hedge_error.sum())
    assert summary.n_gap_intervals == 1 and summary.n_daily_intervals == 1


def test_missing_expiry_spot_rejected_before_mc_and_prepare_is_read_only(tmp_path):
    inputs, maps, pricer, options = _fixed_fixture(pd.Timestamp("2026-08-07").date())
    with patch.object(pricer, "get", side_effect=AssertionError("MC started")):
        with raises(ValueError, "missing settlement spot"):
            run_fixed_step7(inputs, outdir=tmp_path / "bad", options=options,
                            map_store=maps, pricer=pricer, prepare_only=True)
    inputs, maps, pricer, options = _fixed_fixture()
    with patch.object(pricer, "get", side_effect=AssertionError("MC started")):
        result = run_fixed_step7(inputs, outdir=tmp_path / "plan", options=options,
            map_store=maps, pricer=pricer, prepare_only=True, progress=lambda _: None)
        assert not result["validation"]["mc_run"]
        assert (tmp_path / "plan" / "localvol_diagnostics.csv").exists()
        assert set(result["localvol_diagnostics"].strategy) == set(SR_KINDS)
        with raises(ValueError, "different backtest"):
            run_fixed_step7(inputs, outdir=tmp_path / "plan", options=replace(options, half_life=0),
                map_store=maps, pricer=pricer, prepare_only=True)


def test_fixed_today_delta_has_no_future_beta_or_factor_dependency(tmp_path):
    inputs, maps, pricer, options = _fixed_fixture()
    result = run_fixed_step7(inputs, outdir=tmp_path / "first", options=options,
                            map_store=maps, pricer=pricer, progress=lambda _: None)
    first = inputs.forecasts.index[0]
    inputs.forecasts.loc[inputs.forecasts.index > first] = 100.
    inputs.daily.loc[inputs.daily.observation_date > first, "beta_surface_daily"] = 100.
    changed = run_fixed_step7(inputs, outdir=tmp_path / "changed", options=options,
                            map_store=maps, pricer=pricer, progress=lambda _: None)
    a = result["option_pnl"].query("feature_date == @first").delta
    b = changed["option_pnl"].query("feature_date == @first").delta
    assert np.array_equal(a, b)


def test_new_mc_accepts_aging_fixed_strikes_and_hits_cache(tmp_path):
    inputs, _, _, options = _fixed_fixture()
    date, end = inputs.history.dates[2:4]
    surface = inputs.history[date]
    cohort = make_cohort(surface, options.book_tenors, options.book_levels, "contracts", "test")
    marks = fixed_marks(surface, inputs.history[end], cohort)
    profile = AlphaSurface(surface, TENORS, LEVELS, np.ones((2, 3)), "term_sr")
    pricer = SharedSRPricer(inputs.config, tmp_path)
    result = pricer.get(surface, marks, profile)
    assert np.isfinite(result.delta).all()
    with patch.object(pricer, "measure", side_effect=AssertionError("cache miss")):
        assert np.allclose(pricer.get(surface, marks, profile).delta, result.delta)
    assert pricer.computed == pricer.reused == 1


def test_signal_grid_is_explicit_and_legacy_default_unchanged():
    legacy = _config(SimpleNamespace())
    full = _config(SimpleNamespace(surface_grid="full63"))
    assert len(legacy.step4_tenors)*len(legacy.step4_strike_levels) == 56
    assert len(full.step4_tenors)*len(full.step4_strike_levels) == 63
    assert len(FixedStep7Config().book_tenors)*len(FixedStep7Config().book_levels) == 63


def test_renewal_generations_keep_original_tenor_level_quantity_and_finance_premiums(tmp_path):
    inputs, maps, pricer, options = _fixed_fixture()
    options = replace(options, renew_expired=True)
    result = run_fixed_step7(inputs, outdir=tmp_path, options=options,
                            map_store=maps, pricer=pricer, progress=lambda _: None)
    contracts = pd.read_csv(tmp_path / "test_contracts.csv", parse_dates=["inception", "expiry"])
    assert (contracts.generation > 0).any()
    for r in contracts[contracts.generation > 0].itertuples():
        parent = contracts.set_index("contract_id").loc[r.parent_contract_id]
        assert r.inception == parent.expiry
        assert r.initial_tenor == parent.initial_tenor and r.initial_level == parent.initial_level
        assert r.quantity == parent.quantity
        assert np.isclose(r.strike, r.initial_level*inputs.history[r.inception.date()].ref_spot)
    assert contracts.groupby("slot_id").quantity.nunique().eq(1).all()
    option, book = result["option_pnl"], result["book_pnl"]
    assert book.n_options.eq(6).all()
    assert len(book[book.strategy == 'fixed_1']) == 6  # Renewed book continues beyond old last expiry.
    for _, group in book.groupby("strategy"):
        group = group.sort_values("feature_date")
        renewals = group[group.n_renewed > 0]
        assert len(renewals) > 0 and renewals.option_trade_cashflow.lt(0).all()
        assert np.allclose(-renewals.option_trade_cashflow, renewals.renewal_premium)
        assert np.allclose(group.option_pv.iloc[1:].to_numpy(),
                           group.next_option_pv.iloc[:-1].to_numpy()+group.renewal_premium.iloc[1:].to_numpy())
        assert np.allclose(group.wealth_change, group.raw_hedge_error+group.cash_interest+group.stock_income-group.cost)
        assert np.isclose(group.wealth_change.sum(), group.wealth.iloc[-1])
    assert np.allclose(option.raw_hedge_error, option.dV-option.delta*option.dS)
    assert result['summary'].n_contracts_renewed.gt(0).all()


def test_full_expiry_renews_with_one_net_stock_trade_and_zero_spot_keeps_pnl(tmp_path):
    inputs, maps, pricer, options = _fixed_fixture()
    options = replace(options, renew_expired=True, book_tenors=(2/260,))
    for date in inputs.history.dates:
        inputs.history[date] = _surface(date, 1.)
    inputs.daily['beta_surface_daily'] = np.nan  # No realised ratio, but valid closing forecasts and marks.
    result = run_fixed_step7(inputs, outdir=tmp_path, options=options,
                            map_store=maps, pricer=pricer, progress=lambda _: None)
    group = result['book_pnl'].query("strategy == 'constant_sr'").sort_values('feature_date')
    assert len(group) == 6 and group.n_options.eq(3).all()
    assert group.natural_close_units.iloc[:-1].eq(0).all()
    assert (group.n_renewed > 0).sum() == 2
    assert np.allclose(group.hedge_turnover.iloc[1:-1], abs(np.diff(group.book_delta.to_numpy())[:-1]))
    assert np.isfinite(group.raw_hedge_error).all() and np.allclose(group.raw_hedge_error, group.dV)
    options_frame = result['option_pnl']
    assert options_frame.dS.eq(0).all() and options_frame.hedge_pnl.eq(0).all()
    # The terminal close is settlement only: do not open a new unobserved horizon.
    contracts = pd.read_csv(tmp_path/'test_contracts.csv')
    assert contracts.inception.max() < str(group.label_date.iloc[-1])
    assert group.stock_position.iloc[-1] == 0.


def test_renewal_premium_is_not_pnl_and_new_future_contracts_do_not_enter_past_marks():
    first = advance_account(None, pv=10., next_pv=0., settlement=12., delta=.5,
        spot=100., next_spot=100., dt=1/365, rate=0., income_yield=0., cost_bps=0.)
    second = advance_account(first, pv=15., next_pv=15., settlement=0., delta=.3,
        spot=100., next_spot=100., dt=1/365, rate=0., income_yield=0., cost_bps=0., renewal_pv=15.)
    assert second['option_trade_cashflow'] == -15.
    assert second['wealth_change'] == 0. and second['wealth'] == first['wealth']
    inputs, _, _, options = _fixed_fixture()
    dates = inputs.history.dates[2:]
    schedule = contract_schedule(inputs.history, dates, options.book_tenors, LEVELS, 'contracts', 'test', True)
    marked = fixed_marks(inputs.history[dates[0]], inputs.history[dates[1]], schedule)
    assert marked.generation.eq(0).all() and len(marked) == 6
    # Change a future renewal's spot. Original contracts and today's pricing inputs stay identical.
    inputs.history[dates[2]] = _surface(dates[2], 1.2)
    changed = contract_schedule(inputs.history, dates, options.book_tenors, LEVELS, 'contracts', 'test', True)
    pd.testing.assert_frame_equal(marked, fixed_marks(inputs.history[dates[0]], inputs.history[dates[1]], changed))


def test_missing_second_generation_settlement_rejected_before_mc(tmp_path):
    inputs, maps, pricer, options = _fixed_fixture(pd.Timestamp('2026-08-11').date())
    with patch.object(pricer, 'get', side_effect=AssertionError('MC started')):
        with raises(ValueError, 'missing settlement spot'):
            run_fixed_step7(inputs, outdir=tmp_path, options=replace(options, renew_expired=True),
                map_store=maps, pricer=pricer, prepare_only=True, progress=lambda _: None)


def test_equal_vega_slot_quantity_is_preserved_on_renewal():
    inputs, _, _, options = _fixed_fixture()
    dates = inputs.history.dates[2:]
    schedule = contract_schedule(inputs.history, dates, options.book_tenors, (.99, 1., 1.01), 'equal_vega', 'test', True)
    assert schedule.groupby('slot_id').quantity.nunique().eq(1).all()
    initial = schedule[schedule.generation == 0]
    assert np.isclose((initial.quantity*initial.initial_vega).sum(), 100.)


def test_flat_spot_override_uses_exact_fixed_one_delta_and_keeps_ema_state(tmp_path):
    inputs, maps, pricer, options = _fixed_fixture()
    dates = inputs.history.dates
    for date in dates[2:4]:
        inputs.history[date] = _surface(date, inputs.history[dates[1]].ref_spot)
    enabled = run_fixed_step7(inputs, outdir=tmp_path/'enabled', options=options,
        map_store=maps, pricer=pricer, progress=lambda _: None)
    disabled = run_fixed_step7(inputs, outdir=tmp_path/'disabled',
        options=replace(options, flat_spot_alpha_one=False), map_store=maps, pricer=pricer, progress=lambda _: None)
    rows = enabled['option_pnl']
    baseline = rows[rows.strategy == 'fixed_1'].set_index(['feature_date', 'contract_id'])
    adaptive = rows.strategy.str.startswith(('constant_sr', 'term_sr', 'term_spot_sr')) | rows.strategy.eq('rolling_alpha_mean')
    forced = rows[adaptive & rows.feature_date.isin(dates[2:4])]
    assert len(forced) and forced.flat_spot_alpha_one_fallback.all() and forced.alpha.eq(1.).all()
    assert np.array_equal(forced.delta, baseline.loc[list(zip(forced.feature_date, forced.contract_id))].delta)
    assert forced.effective_beta.eq(0).all()
    assert forced.alpha_override_reason.eq('unchanged_observed_spot').all()
    assert (forced.alpha_before_override != 1.).any()
    assert not rows[~adaptive].flat_spot_alpha_one_fallback.any()
    assert not rows[rows.feature_date == dates[4]].flat_spot_alpha_one_fallback.any()
    # The execution override does not feed artificial ones into the model's EMA state.
    normal = rows[rows.strategy.str.endswith('_ema') & rows.feature_date.eq(dates[4])]
    ref = disabled['option_pnl']
    ref = ref[ref.strategy.str.endswith('_ema') & ref.feature_date.eq(dates[4])]
    assert np.array_equal(normal.alpha.to_numpy(), ref.alpha.to_numpy())
    counts = enabled['summary'].set_index('strategy').flat_spot_alpha_one_intervals
    assert counts['term_sr_ema'] == 2 and counts['fixed_0'] == counts['bs_delta'] == 0
    assert not disabled['book_pnl'].flat_spot_alpha_one_fallback.any()


def test_flat_spot_trigger_never_uses_next_close_or_small_nonzero_returns():
    from dynamic_alpha_hedging.step07_fixed_book import _known_flat_spot
    inputs, _, _, _ = _fixed_fixture()
    dates = inputs.history.dates
    # Today's close moved, but tomorrow will be unchanged: no override is known today.
    inputs.history[dates[3]] = _surface(dates[3], inputs.history[dates[2]].ref_spot)
    assert _known_flat_spot(inputs.history, dates[2]) == (dates[1], False)
    assert _known_flat_spot(inputs.history, dates[3]) == (dates[2], True)
    inputs.history[dates[3]] = _surface(dates[3], inputs.history[dates[2]].ref_spot+1e-6)
    assert _known_flat_spot(inputs.history, dates[3]) == (dates[2], False)
    assert _known_flat_spot(inputs.history, dates[0]) == (None, False)

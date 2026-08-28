"""Data-contract and preflight tests for the dynamic-alpha workflow."""

from pathlib import Path

import numpy as np

from dynamic_alpha_hedging.config import (DynamicAlphaConfig,
                                          RESEARCH_STRIKE_LEVELS)
from dynamic_alpha_hedging.data_loader import (
    DEFAULT_TENORS, MarketConventions, implied_vol_panel,
    load_surface_history)
from dynamic_alpha_hedging.preflight import run_preflight

QUOTE_FILE = Path(__file__).resolve().parent.parent / "data" / "svi_data.pkl"


def test_research_strike_levels_match_the_document():
    assert RESEARCH_STRIKE_LEVELS == tuple(x / 10 for x in range(4, 13))


def test_quote_file_loads_into_surfaces():
    history = load_surface_history(QUOTE_FILE, MarketConventions())
    assert len(history) > 0
    assert set(history.skipped.columns) == {"date", "n_slices", "reason"}
    date = history.dates[-1]
    assert history[date].market.pricing_date == date
    assert history[date].ref_spot == history.spots[date]


def test_beta_clamp_never_loses_calibrated_days():
    strict = load_surface_history(QUOTE_FILE, MarketConventions())
    lenient = load_surface_history(QUOTE_FILE, MarketConventions(),
                                   beta_clamp=0.05)
    assert len(lenient) >= len(strict)
    assert len(lenient.skipped) <= len(strict.skipped)


def test_implied_vol_panel_axes():
    history = load_surface_history(QUOTE_FILE, MarketConventions(),
                                   beta_clamp=0.05)
    subset = load_surface_history(QUOTE_FILE, MarketConventions(),
                                  dates=history.dates[:20], beta_clamp=0.05)
    panel = implied_vol_panel(subset, RESEARCH_STRIKE_LEVELS)
    assert panel.shape == (len(subset), len(DEFAULT_TENORS),
                           len(RESEARCH_STRIKE_LEVELS))
    assert panel.expiry_axis == "constant_tau"
    assert np.isfinite(panel.iv).all()
    flat, columns = panel.flattened()
    assert flat.shape == (len(subset),
                          len(DEFAULT_TENORS) * len(RESEARCH_STRIKE_LEVELS))
    assert len(columns) == flat.shape[1]


def test_constant_tau_axis_is_like_for_like():
    history = load_surface_history(QUOTE_FILE, MarketConventions(),
                                   beta_clamp=0.05)
    subset = load_surface_history(QUOTE_FILE, MarketConventions(),
                                  dates=history.dates[:10], beta_clamp=0.05)
    panel = implied_vol_panel(subset, np.array([0.9, 1.0, 1.1]))
    for date_index, date in enumerate(panel.dates):
        surface = subset[date]
        for tenor_index, tenor in enumerate(panel.expiries):
            got = surface.tau_vol(panel.vol_dates[date_index, tenor_index])
            assert abs(got - tenor) <= 1.5 / 260.0


def test_panel_marks_extrapolation_instead_of_filling():
    history = load_surface_history(QUOTE_FILE, MarketConventions(),
                                   beta_clamp=0.05)
    subset = load_surface_history(QUOTE_FILE, MarketConventions(),
                                  dates=history.dates[:5], beta_clamp=0.05)
    panel = implied_vol_panel(subset, np.array([1.0]),
                              tenors=(1 / 260, 0.5), extrapolation="nan")
    assert panel.extrapolated[:, 0].all()
    assert np.isnan(panel.iv[:, 0]).all()
    assert panel.coverage_frame().loc[0, "n_extrapolated"] == len(subset)


def test_preflight_detects_unresolved_observation_dates():
    report = run_preflight(DynamicAlphaConfig(data_path=QUOTE_FILE))
    assert report.audit.alpha_values == (1.0,)
    assert len(report.audit.non_business_dates) > 0
    assert not report.ready_for_step1
    assert len(report.prepared_history) == report.audit.n_records

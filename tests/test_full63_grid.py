"""The default observation, forecast, converter and book grids stay aligned."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from dynamic_alpha_hedging.cli import cli, _config
from dynamic_alpha_hedging.config import DynamicAlphaConfig, LEGACY56_STRIKE_LEVELS
from dynamic_alpha_hedging.grid import validate_cells
from dynamic_alpha_hedging.mc_library import MCLibrary
from dynamic_alpha_hedging.step02 import run_step2
from dynamic_alpha_hedging.step04 import run_step4
from dynamic_alpha_hedging.step05 import run_step5
from dynamic_alpha_hedging.step06 import _ordered_loadings
from dynamic_alpha_hedging.step07_fixed_book import FixedStep7Config
from dynamic_alpha_hedging.step07_shared import _forecast_nodes
from tests.test_precompute import raises


def test_all_cli_defaults_select_the_same_63_cells():
    class Captured(Exception):
        pass

    captured = []

    def capture(args):
        config = _config(args)
        captured.append(config)
        if args.command == "precompute":
            assert tuple(args.tenors) == config.tenors
            assert tuple(args.levels) == config.strike_levels
        raise Captured()

    for command in ("step1", "step2", "step3", "step4", "step5", "step6",
                    "precompute", "step7", "step7-fixed", "step7-sr", "step7-legacy"):
        args = ["dynamic-alpha", command]
        if command in ("step7", "step7-fixed", "step7-sr"):
            args += ["--mc-library", "unused"]
        with patch("sys.argv", args), patch("dynamic_alpha_hedging.cli._config", side_effect=capture), \
                redirect_stdout(io.StringIO()), raises(Captured):
            cli()
    book = FixedStep7Config()
    for config in captured:
        assert config.tenors == config.step4_tenors == book.book_tenors
        assert config.strike_levels == config.step4_strike_levels == book.book_levels
        assert len(config.tenors) == 7 and len(config.strike_levels) == 9


def test_old_56_loadings_and_library_fail_in_full63_mode(tmp_path):
    config = DynamicAlphaConfig()
    cells = pd.MultiIndex.from_product([config.tenors, LEGACY56_STRIKE_LEVELS], names=["tenor", "level"])
    old = cells.to_frame(index=False)
    for column in ("mean_beta_train", "factor_intercept", "atm_beta_loading",
                   "shape_loading_1", "shape_loading_2"):
        old[column] = 0.
    with raises(ValueError, "missing cells"):
        _ordered_loadings(old, config)
    with raises(ValueError, "missing cells"):
        run_step5(None, old, None, None, None, config)
    MCLibrary(tmp_path, config, config.tenors, LEGACY56_STRIKE_LEVELS, writable=True)
    with raises(ValueError, "no extrapolation"):
        MCLibrary(tmp_path, config, config.tenors, config.strike_levels)


def test_step2_rejects_old_72_cell_input_before_calculating_beta():
    config = DynamicAlphaConfig()
    cells = pd.MultiIndex.from_product([(1/12, *config.tenors), config.strike_levels],
                                      names=["tenor", "level"]).to_frame(index=False)
    # Isolate the grid guard from the existing change-decomposition validator.
    with patch("dynamic_alpha_hedging.step02._validate_input"), raises(ValueError, "unexpected tenor"):
        run_step2(cells, config)


def test_full63_factor_reconstruction_includes_all_nine_levels():
    config = DynamicAlphaConfig()
    cells = pd.MultiIndex.from_product([config.tenors, config.strike_levels], names=["tenor", "level"])
    dates = pd.bdate_range("2025-01-02", periods=40).date
    rng = np.random.default_rng(19)
    beta = pd.DataFrame(rng.normal(size=(len(dates), len(cells))), index=dates, columns=cells)
    beta.index.name = "observation_date"
    daily = beta.stack([0, 1]).rename("beta_surface_daily").reset_index()
    result = run_step4(daily, config)
    assert result.validation["surface_cell_count"] == 63
    ordered = _ordered_loadings(result.loadings, config)
    forecast = pd.DataFrame({"atm_beta_factor": [.1], "shape_score_1": [.2], "shape_score_2": [.3]},
                            index=[dates[-1]])
    inputs = SimpleNamespace(forecasts=forecast, loadings=ordered,
                             settings=SimpleNamespace(factor_count=3))
    nodes = _forecast_nodes(inputs, dates[-1])
    assert len(nodes) == 63 and np.isfinite(nodes).all()
    assert len(nodes.xs(1.2, level="level")) == 7
    validate_cells(nodes.rename("beta").reset_index(), config.tenors, config.strike_levels,
                   source="reconstructed forecast")


def test_grid_validation_tolerates_csv_roundoff_but_rejects_missing_models():
    config = DynamicAlphaConfig()
    cells = pd.MultiIndex.from_product([config.tenors, config.strike_levels],
                                      names=["tenor", "level"]).to_frame(index=False)
    cells["tenor"] = cells.tenor.round(15)
    validate_cells(cells, config.tenors, config.strike_levels, source="models")
    with raises(ValueError, "missing cells"):
        validate_cells(cells.iloc[:-1], config.tenors, config.strike_levels, source="models")

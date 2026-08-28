"""Workflow-boundary tests for the four calibration steps."""

from pricing_svi_localvol_calibration.config import (
    DEFAULT_STRIKE_LEVELS, TEST_MARKET, TEST_VOL_PARAMS)
from pricing_svi_localvol_calibration.step01_svi_conversion import (
    run as run_step01, validate as validate_step01)
from pricing_svi_localvol_calibration.step02_implied_surface import (
    run as run_step02, validate as validate_step02)
from pricing_svi_localvol_calibration.step03_localvol_surface import (
    run as run_step03, validate as validate_step03)
from pricing_svi_localvol_calibration.step04_mc_validation import (
    run as run_step04, validate as validate_step04)
from pricing_svi_localvol_calibration.task_api import build_surface


def _surface():
    return build_surface(TEST_MARKET, TEST_VOL_PARAMS)


def test_calibration_step01_is_independently_runnable():
    surface = _surface()
    table = run_step01(surface)
    checks = validate_step01(surface)
    assert len(table) == len(TEST_VOL_PARAMS["VolDate"])
    assert checks["pass"].all()


def test_calibration_step02_is_independently_runnable():
    surface = _surface()
    matrix = run_step02(surface, surface.slices[-1].vol_date,
                        DEFAULT_STRIKE_LEVELS)
    checks = validate_step02(surface)
    assert matrix.shape[1] == len(DEFAULT_STRIKE_LEVELS)
    assert checks["pass"].all()


def test_calibration_step03_is_independently_runnable():
    surface = _surface()
    result = run_step03(surface, surface.slices[-1].vol_date,
                        DEFAULT_STRIKE_LEVELS)
    checks = validate_step03(result)
    assert result["local_vol"].shape == result["flags"].shape
    assert checks["n_points"] == result["flags"].size


def test_calibration_step04_is_independently_runnable():
    surface = _surface()
    result = run_step04(surface.repaired(), surface.slices[-1].vol_date,
                        surface.market.spot, n_paths=2_000, n_substeps=1,
                        raw_surface=surface, comparison_levels=(1.0,))
    checks = validate_step04(result)
    assert len(result["pricing_errors"]) == 1
    assert len(result["delta_comparison"]) == 1
    assert set(checks) == {
        "pv_relative_error", "pv_pass_1pct", "delta_error", "delta_stderr",
        "delta_within_2_stderr",
    }

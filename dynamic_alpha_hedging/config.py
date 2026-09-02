"""Frozen conventions for the dynamic-alpha hedging study."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from svi_localvol.conventions import us_equity_scheduled_holidays
from svi_localvol.params import (ALPHA_STICKY_LOCAL_VOL,
                                 ALPHA_STICKY_MONEYNESS,
                                 ALPHA_STICKY_STRIKE,
                                 validate_stickiness_alpha)

from .data_loader import MarketConventions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "svi_param.pkl"
RESEARCH_TENORS: tuple[float, ...] = (
    1 / 12, 2 / 12, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
# The document's literal grid is 0.4, 0.5, ..., 1.2.  Finer grids belong in
# an explicit sensitivity run rather than silently changing the baseline.
RESEARCH_STRIKE_LEVELS: tuple[float, ...] = tuple(x / 10 for x in range(4, 13))


@dataclass(frozen=True)
class DynamicAlphaConfig:
    """Every convention that must be frozen before dynamic Step 1."""

    data_path: Path = DEFAULT_DATA_PATH
    rate: float = 0.036
    dividend: float = 0.03
    repo: float = 0.0
    holidays: tuple[dt.date, ...] = field(default_factory=lambda:
        us_equity_scheduled_holidays(2023, 2030))
    holiday_calendar_name: str = "scheduled US equity holidays (no one-offs)"
    source_timezone: str = "Asia/Shanghai"
    market_timezone: str = "America/New_York"
    require_weekday_observations: bool = True
    duplicate_vol_date_policy: str = "first"
    allow_skipped_surfaces: bool = True
    beta_clamp: float = 0.0
    expiry_axis: str = "constant_tau"
    tenors: tuple[float, ...] = RESEARCH_TENORS
    strike_levels: tuple[float, ...] = RESEARCH_STRIKE_LEVELS
    level_anchor: str = "spot"
    extrapolation: str = "nan"
    beta_min_abs_dlogS: float = 0.005
    beta_require_consecutive_business_days: bool = True
    beta_window: int = 60
    beta_min_obs: int = 20
    beta_threshold_sensitivity: tuple[float, ...] = (0.0025, 0.005, 0.01)
    beta_window_sensitivity: tuple[int, ...] = (20, 40, 60, 120)
    beta_weight_sensitivity: tuple[str, ...] = (
        "equal", "abs_dlogS", "time_decay")
    beta_time_decay_half_life: float = 20.0
    step3_alphas: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    step3_calibration_date: dt.date | None = None
    step3_spot_bump_fraction: float = 0.01
    # Production Step 3 defaults.  ``--fast`` in the CLI is deliberately a
    # diagnostic run and must not be used to publish the converter.
    step3_n_paths: int = 100_000
    step3_seed: int = 20260807
    step3_antithetic: bool = True
    step3_n_substeps: int = 2
    step3_n_ratio: int = 801
    step3_ratio_min: float = 1e-3
    step3_ratio_max: float = 3.0
    step3_vol_floor: float = 0.0
    step3_vol_cap: float = 5.0
    step3_alpha_one_abs_tolerance: float = 0.03
    step3_max_beta_stderr: float = 0.10
    step3_min_beta_span: float = 0.10
    step3_min_span_z: float = 3.0
    step3_max_grid_undefined_fraction: float = 0.05
    step3_max_grid_clipped_fraction: float = 0.01

    def __post_init__(self):
        object.__setattr__(self, "data_path", Path(self.data_path))
        if self.expiry_axis != "constant_tau":
            raise ValueError("dynamic-alpha study must use constant_tau across days")
        if self.level_anchor != "spot":
            raise ValueError("the study grid is quoted as K / refSpot")
        if self.extrapolation != "nan":
            raise ValueError("study inputs must mark, not fill, tenor extrapolation")
        if self.beta_clamp < 0:
            raise ValueError("beta_clamp must be non-negative")
        if self.duplicate_vol_date_policy not in ("first", "error"):
            raise ValueError("duplicate_vol_date_policy must be 'first' or 'error'")
        if not self.tenors or min(self.tenors) <= 0:
            raise ValueError("tenors must be strictly positive")
        if not self.strike_levels or min(self.strike_levels) <= 0:
            raise ValueError("strike levels must be strictly positive")
        if self.beta_min_abs_dlogS < 0:
            raise ValueError("beta_min_abs_dlogS must be non-negative")
        if self.beta_window < 2:
            raise ValueError("beta_window must be at least 2")
        if not 2 <= self.beta_min_obs <= self.beta_window:
            raise ValueError("beta_min_obs must be between 2 and beta_window")
        if (not self.beta_threshold_sensitivity
                or min(self.beta_threshold_sensitivity) < 0):
            raise ValueError("beta threshold sensitivities must be non-negative")
        if (not self.beta_window_sensitivity
                or min(self.beta_window_sensitivity) < 2):
            raise ValueError("beta window sensitivities must be at least 2")
        allowed_weights = {"equal", "abs_dlogS", "time_decay"}
        if not self.beta_weight_sensitivity or not set(
                self.beta_weight_sensitivity).issubset(allowed_weights):
            raise ValueError("unsupported beta weight sensitivity")
        if self.beta_time_decay_half_life <= 0:
            raise ValueError("beta_time_decay_half_life must be positive")
        if not self.step3_alphas:
            raise ValueError("step3_alphas must not be empty")
        validated_alphas = tuple(
            validate_stickiness_alpha(alpha) for alpha in self.step3_alphas)
        if tuple(sorted(set(validated_alphas))) != validated_alphas:
            raise ValueError("step3_alphas must be unique and increasing")
        if 1.0 not in validated_alphas:
            raise ValueError("step3_alphas must contain the alpha=1 sanity anchor")
        object.__setattr__(self, "step3_alphas", validated_alphas)
        if self.step3_calibration_date is not None:
            object.__setattr__(
                self, "step3_calibration_date",
                dt.date.fromisoformat(self.step3_calibration_date)
                if isinstance(self.step3_calibration_date, str)
                else self.step3_calibration_date)
        if not 0.0 < self.step3_spot_bump_fraction < 1.0:
            raise ValueError("step3_spot_bump_fraction must lie in (0, 1)")
        if self.step3_n_paths < 2:
            raise ValueError("step3_n_paths must be at least 2")
        if self.step3_antithetic and self.step3_n_paths % 2:
            raise ValueError("step3_n_paths must be even with antithetic sampling")
        if self.step3_n_substeps < 1:
            raise ValueError("step3_n_substeps must be at least 1")
        if self.step3_n_ratio < 3:
            raise ValueError("step3_n_ratio must be at least 3")
        if not 0.0 < self.step3_ratio_min < self.step3_ratio_max:
            raise ValueError("step3 ratio bounds must satisfy 0 < min < max")
        if not 0.0 <= self.step3_vol_floor < self.step3_vol_cap:
            raise ValueError("step3 vol bounds must satisfy 0 <= floor < cap")
        if self.step3_alpha_one_abs_tolerance <= 0.0:
            raise ValueError("step3 alpha-one tolerance must be positive")
        if self.step3_max_beta_stderr <= 0.0:
            raise ValueError("step3 maximum beta stderr must be positive")
        if self.step3_min_beta_span < 0.0:
            raise ValueError("step3 minimum beta span must be non-negative")
        if self.step3_min_span_z < 0.0:
            raise ValueError("step3 minimum span z-score must be non-negative")
        for name, value in (
                ("undefined", self.step3_max_grid_undefined_fraction),
                ("clipped", self.step3_max_grid_clipped_fraction)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"step3 maximum grid {name} fraction must lie in [0, 1]")
        for alpha in (ALPHA_STICKY_LOCAL_VOL, ALPHA_STICKY_STRIKE,
                      ALPHA_STICKY_MONEYNESS):
            validate_stickiness_alpha(alpha)

    @property
    def market_conventions(self) -> MarketConventions:
        return MarketConventions(rate=self.rate, dividend=self.dividend,
                                 repo=self.repo, holidays=self.holidays)

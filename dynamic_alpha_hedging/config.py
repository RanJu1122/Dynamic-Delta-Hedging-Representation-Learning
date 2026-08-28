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
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "svi_data.pkl"
RESEARCH_TENORS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
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
        us_equity_scheduled_holidays(2024, 2030))
    holiday_calendar_name: str = "scheduled US equity holidays (no one-offs)"
    observation_date_shift: int = 0
    roll_observation_dates: bool = False
    require_business_observation_dates: bool = True
    beta_clamp: float = 0.0
    expiry_axis: str = "constant_tau"
    tenors: tuple[float, ...] = RESEARCH_TENORS
    strike_levels: tuple[float, ...] = RESEARCH_STRIKE_LEVELS
    level_anchor: str = "spot"
    extrapolation: str = "nan"

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
        if not self.tenors or min(self.tenors) <= 0:
            raise ValueError("tenors must be strictly positive")
        if not self.strike_levels or min(self.strike_levels) <= 0:
            raise ValueError("strike levels must be strictly positive")
        for alpha in (ALPHA_STICKY_LOCAL_VOL, ALPHA_STICKY_STRIKE,
                      ALPHA_STICKY_MONEYNESS):
            validate_stickiness_alpha(alpha)

    @property
    def market_conventions(self) -> MarketConventions:
        return MarketConventions(rate=self.rate, dividend=self.dividend,
                                 repo=self.repo, holidays=self.holidays)

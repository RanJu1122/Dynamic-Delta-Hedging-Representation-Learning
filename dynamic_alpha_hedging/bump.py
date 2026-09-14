"""Deterministic expiry-aware spot bumps for new shared-SR pricing.

This controls a numerical derivative, not the forecast Alpha policy. It uses
only the pricing date and contractual expiry, never the next observed return.
"""
from dataclasses import dataclass
import numpy as np
from svi_localvol.conventions import nb_biz_days


@dataclass(frozen=True)
class SpotBumpPolicy:
    short_business_days: int = 10
    short_fraction: float = .005

    def __post_init__(self):
        if (isinstance(self.short_business_days, bool)
                or not isinstance(self.short_business_days, int)
                or self.short_business_days < 0):
            raise ValueError("short bump business days must be a nonnegative integer")
        if not np.isfinite(self.short_fraction) or not 0 < self.short_fraction < 1:
            raise ValueError("short bump fraction must lie in (0, 1)")

    def fraction(self, surface, expiry, base_fraction):
        remaining = nb_biz_days(surface.market.pricing_date, expiry, surface.market.holidays)
        if remaining <= 0:
            raise ValueError("spot bump requires a future business expiry")
        if not np.isfinite(base_fraction) or not 0 < base_fraction < 1:
            raise ValueError("base bump fraction must lie in (0, 1)")
        if remaining <= self.short_business_days:
            return float(min(base_fraction, self.short_fraction))
        return float(base_fraction)

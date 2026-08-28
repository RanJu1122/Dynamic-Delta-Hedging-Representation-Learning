"""Input containers: market data and the SVI-JW quote set."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from .conventions import to_date

# --------------------------------------------------------------------------- #
# Stickiness parameter alpha (the `StickinessRatio` field).
#
# The dynamic-hedging study parameterises smile dynamics with a single
# per-VolDate scalar alpha, entering the Dupire denominator as
#
#     y_adj = y - alpha * spot_adj,      spot_adj = log(S / refSpot)
#
# The three anchors are defined by the dynamic-alpha document.  Their model
# beta values must be measured empirically; no analytic shortcut is encoded in
# the core library.
# --------------------------------------------------------------------------- #
ALPHA_STICKY_LOCAL_VOL = 0.0
ALPHA_STICKY_STRIKE = 1.0
ALPHA_STICKY_MONEYNESS = 2.0
DEFAULT_ALPHA = ALPHA_STICKY_STRIKE


def validate_stickiness_alpha(alpha: float, *, allow_extrapolation: bool = False) -> float:
    """Validate the dynamic-hedging study's dimensionless alpha convention.

    The research document defines continuous interpolation on [0, 2].  Model
    experiments outside that interval must opt in explicitly so a negative or
    otherwise incompatible value cannot accidentally be passed as alpha.
    """
    value = float(alpha)
    if not np.isfinite(value):
        raise ValueError("stickiness alpha must be finite")
    if not allow_extrapolation and not (
            ALPHA_STICKY_LOCAL_VOL <= value <= ALPHA_STICKY_MONEYNESS):
        raise ValueError("stickiness alpha must lie in [0, 2]")
    return value


@dataclass(frozen=True)
class MarketData:
    """Everything that is not the volatility surface itself."""

    pricing_date: dt.date
    spot: float                       # S0, the live spot used to normalise strikes
    rate: float = 0.0                 # continuously compounded
    dividend: float = 0.0             # continuous dividend yield
    repo: float = 0.0                 # continuous repo / borrow
    holidays: tuple[dt.date, ...] = ()

    @property
    def cost_of_carry(self) -> float:
        """b = r - q - repo, the drift of the forward under the risk-neutral measure."""
        return self.rate - self.dividend - self.repo

    def __post_init__(self):
        object.__setattr__(self, "pricing_date", to_date(self.pricing_date))
        object.__setattr__(self, "holidays",
                           tuple(to_date(h) for h in self.holidays))


@dataclass(frozen=True)
class SVIJWQuote:
    """The five numbers a trading desk quotes for one expiry.

    Field names follow the trading system, including the legacy misnomer
    `kurt`, which is the MINIMUM implied volatility, not a kurtosis.
    """

    vol_date: dt.date
    atm_vol: float          # ATMVol   -- annualised ATM implied vol
    skew: float             # Skew     -- ATM skew  d(sigma)/dk  at k = 0
    putwing: float          # Putwing  -- left wing slope   p_t
    callwing: float         # Callwing -- right wing slope  c_t
    kurt: float             # Kurt     -- MINIMUM implied vol (not kurtosis)
    alpha: float = DEFAULT_ALPHA        # StickinessRatio for this expiry

    @property
    def atm_var(self) -> float:
        return self.atm_vol ** 2

    @property
    def min_imp_var(self) -> float:
        return self.kurt ** 2

    @property
    def stickiness_ratio(self) -> float:
        """Backward-compatible alias for `alpha`."""
        return self.alpha

    def __post_init__(self):
        object.__setattr__(self, "vol_date", to_date(self.vol_date))
        object.__setattr__(self, "alpha", validate_stickiness_alpha(self.alpha))


@dataclass(frozen=True)
class VolQuoteSet:
    """The full surface quote: a reference spot plus one quote per expiry."""

    ref_spot: float
    quotes: tuple[SVIJWQuote, ...]

    def __post_init__(self):
        q = tuple(sorted(self.quotes, key=lambda x: x.vol_date))
        object.__setattr__(self, "quotes", q)

    @property
    def vol_dates(self) -> list[dt.date]:
        return [q.vol_date for q in self.quotes]

    @classmethod
    def from_dict(cls, d: dict) -> "VolQuoteSet":
        """Build from the dict layout used in the task statement."""
        n = len(d["VolDate"])
        stick = d.get("StickinessRatio", [DEFAULT_ALPHA] * n)
        quotes = [
            SVIJWQuote(
                vol_date=d["VolDate"][i],
                atm_vol=d["ATMVol"][i],
                skew=d["Skew"][i],
                putwing=d["Putwing"][i],
                callwing=d["Callwing"][i],
                kurt=d["Kurt"][i],
                alpha=float(stick[i]),
            )
            for i in range(n)
        ]
        return cls(ref_spot=float(d.get("Spot", 1.0)), quotes=tuple(quotes))

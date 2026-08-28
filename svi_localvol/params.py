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
# Measured anchors on the shipped test surface (ATM call, furthest VolDate,
# 200k paths, common random numbers) -- see `tests.py::test_alpha_anchors`:
#
#     alpha = 0  ->  0.3728   local volatility model delta (sigma_loc frozen)
#     alpha = 1  ->  0.5256   sticky strike, matches Black-Scholes 0.5284
#     alpha = 2  ->  0.6759   moves toward sticky moneyness (analytic 0.6408)
#
# Only the alpha = 1 anchor is exact by construction; the study's Step 3 is
# what measures where the other two actually land.
# --------------------------------------------------------------------------- #
ALPHA_STICKY_LOCAL_VOL = 0.0
ALPHA_STICKY_STRIKE = 1.0
ALPHA_STICKY_MONEYNESS = 2.0
DEFAULT_ALPHA = ALPHA_STICKY_STRIKE


def validate_stickiness_alpha(alpha: float, *, allow_extrapolation: bool = False) -> float:
    """Validate the dynamic-hedging study's dimensionless alpha convention.

    The research document defines continuous interpolation on [0, 2].  Model
    experiments outside that interval must opt in explicitly so an old R value
    cannot accidentally be passed as alpha.
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


# --------------------------------------------------------------------------- #
# The test case shipped with the task statement.
# --------------------------------------------------------------------------- #
TEST_VOL_PARAMS: dict = {
    "VolDate": [
        dt.date(2026, 9, 18), dt.date(2026, 10, 16), dt.date(2026, 11, 20),
        dt.date(2026, 12, 18), dt.date(2027, 1, 15), dt.date(2027, 2, 19),
        dt.date(2027, 3, 19), dt.date(2027, 4, 16),
    ],
    "ATMVol": [
        0.13101478256828936, 0.1387398983996047, 0.1469180239887264,
        0.15133376603985407, 0.15438106674725624, 0.15880879795534372,
        0.1619786871451705, 0.16456310845677094,
    ],
    "Skew": [
        -0.5472692091059775, -0.5004875497422803, -0.4412850981019261,
        -0.41108464818084156, -0.38715454613641037, -0.36530950353352926,
        -0.3493661568911403, -0.34279006976244386,
    ],
    "Putwing": [
        1.748011556529109, 1.6176256580662147, 1.462056803327979,
        1.3676788186018394, 1.3733898083855658, 1.3187477856716598,
        1.2623019444335133, 1.2875100239467405,
    ],
    "Callwing": [
        1.4921104419231117, 1.184486436063978, 0.9184304493164512,
        0.7113392544568392, 0.6925504706439176, 0.6389611418448444,
        0.6037929223193157, 1.2154695651542504,
    ],
    "Kurt": [
        0.11768267800001103, 0.12057683387130651, 0.12619802294212767,
        0.12967717064320597, 0.1306643672715086, 0.13245972609810636,
        0.1338227295593484, 0.13482067267086356,
    ],
    "StickinessRatio": [1, 1, 1, 1, 1, 1, 1, 1],
    "Spot": 1.0,
}

TEST_MARKET = MarketData(
    pricing_date=dt.date(2026, 8, 7),
    spot=1.0,
    rate=0.036,
    dividend=0.03,
    repo=0.0,
    holidays=(),
)

DEFAULT_STRIKE_LEVELS: np.ndarray = np.round(np.arange(0.80, 1.3001, 0.05), 4)

# Strike axis for the dynamic-alpha hedging study (K = level * S_t).  The study
# asks for a much wider band than the Task 7 pricing grid because the empirical
# beta surface is measured out to deep wings.
HEDGING_STRIKE_LEVELS: np.ndarray = np.round(np.arange(0.40, 1.2001, 0.05), 4)

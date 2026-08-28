"""Daily SVI quote history -> surfaces -> implied volatility panels.

This is the data layer the dynamic-alpha hedging study starts from.  It turns
``data/svi_data.pkl`` (one SVI-JW quote set per observation) into

    * one `VolSurface` per day, and
    * a descriptive IV[t, i, j] panel on an explicit coordinate grid,

and nothing else: no beta, no regression, no hedging.  Dynamic Step 1 must
add fixed-contract day-over-day IV changes; a naive difference of panels whose
strike is re-anchored to each day's spot is not an empirical stickiness label.

Three conventions have to be pinned down before any of that, because they are
not implied by the data and silently choosing wrong would poison every later
step.

1.  RATES.  The pickle carries only `Spot` and the five SVI-JW fields.  There
    is no rate, dividend or repo in it, so `MarketConventions` must be supplied
    by the caller.  The defaults here are the ones from the Task 7 statement
    (r = 3.6%, q = 3%, repo = 0); they are a placeholder, not a measurement.

2.  HOLIDAYS.  dt_vol is Business/260 and the underlying is a US index, so the
    exchange holiday list belongs in `MarketConventions.holidays`.  Leaving it
    empty overstates the business-day count across holiday weeks and therefore
    understates every tau.  Empty is the default only because the data does not
    ship a calendar.

3.  THE EXPIRY AXIS.  The number of quoted VolDates moves between 8 and 18 over
    the sample and the expiries roll, so there is no fixed set of expiry dates
    to index by.  Two axes are offered:

      'slot'         the i-th nearest expiry.  Rectangular and cheap, but a
                     difference taken along t at fixed slot straddles the roll:
                     on a roll day it compares two DIFFERENT expiries and the
                     jump is an artefact, not a volatility move.
      'constant_tau' a fixed Business/260 tenor grid, read off each day's own
                     surface.  Differences along t are then like-for-like.
                     This is the right rectangular axis for reporting and
                     later cross-sectional factor analysis.  Fixed-contract
                     dIV still requires evaluating adjacent surfaces at the
                     same actual strike and expiry.

    `implied_vol_panel` defaults to 'constant_tau' for that reason and records
    which axis produced the panel.
"""

from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from svi_localvol.conventions import is_biz_day, nb_biz_days, roll, to_date
from svi_localvol.params import MarketData, VolQuoteSet
from svi_localvol.surface import VolSurface

#: Tenor grid in years (Business/260) used by the 'constant_tau' expiry axis.
DEFAULT_TENORS: np.ndarray = np.array([1 / 12, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

REQUIRED_QUOTE_FIELDS: tuple[str, ...] = (
    "Spot", "VolDate", "ATMVol", "Skew", "Putwing", "Callwing", "Kurt",
    "StickinessRatio",
)


@dataclass(frozen=True)
class MarketConventions:
    """Everything the quote file does not carry.  See the module docstring."""

    rate: float = 0.036
    dividend: float = 0.03
    repo: float = 0.0
    holidays: tuple[dt.date, ...] = ()

    def market_data(self, pricing_date, spot: float) -> MarketData:
        return MarketData(pricing_date=pricing_date, spot=float(spot),
                          rate=self.rate, dividend=self.dividend,
                          repo=self.repo, holidays=self.holidays)


@dataclass
class SurfaceHistory:
    """One calibrated surface per trading day, plus what could not be built."""

    surfaces: dict[dt.date, VolSurface]
    spots: pd.Series                      # index = date, value = close
    skipped: pd.DataFrame                 # date, n_slices, reason
    conventions: MarketConventions
    calendar_repair: bool

    @property
    def dates(self) -> list[dt.date]:
        return sorted(self.surfaces)

    def __len__(self) -> int:
        return len(self.surfaces)

    def __getitem__(self, key) -> VolSurface:
        return self.surfaces[to_date(key)]

    def log_spot_returns(self) -> pd.Series:
        """dlogS(t), indexed by t, aligned to the surface dates."""
        s = self.spots.reindex(self.dates).astype(float)
        return np.log(s).diff().rename("dlogS")

    def summary(self) -> str:
        n_skip = len(self.skipped)
        head = (f"{len(self)} surfaces, {self.dates[0]} -> {self.dates[-1]}, "
                f"{n_skip} day(s) skipped")
        if n_skip:
            head += "\n" + self.skipped.to_string(index=False)
        return head


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_quote_file(path: str | Path, *, observation_date_shift: int = 0,
                    roll_observation_dates: bool = False,
                    holidays: Iterable = ()) -> dict[dt.date, dict]:
    """Read the pickled ``{observation date: SVI-JW quote dict}`` file.

    Dates are preserved exactly by default.  The research file contains many
    weekend keys, but its timestamp semantics are not documented; silently
    repairing those keys would risk moving two distinct observations onto the
    same date.  A caller may explicitly shift all keys by a fixed number of
    calendar days and/or roll non-business dates backward.  Any collision then
    raises instead of overwriting data.
    """
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    if not isinstance(raw, dict):
        raise TypeError(f"expected a dict of daily quotes, got {type(raw)!r}")
    out: dict[dt.date, dict] = {}
    sources: dict[dt.date, dt.date] = {}
    for key, record in raw.items():
        source = to_date(key)
        target = source + dt.timedelta(days=int(observation_date_shift))
        if roll_observation_dates and not is_biz_day(target, holidays):
            target = roll(target, "Preceding", holidays)
        if target in out:
            raise ValueError(
                f"observation-date transform collision: {sources[target]} and "
                f"{source} both map to {target}"
            )
        out[target] = record
        sources[target] = source
    return out


def build_surface(pricing_date, record: dict,
                  conventions: MarketConventions = MarketConventions(),
                  calendar_repair: bool = False,
                  beta_clamp: float = 0.0) -> VolSurface:
    """Calibrate one day's quote record into a `VolSurface`.

    `Spot` is used both as the live spot and as the surface's refSpot: the
    quotes were fitted on that day's close, so the two coincide by construction
    in this data set.
    """
    spot = float(record["Spot"])
    quotes = VolQuoteSet.from_dict(record)
    market = conventions.market_data(to_date(pricing_date), spot)
    return VolSurface(market, quotes, calendar_repair=calendar_repair,
                      beta_clamp=beta_clamp)


def load_surface_history(path: str | Path,
                         conventions: MarketConventions = MarketConventions(),
                         calendar_repair: bool = False,
                         on_error: str = "skip",
                         dates: Sequence | None = None,
                         beta_clamp: float = 0.0,
                         observation_date_shift: int = 0,
                         roll_observation_dates: bool = False) -> SurfaceHistory:
    """Calibrate every day in the quote file.

    on_error
        'skip'  record the failure and carry on (default -- a handful of days
                carry a non-convex quoted smile and cannot be converted).
        'raise' propagate, useful when debugging a single day.

    A day is skipped rather than patched: silently repairing an input the desk
    considers valid would hide a data problem the study should see.  Inspect
    `SurfaceHistory.skipped` and decide.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError("on_error must be 'skip' or 'raise'")

    records = load_quote_file(
        path, observation_date_shift=observation_date_shift,
        roll_observation_dates=roll_observation_dates,
        holidays=conventions.holidays)
    wanted = sorted(records) if dates is None else [to_date(d) for d in dates]

    surfaces: dict[dt.date, VolSurface] = {}
    spots: dict[dt.date, float] = {}
    failures: list[dict] = []
    for d in wanted:
        rec = records[d]
        spots[d] = float(rec["Spot"])
        try:
            surfaces[d] = build_surface(d, rec, conventions, calendar_repair,
                                        beta_clamp=beta_clamp)
        except Exception as exc:                                   # noqa: BLE001
            if on_error == "raise":
                raise
            failures.append({"date": d, "n_slices": len(rec["VolDate"]),
                             "reason": f"{type(exc).__name__}: {exc}"})

    return SurfaceHistory(
        surfaces=surfaces,
        spots=pd.Series(spots, name="spot").sort_index(),
        skipped=pd.DataFrame(failures,
                             columns=["date", "n_slices", "reason"]),
        conventions=conventions,
        calendar_repair=calendar_repair,
    )


# --------------------------------------------------------------------------- #
# implied volatility panel  ->  IV[t, i, j]
# --------------------------------------------------------------------------- #
@dataclass
class ImpliedVolPanel:
    """IV[t, i, j] with its axes and the conventions that produced it."""

    iv: np.ndarray                      # (n_dates, n_expiries, n_levels)
    dates: list[dt.date]
    expiries: np.ndarray                # tenors in years, or slot indices
    levels: np.ndarray                  # K / anchor
    expiry_axis: str                    # 'constant_tau' | 'slot'
    level_anchor: str                   # 'spot' | 'forward'
    vol_dates: np.ndarray = field(default=None, repr=False)  # (t, i) actual dates
    extrapolated: np.ndarray = field(default=None, repr=False)  # (t, i)
    extrapolation_policy: str = "flat"

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.iv.shape

    def to_frame(self) -> pd.DataFrame:
        """Long format: one row per (date, expiry, level)."""
        t, i, j = self.iv.shape
        idx = pd.MultiIndex.from_product(
            [self.dates, self.expiries, self.levels],
            names=["date", "expiry", "level"])
        out = pd.DataFrame({"implied_vol": self.iv.reshape(-1)}, index=idx)
        return out.reset_index()

    def cross_section(self, date) -> pd.DataFrame:
        """One day's (expiry x level) slice as a DataFrame."""
        k = self.dates.index(to_date(date))
        return pd.DataFrame(self.iv[k],
                            index=pd.Index(self.expiries, name="expiry"),
                            columns=pd.Index(self.levels, name="level"))

    def flattened(self) -> tuple[np.ndarray, pd.MultiIndex]:
        """(n_dates, n_expiries * n_levels) matrix, ready for PCA."""
        t = len(self.dates)
        cols = pd.MultiIndex.from_product([self.expiries, self.levels],
                                          names=["expiry", "level"])
        return self.iv.reshape(t, -1), cols

    def coverage_frame(self) -> pd.DataFrame:
        """Per-expiry counts of observations read outside quoted maturities."""
        mask = (np.zeros(self.vol_dates.shape, dtype=bool)
                if self.extrapolated is None else self.extrapolated)
        return pd.DataFrame({
            "expiry": self.expiries,
            "n_dates": len(self.dates),
            "n_extrapolated": mask.sum(axis=0),
            "fraction_extrapolated": mask.mean(axis=0),
        })


def implied_vol_panel(history: SurfaceHistory,
                      levels: Sequence[float],
                      expiry_axis: str = "constant_tau",
                      tenors: Sequence[float] = DEFAULT_TENORS,
                      n_slots: int | None = None,
                      level_anchor: str = "spot",
                      extrapolation: str = "flat") -> ImpliedVolPanel:
    """Build IV[t, i, j] from a history of daily surfaces.

    levels
        Strike levels.  K = level * anchor, where the anchor is that day's spot
        ('spot', the default) or that expiry's forward ('forward').  'spot'
        matches the Task 7 convention K = level * S0.

    expiry_axis
        'constant_tau'  read each day's surface at the fixed `tenors` grid
                        (years, Business/260).  Like-for-like across days.
        'slot'          the i-th nearest quoted VolDate.  Rolls.

    See the module docstring for why 'constant_tau' is the default.
    """
    if expiry_axis not in ("constant_tau", "slot"):
        raise ValueError("expiry_axis must be 'constant_tau' or 'slot'")
    if level_anchor not in ("spot", "forward"):
        raise ValueError("level_anchor must be 'spot' or 'forward'")
    if extrapolation not in ("flat", "nan"):
        raise ValueError("extrapolation must be 'flat' or 'nan'")

    levels = np.asarray(levels, dtype=float)
    dates = history.dates

    if expiry_axis == "constant_tau":
        expiries = np.asarray(tenors, dtype=float)
    else:
        n = n_slots if n_slots is not None else min(
            len(history[d].slices) for d in dates)
        expiries = np.arange(n, dtype=float)

    iv = np.full((len(dates), len(expiries), len(levels)), np.nan)
    vol_dates = np.empty((len(dates), len(expiries)), dtype=object)
    extrapolated = np.zeros((len(dates), len(expiries)), dtype=bool)

    for a, d in enumerate(dates):
        surf = history[d]
        anchor_spot = surf.market.spot
        for b, e in enumerate(expiries):
            if expiry_axis == "constant_tau":
                T = _date_at_tau(surf, float(e))
                outside = float(e) < surf.taus[0] or float(e) > surf.taus[-1]
                extrapolated[a, b] = outside
            else:
                T = surf.slices[int(e)].vol_date
            vol_dates[a, b] = T
            if surf.tau_vol(T) <= 0:
                continue
            if extrapolated[a, b] and extrapolation == "nan":
                continue
            anchor = (anchor_spot if level_anchor == "spot"
                      else surf.forward(T, anchor_spot))
            iv[a, b] = np.atleast_1d(surf.implied_vol(T, levels * anchor))

    return ImpliedVolPanel(iv=iv, dates=dates, expiries=expiries,
                           levels=levels, expiry_axis=expiry_axis,
                           level_anchor=level_anchor, vol_dates=vol_dates,
                           extrapolated=extrapolated,
                           extrapolation_policy=extrapolation)


def _date_at_tau(surface: VolSurface, tau: float) -> dt.date:
    """The business date whose dt_vol is closest to `tau`.

    dt_vol is (business days) / 260, so a tenor in years maps to an integer
    business-day count and therefore to an exact date.  Walking the calendar is
    cheap and keeps the holiday list authoritative.
    """
    n_biz = max(1, int(round(tau * 260.0)))
    hol = surface.market.holidays
    d = surface.market.pricing_date
    step = dt.timedelta(days=1)
    # walk forward until the business-day count matches
    guess = d + dt.timedelta(days=int(round(n_biz * 7 / 5)))
    guess = roll(guess, "Following", hol)
    while nb_biz_days(d, guess, hol) > n_biz:
        guess = roll(guess - step, "Preceding", hol)
    while nb_biz_days(d, guess, hol) < n_biz:
        guess = roll(guess + step, "Following", hol)
    return guess


def spot_and_returns(history: SurfaceHistory) -> pd.DataFrame:
    """Daily close and dlogS on the surface date axis."""
    s = history.spots.reindex(history.dates).astype(float)
    return pd.DataFrame({"spot": s, "dlogS": np.log(s).diff()})

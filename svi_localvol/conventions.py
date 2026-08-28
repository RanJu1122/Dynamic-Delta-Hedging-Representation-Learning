"""Date and day-count conventions.

The whole project runs on TWO independent clocks.  They must never be mixed:

    dt_vol  (Business/260)  -- scales total implied variance.  Used for
                               omg = atmvar * tau, for the interpolation axis,
                               and for sigma = sqrt(w / tau).
    dt_r    (Act/365)       -- scales discounting and the forward.  Used for
                               F = refSpot * exp((r - q - repo) * dt_r) and for
                               the discount factor exp(-r * dt_r).

Rationale: variance only accrues on trading days (the market is shut at the
weekend) whereas interest accrues on calendar days.
"""

from __future__ import annotations

import datetime as _dt
import calendar as _calendar
from typing import Iterable, Sequence

import numpy as np

BIZ_DAYS_PER_YEAR = 260.0
CALENDAR_DAYS_PER_YEAR = 365.0

DEFAULT_WEEKMASK = "1111100"  # Mon-Fri


def _observed(d: _dt.date) -> _dt.date:
    """US-market weekend observation rule for a fixed-date holiday."""
    if d.weekday() == 5:
        return d - _dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + _dt.timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _dt.date:
    first = _dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + _dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> _dt.date:
    last = _dt.date(year, month, _calendar.monthrange(year, month)[1])
    return last - _dt.timedelta(days=(last.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> _dt.date:
    """Gregorian Easter (Meeus/Jones/Butcher), used for Good Friday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return _dt.date(year, month, day)


def us_equity_scheduled_holidays(start_year: int, end_year: int) -> tuple[_dt.date, ...]:
    """Scheduled full-day US equity-market holidays, inclusive by year.

    This deliberately excludes extraordinary closures.  A production study
    should append any exchange-announced one-offs supplied by the desk.  The
    helper exists so Business/260 never silently treats July 4 or Christmas as
    trading days merely because no calendar package was installed.
    """
    if end_year < start_year:
        raise ValueError("end_year must not precede start_year")
    holidays: set[_dt.date] = set()
    for year in range(start_year, end_year + 1):
        holidays.update({
            _observed(_dt.date(year, 1, 1)),
            _nth_weekday(year, 1, 0, 3),       # Martin Luther King Jr. Day
            _nth_weekday(year, 2, 0, 3),       # Presidents Day
            _easter_sunday(year) - _dt.timedelta(days=2),
            _last_weekday(year, 5, 0),         # Memorial Day
            _observed(_dt.date(year, 7, 4)),
            _nth_weekday(year, 9, 0, 1),       # Labor Day
            _nth_weekday(year, 11, 3, 4),      # Thanksgiving
            _observed(_dt.date(year, 12, 25)),
        })
        if year >= 2022:
            holidays.add(_observed(_dt.date(year, 6, 19)))
    return tuple(sorted(holidays))


# --------------------------------------------------------------------------- #
# low level helpers
# --------------------------------------------------------------------------- #
def to_date(d) -> _dt.date:
    """Coerce datetime/date/np.datetime64/str into datetime.date."""
    if isinstance(d, _dt.datetime):
        return d.date()
    if isinstance(d, _dt.date):
        return d
    if isinstance(d, np.datetime64):
        return d.astype("datetime64[D]").astype(_dt.date)
    if isinstance(d, str):
        return _dt.date.fromisoformat(d)
    raise TypeError(f"cannot interpret {d!r} as a date")


def _holiday_array(hol: Iterable | None) -> np.ndarray:
    if hol is None:
        return np.array([], dtype="datetime64[D]")
    return np.array([to_date(h) for h in hol], dtype="datetime64[D]")


def is_biz_day(d, hol: Iterable | None = None) -> bool:
    return bool(
        np.is_busday(
            np.datetime64(to_date(d), "D"),
            weekmask=DEFAULT_WEEKMASK,
            holidays=_holiday_array(hol),
        )
    )


def nb_biz_days(start, end, hol: Iterable | None = None) -> int:
    """Number of business days in [start, end).

    Mirrors np.busday_count: the start date is included, the end date is not.
    A negative result is returned when end < start.
    """
    return int(
        np.busday_count(
            np.datetime64(to_date(start), "D"),
            np.datetime64(to_date(end), "D"),
            weekmask=DEFAULT_WEEKMASK,
            holidays=_holiday_array(hol),
        )
    )


def roll(d, bizconv: str = "Following", hol: Iterable | None = None) -> _dt.date:
    """Adjust a date onto a business day."""
    d = to_date(d)
    if bizconv.lower() in ("none", "unadjusted"):
        return d
    if is_biz_day(d, hol):
        return d

    step = 1 if bizconv.lower() in ("following", "modifiedfollowing") else -1
    out = d
    for _ in range(15):
        out = out + _dt.timedelta(days=step)
        if is_biz_day(out, hol):
            break
    if bizconv.lower() == "modifiedfollowing" and out.month != d.month:
        out = d
        while not is_biz_day(out, hol):
            out = out - _dt.timedelta(days=1)
    return out


# --------------------------------------------------------------------------- #
# the two time axes
# --------------------------------------------------------------------------- #
def dt_vol(pricing_date, T, hol: Iterable | None = None) -> float:
    """Volatility time tau, Business/260.  Never used for discounting."""
    return nb_biz_days(pricing_date, T, hol) / BIZ_DAYS_PER_YEAR


def dt_r(pricing_date, T) -> float:
    """Discount / forward time, Act/365.  Never used to scale variance."""
    return (to_date(T) - to_date(pricing_date)).days / CALENDAR_DAYS_PER_YEAR


def timespan(start, end, basis: str = "Act/365", hol: Iterable | None = None) -> float:
    basis = basis.replace(" ", "").lower()
    if basis in ("act/365", "act365"):
        return dt_r(start, end)
    if basis in ("business/260", "bus/260", "biz/260"):
        return dt_vol(start, end, hol)
    if basis in ("act/360", "act360"):
        return (to_date(end) - to_date(start)).days / 360.0
    raise ValueError(f"unknown basis {basis!r}")


# --------------------------------------------------------------------------- #
# schedule generation
# --------------------------------------------------------------------------- #
def gen_schedule(
    start,
    end,
    period: str = "1D",
    bizconv: str = "Following",
    hol: Iterable | None = None,
) -> list[_dt.date]:
    """Generate a date schedule between start and end (both inclusive).

    period : '1D', '1W', '1M', '3M', '1Y' ...
    bizconv: business day convention applied to every generated date.

    With period='1D' the result is every business day in [start, end],
    weekends and `hol` excluded -- this is the date axis used for the implied
    and local volatility matrices.
    """
    start, end = to_date(start), to_date(end)
    if end < start:
        raise ValueError("end must not precede start")

    n, unit = int(period[:-1]), period[-1].upper()

    raw: list[_dt.date] = []
    if unit == "D":
        cur = start
        while cur <= end:
            raw.append(cur)
            cur = cur + _dt.timedelta(days=n)
    elif unit == "W":
        cur = start
        while cur <= end:
            raw.append(cur)
            cur = cur + _dt.timedelta(weeks=n)
    elif unit in ("M", "Y"):
        months = n * (12 if unit == "Y" else 1)
        i = 0
        while True:
            cur = _add_months(start, months * i)
            if cur > end:
                break
            raw.append(cur)
            i += 1
    else:
        raise ValueError(f"unsupported period {period!r}")

    if raw and raw[-1] != end:
        raw.append(end)

    out: list[_dt.date] = []
    for d in raw:
        adj = roll(d, bizconv, hol)
        if adj <= end and (not out or adj > out[-1]):
            out.append(adj)
    # the terminal date is business-adjusted backwards so it never overshoots
    if out and out[-1] != roll(end, "Preceding", hol):
        tail = roll(end, "Preceding", hol)
        if tail > out[-1]:
            out.append(tail)
    return out


def _add_months(d: _dt.date, months: int) -> _dt.date:
    y, m = divmod(d.month - 1 + months, 12)
    y, m = d.year + y, m + 1
    day = min(d.day, [31, 29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return _dt.date(y, m, day)


def year_fractions(dates: Sequence[_dt.date], basis: str = "Act/365") -> np.ndarray:
    """Step sizes between consecutive dates on the given basis (length n-1)."""
    if basis.replace(" ", "").lower() not in ("act/365", "act365"):
        raise ValueError("year_fractions currently only supports Act/365")
    d = [to_date(x) for x in dates]
    return np.array([(d[i + 1] - d[i]).days / CALENDAR_DAYS_PER_YEAR
                     for i in range(len(d) - 1)])

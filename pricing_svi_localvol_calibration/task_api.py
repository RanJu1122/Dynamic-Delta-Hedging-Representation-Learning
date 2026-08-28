"""Compatibility adapter for the four fixed signatures in the pricing task."""

from __future__ import annotations

from contextlib import contextmanager

from svi_localvol.conventions import gen_schedule as _gen_schedule
from svi_localvol.params import MarketData, VolQuoteSet
from svi_localvol.surface import VolSurface
from svi_localvol.svi import jw_to_raw

from .config import TEST_MARKET, TEST_VOL_PARAMS

_SURFACE: VolSurface | None = None


def build_surface(market: MarketData = TEST_MARKET,
                  vol_params: dict = TEST_VOL_PARAMS) -> VolSurface:
    """Build and register the surface used by the fixed-signature API."""
    global _SURFACE
    _SURFACE = VolSurface(market, VolQuoteSet.from_dict(vol_params))
    return _SURFACE


def _surface() -> VolSurface:
    return _SURFACE if _SURFACE is not None else build_surface()


def set_surface(surface: VolSurface) -> VolSurface:
    global _SURFACE
    _SURFACE = surface
    return surface


@contextmanager
def using_surface(surface: VolSurface):
    global _SURFACE
    previous = _SURFACE
    _SURFACE = surface
    try:
        yield surface
    finally:
        _SURFACE = previous


def param_convert(atm_vol, skew, putwing, callwing, kurt, tau):
    """SVI-JW to SVI-raw; returns ``(a, b, rho, m, sigma)``."""
    return jw_to_raw(atm_var=atm_vol ** 2, skew=skew, putwing=putwing,
                     callwing=callwing, min_imp_var=kurt ** 2,
                     tau=tau).as_tuple()


def gen_schedule(start, end, period="1D", bizconv="Following", hol=None):
    return _gen_schedule(start, end, period=period, bizconv=bizconv, hol=hol)


def ImpliedVol(T, K):  # noqa: N802
    return _surface().implied_vol(T, K)


def localvol(T, K, spot_adj: float = 0.0, alpha: float | None = None):
    return _surface().local_vol(T, K, spot_adj=spot_adj, alpha=alpha)

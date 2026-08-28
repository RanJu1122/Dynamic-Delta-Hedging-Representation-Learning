"""Delta hedging backtest framework.

The point of the exercise is NOT to find "the" delta.  It is to find, for each
market regime, the stickiness ratio R whose delta minimises the variance of the
daily hedging P&L.

Daily P&L of a delta-hedged long option position decomposes roughly as

    dPnL ~  1/2 * Gamma * S^2 * (realised_var - implied_var) * dt      (gamma/theta)
          + Vega * ( d sigma_imp - E_R[d sigma_imp] )                  (delta error)
          + vanna / volga cross terms

The first line does not depend on R.  Choosing R only shrinks the second line,
which is exactly what the objective function below targets.

Plug real data in via `MarketState`; `simulate_states_from_local_vol` produces
a self-consistent synthetic history so the framework can be exercised before
any historical file exists.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .conventions import to_date
from .deltas import smile_greeks
from .params import MarketData
from .surface import VolSurface


# --------------------------------------------------------------------------- #
# instrument and state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OptionSpec:
    strike: float
    maturity: dt.date
    is_call: bool = True
    quantity: float = 1.0


@dataclass
class MarketState:
    """One observation date: the spot and the surface prevailing that day."""

    date: dt.date
    spot: float
    surface: VolSurface
    tag: str = ""              # optional regime label ('high_vol', 'trend', ...)


# --------------------------------------------------------------------------- #
# the backtester
# --------------------------------------------------------------------------- #
@dataclass
class HedgeResult:
    stickiness: float
    frame: pd.DataFrame
    total_pnl: float
    pnl_std: float
    pnl_mad: float

    def summary(self) -> dict:
        return {"stickiness": self.stickiness, "total_pnl": self.total_pnl,
                "pnl_std": self.pnl_std, "pnl_mad": self.pnl_mad,
                "n_days": len(self.frame)}


class HedgeBacktester:
    """Daily rebalanced delta hedge of a single vanilla."""

    def __init__(self, states: Sequence[MarketState], option: OptionSpec,
                 financing: bool = True):
        self.states = sorted(states, key=lambda s: s.date)
        self.option = option
        self.financing = financing

    # ------------------------------------------------------------------ #
    def run(self, stickiness: float) -> HedgeResult:
        opt = self.option
        rows = []
        prev = None

        for st in self.states:
            if st.surface.tau_vol(opt.maturity) <= 0:
                break
            # the state's surface is already anchored on today's spot, so the
            # evaluation point needs no further shift; `stickiness` only enters
            # the delta sensitivity
            g = smile_greeks(st.surface, opt.maturity, opt.strike,
                             spot=st.spot, stickiness=stickiness,
                             is_call=opt.is_call, surface_shift=0.0)
            v = float(np.atleast_1d(g["price"])[0]) * opt.quantity
            d = float(np.atleast_1d(g["delta"])[0]) * opt.quantity

            row = {
                "date": st.date, "spot": st.spot, "tag": st.tag,
                "option_value": v, "delta": d,
                "sigma_imp": float(np.atleast_1d(g["sigma_imp"])[0]),
                "gamma": float(np.atleast_1d(g["gamma_bs"])[0]) * opt.quantity,
                "vega": float(np.atleast_1d(g["vega_sigma"])[0]) * opt.quantity,
            }

            if prev is not None:
                r = st.surface.market.rate
                days = (to_date(st.date) - to_date(prev["date"])).days
                accr = days / 365.0
                d_spot = st.spot - prev["spot"]

                pnl_option = v - prev["option_value"]
                pnl_hedge = -prev["delta"] * d_spot
                # cash leg: borrow to buy the option, lend the short stock proceeds
                cash_prev = prev["delta"] * prev["spot"] - prev["option_value"]
                pnl_carry = (cash_prev * (np.exp(r * accr) - 1.0)
                             if self.financing else 0.0)

                row["d_spot"] = d_spot
                row["d_log_spot"] = float(np.log(st.spot / prev["spot"]))
                row["d_sigma_imp"] = row["sigma_imp"] - prev["sigma_imp"]
                row["pnl_option"] = pnl_option
                row["pnl_hedge"] = pnl_hedge
                row["pnl_carry"] = pnl_carry
                row["pnl"] = pnl_option + pnl_hedge + pnl_carry
                row["pnl_gamma_approx"] = 0.5 * prev["gamma"] * d_spot ** 2
                row["pnl_vega_approx"] = prev["vega"] * row["d_sigma_imp"]

            rows.append(row)
            prev = row

        frame = pd.DataFrame(rows).set_index("date")
        pnl = frame["pnl"].dropna()
        return HedgeResult(
            stickiness=stickiness, frame=frame,
            total_pnl=float(pnl.sum()),
            pnl_std=float(pnl.std(ddof=1)) if len(pnl) > 1 else np.nan,
            pnl_mad=float(pnl.abs().mean()) if len(pnl) else np.nan,
        )

    # ------------------------------------------------------------------ #
    def sweep(self, stickiness_grid: Sequence[float]) -> pd.DataFrame:
        """Run the hedge across a grid of R and report the objective."""
        return pd.DataFrame([self.run(R).summary() for R in stickiness_grid])

    def optimal_stickiness(self, stickiness_grid: Sequence[float],
                           objective: str = "pnl_std") -> tuple[float, pd.DataFrame]:
        tbl = self.sweep(stickiness_grid)
        best = float(tbl.loc[tbl[objective].idxmin(), "stickiness"])
        return best, tbl

    def sweep_by_regime(self, stickiness_grid: Sequence[float],
                        objective: str = "pnl_std") -> pd.DataFrame:
        """Optimal R per regime tag -- the R*(regime) table the desk wants."""
        runs = {R: self.run(R).frame for R in stickiness_grid}
        tags = sorted({s.tag for s in self.states if s.tag})
        out = []
        for tag in tags or [""]:
            scores = []
            for R, fr in runs.items():
                sub = fr[fr["tag"] == tag]["pnl"].dropna() if tag \
                    else fr["pnl"].dropna()
                if len(sub) < 2:
                    continue
                scores.append({"stickiness": R,
                               "pnl_std": float(sub.std(ddof=1)),
                               "pnl_mad": float(sub.abs().mean()),
                               "n_days": len(sub)})
            if not scores:
                continue
            sc = pd.DataFrame(scores)
            out.append({"regime": tag or "all",
                        "best_stickiness": float(sc.loc[sc[objective].idxmin(),
                                                        "stickiness"]),
                        "best_" + objective: float(sc[objective].min()),
                        "n_days": int(sc["n_days"].iloc[0])})
        return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# synthetic history so the framework is runnable today
# --------------------------------------------------------------------------- #
def roll_surface(base: VolSurface, date: dt.date, spot: float,
                 stickiness_truth: float) -> VolSurface | None:
    """The surface an observer would see on `date` with the spot at `spot`.

    Two assumptions define the synthetic world:

    * the SVI-JW quotes themselves are unchanged (the desk re-quotes the same
      five numbers per expiry, so the surface simply rolls down as tau shrinks);
    * the smile sits at   y - R_true * ln(spot / S0)   in log-moneyness space.

    The second is implemented by setting the reference spot of the quote set to
    S0 * (spot / S0) ** R_true, which reproduces exactly that shift.  R_true = 0
    is sticky strike, R_true = 1 is sticky moneyness.

    Returns None once every quoted expiry has rolled off.
    """
    live = [q for q in base.quotes.quotes if q.vol_date > date]
    if not live:
        return None

    s0 = base.market.spot
    ref = s0 * (spot / s0) ** stickiness_truth
    mkt = MarketData(pricing_date=date, spot=spot, rate=base.market.rate,
                     dividend=base.market.dividend, repo=base.market.repo,
                     holidays=base.market.holidays)
    from .params import VolQuoteSet
    return VolSurface(mkt, VolQuoteSet(ref_spot=ref, quotes=tuple(live)),
                      calendar_repair=base.calendar_repair)


def simulate_states_from_local_vol(surface: VolSurface, maturity,
                                   n_days: int | None = None,
                                   seed: int = 7,
                                   stickiness_truth: float = 0.0
                                   ) -> list[MarketState]:
    """One spot path from the calibrated local vol model, wrapped as states.

    Each state carries a surface rolled to that date under `stickiness_truth`,
    so `HedgeBacktester.sweep` should recover R* ~= stickiness_truth.  That is
    the self-validation of the framework; replace this function with real data
    to get an answer that means something about the market.

    Regime tags come from realised volatility terciles so `sweep_by_regime`
    has something to bucket on.
    """
    from .montecarlo import LocalVolGrid, LocalVolMC

    grid = LocalVolGrid.build(surface, maturity)
    mc = LocalVolMC(surface, grid, n_paths=2, seed=seed, antithetic=False)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((len(mc.dt_r), 1))
    _, paths = mc.terminal_spots(surface.market.spot, z, store_paths=True)
    spots = paths[:, 0]

    dates = grid.dates
    if n_days is not None:
        dates, spots = dates[:n_days], spots[:n_days]

    logret = np.diff(np.log(spots), prepend=np.log(spots[0]))
    rolling = pd.Series(logret).rolling(10, min_periods=3).std().to_numpy()
    q1, q2 = np.nanquantile(rolling, [1 / 3, 2 / 3])
    tags = np.where(np.isnan(rolling) | (rolling < q1), "calm",
                    np.where(rolling < q2, "normal", "stressed"))

    states = []
    for d, s, t in zip(dates, spots, tags):
        surf = roll_surface(surface, d, float(s), stickiness_truth)
        if surf is None:
            break
        states.append(MarketState(date=d, spot=float(s), surface=surf,
                                  tag=str(t)))
    return states

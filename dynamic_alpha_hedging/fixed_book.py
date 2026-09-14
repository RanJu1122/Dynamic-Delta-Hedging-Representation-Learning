"""Contracts held to expiry, optional renewal, and a self-financing cash/stock ledger."""

import numpy as np
import pandas as pd

from svi_localvol.blackscholes import bs_price_w
from svi_localvol.conventions import nb_biz_days
from .data_loader import date_at_tau
from .hedging import vanilla_mark


def make_cohort(surface, tenors, levels, weights, cohort_id):
    """Set K, calendar expiry and quantity once, independently of the strategy."""
    rows = []
    for i, tenor in enumerate(tenors):
        expiry = date_at_tau(surface, tenor)
        for j, level in enumerate(levels):
            strike = level * surface.ref_spot
            mark = vanilla_mark(surface, expiry, strike)
            rows.append(dict(cohort_id=cohort_id, contract_id=f"{cohort_id}_{i}_{j}",
                slot_id=f"{cohort_id}_{i}_{j}", generation=0, parent_contract_id="",
                inception=surface.market.pricing_date, initial_tenor=tenor, initial_level=level,
                expiry=expiry, strike=strike, initial_spot=surface.ref_spot,
                initial_pv=mark["pv"], initial_vega=mark["vega"]))
    frame = pd.DataFrame(rows)
    if weights == "equal_vega":
        if (frame.initial_vega <= 1e-8).any():
            raise ValueError("initial equal-vega quantity is undefined; use contracts weights")
        frame["quantity"] = 100. / (len(frame) * frame.initial_vega)
    elif weights == "contracts":
        frame["quantity"] = 1.
    else:
        raise ValueError("unknown fixed book weights")
    return frame


def renew_contracts(surface, expired, strike_rule="level"):
    """Replace each expired slot at its original tenor and quantity, using close-t marks."""
    if strike_rule not in ("level", "fixed"):
        raise ValueError("renewal strike must be level or fixed")
    rows = []
    for old in expired.to_dict("records"):
        if old["expiry"] != surface.market.pricing_date:
            raise ValueError("renewals must be opened on the observed settlement date")
        expiry = date_at_tau(surface, old["initial_tenor"])
        strike = old["initial_level"]*surface.ref_spot if strike_rule == "level" else old["strike"]
        mark = vanilla_mark(surface, expiry, strike)
        generation = old["generation"]+1
        rows.append({**old, "contract_id": f"{old['slot_id']}_g{generation}",
            "generation": generation, "parent_contract_id": old["contract_id"],
            "inception": surface.market.pricing_date, "expiry": expiry, "strike": strike,
            "initial_spot": surface.ref_spot, "initial_pv": mark["pv"], "initial_vega": mark["vega"]})
    return pd.DataFrame(rows)


def scheduled_expiries(history, dates, tenors, renew):
    """Calendar coverage of every generation, before creating marks or running MC."""
    expiries = []
    available = set(dates)
    for tenor in tenors:
        expiry = date_at_tau(history[dates[0]], tenor)
        while True:
            expiries.append(expiry)
            if not renew or expiry >= dates[-1] or expiry not in available:
                break
            expiry = date_at_tau(history[expiry], tenor)
    return pd.DataFrame({"expiry": expiries})


def contract_schedule(history, dates, tenors, levels, weights, cohort_id, renew=False, strike_rule="level"):
    """Record generations; future inceptions must never enter earlier pricing frames.

    The final date only marks the last holding interval: no new horizon is opened.
    """
    missing = missing_settlements(scheduled_expiries(history, dates, tenors, renew), dates)
    if missing:
        raise ValueError(f"{cohort_id}: missing settlement spot on {missing}; supply settlement data")
    contracts = make_cohort(history[dates[0]], tenors, levels, weights, cohort_id)
    if renew:
        for date in dates[1:-1]:
            expired = contracts[contracts.expiry == date]
            if not expired.empty:
                contracts = pd.concat([contracts, renew_contracts(history[date], expired, strike_rule)],
                                      ignore_index=True)
    return contracts


def missing_settlements(cohort, dates):
    """A later observation is never a substitute for an absent expiry spot."""
    available = set(dates)
    return sorted({d for d in cohort.expiry if dates[0] < d <= dates[-1] and d not in available})


def fixed_marks(previous, current, cohort, *, extrapolation="flat_iv"):
    """Mark only surviving original contracts; expiry payoff is a separate cash flow."""
    date, end = previous.market.pricing_date, current.market.pricing_date
    business_days = nb_biz_days(date, end, previous.market.holidays)
    if business_days < 1:
        raise ValueError("holding interval must advance at least one business day")
    rows = []
    for contract in cohort[(cohort.inception <= date) & (cohort.expiry > date)].to_dict("records"):
        expiry, strike = contract["expiry"], contract["strike"]
        if expiry < end:
            raise ValueError(f"missing settlement spot on {expiry}; cannot settle at {end}")
        expired = expiry == end
        tau0, tau1 = previous.tau_vol(expiry), current.tau_vol(expiry)
        outside0 = not previous.taus[0] <= tau0 <= previous.taus[-1]
        outside1 = not expired and not current.taus[0] <= tau1 <= current.taus[-1]
        if extrapolation not in ("flat_iv", "reject"):
            raise ValueError("unknown mark extrapolation policy")
        if extrapolation == "reject" and (outside0 or outside1):
            raise ValueError(f"{date}: fixed contract outside quoted tenor range")
        start = vanilla_mark(previous, expiry, strike)
        payoff = max(current.ref_spot-strike, 0.) if expired else 0.
        finish = None if expired else vanilla_mark(current, expiry, strike)
        next_pv = 0. if expired else finish["pv"]
        frozen_pv = (max(previous.ref_spot-strike, 0.) if expired else float(bs_price_w(
            previous.ref_spot*np.exp(previous.market.cost_of_carry*current.tau_r(expiry)),
            strike, start["iv"]**2*tau1, np.exp(-previous.market.rate*current.tau_r(expiry)))))
        # The smooth Vega/term-roll decomposition has no terminal IV at expiry.
        roll_iv = (np.nan if expired else float(previous.implied_vol(
            date_at_tau(previous, tau1), strike)-start["iv"]))
        rows.append({**contract, "feature_date": date, "label_date": end,
            "tenor": tau0, "level": strike/previous.ref_spot, "remaining_tau_end": tau1,
            "spot": previous.ref_spot, "next_spot": current.ref_spot,
            "dS": current.ref_spot-previous.ref_spot,
            "opened_today": contract["inception"] == date,
            "renewed_today": contract["inception"] == date and contract["generation"] > 0,
            "dlogS": float(np.log(current.ref_spot/previous.ref_spot)),
            "dt_r": (end-date).days/365., "rate": previous.market.rate,
            "income_yield": previous.market.rate-previous.market.cost_of_carry,
            "business_days": business_days, "is_gap": business_days != 1,
            **start, "next_pv": next_pv, "expiry_cashflow": payoff,
            "next_iv": np.nan if expired else finish["iv"],
            "dV": next_pv+payoff-start["pv"], "expired": expired,
            "time_pnl": frozen_pv-start["pv"], "term_roll_iv": roll_iv,
            "attribution_valid": not expired, "mark_extrapolated": outside0 or outside1})
    return pd.DataFrame(rows)


def interpolate_beta(surface, tenors, levels, beta, marks):
    """Interpolate a signed Beta field at today's actual tau and log(K/F), no clipping."""
    expiries = [date_at_tau(surface, t) for t in tenors]
    taus = np.array([surface.tau_vol(e) for e in expiries])
    y_nodes = [np.log(np.asarray(levels)*surface.ref_spot/surface.forward(e)) for e in expiries]
    values = beta.reindex(pd.MultiIndex.from_product([tenors, levels])).to_numpy(float)
    values = values.reshape(len(tenors), len(levels))
    if not np.isfinite(values).all():
        raise ValueError("nonfinite prediction nodes in fixed-contract attribution")
    result, outside = [], []
    for r in marks.itertuples():
        tau, y = surface.tau_vol(r.expiry), np.log(r.strike/surface.forward(r.expiry))
        result.append(float(np.interp(tau, taus,
            [np.interp(y, nodes, row) for nodes, row in zip(y_nodes, values)])))
        hi = min(int(np.searchsorted(taus, tau)), len(taus)-1)
        lo = max(hi-1, 0)
        used = {hi} if tau >= taus[hi] else ({lo} if tau <= taus[lo] else {lo, hi})
        outside.append(tau < taus[0]-1e-12 or tau > taus[-1]+1e-12 or
                       any(y < y_nodes[i][0]-1e-12 or y > y_nodes[i][-1]+1e-12 for i in used))
    return np.asarray(result), np.asarray(outside)


def advance_account(previous, *, pv, next_pv, settlement, delta, spot, next_spot,
                    dt, rate, income_yield, cost_bps, all_expired=False, renewal_pv=0.):
    """Finance initial/renewal premiums, rebalance net stock and receive payoff once.

    H=-Delta. Interest/dividends follow the existing Step 7 discrete carry convention.
    Only natural exhaustion closes the hedge; a data-end mark never does.
    """
    opening = previous is None
    prior = previous or {"wealth": 0., "cash_close": 0., "stock_position": 0., "next_option_pv": 0.}
    wealth0, cash0, stock0 = prior["wealth"], prior["cash_close"], prior["stock_position"]
    if not np.isfinite(renewal_pv) or renewal_pv < 0 or (opening and renewal_pv != 0):
        raise ValueError("renewal premium must be finite, nonnegative and absent at initial inception")
    if not opening and not np.isclose(pv, prior["next_option_pv"]+renewal_pv, rtol=1e-10, atol=1e-8):
        raise ValueError("opening marks do not equal prior surviving marks plus renewal premiums")
    stock = -delta
    trade = stock-stock0
    option_trade = -pv if opening else -renewal_pv
    stock_trade = -trade*spot
    open_cost = abs(trade)*spot*cost_bps/10000.
    cash_open = cash0+option_trade+stock_trade-open_cost
    interest = cash_open*np.expm1(rate*dt)
    income = stock*spot*np.expm1(income_yield*dt)
    close_units = -stock if all_expired else 0.
    close_cashflow = -close_units*next_spot
    close_cost = abs(close_units)*next_spot*cost_bps/10000.
    cash_close = cash_open+interest+income+settlement+close_cashflow-close_cost
    stock_end = stock+close_units
    wealth = next_pv+stock_end*next_spot+cash_close
    dv = next_pv+settlement-pv
    raw = dv+stock*(next_spot-spot)
    cost = open_cost+close_cost
    wealth_change = wealth-wealth0
    if not np.isclose(wealth_change, raw+interest+income-cost, rtol=1e-9, atol=1e-8):
        raise ArithmeticError("fixed book cash/stock/P&L reconciliation failed")
    return dict(wealth_open=wealth0, cash_before=cash0, stock_before=stock0,
        cash_open=cash_open, cash_interest=interest, stock_income=income,
        option_trade_cashflow=option_trade, renewal_premium=renewal_pv, stock_trade_units=trade,
        stock_trade_cashflow=stock_trade, stock_position_held=stock,
        natural_close_units=close_units, natural_close_cashflow=close_cashflow,
        stock_position=stock_end, cash_close=cash_close, next_option_pv=next_pv,
        expiry_cashflow=settlement, wealth=wealth, wealth_change=wealth_change,
        net_error=wealth-wealth0*np.exp(rate*dt), cost=cost,
        cost_open=open_cost, cost_natural_close=close_cost,
        hedge_turnover=abs(trade)+abs(close_units),
        hedge_notional_turnover=abs(trade)*spot+abs(close_units)*next_spot)

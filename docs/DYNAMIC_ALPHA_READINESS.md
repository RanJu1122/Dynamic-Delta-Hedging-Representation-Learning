# Dynamic-alpha hedging readiness

This is the boundary between the completed pricing calibration and
`docs/source/动态Alpha对冲研究.docx`.  Preflight is a gate, not Dynamic Step 0.

## Frozen notation

| Symbol | Meaning |
|---|---|
| `spot_adj` | `log(S / refSpot)` |
| `alpha` | denominator shift in `y_adj = y - alpha * spot_adj` |
| `alpha=0` | sticky/frozen local vol |
| `alpha=1` | sticky strike; BS-delta anchor |
| `alpha=2` | sticky moneyness endpoint |
| raw grid `beta` | total same-`(tau, level)` IV response; diagnostic only |
| empirical `beta` | skew-adjusted surface response; Step 3-7 primary label |

The runtime contains no analytic `R` convention.  SVI-JW intermediate variables
are named `jw_beta`/`jw_shape` so they cannot be confused with empirical beta or
stickiness alpha.

## Current input contract

- Source: `data/svi_param.pkl`; preflight prints its SHA-256.
- Rates are assumptions because the pickle contains no rate/dividend/repo:
  `r=3.6%`, `q=3%`, `repo=0`.
- Volatility time is Business/260; forward and discount time are Act/365.
- Descriptive panel tenors are `1M, 2M, 3M, 6M, 9M, 1Y, 1.5Y, 2Y` on
  Business/260.  The added 2M node has near-complete historical coverage and
  resolves the fast short-end beta decay without duplicating the long end.
- Baseline strike levels follow the document literally: `0.4, 0.5, ..., 1.2`.
- Nodes outside a day's quoted maturity range are marked `NaN`, not flat-filled.
- Expiries and all associated quote fields are sorted together.

Step 1 compares the same `(tau, level)` matrix cell across dates.  Its actual
strike and expiry roll.  The raw change is decomposed exactly as
`dIV_grid = smile_crossing_iv + dIV_surface`, where the crossing counterfactual
uses only the previous surface.  `dIV_surface` is the Step 2 primary label and
has the required sticky-strike target `alpha=1 -> beta_surface ~= 0`.

## Implemented boundary

`step01.py` is the only Step 1 implementation and writes rolling-grid changes.
It also exports the otherwise opaque pickle to `raw_svi_quotes.csv`, preserving
the original key text and flagging whether a source timestamp was available.
`step02.py` is the only Step 2 implementation and writes daily-ratio and trailing
OLS raw-grid and surface beta, including regression diagnostics.  Step 2 reads
the saved Step 1 artefact instead of silently rerunning it.  `step03.py` fixes a
representative SVI surface, reprices fixed strikes under bumped spot for each
Alpha, inverts prices to IV and writes both `beta(alpha)` and the usable inverse
`alpha(beta)` knots.  Very remote strikes remain in diagnostics but are withheld
from the inverse when ordinary MC cannot identify IV reliably.

The input contract is one US market-date observation on Monday through Friday.
Step 1 never shifts or rolls those dates and refuses weekend keys.  The current
file satisfies that date contract.  Its explicit tolerance policy keeps the first
row of a repeated VolDate and records every dropped duplicate in the raw export;
the small number of still-unbuildable observations are listed and excluded.
Multi-business-day gaps remain visible in Step 1 but are excluded from the
default daily/rolling beta regressions.

Run:

```bash
python3 -m dynamic_alpha_hedging preflight
python3 -m dynamic_alpha_hedging step1
python3 -m dynamic_alpha_hedging step2
python3 -m dynamic_alpha_hedging step3 --fast
```

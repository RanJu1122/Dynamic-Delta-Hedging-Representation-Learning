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
| empirical `beta` | `-dIV/dlogS`; not the SVI-JW algebra variable |

The runtime contains no analytic `R` convention.  SVI-JW intermediate variables
are named `jw_beta`/`jw_shape` so they cannot be confused with empirical beta or
stickiness alpha.

## Current input contract

- Source: `data/svi_data.pkl`; preflight prints its SHA-256.
- Rates are assumptions because the pickle contains no rate/dividend/repo:
  `r=3.6%`, `q=3%`, `repo=0`.
- Volatility time is Business/260; forward and discount time are Act/365.
- Descriptive panel tenors are `0.25, 0.50, 0.75, 1.00, 1.50, 2.00`.
- Baseline strike levels follow the document literally: `0.4, 0.5, ..., 1.2`.
- Nodes outside a day's quoted maturity range are marked `NaN`, not flat-filled.
- Expiries and all associated quote fields are sorted together.

The descriptive IV panel is not yet the empirical beta label.  Dynamic Step 1
must evaluate adjacent daily surfaces at the same actual strike and expiry;
subtracting two panels re-anchored to different daily spots would compare
different options.

## Current blocker

The file contains 406 observations and calibrates all of them under the declared
scheduled US-equity calendar.  However, 85 observation keys are not business
days.  Their source timestamp semantics must be confirmed before computing
daily returns or IV changes.  Proposed date transforms are collision-checked
and never overwrite records silently.

Run:

```bash
python3 -m dynamic_alpha_hedging preflight
```

Do not use `--allow-non-business-observation-dates` for final results merely to
turn the status green.

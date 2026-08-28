# Dynamic Alpha research: pre-Step-1 readiness

This file records the boundary between the completed SVI/local-vol pricing task
and `动态Alpha对冲研究.docx`.  It is a conventions contract, not a research
result.

## Frozen notation

| Symbol | Meaning in the new study | Code |
|---|---|---|
| `spot_adj` | observed `log(S / refSpot)` | argument to `VolSurface.local_vol` |
| `alpha` | dimensionless local-vol stickiness in `y_adj = y - alpha*spot_adj` | `SVIJWQuote.alpha`, `Slice.alpha` |
| `alpha=0` | frozen/sticky local-vol surface | `ALPHA_STICKY_LOCAL_VOL` |
| `alpha=1` | sticky strike; BS-delta anchor | `ALPHA_STICKY_STRIKE` |
| `alpha=2` | sticky moneyness endpoint | `ALPHA_STICKY_MONEYNESS` |
| empirical `beta` | `-dIV/dlogS`, with volatility units | not estimated before Step 2 |

Two similarly named SVI-JW algebra variables have been renamed `jw_beta` and
`jw_shape`.  They have no relationship to empirical hedge beta or stickiness
alpha.

`deltas.py` and `backtest.py` still contain the old analytic smile-shift
parameter `R`, where `R=-1/0/1` labels local-vol-like/sticky-strike/sticky-
moneyness.  They are retained only to reproduce the original pricing-task
extension.  `alpha = R + 1` lines up names at three anchors; it does **not** make
the analytic R delta equal to the local-vol MC alpha delta.  New research code
must not import the legacy backtester.

## Step-1 input contract

- Source: `svi_data.pkl`; its SHA-256 is printed by the preflight.
- Each daily record must contain aligned `VolDate`, `ATMVol`, `Skew`, `Putwing`,
  `Callwing`, `Kurt`, and `StickinessRatio` arrays plus scalar `Spot`.
- `refSpot` is that record's `Spot`.
- `F(T) = refSpot * exp((r-q-repo)*dt_r)`; moneyness is `log(K/F)`.
- Strike axis is `0.40, 0.45, ..., 1.20`, with `K = level * refSpot`.
- Volatility/SVI interpolation uses Business/260; forward and discounting use
  Act/365.
- Cross-day changes use the constant Business/260 tenor axis
  `0.25, 0.50, 0.75, 1.00, 1.50, 2.00` years.  The quoted
  `VolDate` slots roll and their count varies, so differencing the i-th listed
  expiry would compare different contracts on roll days.
- A constant-tenor node outside that day's quoted maturity range is `NaN` and
  is recorded in `ImpliedVolPanel.extrapolated`; it is never silently flat-
  filled for research.
- SVI-JW expiries and their associated parameters are sorted together.  The
  source file contains out-of-order lists, so sorting `VolDate` alone would
  corrupt the quotes.

The existing `dataset.implied_vol_panel` is the reusable Step-1 engine, but the
preflight does not call it.  Starting Step 1 is a separate, explicit action.

## Repairs and assumptions that remain visible

1. With a Mon-Fri-only calendar, five observations appear to breach the exact
   SVI-JW convexity transform.  Once scheduled US equity holidays are included,
   all 406 observations calibrate strictly.  The research default is therefore
   `beta_clamp=0`; a nonzero clamp remains an explicit fallback, and the
   preflight always reports whether it was used.
2. The pickle contains no rate, dividend, repo or exchange calendar.  Current
   defaults (`r=3.6%`, `q=3%`, `repo=0`) come from the earlier Task 7 and are
   assumptions.  The bundled calendar covers scheduled US equity holidays but
   not extraordinary closures.
3. The generic data layer can also evaluate a 1-month tenor, but it lies before
   the first quote on 254 of 406 days.  It is excluded from the research default
   rather than being silently extrapolated.
4. The stored observation dates include many weekends.  Their timestamp/source
   convention must be confirmed before daily changes are meaningful.  The
   loader supports an explicit fixed date shift and preceding-business-day
   roll, but raises on collisions and never changes keys by default.

Because item 4 is unresolved, the current data preflight correctly reports
`BLOCKED` even though the code path itself is prepared.  Do not bypass it with
`--allow-non-business-observation-dates` for final research merely to make the
status green.

## Commands

Read-only preflight:

```bash
python3 -m svi_localvol.research
```

After the source owner confirms a date policy, encode it explicitly, rerun the
preflight, and only then begin Step 1.  For example, the CLI can test a proposed
policy without changing the pickle:

```bash
python3 -m svi_localvol.research \
  --observation-date-shift -1 \
  --roll-observation-dates
```

If that proposal maps two source observations onto one date, it fails with a
collision instead of overwriting either observation.

# Pricing SVI/local-vol calibration workflow

This workflow is the executable form of
`docs/source/【中】Pricing Practice-LocalVol Calibration.docx`.

| Document step | Module | Acceptance output |
|---|---|---|
| 1: SVI-JW to raw | `step01_svi_conversion.py` | raw parameters and round-trip checks |
| 2: implied surface | `step02_implied_surface.py` | date/strike IV matrix and ATM checks |
| 3: local-vol surface | `step03_localvol_surface.py` | local vol and arbitrage flags |
| 4: MC validation | `step04_mc_validation.py` | BS/MC prices, deltas and standard errors |

Commands:

```bash
python3 -m pricing_svi_localvol_calibration step01
python3 -m pricing_svi_localvol_calibration step02
python3 -m pricing_svi_localvol_calibration step03
python3 -m pricing_svi_localvol_calibration step04 --fast
python3 -m pricing_svi_localvol_calibration --fast
```

The MC engine carries two clocks consistently with the calibrated total
variance: drift/discounting use Act/365 and diffusion/quadratic variation use
Business/260.  This resolves the source document's Step 4 single-clock formula,
which conflicts with its own Business/260 definition of total variance.

Raw quotes remain visible in Step 3 so calendar/butterfly violations are not
hidden.  Step 4 uses a running-maximum calendar repair and compares against IV
read from the same repaired surface.  The repair policy is a model assumption,
not a modification of source data.

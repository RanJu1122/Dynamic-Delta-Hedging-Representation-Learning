# SVI local volatility and dynamic-alpha hedging

This repository has one reusable pricing engine and two clearly separated
workflows:

- `svi_localvol/`: SVI, implied/local volatility, Black-Scholes and Monte Carlo;
- `pricing_svi_localvol_calibration/`: the four validation steps from
  *Pricing Practice 3 - LocalVol Calibration*;
- `dynamic_alpha_hedging/`: preparation for the eight-step dynamic-alpha study.

The pricing workflow is complete.  The dynamic workflow is intentionally still
at preflight: the source contains observation keys whose trading-date semantics
must be confirmed before empirical `dlogS` or `dIV` is produced.

## Install and validate

Run from the repository root:

```bash
python3 -m pip install -e ".[plots,dev]"
python3 -m tests.run_all
```

The dependency-free test runner is provided because some research environments
do not have pytest installed; `pytest` remains supported through the `dev` extra.

Run the complete pricing acceptance workflow:

```bash
python3 -m pricing_svi_localvol_calibration --fast
```

Or run one calibration step independently:

```bash
python3 -m pricing_svi_localvol_calibration step01
python3 -m pricing_svi_localvol_calibration step02
python3 -m pricing_svi_localvol_calibration step03
python3 -m pricing_svi_localvol_calibration step04 --fast
```

The original delivery command remains compatible:

```bash
python3 solution.py --fast
```

Run the read-only dynamic-alpha gate:

```bash
python3 -m dynamic_alpha_hedging preflight
```

Generated files live under `output/` and are ignored by Git.  Source data lives
under `data/`; it is not generated output and is versioned deliberately.

See [architecture](docs/ARCHITECTURE.md),
[calibration workflow](docs/PRICING_CALIBRATION.md), and
[dynamic-alpha readiness](docs/DYNAMIC_ALPHA_READINESS.md).

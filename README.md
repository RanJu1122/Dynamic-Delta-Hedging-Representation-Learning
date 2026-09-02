# SVI local volatility and dynamic-alpha hedging

This repository has one reusable pricing engine and two clearly separated
workflows:

- `svi_localvol/`: SVI, implied/local volatility, Black-Scholes and Monte Carlo;
- `pricing_svi_localvol_calibration/`: the four validation steps from
  *Pricing Practice 3 - LocalVol Calibration*;
- `dynamic_alpha_hedging/`: preflight, rolling-grid Step 1, empirical beta
  Step 2 and the measured Alpha/Beta converter in Step 3.

The pricing workflow is complete.  Dynamic Step 1 compares a fixed
`(tenor, K/spot)` grid and removes mechanical smile traversal with the previous
surface; Step 2 reports both raw grid beta and the skew-adjusted primary beta.

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

Run the first three dynamic stages independently:

```bash
python3 -m dynamic_alpha_hedging step1
python3 -m dynamic_alpha_hedging step2
python3 -m dynamic_alpha_hedging step3 --fast
```

Step 3's `--fast` mode is only a 10k-path development check.  Omit it for the
formal 100k-path, two-substep, 801-node converter, and use
`--calibration-date YYYY-MM-DD` to override the robust representative-surface
selection.  Quality checks are retained as audit columns and never hide raw
curves; the inverse excludes only non-monotone cells, which have no unique
piecewise-linear inverse.  No monotonic projection is used.

Generated files live under `output/` and are ignored by Git.  Source data lives
under `data/`; it is not generated output and is versioned deliberately.

See [architecture](docs/ARCHITECTURE.md),
[calibration workflow](docs/PRICING_CALIBRATION.md), and
[dynamic-alpha readiness](docs/DYNAMIC_ALPHA_READINESS.md).  The current
Step 3 findings and their comparison with empirical Step 2 beta are documented
in [the Chinese Step 3 report](docs/STEP3_RESULTS_AND_STEP2_COMPARISON_CN.md).

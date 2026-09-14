# SVI local volatility and dynamic-alpha hedging

> **2026-09-14 current defaults:** restored March 31 / April 1 / April 2 snapshots;
> 40k MC; `step7` now aliases the fixed/renewing-book runner and defaults to only
> `constant_sr`, `term_sr`, `term_spot_sr`. The old per-option command is
> `step7-legacy`; add `--strategy-set full` for the optional controls/EMA suite.
> Short expiries (<=10 business days) use 0.5% bumps in the fixed-book runner.
> The default signal grid remains legacy56 (the held book has 63 slots). Rebuild upstream and
> the converter library in a new folder before running. See the complete
> [current migration and commands](docs/MC_REBUILD_40K_AND_THREE_STRATEGIES_CN.md).

This repository has one reusable pricing engine and two clearly separated
workflows:

- `svi_localvol/`: SVI, implied/local volatility, Black-Scholes and Monte Carlo;
- `pricing_svi_localvol_calibration/`: the four validation steps from
  *Pricing Practice 3 - LocalVol Calibration*;
- `dynamic_alpha_hedging/`: preflight, rolling-grid Step 1, empirical beta
  Step 2, the measured Alpha/Beta converter in Step 3, an ATM-anchored
  three-factor daily-Beta surface model in Step 4, the Step 5 predictability
  gate, the Step 6 next-observation factor forecast, and a reusable Step 7
  alpha-hedging/book-attribution framework.

The pricing workflow is complete.  Dynamic Step 1 compares a fixed
`(tenor, K/spot)` grid and removes mechanical smile traversal with the previous
surface; Step 2 reports daily-ratio raw-grid beta and skew-adjusted daily beta.
The current workflow is daily-only: no rolling-regression Beta calculation,
feature, input file or comparison strategy. Models are evaluated against
last-observed daily Beta/factors. See [daily-only requirements and migration](docs/DAILY_ONLY_PIPELINE_CN.md).

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

Run the first six dynamic stages independently:

```bash
python3 -m dynamic_alpha_hedging step1
python3 -m dynamic_alpha_hedging step2
python3 -m dynamic_alpha_hedging step3 --paths 40000
python3 -m dynamic_alpha_hedging step4
python3 -m dynamic_alpha_hedging step5
python3 -m dynamic_alpha_hedging step6
```

The optional legacy workflow can prepare and launch a 3M ATM backtest:

```bash
python3 -m dynamic_alpha_hedging step7-legacy --prepare
python3 -m dynamic_alpha_hedging step7-legacy
```

The same engine supports `--book near_atm` and `--book full`, daily cached
Alpha/Beta/Delta maps, frozen Step 6 forecasts, smoothing and net book costs.
Full historical MC is substantially slower than single-date Step 3. See the
[Chinese Step 7 guide](docs/STEP7_GUIDE_CN.md) for commands, assumptions,
converter-precision checks and outputs. Framework tests are not evidence of
hedging improvement; interpret each completed run through its own manifest.

The independent `precompute` command prepares a model-independent pricing library.
`precompute --plan-only` writes the plan without MC. The optional legacy Step 7 can read it with
`--mc-library PATH`, with HGB, Ridge or training-mean predictors; missing shards
fail instead of starting MC. See [预计算接口、参数与异常日期处理](docs/MC_PRECOMPUTE_CN.md).
The March 31 / April 1 / April 2 snapshots are restored in the executable loader
as of 2026-09-14. Rebuild Steps 1, 2, 4, 5 and 6 and create the new 40k converter
library before another Step 7 run; existing reports retain their original exclusions.

The independent `step7-sr --mc-library PATH` experiment reuses those Beta–Alpha
converters and runs new MC for whole-book `constant_sr`, `term_sr`, and
`term_spot_sr` policies. Maturities share paths; raw/EMA policies, fixed-alpha
controls, training-only selection, attribution and net book PnL are reported.
`--prepare` validates the new run without MC. See
[新 Step 7：统一 SR 曲面、MC 与 PnL](docs/STEP7_SHARED_SR_CN.md) and the
[full-book accounting, turnover and attribution guide](docs/STEP7_FULL_BOOK_BACKTEST_ACCOUNTING_CN.md).

`step7-fixed` runs the same three shared SR tests on 63 slots (2M–2Y, levels
0.4–1.2). Each contract is held to expiry, then renewed at its original tenor,
original level times current spot, and unchanged quantity. `--no-renew-expired`
retains the one-cohort runoff experiment. The book carries across data gaps and
open positions are marked without forced liquidation at data end.
Its default `--surface-grid legacy56` keeps the current 56 signal nodes with
boundary extension while pricing 63 fixed contracts. Explicit `full63` requires
matching Step 4–6 artifacts and converters including level 1.2.
The older per-option and shared rolling-book workflows remain available as optional commands. See
[到期续开、恢复数据与运行命令](docs/STEP7_RENEWAL_CN.md).

Step 7 also reports direct shadow-delta controls and MC consistency checks.
`--converter pooled --converter-dates 10` averages training-date forward maps;
`--mc-cache PATH` shares validated MC results across separate output folders.
See [解析 Delta 与多日期转换器对照](docs/STEP7_DELTA_COMPARISON_CN.md)
for commands, output definitions and the cached ATM comparison results.

Step 3's `--fast` mode is only a 10k-path development check.  Omit it for the
normal 40k-path, two-substep, 801-node converter, and use
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
The complete Step 6–7 code walkthrough, current ATM backtest review and ordered
improvement plan are in the
[Chinese Step 6–7 report](docs/STEP6_STEP7_CODE_WALKTHROUGH_AND_REVIEW_CN.md).
The six-contract near-ATM daily-converter experiment is documented in the
[Chinese near-ATM settings and results report](docs/STEP7_NEAR_ATM_FAST_RESULTS_CN.md).

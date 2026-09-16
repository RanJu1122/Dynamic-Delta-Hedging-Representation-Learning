# SVI local volatility and dynamic-alpha hedging

> **2026-09-14 current defaults:** restored March 31 / April 1 / April 2 snapshots;
> 40k MC; `step7` now aliases the fixed/renewing-book runner and defaults to only
> `constant_sr`, `term_sr`, `term_spot_sr`. The old per-option command is
> `step7-legacy`; add `--strategy-set full` for the optional controls/EMA suite.
> Short expiries (<=10 business days) use 0.5% bumps in the fixed-book runner.
> The default signal grid remains legacy56 (the held book has 63 slots). Rebuild upstream and
> the converter library in a new folder before running. See the complete
> [current Step7 guide](docs/STEP7_GUIDE_CN.md).

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

Step 4 supports two three-factor decompositions. `--factor-method atm_anchored`
(the backward-compatible default) uses observed reference ATM beta plus two
residual PCs. `--factor-method pca` uses ordinary PCA of the training-centered
Beta surface, without variance standardization or whitening. Both fit only the
chronological training sample and report nested 1/2/3-factor reconstructions;
these are reconstruction scores, not next-day forecast scores.

For example, from the project root:

```bash
.venv/bin/python -m dynamic_alpha_hedging step4 \
  --factor-method pca \
  --input output/rebuild_20260914_restored_40k_full63/step02/beta_daily.csv \
  --output output/step04_dual_20260915/pca/step04

.venv/bin/python -m dynamic_alpha_hedging step4 \
  --factor-method atm_anchored \
  --input output/rebuild_20260914_restored_40k_full63/step02/beta_daily.csv \
  --output output/step04_dual_20260915/atm_anchored/step04
```

PCA artifacts use `pc_score_1..3` and `pc_loading_1..3`; observed ATM beta
remains a separate diagnostic. Step 5 keeps its direct per-cell prediction
benchmark and diagnoses the selected factors. Step 6 predicts those factors;
its PCA ATM metric comes from the reconstructed ATM node, not PC1. Step 7
uses the same selected basis for model and persistence signals.

Step 5/6/7 CLI commands infer the method from the upstream manifest. Explicit
`--factor-method` must agree. New scores, loadings and panels carry
`factor_method` and `factor_basis_id`; mixed fitted bases fail before training.
Old untagged artifacts retain ATM-anchored semantics. Python API callers must
set `DynamicAlphaConfig(step4_factor_method="pca")` explicitly.

Each Step 4 run also writes `surface_examples.csv` and
`surface_cross_sections.png`, using five fixed training dates. The plot shows
raw sections and ratios to observed reference ATM beta; near-zero denominator
ratios are omitted. This is a visual stability diagnostic, not a stability test.

Start with `READ_ME_FIRST_CN.html` for a self-contained Chinese report with
embedded figures, parameter explanations and the complete metric glossary.
`step4_dashboard_cn.png` shows reconstruction, weak nodes and variance weights;
`shape_stability_cn.png` adds intercept-adjusted normalization and single-factor
residuals. `decision_summary.csv` and `variance_contribution.csv` provide the
corresponding tables. These reports do not change the fitted factors.
The full definitions are in [Step 4 指标词典](docs/STEP4_METRICS_GUIDE_CN.md).

Use separate output roots for the two methods and rebuild Step 5/6 from their
respective Step 4 outputs. Step 7's input root still requires the full
Step 1/2/4/5/6 artifact chain. Step 1/2 and the MC converter library can be
reused; changing factor method does not change the pricing-engine fingerprint.
This option does not add automatic factor-count selection or fold-wise PCA
refitting for model search; a future rolling-validation experiment must refit
the decomposition within each training fold.

The optional legacy workflow can prepare and launch a 3M ATM backtest:

```bash
python3 -m dynamic_alpha_hedging step7-legacy --prepare
python3 -m dynamic_alpha_hedging step7-legacy
```

The same engine supports `--book near_atm` and `--book full`, daily cached
Alpha/Beta/Delta maps, frozen Step 6 forecasts, smoothing and net book costs.
These ATM/near-ATM options apply to `step7-legacy`; the current default `step7` uses the fixed renewal full book. Full historical MC is substantially slower than single-date Step 3. See the
[Chinese Step 7 guide](docs/STEP7_GUIDE_CN.md) for commands, assumptions,
converter-precision checks and outputs. Framework tests are not evidence of
hedging improvement; interpret each completed run through its own manifest.

The independent `precompute` command prepares a model-independent pricing library.
The 2026-09-15 control-variate fix separates paired Delta controls from individual
price controls used for IV inversion. Clipped/non-finite Beta estimates remain
auditable but cannot enter the converter or model attribution. Rebuild the MC
library in a new directory; Step 1/2/4/5/6 artifacts remain reusable. See the
[fix details and exact rerun commands](MC_CONTROL_FIX_CN.md).
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
[当前Step7：统一SR与固定续开book](docs/STEP7_GUIDE_CN.md) and the
[full-book accounting, turnover and attribution guide](docs/STEP7_FULL_BOOK_BACKTEST_ACCOUNTING_CN.md).

`step7-fixed` runs the same three shared SR tests on 63 slots (2M–2Y, levels
0.4–1.2). Each contract is held to expiry, then renewed at its original tenor,
original level times current spot, and unchanged quantity. `--no-renew-expired`
retains the one-cohort runoff experiment. The book carries across data gaps and
open positions are marked without forced liquidation at data end.
Its default `--surface-grid full63` requires matching Step 4–6 artifacts and
converters including level 1.2. Explicit `legacy56` retains 56 signal nodes with
boundary extension while pricing 63 fixed contracts.
The older per-option and shared rolling-book workflows remain available as optional commands. See
[到期续开、恢复数据与运行命令](docs/STEP7_GUIDE_CN.md).

The legacy Step 7 also reports direct shadow-delta controls and MC consistency checks.
`--converter pooled --converter-dates 10` averages training-date forward maps;
`--mc-cache PATH` shares validated MC results across separate output folders.
See [current and legacy entrypoint distinctions](docs/STEP7_GUIDE_CN.md). These converter options apply to `step7-legacy`; the default shared-SR workflow uses a separate pricing path.

Step 3's `--fast` mode is only a 10k-path development check.  Omit it for the
normal 40k-path, two-substep, 801-node converter, and use
`--calibration-date YYYY-MM-DD` to override the robust representative-surface
selection.  Quality checks remain auditable. Clipped/non-finite Beta estimates are excluded from production inverse inputs, and invalid alpha=1 anchors invalidate centered curves. Non-monotone curves have no unique piecewise-linear inverse. No monotonic projection is used.

Generated files live under `output/` and are ignored by Git.  Source data lives
under `data/`; it is not generated output and is versioned deliberately.

Start with the [document index and cleanup list](docs/README.md).
The maintained references are [architecture](docs/ARCHITECTURE.md),
[calibration](docs/PRICING_CALIBRATION.md), [input contract](docs/DYNAMIC_ALPHA_READINESS.md),
[Step3](docs/step3_guide.md), [Step4 metric definitions](docs/STEP4_METRICS_GUIDE_CN.md),
and [current Step7](docs/STEP7_GUIDE_CN.md).
Old experiment and implementation-plan Markdown files were consolidated on
2026-09-15; their originals and the old deck are kept in the compressed revision
archive listed in the document index. Historical numerical outputs remain in `output/`.

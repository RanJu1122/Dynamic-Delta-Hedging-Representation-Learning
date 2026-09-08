# Architecture

## Dependency rule

```text
                         pricing_svi_localvol_calibration
                        /
svi_localvol (core) ----
                        \
                         dynamic_alpha_hedging
```

The two workflows may import the core.  The core imports neither workflow, and
the workflows do not import each other.

## Core library

| Module | Responsibility |
|---|---|
| `conventions.py` | Business/260, Act/365, holidays and schedules |
| `params.py` | generic market and quote containers; alpha boundaries |
| `svi.py` | SVI-JW/raw conversion, derivatives and static-arbitrage algebra |
| `surface.py` | total variance, implied vol, Dupire local vol and repair |
| `blackscholes.py` | prices, Greeks and implied-total-variance inversion |
| `montecarlo.py` | local-vol grid, paths and reusable option-pricing primitives |

Task fixtures, CLIs, plots and output paths do not belong in the core.

## Workflow boundaries

`pricing_svi_localvol_calibration` mirrors Calibration Steps 1-4.  Every step
has `run()` and `validate()` and can be invoked independently through the CLI.
`pipeline.py` only orchestrates those modules and writes artefacts.

`dynamic_alpha_hedging` contains configuration, data loading, manifests,
preflight and one canonical module for each implemented Step 1-7.  Later steps
will be added one at a time; empty placeholder modules are deliberately avoided.

The planned dynamic dependency graph is not purely linear:

```text
preflight -> Step 1 -> Step 2 -> Step 4 -> Step 5 -> Step 6 --\
                  \-> Step 3 -------------------------------> Step 7 -> Step 8
```

Step 1 decomposes rolling-grid IV changes into smile traversal and surface
motion without future data.  Step 2 estimates both diagnostic raw-grid beta and
the primary surface beta.  Step 3 builds `beta(alpha)` under that same primary
definition by fixed-strike MC repricing and implied-vol inversion.  Its raw
curve remains auditable; the inverse converter subtracts the raw alpha=1
anchor, then inverts strictly decreasing measured knots without a monotone
projection. Numerical quality is audited separately from invertibility.
Step 4 uses dates with a complete 2M-2Y, level 0.4-1.1 daily-beta surface.  The
poorly covered 1M slice and noisy level 1.2 remain in upstream diagnostics but
do not enter the factor model; no missing-value imputation is used.  Factor 1
is exactly the observed 3M ATM daily beta.  Every other cell is regressed on
that anchor, and two centered, unstandardized PCs are fitted to the residual
surface.  Thus the model has one economically identified ATM factor and two
shape factors.  Loadings are estimated on the first 75% of complete dates;
later dates are projected onto frozen loadings and reported separately to
prevent factor-loading leakage.  Steps 6-7 must compare nested one-, two- and
three-factor forecasts rather than assume that the third historical factor is
predictable.

Step 5 builds close-of-day state features and predicts the next observation's
filtered `beta_surface_daily` on a chronological split. The last known daily
beta is a feature and persistence benchmark. Ridge and histogram gradient
boosting are compared with sticky-strike, training-mean and daily persistence.  The per-cell forecasts remain the benchmark for Step
6.  The factor-state panel also exposes the three next-observation factor
labels needed to compare one-, two- and three-factor forecasting models.

Step 6 implements `g(state_t) -> z(t+1)` with features available through close
`t`.  It treats 3M ATM beta as the primary target, forecasts both residual
shape factors as predefined extensions, and reconstructs nested one-, two- and
three-factor beta surfaces.  Ridge and histogram gradient boosting are tested
against training-mean and last-observed-daily-factor
baselines on the Step 4 chronological holdout.  Future `dlogS` is used only to
score reconstructed `dIV`, never as a prediction feature.  Beta-to-alpha
conversion remains in Step 7.

Step 7 reuses the frozen Step 6 primary predictor on all holding dates, not
only dates with usable next-day beta labels. `hedging.py` supplies fixed-K/T
marks and cached, joint Beta/Delta bump measurements through the existing
core MC and Step 3 anchor/quality methods. `step07.py` handles training-only
fixed-alpha selection, inverse/smoothing strategies, self-financing net book
hedges, separate attribution and paired-block uncertainty estimates. ATM,
near-ATM and full retained-grid books use the same engine. The default
converter is rebuilt from each close-t surface; fixed training-date inverse
tables are optional sensitivity inputs, never sources of today's Delta.
See [Step 7 guide](STEP7_GUIDE_CN.md) for assumptions and run commands.

## Outputs

```text
output/
├── pricing_calibration/
└── dynamic_alpha/
    └── stepNN/
```

Dynamic stages use `manifest.json` to record data/config/upstream hashes, model
policy and validation status.

## Independent MC pricing library

`precompute.py` plans and writes date/tenor shards through `mc_library.py`. The library freezes pricing inputs, axes and engine identity while allowing model/strategy changes. `step7 --mc-library` is read-only. The CLI exposes HGB, Ridge and training-mean models with explicit parameters. Observational gaps reset market-state windows and past-daily-factor fill; Step 7 closes/reopens independent segments and reports centered/raw attribution separately. See [the complete parameter and data-policy guide](MC_PRECOMPUTE_CN.md).

Current beta workflow: `daily_only_v1`. Step 2 no longer writes regression-beta outputs. Step 5/6 no longer accept a rolling-beta input. Step 6 uses 13 features, and Step 7 adds a same-factor-count persistence hedge. See [daily-only guide](DAILY_ONLY_PIPELINE_CN.md).

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
preflight and one canonical module for each implemented Step 1-3.  Later steps
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
curve remains auditable; the inverse converter uses a monotone fit anchored at
`alpha=1 -> beta=0` and excludes statistically unidentified far-wing cells.
Step 7 uses the map with factor shapes and forecasts for hedge delta.

## Outputs

```text
output/
├── pricing_calibration/
└── dynamic_alpha/
    └── stepNN/
```

Dynamic stages use `manifest.json` to record data/config/upstream hashes, model
policy and validation status.

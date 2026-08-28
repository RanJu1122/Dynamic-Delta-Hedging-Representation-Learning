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

`dynamic_alpha_hedging` currently contains only code that genuinely exists:
configuration, data loading, manifests and preflight.  Dynamic Steps 1-8 will be
added one at a time; empty placeholder modules are deliberately avoided.

The planned dynamic dependency graph is not purely linear:

```text
preflight -> Step 1 -> Step 2 -> Step 4 -> Step 5 -> Step 6 --\
                  \-> Step 3 -------------------------------> Step 7 -> Step 8
```

Step 3 builds `beta(alpha)` by MC repricing and implied-vol inversion.  Step 7
uses that map together with factor shapes and forecasts to produce hedge deltas.

## Outputs

```text
output/
├── pricing_calibration/
└── dynamic_alpha_hedging/
    └── stepNN/
```

Future dynamic stages use `manifest.json` to record data/config/upstream hashes,
random seeds, model policy and validation status.

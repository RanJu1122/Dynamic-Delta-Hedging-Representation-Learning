#!/usr/bin/env bash
# Reuse verified Step1/2 and the repaired MC library; refresh Step4 and Step5.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
run_root="output/rebuild_20260914_restored_40k_full63"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

.venv/bin/python -m dynamic_alpha_hedging step4 \
  --input "$run_root/step02/beta_daily.csv" \
  --output "$run_root/step04" \
  --factor-method atm_anchored --surface-grid full63 --min-abs-dlogS 0.0025

.venv/bin/python -m dynamic_alpha_hedging step5 \
  --factors "$run_root/step04/factor_scores.csv" \
  --loadings "$run_root/step04/factor_loadings.csv" \
  --iv-state "$run_root/step01/iv_state.csv" \
  --changes "$run_root/step01/grid_changes.csv" \
  --daily-beta "$run_root/step02/beta_daily.csv" \
  --output "$run_root/step05" \
  --factor-method atm_anchored --surface-grid full63 --min-abs-dlogS 0.0025 \
  --mc-library "$run_root/mc_library_40k_shared_cv_fix" \
  --attribution-window 60 --attribution-min-observations 20

#!/usr/bin/env bash
# Full comparison grid (18 runs with the default 3 seeds):
#   {A, B} × {none, scaled} × d_model 64     → 12 runs
#   {A, B} × scaled         × d_model 128    →  6 runs
# then writes runs/summary.csv and runs/summary_by_config.csv.
#
# Seed is the outer loop, so every configuration gets seed 0 first.
# Finished runs (metrics.json present) are skipped — after a Colab disconnect,
# just run the same command again to resume.
#
# Colab:
#   from google.colab import drive; drive.mount('/content/drive')
#   !bash /content/drive/Shareddrives/Baiying/phd_thesis/multivariate_baselines/command/run_grid.sh
#
#   !SEEDS="0" bash .../run_grid.sh                 # one seed only
#   !bash .../run_grid.sh --max_epochs 2            # extra arguments go to every train.py call
set -euo pipefail
source "$(dirname "$0")/common.sh"
SEEDS=${SEEDS:-"0 1 2"}
EXTRA_ARGS=("$@")

run() {  # run <norm> <d_model> <seed> <features...>
  local norm=$1 d_model=$2 seed=$3
  shift 3
  "$PYTHON" "$CODE_DIR/train.py" \
    --data_dir "$DATA_DIR" --runs_dir "$RUNS_DIR" \
    --norm "$norm" --d_model "$d_model" --seed "$seed" --skip_existing \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    --features "$@"
}

for seed in $SEEDS; do
  for norm in none scaled; do
    run "$norm" 64 "$seed" bg                 # Baseline A
    run "$norm" 64 "$seed" bg carbs bolus     # Baseline B
  done
  run scaled 128 "$seed" bg                   # Baseline A, larger model
  run scaled 128 "$seed" bg carbs bolus       # Baseline B, larger model
done

"$PYTHON" "$CODE_DIR/summarize_runs.py" --runs_dir "$RUNS_DIR"

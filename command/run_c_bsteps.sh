#!/usr/bin/env bash
# Follow-up baselines (scaled, d_model 64, seeds 0–2 → 6 runs):
#   C       bg iob cob                  physiological alignment (same modalities as B, different encoding)
#   Bsteps  bg carbs bolus steps        B + activity (checks the EXERCISE-window regression; = LSTM raw features)
#
# Results go to runs_c_bsteps/ (not runs/), with run names C_scaled_d64_... and Bsteps_scaled_d64_...
# Tables written at the end (runs/summary*.csv is never touched):
#   runs_c_bsteps/summary.csv, summary_by_config.csv                    these 6 runs only
#   runs_c_bsteps/combined_summary.csv, combined_summary_by_config.csv  A/B (runs/) + C/Bsteps side by side
#
# Finished runs are skipped, so re-running after a Colab disconnect resumes.
#
# Colab (after the usual mount + git pull cell):
#   !bash {REPO}/command/run_c_bsteps.sh
#   !SEEDS="0" bash {REPO}/command/run_c_bsteps.sh      # one seed only
#   !bash {REPO}/command/run_c_bsteps.sh --max_epochs 2  # extra arguments go to every train.py call
set -euo pipefail
RUNS_SUBDIR=${RUNS_SUBDIR:-runs_c_bsteps}
source "$(dirname "$0")/common.sh"
SEEDS=${SEEDS:-"0 1 2"}
EXTRA_ARGS=("$@")
AB_RUNS_DIR=${AB_RUNS_DIR:-$PHD_ROOT/multivariate_baselines/runs}   # earlier A/B results, read-only here

run() {  # run <seed> <features...>
  local seed=$1
  shift
  "$PYTHON" "$CODE_DIR/train.py" \
    --data_dir "$DATA_DIR" --runs_dir "$RUNS_DIR" \
    --norm scaled --d_model 64 --seed "$seed" --skip_existing \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    --features "$@"
}

for seed in $SEEDS; do
  run "$seed" bg iob cob                 # Baseline C
  run "$seed" bg carbs bolus steps       # Baseline B + steps
done

"$PYTHON" "$CODE_DIR/summarize_runs.py" --runs_dir "$RUNS_DIR"
if [ -d "$AB_RUNS_DIR" ]; then
  "$PYTHON" "$CODE_DIR/summarize_runs.py" --runs_dir "$AB_RUNS_DIR" "$RUNS_DIR" \
    --out_dir "$RUNS_DIR" --prefix combined_
fi

#!/usr/bin/env bash
# Training-target experiment: is it better to train on the evaluated endpoint only?
#   --loss_target last  (MSE on the endpoint only; early stopping on the endpoint)
#   × features  A       bg
#               Bsteps  bg carbs bolus steps
#   × horizon 6 / 12 steps (30 / 60 min), history 24 steps (2 h), scaled, d_model 64
#   × seeds 0–2                                              → 12 runs
#
# The comparison runs with --loss_target all (MSE over every step) already exist in
# runs_history/ with the same aligned samples, hyperparameters and seeds, so they are
# reused, not retrained. runs_history/ is only read.
#
# Results: runs_loss_target/{A|Bsteps}_scaled_d64_h24_f{6|12}_losslast_seed{S}/
#          runs_loss_target/summary*.csv           the 12 endpoint-only runs
#          runs_loss_target/combined_summary*.csv  side by side with the multi-step runs
#                                                  from runs_history/ (history 24, d_model 64)
# Finished runs are skipped, so re-running resumes.
#
# Colab (after the usual mount + git pull cell):
#   !bash {REPO}/command/run_loss_target.sh
#   !SEEDS="0" bash {REPO}/command/run_loss_target.sh               # seed 0 only
#   !bash {REPO}/command/run_loss_target.sh --max_epochs 1          # extra arguments go to train.py
set -euo pipefail
RUNS_SUBDIR=${RUNS_SUBDIR:-runs_loss_target}
SAMPLES_SUBDIR=${SAMPLES_SUBDIR:-new_sample}
DEFER_SAMPLES=1
source "$(dirname "$0")/common.sh"
SEEDS=${SEEDS:-"0 1 2"}
HISTORY=${HISTORY:-24}
HORIZONS=${HORIZONS:-"6 12"}
EXTRA_ARGS=("$@")
HISTORY_RUNS_DIR=${HISTORY_RUNS_DIR:-$PHD_ROOT/multivariate_baselines/runs_history}   # multi-step runs, read-only
mkdir -p "$RUNS_DIR"

run() {  # run <seed> <features...>
  local seed=$1
  shift
  "$PYTHON" "$CODE_DIR/train.py" \
    --data_dir "$DATA_DIR" --runs_dir "$RUNS_DIR" \
    --norm scaled --d_model 64 --seed "$seed" --loss_target last --skip_existing \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    --features "$@"
}

for seed in $SEEDS; do
  for f in $HORIZONS; do
    ensure_aligned_samples "$HISTORY" "$f"
    run "$seed" bg                              # A
    run "$seed" bg carbs bolus steps            # B + steps
  done
done

"$PYTHON" "$CODE_DIR/summarize_runs.py" --runs_dir "$RUNS_DIR"
if [ -d "$HISTORY_RUNS_DIR" ]; then
  "$PYTHON" "$CODE_DIR/summarize_runs.py" --runs_dir "$HISTORY_RUNS_DIR" "$RUNS_DIR" \
    --out_dir "$RUNS_DIR" --prefix combined_ \
    --where history="$HISTORY" --where d_model=64 --windows
else
  echo "No $HISTORY_RUNS_DIR — skipping the side-by-side comparison with multi-step runs"
fi

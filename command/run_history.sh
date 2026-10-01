#!/usr/bin/env bash
# History-length × horizon experiment (all scaled, d_model 64):
#   history 12 / 24 / 48 steps (1 h / 2 h / 4 h)  ×  horizon 6 / 12 steps (30 / 60 min)
#   × features  A       bg
#               Bsteps  bg carbs bolus steps
#               Csteps  bg iob cob smoothed_step
#   × seeds 0–2                                              → 54 runs
#
# The six sample sets are aligned (--min_history 48 --min_horizon 12): they share the same
# forecast origins and, per horizon, the same targets — only the input length differs.
# Each set h{H}_f{F}_trs1_tes1_mh48_mf12 is taken from, in order:
#   1. /content/samples/<set>/                         (already on local disk this session)
#   2. sample_prepare/new_sample/<set>/ on Drive       (uploaded; copied to local disk)
#   3. built from combined/upgrade_filter_{train,test} with prepare_samples.py (fallback)
# All six sets take ≈ 7.5 GB of local disk.
#
# Results: runs_history/{A|Bsteps|Csteps}_scaled_d64_h{H}_f{F}_seed{S}/
#          runs_history/summary.csv, summary_by_config.csv
#          runs_history/_samples/<sample set>/  (config + per-subject counts of each sample set)
# runs/ and runs_c_bsteps/ are never touched. Finished runs are skipped, so re-running resumes.
#
# Colab (after the usual mount + git pull cell):
#   !bash {REPO}/command/run_history.sh
#   !SEEDS="0" bash {REPO}/command/run_history.sh                   # seed 0 only
#   !HORIZONS="6" HISTORIES="12 24" bash {REPO}/command/run_history.sh
#   !bash {REPO}/command/run_history.sh --max_epochs 1              # extra arguments go to train.py
set -euo pipefail
RUNS_SUBDIR=${RUNS_SUBDIR:-runs_history}
SAMPLES_SUBDIR=${SAMPLES_SUBDIR:-new_sample}
DEFER_SAMPLES=1
source "$(dirname "$0")/common.sh"
SEEDS=${SEEDS:-"0 1 2"}
HISTORIES=${HISTORIES:-"12 24 48"}
HORIZONS=${HORIZONS:-"6 12"}
EXTRA_ARGS=("$@")
mkdir -p "$RUNS_DIR"

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
  for f in $HORIZONS; do
    for h in $HISTORIES; do
      ensure_aligned_samples "$h" "$f"
      run "$seed" bg                              # A
      run "$seed" bg carbs bolus steps            # B + steps
      run "$seed" bg iob cob smoothed_step        # C + steps
    done
  done
done

"$PYTHON" "$CODE_DIR/summarize_runs.py" --runs_dir "$RUNS_DIR" --pivot

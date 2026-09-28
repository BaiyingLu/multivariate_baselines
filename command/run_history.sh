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
# They are built on the fly from the combined/upgrade_filter_{train,test} CSVs on Drive into
# /content/samples/ (≈ 7.5 GB for all six on local disk; nothing large is written to Drive).
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
DEFER_SAMPLES=1
source "$(dirname "$0")/common.sh"
SEEDS=${SEEDS:-"0 1 2"}
HISTORIES=${HISTORIES:-"12 24 48"}
HORIZONS=${HORIZONS:-"6 12"}
MIN_HISTORY=48
MIN_HORIZON=12
EXTRA_ARGS=("$@")
PREP_SCRIPT="$PREPROC_DIR/sample_prepare/prepare_samples.py"
COMBINED_DIR="$PREPROC_DIR/combined"

# ── preflight ────────────────────────────────────────────────────────────────
if [ ! -f "$PREP_SCRIPT" ] || ! grep -q -- "--min_history" "$PREP_SCRIPT"; then
  echo "Need the updated prepare_samples.py (with --min_history) at $PREP_SCRIPT"
  exit 1
fi
for split in upgrade_filter_train upgrade_filter_test; do
  if [ ! -d "$COMBINED_DIR/$split" ]; then
    echo "Missing $COMBINED_DIR/$split — upload the combined CSVs first"
    exit 1
  fi
done
mkdir -p "$RUNS_DIR"

# ensure_samples <history> <horizon>: set DATA_DIR to the aligned sample set, building it if needed
ensure_samples() {
  local h=$1 f=$2
  local name="h${h}_f${f}_trs1_tes1_mh${MIN_HISTORY}_mf${MIN_HORIZON}"
  DATA_DIR="$LOCAL_SAMPLES_ROOT/$name"
  if [ ! -f "$DATA_DIR/config.json" ]; then           # config.json is written last
    if [ -f "$SAMPLES_ROOT/$name/test.npz" ]; then
      use_samples "$name"
    else
      echo "Building samples $name ..."
      "$PYTHON" "$PREP_SCRIPT" --data_root "$COMBINED_DIR" --history "$h" --horizon "$f" \
        --min_history "$MIN_HISTORY" --min_horizon "$MIN_HORIZON" --out_dir "$DATA_DIR"
    fi
  fi
  mkdir -p "$RUNS_DIR/_samples/$name"
  cp "$DATA_DIR/config.json" "$RUNS_DIR/_samples/$name/"
  if [ -f "$DATA_DIR/summary.csv" ]; then
    cp "$DATA_DIR/summary.csv" "$RUNS_DIR/_samples/$name/"
  fi
}

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
      ensure_samples "$h" "$f"
      run "$seed" bg                              # A
      run "$seed" bg carbs bolus steps            # B + steps
      run "$seed" bg iob cob smoothed_step        # C + steps
    done
  done
done

"$PYTHON" "$CODE_DIR/summarize_runs.py" --runs_dir "$RUNS_DIR" --pivot

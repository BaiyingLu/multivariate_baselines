# Shared paths for the command scripts (sourced, not run directly).
# Every variable can be overridden from the environment, e.g.
#   PHD_ROOT=/Users/baiyinglu/Desktop/AugmentedHealthLab/phd_thesis bash run_baseline_a.sh

PHD_ROOT=${PHD_ROOT:-/content/drive/Shareddrives/Baiying/phd_thesis}
SAMPLES=${SAMPLES:-h12_f6_trs1_tes1}                     # which prepare_samples.py output to use
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../code" && pwd)"   # code next to this script (git clone)
RUNS_SUBDIR=${RUNS_SUBDIR:-runs}                                    # set by scripts that keep their own results folder
RUNS_DIR=${RUNS_DIR:-$PHD_ROOT/multivariate_baselines/$RUNS_SUBDIR} # results stay on Drive
PREPROC_DIR="$PHD_ROOT/multivariate_data_preprocessing"
SAMPLES_SUBDIR=${SAMPLES_SUBDIR:-samples}                           # folder under sample_prepare/ holding the sample sets
SAMPLES_ROOT="$PREPROC_DIR/sample_prepare/$SAMPLES_SUBDIR"
PYTHON=${PYTHON:-$(command -v python || command -v python3)}

# On Colab, samples are read from local disk — reading GBs from Drive is slow.
if [ -d /content ] && [ "${COPY_TO_LOCAL:-1}" = "1" ]; then
  LOCAL_SAMPLES_ROOT=/content/samples
else
  LOCAL_SAMPLES_ROOT="$SAMPLES_ROOT"
fi

# use_samples <name>: point DATA_DIR at samples/<name>, copying it from Drive to local disk once on Colab
use_samples() {
  local name=$1
  local src="$SAMPLES_ROOT/$name"
  DATA_DIR="$LOCAL_SAMPLES_ROOT/$name"
  if [ -f "$DATA_DIR/test.npz" ]; then
    return 0
  fi
  if [ ! -f "$src/test.npz" ]; then
    echo "Samples not found: $src"
    echo "Generate them first with multivariate_data_preprocessing/sample_prepare/prepare_samples.py"
    return 1
  fi
  echo "Copying samples to $DATA_DIR ..."
  mkdir -p "$DATA_DIR.tmp"
  cp "$src"/*.npz "$src"/config.json "$DATA_DIR.tmp"/
  if [ -f "$src/summary.csv" ]; then
    cp "$src/summary.csv" "$DATA_DIR.tmp"/
  fi
  mv "$DATA_DIR.tmp" "$DATA_DIR"
}

# ── Aligned sample sets (history × horizon experiments) ──────────────────────
# h{H}_f{F}_trs1_tes1_mh48_mf12: all sets share the same forecast origins.
MIN_HISTORY=${MIN_HISTORY:-48}
MIN_HORIZON=${MIN_HORIZON:-12}
PREP_SCRIPT="$PREPROC_DIR/sample_prepare/prepare_samples.py"
COMBINED_DIR="$PREPROC_DIR/combined"

# Only needed when a sample set is neither on local disk nor uploaded to Drive
check_can_build() {
  if [ ! -f "$PREP_SCRIPT" ] || ! grep -q -- "--min_history" "$PREP_SCRIPT"; then
    echo "Sample set not found in $SAMPLES_ROOT and cannot build it:"
    echo "need the updated prepare_samples.py (with --min_history) at $PREP_SCRIPT"
    exit 1
  fi
  for split in upgrade_filter_train upgrade_filter_test; do
    if [ ! -d "$COMBINED_DIR/$split" ]; then
      echo "Sample set not found in $SAMPLES_ROOT and cannot build it: missing $COMBINED_DIR/$split"
      exit 1
    fi
  done
}

# ensure_aligned_samples <history> <horizon>: set DATA_DIR (local disk → uploaded on Drive → build),
# and keep a copy of the sample set's config / per-subject counts in $RUNS_DIR/_samples/
ensure_aligned_samples() {
  local h=$1 f=$2
  local name="h${h}_f${f}_trs1_tes1_mh${MIN_HISTORY}_mf${MIN_HORIZON}"
  DATA_DIR="$LOCAL_SAMPLES_ROOT/$name"
  if [ ! -f "$DATA_DIR/config.json" ]; then           # config.json is written last
    if [ -f "$SAMPLES_ROOT/$name/test.npz" ]; then
      use_samples "$name"
    else
      check_can_build
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

# Scripts that switch between several sample sets set DEFER_SAMPLES=1 and call
# use_samples / ensure_aligned_samples themselves.
if [ "${DEFER_SAMPLES:-0}" != "1" ]; then
  use_samples "$SAMPLES" || exit 1
fi

# Shared paths for the command scripts (sourced, not run directly).
# Every variable can be overridden from the environment, e.g.
#   PHD_ROOT=/Users/baiyinglu/Desktop/AugmentedHealthLab/phd_thesis bash run_baseline_a.sh

PHD_ROOT=${PHD_ROOT:-/content/drive/Shareddrives/Baiying/phd_thesis}
SAMPLES=${SAMPLES:-h12_f6_trs1_tes1}                     # which prepare_samples.py output to use
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../code" && pwd)"   # code next to this script (git clone)
RUNS_SUBDIR=${RUNS_SUBDIR:-runs}                                    # set by scripts that keep their own results folder
RUNS_DIR=${RUNS_DIR:-$PHD_ROOT/multivariate_baselines/$RUNS_SUBDIR} # results stay on Drive
PREPROC_DIR="$PHD_ROOT/multivariate_data_preprocessing"
SAMPLES_ROOT="$PREPROC_DIR/sample_prepare/samples"
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
  mv "$DATA_DIR.tmp" "$DATA_DIR"
}

# Scripts that switch between several sample sets set DEFER_SAMPLES=1 and call use_samples themselves.
if [ "${DEFER_SAMPLES:-0}" != "1" ]; then
  use_samples "$SAMPLES" || exit 1
fi

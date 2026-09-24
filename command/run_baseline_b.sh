#!/usr/bin/env bash
# Baseline B — naive multivariate Transformer (encoder + MLP head),
# features: bg carbs bolus (grid-aligned, zero where no event)
#
# Colab:
#   from google.colab import drive; drive.mount('/content/drive')
#   !bash /content/drive/Shareddrives/Baiying/phd_thesis/multivariate_baselines/command/run_baseline_b.sh
#
# Change settings with environment variables (defaults: NORM=scaled D_MODEL=64 SEED=0):
#   !NORM=none SEED=1 bash .../run_baseline_b.sh
#   !D_MODEL=128 bash .../run_baseline_b.sh
#   !SAMPLES=h24_f6_trs1_tes1 bash .../run_baseline_b.sh      # other history / horizon
# Extra arguments are passed straight to train.py:
#   !bash .../run_baseline_b.sh --max_epochs 5 --batch_size 1024
set -euo pipefail
source "$(dirname "$0")/common.sh"

"$PYTHON" "$CODE_DIR/train.py" \
  --data_dir "$DATA_DIR" --runs_dir "$RUNS_DIR" \
  --norm "${NORM:-scaled}" --d_model "${D_MODEL:-64}" --seed "${SEED:-0}" \
  "$@" \
  --features bg carbs bolus

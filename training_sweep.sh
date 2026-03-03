#!/bin/bash
set -euo pipefail

# SEEDS=(1)
# HIDDEN_DIMS=(128)
# NUM_LAYERS=(4)
# NUM_HEADS=(4)
SEEDS=(1 2)
HIDDEN_DIMS=(512)
NUM_LAYERS=(6)
NUM_HEADS=(4)
DROPOUTS=(0.0 0.1 0.2 0.4)
DATASET_GROUPS=(
  '["episodes/20260302_153804/episodes_000000_trim.pt"]'
  '["episodes/20260302_153804/episodes_000000_trim.pt", "episodes/20260302_153804/episodes_000001_trim.pt",]'
  '["episodes/20260302_153804/episodes_000000_trim.pt", "episodes/20260302_153804/episodes_000001_trim.pt", "episodes/20260302_153804/episodes_000002_trim.pt", "episodes/20260302_153804/episodes_000003_trim.pt",]'
  '["episodes/20260302_153804/episodes_000000_trim.pt", "episodes/20260302_153804/episodes_000001_trim.pt", "episodes/20260302_153804/episodes_000002_trim.pt", "episodes/20260302_153804/episodes_000003_trim.pt", "episodes/20260302_153804/episodes_000004_trim.pt", "episodes/20260302_153804/episodes_000005_trim.pt", "episodes/20260302_153804/episodes_000006_trim.pt", "episodes/20260302_153804/episodes_000007_trim.pt", "episodes/20260302_153804/episodes_000008_trim.pt",]'
  '["episodes/20260302_165916/episodes_000000_trim.pt"]'
  '["episodes/20260302_165916/episodes_000000_trim.pt", "episodes/20260302_165916/episodes_000001_trim.pt",]'
  '["episodes/20260302_165916/episodes_000000_trim.pt", "episodes/20260302_165916/episodes_000001_trim.pt", "episodes/20260302_165916/episodes_000002_trim.pt", "episodes/20260302_165916/episodes_000003_trim.pt",]'
  '["episodes/20260302_165916/episodes_000000_trim.pt", "episodes/20260302_165916/episodes_000001_trim.pt", "episodes/20260302_165916/episodes_000002_trim.pt", "episodes/20260302_165916/episodes_000003_trim.pt", "episodes/20260302_165916/episodes_000004_trim.pt", "episodes/20260302_165916/episodes_000005_trim.pt", "episodes/20260302_165916/episodes_000006_trim.pt", "episodes/20260302_165916/episodes_000007_trim.pt", "episodes/20260302_165916/episodes_000008_trim.pt",]'
)

SBATCH_FILE="/gscratch/weirdlab/yanda/lti/UWLab-yanda/train.sbatch"
mkdir -p logs

for DATASET_GROUP_INDEX in "${!DATASET_GROUPS[@]}"; do
  DATASET_GROUP="${DATASET_GROUPS[$DATASET_GROUP_INDEX]}"
  for SEED in "${SEEDS[@]}"; do
    for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
      for NUM_LAYER in "${NUM_LAYERS[@]}"; do
        for NUM_HEAD in "${NUM_HEADS[@]}"; do
          for DROPOUT in "${DROPOUTS[@]}"; do
            RUN_NAME="ds${DATASET_GROUP_INDEX}_seed${SEED}_hd${HIDDEN_DIM}_L${NUM_LAYER}_H${NUM_HEAD}_do${DROPOUT}"
            # Keep job name <= ~128 chars (Slurm limit varies)
            sbatch --job-name="$RUN_NAME" "$SBATCH_FILE" \
              "$SEED" "$HIDDEN_DIM" "$NUM_LAYER" "$NUM_HEAD" "$DROPOUT" "$RUN_NAME" "$DATASET_GROUP"
            echo "submitted $RUN_NAME"
          done
        done
      done
    done
  done
done

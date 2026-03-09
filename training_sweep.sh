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
DROPOUTS=(0.0 0.2 0.4)
LEARNING_RATES=(3e-4 1e-4 5e-5)
DATASET_PATH_FORMAT='episodes/20260303_022952/episodes_0000{x}.pt'
# Each element is one dataset group: values replace {x} in DATASET_PATH_FORMAT.
DATASET_INDEX_GROUPS=(
  "01 02 03 04 05 06 07 08 09 10 11 12 13 14"
  "01 02 03 04 05 06 07 08 09"
  "01 02 03 04"
  "01 02"
)

DATASET_GROUPS=()
for DATASET_INDEX_GROUP in "${DATASET_INDEX_GROUPS[@]}"; do
  GROUP='['
  SEP=""
  for INDEX in $DATASET_INDEX_GROUP; do
    DATASET_PATH="${DATASET_PATH_FORMAT//\{x\}/$INDEX}"
    GROUP+="${SEP}\"${DATASET_PATH}\""
    SEP=", "
  done
  GROUP+=']'
  DATASET_GROUPS+=("$GROUP")
done

# echo "${DATASET_GROUPS[@]}"

# exit 0

SBATCH_FILE="/gscratch/weirdlab/yanda/lti/UWLab-yanda/train.sbatch"
mkdir -p logs

for DATASET_GROUP_INDEX in "${!DATASET_GROUPS[@]}"; do
  DATASET_GROUP="${DATASET_GROUPS[$DATASET_GROUP_INDEX]}"
  for SEED in "${SEEDS[@]}"; do
    for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
      for NUM_LAYER in "${NUM_LAYERS[@]}"; do
        for NUM_HEAD in "${NUM_HEADS[@]}"; do
          for DROPOUT in "${DROPOUTS[@]}"; do
            for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
              RUN_NAME="group_eps_ds${DATASET_GROUP_INDEX}_seed${SEED}_hd${HIDDEN_DIM}_L${NUM_LAYER}_H${NUM_HEAD}_do${DROPOUT}_lr${LEARNING_RATE}"
              # Keep job name <= ~128 chars (Slurm limit varies)
              sbatch --job-name="$RUN_NAME" "$SBATCH_FILE" \
                "$SEED" "$HIDDEN_DIM" "$NUM_LAYER" "$NUM_HEAD" "$DROPOUT" "$LEARNING_RATE" "$RUN_NAME" "$DATASET_GROUP"
              echo "submitted $RUN_NAME"
            done
          done
        done
      done
    done
  done
done

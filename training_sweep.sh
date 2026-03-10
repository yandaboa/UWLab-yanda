#!/bin/bash
set -euo pipefail

# SEEDS=(1)
# HIDDEN_DIMS=(128)
# NUM_LAYERS=(4)
# NUM_HEADS=(4)
SEEDS=(1 2 3)
HIDDEN_DIMS=(64 128 256)
NUM_LAYERS=(6 8 12)
NUM_HEADS=(4)
DROPOUTS=(0.2)
LEARNING_RATES=(1e-4)
WEIGHT_DECAYS=(0.0)
INCLUDE_CURRENT_TRAJECTORY_VALUES=(true false)
TRAIN_PM=true
DATASET_PATH_FORMAT='episodes/20260303_022952/episodes_0000{x}.pt'
DEFAULT_BASE_CONFIG_PATH="/gscratch/weirdlab/yanda/lti/UWLab-yanda/source/uwlab_rl/uwlab_rl/rsl_rl/configs/point_mass.yaml"
BASE_CONFIG_PATH="${1:-$DEFAULT_BASE_CONFIG_PATH}"
# Each element is one dataset group: values replace {x} in DATASET_PATH_FORMAT.
DATASET_INDEX_GROUPS=(
  "01 02 03 04 05 06 07 08 09 10 11 12 13 14"
  "01 02 03 04 05 06 07 08 09"
  "01 02 03 04"
  "01 02"
)

# Optional manual override. Each element must be a JSON-like list string consumed by
# Hydra dotlist override, e.g. '["episodes/a.pt", "episodes/b.pt"]'.
DATASET_GROUPS=(
  "[\"/gscratch/weirdlab/yanda/lti/dm_control/datasets/pm_lower_freq_train.pt\"]"
)

HAS_MANUAL_DATASET_GROUPS=false
for DATASET_GROUP in "${DATASET_GROUPS[@]}"; do
  if [[ -n "$DATASET_GROUP" ]]; then
    HAS_MANUAL_DATASET_GROUPS=true
    break
  fi
done

if [[ "$HAS_MANUAL_DATASET_GROUPS" == false ]]; then
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
fi

# echo "${DATASET_GROUPS[@]}"

# exit 0

SBATCH_FILE="/gscratch/weirdlab/yanda/lti/UWLab-yanda/train.sbatch"
mkdir -p logs

echo "using base config: $BASE_CONFIG_PATH"
[[ -f "$BASE_CONFIG_PATH" ]] || { echo "missing base config: $BASE_CONFIG_PATH"; exit 1; }

for DATASET_GROUP_INDEX in "${!DATASET_GROUPS[@]}"; do
  DATASET_GROUP="${DATASET_GROUPS[$DATASET_GROUP_INDEX]}"
  for SEED in "${SEEDS[@]}"; do
    for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
      for NUM_LAYER in "${NUM_LAYERS[@]}"; do
        for NUM_HEAD in "${NUM_HEADS[@]}"; do
          for DROPOUT in "${DROPOUTS[@]}"; do
            for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
              for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                for INCLUDE_CURRENT_TRAJECTORY in "${INCLUDE_CURRENT_TRAJECTORY_VALUES[@]}"; do
                  RUN_NAME="pm_sweep_${DATASET_GROUP_INDEX}_seed${SEED}_hd${HIDDEN_DIM}_L${NUM_LAYER}_H${NUM_HEAD}_do${DROPOUT}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_ict${INCLUDE_CURRENT_TRAJECTORY}"
                  # Keep job name <= ~128 chars (Slurm limit varies)
                  sbatch --job-name="$RUN_NAME" "$SBATCH_FILE" \
                    "$SEED" "$HIDDEN_DIM" "$NUM_LAYER" "$NUM_HEAD" "$DROPOUT" "$LEARNING_RATE" "$WEIGHT_DECAY" "$INCLUDE_CURRENT_TRAJECTORY" "$RUN_NAME" "$DATASET_GROUP" "$BASE_CONFIG_PATH" "$TRAIN_PM"
                  echo "submitted $RUN_NAME"
                done
              done
            done
          done
        done
      done
    done
  done
done

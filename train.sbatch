#!/bin/bash
#SBATCH --job-name=ctx-train
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --chdir=/gscratch/weirdlab/yanda/lti/UWLab-yanda
#SBATCH --export=ALL
#SBATCH --exclude=g3118,g3092

SEED="$1"
HIDDEN_DIM="$2"
NUM_LAYERS="$3"
NUM_HEADS="$4"
DROPOUT="$5"
LEARNING_RATE="$6"
WEIGHT_DECAY="$7"
INCLUDE_CURRENT_TRAJECTORY="$8"
RUN_NAME="$9"
DATASET_GROUP="${10}"
BASE_CONFIG_PATH="${11:-}"
TRAIN_PM="${12:-false}"

set -e
set -o pipefail
mkdir -p logs

source /gscratch/weirdlab/yanda/miniforge3/etc/profile.d/conda.sh
conda activate uwlab

cd /gscratch/weirdlab/yanda/lti/UWLab-yanda

echo "RUN_NAME=$RUN_NAME"
echo "seed=$SEED hidden_dim=$HIDDEN_DIM num_layers=$NUM_LAYERS num_heads=$NUM_HEADS dropout=$DROPOUT learning_rate=$LEARNING_RATE weight_decay=$WEIGHT_DECAY include_current_trajectory=$INCLUDE_CURRENT_TRAJECTORY"
echo "dataset_group=$DATASET_GROUP"
echo "base_config_path=${BASE_CONFIG_PATH:-<default dataclass cfg>}"
echo "train_pm=$TRAIN_PM"

PY_ARGS=(
  --seed "$SEED"
)
if [[ -n "$BASE_CONFIG_PATH" ]]; then
  PY_ARGS+=(--config "$BASE_CONFIG_PATH")
fi
PY_ARGS+=(
  model.hidden_dim="$HIDDEN_DIM" \
  model.num_layers="$NUM_LAYERS" \
  model.num_heads="$NUM_HEADS" \
  model.embedding_dropout="$DROPOUT" \
  model.attention_dropout="$DROPOUT" \
  model.residual_dropout="$DROPOUT" \
  optim.learning_rate="$LEARNING_RATE" \
  optim.weight_decay="$WEIGHT_DECAY" \
  input.include_current_trajectory="$INCLUDE_CURRENT_TRAJECTORY" \
  data.train_episode_paths="$DATASET_GROUP" \
  logging.run_name="$RUN_NAME"
)

python -u scripts/reinforcement_learning/rsl_rl/train_supervised_context.py "${PY_ARGS[@]}"

if [[ "${TRAIN_PM,,}" == "true" ]]; then
  LATEST_CHECKPOINT="$(find logs/rsl_rl -type f -path "*/${RUN_NAME}/model_*.pt" | sort | tail -n 1)"
  if [[ -z "$LATEST_CHECKPOINT" ]]; then
    echo "ERROR: could not find checkpoint for run_name=$RUN_NAME under logs/rsl_rl/*/$RUN_NAME/model_*.pt"
    exit 1
  fi

  EVAL_CONTEXT_PATH="$(python - "$DATASET_GROUP" <<'PY'
import json
import sys

raw = sys.argv[1].strip()
try:
    parsed = json.loads(raw)
except Exception:
    parsed = raw.strip('"').strip("'")
if isinstance(parsed, list):
    if len(parsed) == 0:
        raise SystemExit("DATASET_GROUP resolved to an empty list.")
    value = parsed[0]
elif isinstance(parsed, str):
    value = parsed
else:
    raise SystemExit(f"Unsupported DATASET_GROUP type: {type(parsed)}")
print(value)
PY
)"

  echo "Running post-train inference eval with checkpoint=$LATEST_CHECKPOINT context_path=$EVAL_CONTEXT_PATH"
  python -u scripts/reinforcement_learning/rsl_rl/infer_point_mass_supervised.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --context-path "$EVAL_CONTEXT_PATH"
else
  echo "Skipping post-train inference eval because train_pm=$TRAIN_PM"
fi
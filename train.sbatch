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
RUN_NAME="$6"
DATASET_GROUP="$7"

set -e
set -o pipefail
mkdir -p logs

source /gscratch/weirdlab/yanda/miniforge3/etc/profile.d/conda.sh
conda activate uwlab

cd /gscratch/weirdlab/yanda/lti/UWLab-yanda

echo "RUN_NAME=$RUN_NAME"
echo "seed=$SEED hidden_dim=$HIDDEN_DIM num_layers=$NUM_LAYERS num_heads=$NUM_HEADS dropout=$DROPOUT"
echo "dataset_group=$DATASET_GROUP"

python -u scripts/reinforcement_learning/rsl_rl/train_supervised_context.py \
  --seed "$SEED" \
  model.hidden_dim="$HIDDEN_DIM" \
  model.num_layers="$NUM_LAYERS" \
  model.num_heads="$NUM_HEADS" \
  model.embedding_dropout="$DROPOUT" \
  model.attention_dropout="$DROPOUT" \
  model.residual_dropout="$DROPOUT" \
  data.train_episode_paths="$DATASET_GROUP" \
  logging.run_name="$RUN_NAME"

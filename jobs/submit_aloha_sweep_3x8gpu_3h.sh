#!/bin/bash
set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/mnt/data/${USER}/openpi}"
WANDB_ENTITY="${WANDB_ENTITY:?Set WANDB_ENTITY}"
WANDB_PROJECT="${WANDB_PROJECT:-openpi-aloha-wandb-integration}"
OPENPI_VIDEO_BACKEND="${OPENPI_VIDEO_BACKEND:-pyav}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-jobs/train_aloha_sim_jax_8gpu_6h.sbatch}"

export OPENPI_ROOT
export WANDB_ENTITY
export WANDB_PROJECT
export OPENPI_VIDEO_BACKEND

# Keep one H100 node free so periodic/final eval jobs can start immediately.
declare -a NAMES=(
  "aloha-b32-lr2p5e5"
  "aloha-b32-lr5e5"
  "aloha-b64-lr2p5e5"
)

declare -a BATCH_SIZES=("32" "32" "64")
declare -a PEAK_LRS=("2.5e-5" "5e-5" "2.5e-5")
declare -a DECAY_LRS=("2.5e-6" "5e-6" "2.5e-6")

for i in "${!NAMES[@]}"; do
  run_name="${NAMES[$i]}"
  batch_size="${BATCH_SIZES[$i]}"
  peak_lr="${PEAK_LRS[$i]}"
  decay_lr="${DECAY_LRS[$i]}"

  train_args="--fsdp-devices=8 \
    --batch-size=${batch_size} \
    --num-train-steps=10000 \
    --log-interval=20 \
    --save-interval=2500 \
    --periodic-eval-num-examples=4 \
    --wandb-checkpoint-artifact-interval=2500 \
    --lr-schedule.warmup-steps=500 \
    --lr-schedule.peak-lr=${peak_lr} \
    --lr-schedule.decay-steps=10000 \
    --lr-schedule.decay-lr=${decay_lr}"

  echo "Submitting ${run_name}"
  sbatch \
    --time=03:00:00 \
    --export=ALL,WANDB_API_KEY,OPENPI_ROOT,WANDB_ENTITY,WANDB_PROJECT,OPENPI_VIDEO_BACKEND,RUN_NAME="${run_name}",TRAIN_ARGS="${train_args}" \
    "${SBATCH_SCRIPT}"
done

#!/bin/bash

set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/mnt/data/${USER}/openpi}"
WANDB_ENTITY="${WANDB_ENTITY:?Set WANDB_ENTITY before running.}"
WANDB_PROJECT="${WANDB_PROJECT:-openpi-aloha-wandb-integration}"
PYTORCH_WEIGHT_PATH="${PYTORCH_WEIGHT_PATH:-${OPENPI_ROOT}/cache/pi0_base_pytorch_smoke}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
PARTITION_ARGS="${PARTITION_ARGS:-}"

export OPENPI_ROOT WANDB_ENTITY WANDB_PROJECT PYTORCH_WEIGHT_PATH

cd "${OPENPI_ROOT}"

COMMON_ARGS="--num-train-steps=20000 --save-interval=500 --log-interval=20 --periodic-eval-num-examples=4 --final-eval-num-examples=0 --wandb-checkpoint-artifact-interval=500 --model.pytorch-compile-mode=None"

submit_job() {
  local run_name="$1"
  local batch_size="$2"
  local peak_lr="$3"

  local train_args="${COMMON_ARGS} --batch-size=${batch_size} --lr-schedule.peak-lr=${peak_lr}"
  sbatch ${PARTITION_ARGS} --time="${TIME_LIMIT}" \
    --export=ALL,OPENPI_ROOT,WANDB_ENTITY,WANDB_PROJECT,PYTORCH_WEIGHT_PATH,RUN_NAME="${run_name}",TRAIN_ARGS="${train_args}" \
    jobs/train_aloha_sim_pytorch_8gpu_6h.sbatch
}

submit_job "aloha-pytorch-b32-lr1e5" 32 1e-5
submit_job "aloha-pytorch-b32-lr1p25e5" 32 1.25e-5
submit_job "aloha-pytorch-b32-lr1p5e5" 32 1.5e-5
submit_job "aloha-pytorch-b32-lr1p75e5" 32 1.75e-5

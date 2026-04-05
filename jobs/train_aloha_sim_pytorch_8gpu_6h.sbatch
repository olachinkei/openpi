#!/bin/bash
#SBATCH --job-name=openpi_aloha_train_pytorch_8gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=256G

set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-$PWD}"
RUN_NAME="${RUN_NAME:-aloha-train-pi0_aloha_sim-pytorch-8gpu-${SLURM_JOB_ID}}"
WANDB_ENTITY="${WANDB_ENTITY:?Set WANDB_ENTITY before submitting.}"
WANDB_PROJECT="${WANDB_PROJECT:-openpi-aloha-wandb-integration}"
PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${OPENPI_ROOT}/.venv/bin/torchrun}"
PYTORCH_WEIGHT_PATH="${PYTORCH_WEIGHT_PATH:?Set PYTORCH_WEIGHT_PATH to a converted PyTorch base checkpoint directory before submitting.}"
OPENPI_VIDEO_BACKEND="${OPENPI_VIDEO_BACKEND:-pyav}"
OPENPI_RUNTIME_ROOT="${OPENPI_RUNTIME_ROOT:-${OPENPI_ROOT}}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_RUNTIME_ROOT}/cache}"
OPENPI_CHECKPOINT_BASE_DIR="${OPENPI_CHECKPOINT_BASE_DIR:-${OPENPI_RUNTIME_ROOT}/checkpoints}"
WANDB_DIR="${WANDB_DIR:-${OPENPI_RUNTIME_ROOT}/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${OPENPI_RUNTIME_ROOT}/wandb-cache}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-${OPENPI_RUNTIME_ROOT}/wandb-data}"
WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${OPENPI_RUNTIME_ROOT}/wandb-artifacts}"
TMPDIR="${TMPDIR:-${OPENPI_RUNTIME_ROOT}/tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OPENPI_RUNTIME_ROOT}/xdg-cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${OPENPI_RUNTIME_ROOT}/uv-cache}"
REMOTE_USER_NAME="${REMOTE_USER_NAME:-${SLURM_JOB_USER:-${USER}}}"
PRIMARY_VENDOR_GL_LIBDIR="${PRIMARY_VENDOR_GL_LIBDIR:-/mnt/data/${REMOTE_USER_NAME}/vendor-gl/lib}"
FALLBACK_VENDOR_GL_LIBDIR="${FALLBACK_VENDOR_GL_LIBDIR:-$HOME/vendor-gl/root/usr/lib/x86_64-linux-gnu}"
VENDOR_GL_LIBDIR="${VENDOR_GL_LIBDIR:-${PRIMARY_VENDOR_GL_LIBDIR}}"
TRAIN_ARGS="${TRAIN_ARGS:---log-interval=20 --save-interval=500 --wandb-checkpoint-artifact-interval=500}"

mkdir -p "${OPENPI_ROOT}/logs"
mkdir -p "${OPENPI_DATA_HOME}" "${OPENPI_CHECKPOINT_BASE_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_ARTIFACT_DIR}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}"
cd "${OPENPI_RUNTIME_ROOT}"

if [[ "${VENDOR_GL_LIBDIR}" == "${PRIMARY_VENDOR_GL_LIBDIR}" ]]; then
  mkdir -p "${PRIMARY_VENDOR_GL_LIBDIR}"
  if [[ ! -e "${PRIMARY_VENDOR_GL_LIBDIR}/libOSMesa.so.0" && -e "${FALLBACK_VENDOR_GL_LIBDIR}/libOSMesa.so.8.0.0" ]]; then
    ln -sfn "${FALLBACK_VENDOR_GL_LIBDIR}/libOSMesa.so.8.0.0" "${PRIMARY_VENDOR_GL_LIBDIR}/libOSMesa.so.0"
  fi
  if [[ ! -e "${PRIMARY_VENDOR_GL_LIBDIR}/libOpenGL.so.0" && -e "${FALLBACK_VENDOR_GL_LIBDIR}/libOpenGL.so.0" ]]; then
    ln -sfn "${FALLBACK_VENDOR_GL_LIBDIR}/libOpenGL.so.0" "${PRIMARY_VENDOR_GL_LIBDIR}/libOpenGL.so.0"
  fi
  if [[ ! -e "${PRIMARY_VENDOR_GL_LIBDIR}/libEGL.so.0" && -e "${FALLBACK_VENDOR_GL_LIBDIR}/libEGL.so.0" ]]; then
    ln -sfn "${FALLBACK_VENDOR_GL_LIBDIR}/libEGL.so.0" "${PRIMARY_VENDOR_GL_LIBDIR}/libEGL.so.0"
  fi
fi

if [[ -d "${VENDOR_GL_LIBDIR}" ]]; then
  export LD_LIBRARY_PATH="${VENDOR_GL_LIBDIR}:${LD_LIBRARY_PATH:-}"
fi

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
if [[ "${MUJOCO_GL}" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
fi

export OPENPI_ROOT
export OPENPI_RUNTIME_ROOT
export OPENPI_DATA_HOME
export OPENPI_CHECKPOINT_BASE_DIR
export PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}"
export WANDB_ENTITY
export WANDB_PROJECT
export WANDB_DIR
export WANDB_CACHE_DIR
export WANDB_DATA_DIR
export WANDB_ARTIFACT_DIR
export OPENPI_VIDEO_BACKEND
export TMPDIR
export XDG_CACHE_HOME
export UV_CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_TIMEOUT_SECS="${TORCH_DISTRIBUTED_TIMEOUT_SECS:-7200}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"

"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=8 "${OPENPI_ROOT}/scripts/train_pytorch.py" pi0_aloha_sim \
  --exp-name="${RUN_NAME}" \
  --overwrite \
  --checkpoint-base-dir="${OPENPI_CHECKPOINT_BASE_DIR}" \
  --pytorch-weight-path="${PYTORCH_WEIGHT_PATH}" \
  ${TRAIN_ARGS}

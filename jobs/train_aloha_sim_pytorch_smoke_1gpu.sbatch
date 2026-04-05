#!/bin/bash
#SBATCH --job-name=openpi_aloha_train_pytorch_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-$PWD}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi0_aloha_sim}"
RUN_NAME="${RUN_NAME:-aloha-train-pi0_aloha_sim-pytorch-smoke-${SLURM_JOB_ID}}"
WANDB_ENTITY="${WANDB_ENTITY:?Set WANDB_ENTITY before submitting.}"
WANDB_PROJECT="${WANDB_PROJECT:-openpi-aloha-wandb-integration-test}"
PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
PYTORCH_WEIGHT_PATH="${PYTORCH_WEIGHT_PATH:-}"
OPENPI_VIDEO_BACKEND="${OPENPI_VIDEO_BACKEND:-pyav}"
OPENPI_RUNTIME_ROOT="${OPENPI_RUNTIME_ROOT:-/mnt/data/${USER}/openpi}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_RUNTIME_ROOT}/cache}"
OPENPI_CHECKPOINT_BASE_DIR="${OPENPI_CHECKPOINT_BASE_DIR:-${OPENPI_RUNTIME_ROOT}/checkpoints}"
WANDB_DIR="${WANDB_DIR:-${OPENPI_RUNTIME_ROOT}/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${OPENPI_RUNTIME_ROOT}/wandb-cache}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-${OPENPI_RUNTIME_ROOT}/wandb-data}"
WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${OPENPI_RUNTIME_ROOT}/wandb-artifacts}"
TMPDIR="${TMPDIR:-${OPENPI_RUNTIME_ROOT}/tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OPENPI_RUNTIME_ROOT}/xdg-cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${OPENPI_RUNTIME_ROOT}/uv-cache}"
PRIMARY_VENDOR_GL_LIBDIR="${PRIMARY_VENDOR_GL_LIBDIR:-/mnt/data/${USER}/vendor-gl/lib}"
FALLBACK_VENDOR_GL_LIBDIR="${FALLBACK_VENDOR_GL_LIBDIR:-$HOME/vendor-gl/root/usr/lib/x86_64-linux-gnu}"
VENDOR_GL_LIBDIR="${VENDOR_GL_LIBDIR:-${PRIMARY_VENDOR_GL_LIBDIR}}"
TRAIN_ARGS="${TRAIN_ARGS:---num-train-steps=6 --save-interval=5 --log-interval=1 --batch-size=1 --num-workers=0 --periodic-eval-num-examples=1 --final-eval-num-examples=1 --wandb-checkpoint-artifact-interval=-1 --model.pytorch-compile-mode=None}"

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

PYTORCH_WEIGHT_ARGS=()
if [[ -n "${PYTORCH_WEIGHT_PATH}" ]]; then
  PYTORCH_WEIGHT_ARGS+=("--pytorch-weight-path=${PYTORCH_WEIGHT_PATH}")
fi

"${PYTHON_BIN}" "${OPENPI_ROOT}/scripts/train_pytorch.py" "${TRAIN_CONFIG}" \
  --exp-name="${RUN_NAME}" \
  --overwrite \
  --checkpoint-base-dir="${OPENPI_CHECKPOINT_BASE_DIR}" \
  "${PYTORCH_WEIGHT_ARGS[@]}" \
  ${TRAIN_ARGS}

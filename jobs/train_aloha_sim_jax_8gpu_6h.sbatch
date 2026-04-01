#!/bin/bash
#SBATCH --job-name=openpi_aloha_train_8gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=256G

set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-$PWD}"
RUN_NAME="${RUN_NAME:-aloha-train-pi0_aloha_sim-8gpu-${SLURM_JOB_ID}}"
WANDB_ENTITY="${WANDB_ENTITY:?Set WANDB_ENTITY before submitting.}"
WANDB_PROJECT="${WANDB_PROJECT:-openpi-aloha-wandb-integration}"
PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
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
JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${OPENPI_RUNTIME_ROOT}/jax-compilation-cache}"
XLA_GPU_AUTOTUNE_CACHE_DIR="${XLA_GPU_AUTOTUNE_CACHE_DIR:-${JAX_COMPILATION_CACHE_DIR}/xla_gpu_per_fusion_autotune_cache_dir}"
PRIMARY_VENDOR_GL_LIBDIR="${PRIMARY_VENDOR_GL_LIBDIR:-/mnt/data/${USER}/vendor-gl/lib}"
FALLBACK_VENDOR_GL_LIBDIR="${FALLBACK_VENDOR_GL_LIBDIR:-$HOME/vendor-gl/root/usr/lib/x86_64-linux-gnu}"
VENDOR_GL_LIBDIR="${VENDOR_GL_LIBDIR:-${PRIMARY_VENDOR_GL_LIBDIR}}"
TRAIN_ARGS="${TRAIN_ARGS:---fsdp-devices=8 --log-interval=20 --save-interval=500 --wandb-checkpoint-artifact-interval=500}"

mkdir -p "${OPENPI_ROOT}/logs"
mkdir -p "${OPENPI_DATA_HOME}" "${OPENPI_CHECKPOINT_BASE_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_ARTIFACT_DIR}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}" "${XLA_GPU_AUTOTUNE_CACHE_DIR}/tmp"
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
export JAX_COMPILATION_CACHE_DIR
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=${XLA_GPU_AUTOTUNE_CACHE_DIR} ${XLA_FLAGS:-}"

"${PYTHON_BIN}" "${OPENPI_ROOT}/scripts/train.py" pi0_aloha_sim \
  --exp-name="${RUN_NAME}" \
  --overwrite \
  --checkpoint-base-dir="${OPENPI_CHECKPOINT_BASE_DIR}" \
  ${TRAIN_ARGS}

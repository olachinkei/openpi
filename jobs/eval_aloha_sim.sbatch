#!/bin/bash
#SBATCH --job-name=openpi_aloha_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-$PWD}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-pi0_aloha_sim}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR before submitting.}"
MANIFEST_PATH="${MANIFEST_PATH:?Set MANIFEST_PATH before submitting.}"
SPLIT_NAME="${SPLIT_NAME:-subsample}"
NUM_EXAMPLES="${NUM_EXAMPLES:-4}"
CHECKPOINT_ALIAS="${CHECKPOINT_ALIAS:-}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
OPENPI_RUNTIME_ROOT="${OPENPI_RUNTIME_ROOT:-/mnt/data/${USER}/openpi}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_RUNTIME_ROOT}/cache}"
WANDB_DIR="${WANDB_DIR:-${OPENPI_RUNTIME_ROOT}/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${OPENPI_RUNTIME_ROOT}/wandb-cache}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-${OPENPI_RUNTIME_ROOT}/wandb-data}"
WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${OPENPI_RUNTIME_ROOT}/wandb-artifacts}"
TMPDIR="${TMPDIR:-${OPENPI_RUNTIME_ROOT}/tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OPENPI_RUNTIME_ROOT}/xdg-cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${OPENPI_RUNTIME_ROOT}/uv-cache}"
VIDEO_ROOT="${VIDEO_ROOT:-${OPENPI_RUNTIME_ROOT}/aloha_eval/${SPLIT_NAME}-${SLURM_JOB_ID}}"
RESULTS_PATH="${RESULTS_PATH:-$VIDEO_ROOT/eval_${SPLIT_NAME}_results.json}"
VENDOR_GL_LIBDIR="${VENDOR_GL_LIBDIR:-$HOME/vendor-gl/root/usr/lib/x86_64-linux-gnu}"
RENDER_MODE="${RENDER_MODE:-rgb_array}"
VISUALIZATION_WIDTH="${VISUALIZATION_WIDTH:-640}"
VISUALIZATION_HEIGHT="${VISUALIZATION_HEIGHT:-336}"
VISUALIZATION_CAMERA_ID="${VISUALIZATION_CAMERA_ID:-angle}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

mkdir -p "${OPENPI_ROOT}/logs"
mkdir -p "${OPENPI_DATA_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_ARTIFACT_DIR}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" "${VIDEO_ROOT}"
cd "${OPENPI_RUNTIME_ROOT}"

if [[ -d "${VENDOR_GL_LIBDIR}" ]]; then
  export LD_LIBRARY_PATH="${VENDOR_GL_LIBDIR}:${LD_LIBRARY_PATH:-}"
fi

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
if [[ "${MUJOCO_GL}" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
fi

export OPENPI_RUNTIME_ROOT
export OPENPI_DATA_HOME
export PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}"
export WANDB_DIR
export WANDB_CACHE_DIR
export WANDB_DATA_DIR
export WANDB_ARTIFACT_DIR
export TMPDIR
export XDG_CACHE_HOME
export UV_CACHE_DIR

CMD=(
  "${PYTHON_BIN}" "${OPENPI_ROOT}/scripts/eval_aloha_dataset.py"
  --config-name "${CONFIG_NAME}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --manifest "${MANIFEST_PATH}"
  --video-dir "${VIDEO_ROOT}"
  --split "${SPLIT_NAME}"
  --num-examples "${NUM_EXAMPLES}"
  --results-path "${RESULTS_PATH}"
)

if [[ -n "${CHECKPOINT_ALIAS}" ]]; then
  CMD+=(--checkpoint-alias "${CHECKPOINT_ALIAS}")
fi

if [[ -n "${CHECKPOINT_STEP}" ]]; then
  CMD+=(--checkpoint-step "${CHECKPOINT_STEP}")
fi

if [[ -n "${RENDER_MODE}" ]]; then
  CMD+=(--render-mode "${RENDER_MODE}")
fi

CMD+=(--visualization-width "${VISUALIZATION_WIDTH}")
CMD+=(--visualization-height "${VISUALIZATION_HEIGHT}")
CMD+=(--visualization-camera-id "${VISUALIZATION_CAMERA_ID}")

if [[ -n "${EXTRA_EVAL_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EXTRA_EVAL_ARGS} )
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

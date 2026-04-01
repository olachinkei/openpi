#!/bin/bash
#SBATCH --job-name=openpi_aloha_norm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-pi0_aloha_sim}"
OPENPI_VIDEO_BACKEND="${OPENPI_VIDEO_BACKEND:-pyav}"
OPENPI_RUNTIME_ROOT="${OPENPI_RUNTIME_ROOT:-/mnt/data/${USER}/openpi}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_RUNTIME_ROOT}/cache}"
TMPDIR="${TMPDIR:-${OPENPI_RUNTIME_ROOT}/tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OPENPI_RUNTIME_ROOT}/xdg-cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${OPENPI_RUNTIME_ROOT}/uv-cache}"

mkdir -p "${OPENPI_ROOT}/logs"
mkdir -p "${OPENPI_DATA_HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" "${OPENPI_RUNTIME_ROOT}/assets"

cd "${OPENPI_RUNTIME_ROOT}"

export OPENPI_VIDEO_BACKEND
export OPENPI_DATA_HOME
export TMPDIR
export XDG_CACHE_HOME
export UV_CACHE_DIR
export PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" "${OPENPI_ROOT}/scripts/compute_norm_stats.py" --config-name "${CONFIG_NAME}"

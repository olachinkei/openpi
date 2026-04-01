#!/bin/bash
#SBATCH --job-name=openpi_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-$PWD}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

cd "${OPENPI_ROOT}"
mkdir -p logs

nvidia-smi
"${UV_BIN}" run python -c "import jax; print(jax.devices())"
"${UV_BIN}" run scripts/train.py debug --exp-name="cw-smoke-${SLURM_JOB_ID}" --overwrite --num-workers=0

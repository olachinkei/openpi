#!/bin/bash
#SBATCH --job-name=openpi_aloha_demo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-$PWD}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
PORT="${PORT:-8000}"
VIDEO_ROOT="${VIDEO_ROOT:-$OPENPI_ROOT/data/aloha_sim/videos/demo-${SLURM_JOB_ID}}"
VENDOR_GL_LIBDIR="${VENDOR_GL_LIBDIR:-$HOME/vendor-gl/root/usr/lib/x86_64-linux-gnu}"

cd "${OPENPI_ROOT}"
mkdir -p logs
mkdir -p "${VIDEO_ROOT}"

if [[ -d "${VENDOR_GL_LIBDIR}" ]]; then
  export LD_LIBRARY_PATH="${VENDOR_GL_LIBDIR}:${LD_LIBRARY_PATH:-}"
fi

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
if [[ "${MUJOCO_GL}" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[demo] starting baseline policy server on port ${PORT}"
"${UV_BIN}" run scripts/serve_policy.py --env ALOHA_SIM --port "${PORT}" &
SERVER_PID=$!

echo "[demo] waiting for policy server"
server_ready=0
for _ in $(seq 1 30); do
  if python3 -c "import socket; s=socket.socket(); print(s.connect_ex(('127.0.0.1', ${PORT})))" 2>/dev/null | grep -q '^0$'; then
    server_ready=1
    break
  fi
  sleep 2
done

if [[ "${server_ready}" -ne 1 ]]; then
  echo "[demo] policy server did not become ready on port ${PORT}" >&2
  exit 1
fi

echo "[demo] running ALOHA Sim rollout"
"${UV_BIN}" run python examples/aloha_sim/main.py \
  --args.out-dir "${VIDEO_ROOT}" \
  --args.host 127.0.0.1 \
  --args.port "${PORT}"

echo "[demo] rollout videos saved under ${VIDEO_ROOT}"

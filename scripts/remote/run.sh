#!/usr/bin/env bash
# Launch the 2-process Go2 closed-loop MPPI visualisation on the DGX Spark:
#   Process B = mppi_server.py  (JAX venv, GPU, numpy 2.x)
#   Process A = simulate_go2_remote.py  (Isaac python, viewport, numpy 1.26)
# They talk over a localhost socket (see ipc.py). Required because the numpy ABI
# wall prevents JAX (numpy>=2) and IsaacLab (numpy<2) from sharing one Python.
#
# Usage:  ./run.sh [--headless] [--video --video_length N] [hydra overrides]
#   No display? record offscreen, e.g.:
#     ./run.sh --headless --video --video_length 300
#   Video is written to <cwd>/videos/rl-video-step-0.mp4 (override --video_dir).
# Env:    PORT (default 5599)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMDIST="$(cd "$HERE/../.." && pwd)"

JAX_PY="${JAX_PY:-/shared/giacomo/jax_spark_test/bin/python}"
ISAAC_PY="${ISAAC_PY:-/shared/giacomo/isaac/IsaacSim/_build/linux-aarch64/release/python.sh}"
PORT="${PORT:-5599}"

echo "[run] starting MPPI server (Process B, JAX/GPU) on port $PORT ..."
"$JAX_PY" "$HERE/mppi_server.py" --port "$PORT" --simdist-dir "$SIMDIST" &
SERVER_PID=$!
trap 'echo "[run] stopping server ($SERVER_PID)"; kill "$SERVER_PID" 2>/dev/null || true' EXIT

# wait for the server to bind
for _ in $(seq 1 30); do
  if (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then break; fi
  sleep 1
done

echo "[run] starting Isaac sim (Process A, viewport) ..."
LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}" \
  "$ISAAC_PY" "$HERE/simulate_go2_remote.py" --mppi-port "$PORT" "$@"

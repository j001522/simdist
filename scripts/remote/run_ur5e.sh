#!/usr/bin/env bash
# Launch the 2-process UR5e closed-loop MPPI run on the DGX Spark:
#   Process B = mppi_server.py           (JAX venv, GPU, numpy 2.x)
#   Process A = simulate_ur5e_remote.py  (Isaac python, cameras, numpy 1.26)
# They talk over a localhost socket (see ipc.py). Required because the numpy ABI
# wall prevents JAX (numpy>=2) and IsaacLab (numpy<2) from sharing one Python.
#
# Usage:  ./run_ur5e.sh model.checkpoint=wm_manip_5hz_ln [--headless] [hydra overrides]
#   headless (no display):
#     ./run_ur5e.sh model.checkpoint=wm_manip_5hz_ln --headless
#   one episode, watch it:
#     ./run_ur5e.sh model.checkpoint=wm_manip_5hz_ln sim.num_episodes=1
#   record offscreen:
#     ./run_ur5e.sh model.checkpoint=wm_manip_5hz_ln --headless --video --video_length 400
# Env:    PORT (default 5599)
#
# Cameras are always enabled (the world model is vision-based), so unlike the Go2
# runner this costs render time even headless.
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

echo "[run] starting Isaac sim (Process A, cameras) ..."
LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}" \
  "$ISAAC_PY" "$HERE/simulate_ur5e_remote.py" --mppi-port "$PORT" "$@"

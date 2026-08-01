#!/usr/bin/env bash
# Evaluate EVERY checkpoint step of a world-model run under closed-loop MPPI and write
# a report next to the checkpoints.
#
# Same 2-process split as run_ur5e.sh (see there for why), except both processes stay up
# for the whole sweep: the Isaac side sends a "build" per checkpoint step, so Isaac boots
# once instead of once per step.
#
# Usage:  ./sweep_ur5e.sh <checkpoint-run-dir-or-name> [hydra overrides / flags]
#   ./sweep_ur5e.sh /shared/giacomo/simdist/checkpoints/models/wm_resnet_l64_25083707
#   ./sweep_ur5e.sh wm_resnet_l64_25083707 --sweep-episodes 10
#   ./sweep_ur5e.sh wm_dino_l256_25083733 --sweep-steps 38000,40000,42000
#
# Writes <ckpt_dir>/mppi_sweep.json and <ckpt_dir>/mppi_sweep.png. The JSON is rewritten
# after every checkpoint, so killing a long sweep still leaves a usable partial report.
#
# Env:  PORT (default 5599), EPISODES (default 5)
#
# Always headless and never records video -- a sweep is a measurement, not a demo.
set -euo pipefail

if [ $# -lt 1 ]; then
  sed -n '2,18p' "$0"; exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMDIST="$(cd "$HERE/../.." && pwd)"

CKPT_ARG="$1"; shift
# Accept either a full path or a bare run name; hydra wants the bare name.
CKPT_NAME="$(basename "${CKPT_ARG%/}")"
CKPT_DIR="$SIMDIST/checkpoints/models/$CKPT_NAME"
[ -d "$CKPT_DIR" ] || { echo "no such checkpoint dir: $CKPT_DIR" >&2; exit 1; }

JAX_PY="${JAX_PY:-/shared/giacomo/jax_spark_test/bin/python}"
ISAAC_PY="${ISAAC_PY:-/shared/giacomo/isaac/IsaacSim/_build/linux-aarch64/release/python.sh}"
PORT="${PORT:-5599}"
EPISODES="${EPISODES:-5}"

N_STEPS=$(find "$CKPT_DIR" -maxdepth 1 -mindepth 1 -type d -regex '.*/[0-9]+' | wc -l)
echo "[sweep] $CKPT_NAME: $N_STEPS checkpoints x $EPISODES episodes"

echo "[sweep] starting MPPI server (Process B, JAX/GPU) on port $PORT ..."
# No --ckpt-step here on purpose: the step travels with each build message instead.
"$JAX_PY" "$HERE/mppi_server.py" --port "$PORT" --simdist-dir "$SIMDIST" &
SERVER_PID=$!
trap 'echo "[sweep] stopping server ($SERVER_PID)"; kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  if (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then break; fi
  sleep 1
done

echo "[sweep] starting Isaac sim (Process A, cameras) ..."
LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}" \
  "$ISAAC_PY" "$HERE/simulate_ur5e_remote.py" \
    --mppi-port "$PORT" --headless \
    --sweep --sweep-episodes "$EPISODES" \
    "model.checkpoint=$CKPT_NAME" "$@"

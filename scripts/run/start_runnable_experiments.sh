#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
RUNNER="/home/gwx/miniconda3/envs/sc-diff/bin/python"
LAUNCHER="$ROOT_DIR/scripts/run/run_experiment_config.py"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/outputs/logs/runnable_queue_${STAMP}"

mkdir -p "$LOG_DIR"

run_config() {
  local name="$1"
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${name}" | tee -a "$LOG_DIR/queue.log"
  "$RUNNER" "$LAUNCHER" "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  local cmd_status=${PIPESTATUS[0]}
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END ${name} exit=${cmd_status}" | tee -a "$LOG_DIR/queue.log"
  return "${cmd_status}"
}

run_config cifar10_partial \
  --experiment-config "$ROOT_DIR/configs/experiments/cifar10_partial.yaml" \
  --execute \
  --materialize-schedules

run_config cifar10_mainline \
  --experiment-config "$ROOT_DIR/configs/experiments/cifar10_mainline.yaml" \
  --execute \
  --materialize-schedules

run_config modern_diffusers_practical \
  --experiment-config "$ROOT_DIR/configs/experiments/modern_diffusers_practical.yaml" \
  --execute \
  --materialize-schedules

echo "$LOG_DIR" > "$ROOT_DIR/outputs/logs/latest_runnable_queue_path.txt"

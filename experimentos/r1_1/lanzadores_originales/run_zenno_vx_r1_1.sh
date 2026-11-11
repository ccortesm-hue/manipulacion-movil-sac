#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/claudio/Documentos/Zenno_Ambient_v1/Zenno_v1/bin/python}"
export MUJOCO_GL=egl MPLCONFIGDIR=/tmp/vx_r11 TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
cd "$HERE"
MODE="${1:-diagnose}"
if [[ $# -gt 0 ]]; then shift; fi
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
case "$MODE" in
  diagnose)
    exec "$PYTHON_BIN" -u evaluate_zenno_vx_r1_1.py \
      --output "data_vx/r1_1/horizon_${RUN_TAG}" --device cuda \
      --cases-json cases_vx_r1_1_horizon.json --budgets 3000 4500 6000 "$@"
    ;;
  train)
    exec "$PYTHON_BIN" -u train_zenno_vx_r1_1.py \
      --output "models/zenno_vx_r1_1_${RUN_TAG}" --device cuda "$@"
    ;;
  *) echo "Uso: bash run_zenno_vx_r1_1.sh {diagnose|train} [opciones]" >&2; exit 2 ;;
esac

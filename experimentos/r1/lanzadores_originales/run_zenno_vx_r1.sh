#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/claudio/Documentos/Zenno_Ambient_v1/Zenno_v1/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export MUJOCO_GL=egl MPLCONFIGDIR=/tmp/vx_r1 TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
cd "$HERE"
exec "$PYTHON_BIN" -u train_zenno_vx_r1.py --output "models/zenno_vx_r1_${RUN_TAG}" "$@"

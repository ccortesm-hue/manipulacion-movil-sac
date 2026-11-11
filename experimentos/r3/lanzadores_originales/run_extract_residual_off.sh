#!/usr/bin/env bash
set -euo pipefail
ANALYSIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ZENNO_PYTHON:-/home/claudio/Documentos/Zenno_Ambient_v1/Zenno_v1/bin/python}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec "$PYTHON_BIN" -u "$ANALYSIS_DIR/EXTRACT_RESIDUAL_OFF_R11_R3.py" "$@"

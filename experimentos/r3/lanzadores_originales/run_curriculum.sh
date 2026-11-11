#!/usr/bin/env bash
set -euo pipefail
R3_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ZENNO_PYTHON:-/home/claudio/Documentos/Zenno_Ambient_v1/Zenno_v1/bin/python}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
exec "$PYTHON_BIN" -u "$R3_DIR/TRAIN_R3.py" "$@"

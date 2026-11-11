#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${VX_PYTHON:-/home/claudio/Documentos/Zenno_Ambient_v1/Zenno_v1/bin/python}"
run_dir="$experiment_dir/runs/r21_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$experiment_dir/runs"
export MUJOCO_GL=egl MPLCONFIGDIR=/tmp/vxr21_mpl TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$python_bin" -u "$experiment_dir/TRAIN_R21.py" --output "$run_dir" "$@" 2>&1 | tee "$run_dir.launcher.log"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${STUDY_PYTHON:-/usr/bin/python3}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs
"$PYTHON" study.py --stage tune --device cuda:0
"$PYTHON" study.py --stage select
"$PYTHON" study.py --stage main --device cuda:0
"$PYTHON" baselines.py
"$PYTHON" analyze.py --diagnostics
"$PYTHON" analyze.py
printf 'Pipeline completed\n'

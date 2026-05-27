#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/ODIR/ODIR_BankCausal.py}
WORK_DIR=${2:-work_dirs/bankcausal}

python tools/Train.py "$CONFIG" --work-dir "$WORK_DIR" "${@:3}"

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash scripts/test_bankcausal.sh CONFIG CHECKPOINT [WORK_DIR] [extra args...]"
    exit 1
fi

CONFIG=$1
CHECKPOINT=$2
WORK_DIR=work_dirs/bankcausal_eval

shift 2
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    WORK_DIR=$1
    shift 1
fi

python tools/Test.py "$CONFIG" "$CHECKPOINT" --work-dir "$WORK_DIR" "$@"

#!/usr/bin/env bash
set -euo pipefail

MASTER_PORT=24999
TRAIN_SCRIPT="training/train_edit.py"
CONFIG_FILE="./configs/RGenie_tuning_for_editing.yaml"
EXP_NAME="showo-edit"

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "deepspeed is not available. Activate your R-Genie environment first." >&2
  exit 1
fi

deepspeed --master_port=$MASTER_PORT $TRAIN_SCRIPT config=$CONFIG_FILE exp_name=$EXP_NAME

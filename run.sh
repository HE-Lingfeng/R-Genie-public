#!/bin/bash

MASTER_PORT=24999

TRAIN_SCRIPT="training/train_edit.py"

CONFIG_FILE="./configs/RGenie_tuning_for_editing.yaml"

EXP_NAME="showo-edit"

# 运行 deepspeed 命令
# source /root/miniconda3/envs/RIE/bin/activate
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RIE
deepspeed --master_port=$MASTER_PORT $TRAIN_SCRIPT config=$CONFIG_FILE exp_name=$EXP_NAME

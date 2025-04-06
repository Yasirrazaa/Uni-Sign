#!/bin/bash

# Directory for outputs
output_dir="outputs/wlasl_finetuning"
mkdir -p $output_dir

# Choose checkpoint type
checkpoint_type=${1:-"rgb"}  # Options: rgb or pose
use_future_mask=${2:-"false"}  # Options: true or false

if [ "$checkpoint_type" = "rgb" ]; then
    echo "Using RGB-Pose checkpoint"
    ckpt_path="wlasl_rgb_pose_islr.pth"
    rgb_flag="--rgb_support"
else
    echo "Using Pose-only checkpoint"
    ckpt_path="wlasl_pose_only_islr.pth"
    rgb_flag=""
fi

# Future masking flag
if [ "$use_future_mask" = "true" ]; then
    echo "Using future masking"
    mask_flag="--use_future_mask"
    # Add future masking to output directory name
    output_dir="${output_dir}_with_future_mask"
    mkdir -p $output_dir
else
    mask_flag=""
fi

# Training command - using fine_tuning.py instead of finetune_wlasl.py
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
    --batch-size 8 \
    --gradient-accumulation-steps 1 \
    --epochs 20 \
    --opt AdamW \
    --lr 5e-5 \
    --warmup-epochs 2 \
    --output_dir $output_dir \
    --finetune $ckpt_path \
    --dataset WLASL \
    --task ISLR \
    $rgb_flag \
    $mask_flag

# Usage:
# For RGB-pose model without future masking: ./train_wlasl.sh rgb false
# For RGB-pose model with future masking: ./train_wlasl.sh rgb true
# For Pose-only model without future masking: ./train_wlasl.sh pose false
# For Pose-only model with future masking: ./train_wlasl.sh pose true

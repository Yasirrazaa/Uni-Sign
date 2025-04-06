#!/bin/bash

# Directory for outputs
output_dir="outputs/wlasl_improved"
mkdir -p $output_dir

# Choose checkpoint type
checkpoint_type=${1:-"rgb"}  # Options: rgb or pose

if [ "$checkpoint_type" = "rgb" ]; then
    echo "Using RGB-Pose checkpoint"
    ckpt_path="wlasl_rgb_pose_islr.pth"
    rgb_flag="--rgb_support"
else
    echo "Using Pose-only checkpoint"
    ckpt_path="wlasl_pose_only_islr.pth"
    rgb_flag=""
fi

# Set number of classes for WLASL dataset
num_classes=2000

# Always use future masking for improved temporal understanding
mask_flag="--use_future_mask"

# Enable mixup and focal loss
mixup_flag="--use_mixup"
focal_loss_flag="--use_focal_loss"

# Training command with improved hyperparameters
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    --epochs 30 \
    --opt AdamW \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --warmup-epochs 3 \
    --dropout 0.3 \
    --label-smoothing 0.1 \
    --clip-grad 1.0 \
    --output_dir $output_dir \
    --finetune $ckpt_path \
    --dataset WLASL \
    --task ISLR \
    --max_length 64 \
    --dropout 0.3 \
    --num_classes $num_classes \
    $rgb_flag \
    $mask_flag \
    $mixup_flag \
    $focal_loss_flag

# Usage:
# For RGB-pose model: ./train_wlasl_improved.sh rgb
# For Pose-only model: ./train_wlasl_improved.sh pose

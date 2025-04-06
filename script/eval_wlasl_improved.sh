#!/bin/bash
# Path to model checkpoint
ckpt_path=${1:-"outputs/wlasl_improved/best_checkpoint.pth"}

# Choose checkpoint type
checkpoint_type=${2:-"rgb"}  # Options: rgb or pose

# Set number of classes for WLASL dataset
num_classes=2000

# Always use future masking for improved models
use_future_mask="true"

# Set up output directory
if [ "$checkpoint_type" = "rgb" ]; then
    echo "Evaluating RGB-Pose improved model"
    rgb_flag="--rgb_support"
    output_dir="out/wlasl_improved_rgb_pose_eval"
else
    echo "Evaluating Pose-only improved model"
    rgb_flag=""
    output_dir="out/wlasl_improved_pose_only_eval"
fi

# Set up future masking flag
mask_flag="--use_future_mask"

# Single GPU inference for improved model
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
   --batch-size 16 \
   --gradient-accumulation-steps 1 \
   --epochs 30 \
   --opt AdamW \
   --lr 1e-4 \
   --weight-decay 0.05 \
   --dropout 0.3 \
   --output_dir $output_dir \
   --finetune $ckpt_path \
   --dataset WLASL \
   --task ISLR \
   --max_length 64 \
   --num_classes $num_classes \
   --eval \
   $rgb_flag \
   $mask_flag

# Usage:
# Evaluate RGB-Pose model: ./script/eval_wlasl_improved.sh path/to/checkpoint.pth rgb
# Evaluate Pose-only model: ./script/eval_wlasl_improved.sh path/to/checkpoint.pth pose

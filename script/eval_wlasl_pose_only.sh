#!/bin/bash
# Path to model checkpoint
ckpt_path=${1:-"path/to/wlasl_pose_only_islr.pth"}

# Check if we should use future masking (for models trained with it)
use_future_mask=${2:-"false"}

# Set up future masking flag
if [ "$use_future_mask" = "true" ]; then
    echo "Evaluating with future masking"
    mask_flag="--use_future_mask"
    output_dir="out/wlasl_eval_pose_only_with_future_mask"
else
    mask_flag=""
    output_dir="out/wlasl_eval_pose_only"
fi

# Single GPU inference for Pose-only model
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
   --batch-size 8 \
   --gradient-accumulation-steps 1 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --output_dir $output_dir \
   --finetune $ckpt_path \
   --dataset WLASL \
   --task ISLR \
   --eval \
   $mask_flag

# Usage:
# Evaluate without future masking: ./script/eval_wlasl_pose_only.sh path/to/checkpoint.pth false
# Evaluate with future masking: ./script/eval_wlasl_pose_only.sh path/to/checkpoint.pth true
#!/bin/bash
# Path to model checkpoint
ckpt_path=${1:-"path/to/wlasl_pose_only_islr.pth"}

# Check if we should use future masking (for models trained with it)
use_future_mask=${2:-"false"}

# Set number of classes for WLASL dataset
num_classes=2000

# Set vocabulary path
vocab_path="data/WLASL/gloss_vocab.json"

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
   --vocab_path $vocab_path \
   --eval \
   $mask_flag

# Usage:
# Evaluate without future masking: ./script/eval_wlasl_pose_only.sh path/to/checkpoint.pth false
# Evaluate with future masking: ./script/eval_wlasl_pose_only.sh path/to/checkpoint.pth true
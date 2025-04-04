#!/bin/bash -l
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --time=20:00:00
#SBATCH --account=iwi5
#SBATCH --qos=normal
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=wlasltrain
#SBATCH --output=slurm%j.out
#SBATCH --error=slurm_%j.err

unset SLURM_EXPORT_ENV

# Load required modules
module purge
module load cuda
module load cudnn
module load python

# Set CUDA environment variables
export CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set PyTorch cache directory to a writable location
export TORCH_HOME=/home/woody/iwi5/iwi5286h/.cache/torch
mkdir -p $TORCH_HOME

# Force PyTorch to use CUDA
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Activate conda environment
conda activate unisign

# Set cache directories to writable locations
export TORCH_HOME=/home/woody/iwi5/iwi5286h/.cache/torch
export HF_HOME=/home/woody/iwi5/iwi5286h/.cache/huggingface
mkdir -p $TORCH_HOME $HF_HOME

# Print debug information
nvidia-smi
echo "Python Path: $(which python)"
echo "CUDA Path: $(which nvcc)"
echo "CUDA Version: $(nvcc --version)"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "TORCH_HOME: $TORCH_HOME"
echo "HF_HOME: $HF_HOME"
echo "TORCH_HOME: $TORCH_HOME"

# Set NCCL environment variables (adjust if needed, eth0 is common)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# Let Slurm handle CPU binding if necessary, removing explicit setting
# export SLURM_CPU_BIND="cores"

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
    output_dir="outputs/wlasl_finetuning_with_future_mask"
else
    mask_flag=""
    output_dir="outputs/wlasl_finetuning"
fi

# Create output directory
mkdir -p $output_dir

# Launch with DeepSpeed (remove srun and --master_port)
# DeepSpeed will use Slurm environment variables to find nodes/GPUs
deepspeed \
    --num_gpus 2 \
    fine_tuning.py \
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
    --max_length 64 \
    --clip-grad 1.0 \
    $rgb_flag \
    $mask_flag

# Usage:
# sbatch train_wlasl_slurm.sh rgb false  # For RGB-pose model without future masking
# sbatch train_wlasl_slurm.sh rgb true   # For RGB-pose model with future masking
# sbatch train_wlasl_slurm.sh pose false # For Pose-only model without future masking
# sbatch train_wlasl_slurm.sh pose true  # For Pose-only model with future masking

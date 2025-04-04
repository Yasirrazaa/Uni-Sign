#!/bin/bash

echo "Setting up environment for WLASL dataset preparation..."

# Check if conda environment exists and create if not
if ! conda env list | grep -q "Uni-Sign"; then
    echo "Creating Uni-Sign conda environment..."
    conda create -n Uni-Sign python=3.9 -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate Uni-Sign

# Install requirements
echo "Installing required packages..."
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install openmim
mim install mmpose

# Create directories
mkdir -p mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/
mkdir -p mmpose/checkpoints

# Download MMPose model and config
echo "Downloading MMPose model and configs..."
wget -nc https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth -P mmpose/checkpoints/
wget -nc https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py -P mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/

# Check if video directory exists
if [ ! -d "video" ]; then
    echo "Error: 'video' directory not found!"
    echo "Please create a 'video' directory containing:"
    echo "  - WLASL_v0.3.json"
    echo "  - All WLASL video files (*.mp4)"
    exit 1
fi

# Check if JSON file exists
if [ ! -f "video/WLASL_v0.3.json" ]; then
    echo "Error: WLASL_v0.3.json not found in video directory!"
    exit 1
fi

echo "Running dataset preparation script..."
python prepare_wlasl.py

echo "Setup complete! Check logs for any errors."

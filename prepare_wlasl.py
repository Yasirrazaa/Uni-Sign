"""
Script to prepare WLASL dataset structure and extract pose features
"""
import json
import shutil
import torch
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
import cv2
from mmpose.apis import inference_topdown, init_model
import mmcv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create required directory structure"""
    base_dir = Path('Uni-Sign/dataset/WLASL')
    
    # Create directories for RGB
    for split in ['train', 'val', 'test']:
        (base_dir / 'rgb_format' / split).mkdir(parents=True, exist_ok=True)
        (base_dir / 'pose_format' / split).mkdir(parents=True, exist_ok=True)
    
    return base_dir

def load_wlasl_data(json_path):
    """Load and process WLASL dataset JSON"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise ValueError("Invalid dataset format")
        logger.info("Loaded WLASL dataset")
        return data
    except Exception as e:
        logger.error(f"Error loading WLASL data: {e}")
        raise

def organize_videos(data, source_dir, base_dir):
    """Organize videos into train/val/test splits"""
    processed = 0
    errors = 0
    
    for entry in tqdm(data, desc="Organizing videos"):
        for instance in entry['instances']:
            video_id = instance['video_id']
            split = instance['split']
            
            # Source and destination paths
            src_path = source_dir / f"{video_id}.mp4"
            dst_path = base_dir / 'rgb_format' / split / f"{video_id}.mp4"
            
            if src_path.exists():
                try:
                    shutil.copy2(src_path, dst_path)
                    processed += 1
                except Exception as e:
                    logger.error(f"Error copying {video_id}: {e}")
                    errors += 1
            else:
                logger.warning(f"Video not found: {video_id}")
                errors += 1
                
    logger.info(f"Processed {processed} videos with {errors} errors")
    return processed, errors

def extract_pose_features(model, video_path, save_path):
    """Extract pose features using MMPose"""
    try:
        # Load video
        video = mmcv.VideoReader(str(video_path))
        frames = [frame for frame in video]
        
        # Extract keypoints for each frame
        keypoints = []
        scores = []
        
        for frame in frames:
            result = inference_topdown(model, frame)[0]
            kpts = result.pred_instances.keypoints[0].cpu().numpy()  # Shape: (K, 3)
            score = result.pred_instances.keypoint_scores[0].cpu().numpy()  # Shape: (K,)
            
            keypoints.append(kpts)
            scores.append(score)
        
        # Save features
        pose_data = {
            'keypoints': np.array(keypoints),
            'scores': np.array(scores)
        }
        
        torch.save(pose_data, save_path)
        return True
    except Exception as e:
        logger.error(f"Error processing {video_path}: {e}")
        return False

def main():
    # Setup directories
    base_dir = setup_directories()
    source_dir = Path('video')  # Directory containing all WLASL videos
    wlasl_path = source_dir / 'WLASL_v0.3.json'
    
    # Load dataset
    data = load_wlasl_data(wlasl_path)
    
    # Organize videos
    logger.info("Organizing videos into splits...")
    processed, errors = organize_videos(data, source_dir, base_dir)
    
    # Initialize MMPose model
    logger.info("Initializing MMPose model...")
    pose_config = 'mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
    pose_checkpoint = 'mmpose/checkpoints/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
    model = init_model(pose_config, pose_checkpoint, device='cuda:0')
    
    # Extract pose features
    logger.info("Extracting pose features...")
    for split in ['train', 'val', 'test']:
        split_dir = base_dir / 'rgb_format' / split
        pose_dir = base_dir / 'pose_format' / split
        
        for video_path in tqdm(list(split_dir.glob('*.mp4')), desc=f"Processing {split}"):
            save_path = pose_dir / f"{video_path.stem}.pkl"
            if not save_path.exists():
                success = extract_pose_features(model, video_path, save_path)
                if not success:
                    logger.error(f"Failed to process {video_path}")

    logger.info("Dataset preparation complete!")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Create a mapping from WLASL video IDs to class indices.
This script analyzes the WLASL dataset structure and creates a mapping
from video IDs to class indices based on the wlasl_class_list.txt file.
"""

import os
import json
import argparse
import gzip
import pickle
from pathlib import Path

def load_dataset_file(filename):
    """Load a gzipped pickle file containing dataset information."""
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def main():
    parser = argparse.ArgumentParser(description='Create WLASL video ID to class mapping')
    parser.add_argument('--train_path', default='data/WLASL/labels-2000.train', 
                        help='Path to training data')
    parser.add_argument('--val_path', default='data/WLASL/labels-2000.val', 
                        help='Path to validation data')
    parser.add_argument('--test_path', default='data/WLASL/labels-2000.test', 
                        help='Path to test data')
    parser.add_argument('--class_list', default='wlasl_class_list.txt', 
                        help='Path to class list file')
    parser.add_argument('--output_path', default='data/WLASL/video_to_class.json',
                        help='Path to save video ID to class mapping')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load class list
    print(f"Loading class list from {args.class_list}")
    class_to_idx = {}
    idx_to_class = {}
    
    with open(args.class_list, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Warning: Skipping invalid line: {line}")
                continue
                
            idx, gloss = parts
            idx = int(idx)
            class_to_idx[gloss] = idx
            idx_to_class[idx] = gloss
    
    print(f"Loaded {len(class_to_idx)} classes from class list")
    
    # Load dataset files
    video_to_class = {}
    
    for split, path in [('train', args.train_path), ('val', args.val_path), ('test', args.test_path)]:
        if not os.path.exists(path):
            print(f"Warning: {split} dataset file {path} not found, skipping")
            continue
            
        print(f"Loading {split} dataset from {path}")
        try:
            data = load_dataset_file(path)
            
            # Extract video IDs and their corresponding classes
            for key, sample in data.items():
                video_path = sample.get('video_path', '')
                video_id = os.path.splitext(os.path.basename(video_path))[0]
                
                # Try to get class from gloss field
                if 'gloss' in sample and sample['gloss']:
                    gloss = " ".join(sample['gloss'])
                    if gloss in class_to_idx:
                        video_to_class[video_id] = class_to_idx[gloss]
                        continue
                
                # If no gloss or gloss not in class list, try to infer from other information
                # For now, we'll just assign a default class (0)
                if video_id not in video_to_class:
                    video_to_class[video_id] = 0
                    print(f"Warning: Could not determine class for video {video_id}, defaulting to class 0")
        
        except Exception as e:
            print(f"Error loading {split} dataset: {e}")
    
    # Save mapping to file
    with open(args.output_path, 'w') as f:
        json.dump(video_to_class, f, indent=2)
    
    print(f"Created mapping for {len(video_to_class)} videos")
    print(f"Saved mapping to {args.output_path}")
    
    # Print sample of mapping
    print("\nSample of video ID to class mapping:")
    sample_items = list(video_to_class.items())[:10]
    for video_id, class_idx in sample_items:
        class_name = idx_to_class.get(class_idx, "Unknown")
        print(f"  {video_id}: Class {class_idx} ({class_name})")

if __name__ == '__main__':
    main()

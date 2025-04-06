#!/usr/bin/env python3
"""
Create a vocabulary mapping for WLASL dataset from the class list.
This script loads the WLASL class list and creates a mapping from gloss to index.
"""

import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Create WLASL vocabulary from class list')
    parser.add_argument('--class_list', default='wlasl_class_list.txt',
                        help='Path to class list file')
    parser.add_argument('--output_path', default='data/WLASL/gloss_vocab.json',
                        help='Path to save vocabulary')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load class list
    print(f"Loading class list from {args.class_list}")
    gloss_to_idx = {}
    idx_to_gloss = {}

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
            gloss_to_idx[gloss] = idx
            idx_to_gloss[idx] = gloss

    # Save vocabulary to file
    with open(args.output_path, 'w') as f:
        json.dump(gloss_to_idx, f, indent=2)

    print(f"Created vocabulary with {len(gloss_to_idx)} unique glosses")
    print(f"Saved vocabulary to {args.output_path}")

    # Print sample of vocabulary
    print("\nSample of vocabulary:")
    sample_items = list(gloss_to_idx.items())[:10]
    for gloss, idx in sample_items:
        print(f"  {gloss}: {idx}")

if __name__ == '__main__':
    main()

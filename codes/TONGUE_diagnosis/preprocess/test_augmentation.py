#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data augmentation test script
"""

import json
import os
from augment_tongue_dataset import TongueImageAugmenter

def create_test_dataset():
    """Create small test dataset"""
    test_data = [
        {
            "messages": [
                {
                    "content": "<image>根据图片判断舌诊内容",
                    "role": "user"
                },
                {
                    "content": "舌淡红，胖，边有齿印，苔薄白",
                    "role": "assistant"
                }
            ],
            "images": ["25.1.8之前所有with上中医三院/433-舌.jpg"]
        },
        {
            "messages": [
                {
                    "content": "<image>根据图片判断舌诊内容",
                    "role": "user"
                },
                {
                    "content": "舌暗红苔薄少",
                    "role": "assistant"
                }
            ],
            "images": ["25.1.8之前所有with上中医三院/145-舌.jpg"]
        }
    ]
    
    # Save test JSON file
    test_json_path = "/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/test_dataset.json"
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    return test_json_path

def main():
    # Create test dataset
    test_json_path = create_test_dataset()
    
    # Initialize augmenter (set augmentation factor to 2 for quick test)
    augmenter = TongueImageAugmenter(augmentation_factor=2)
    
    # Run test
    image_dir = "/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/25.1.8之前所有with上中医三院"
    output_dir = "/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/test_augmented"
    
    print("Starting test data augmentation...")
    augmenter.augment_dataset(test_json_path, image_dir, output_dir)
    print("Test completed!")

if __name__ == "__main__":
    main() 
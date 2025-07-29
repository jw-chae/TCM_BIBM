#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to find and check corrupted image files
"""
import os
import argparse
from PIL import Image
import json

def check_image_file(image_path):
    """Check if image file is corrupted"""
    try:
        with Image.open(image_path) as img:
            img.load()  # Try to load image
            img.verify()  # Verify image
        return True, None
    except Exception as e:
        return False, str(e)

def check_corrupted_images(image_dir, json_file=None):
    """Find and check corrupted image files"""
    
    print(f"Image directory: {image_dir}")
    
    # Image file list
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
    print(f"Total image files: {len(image_files)}")
    
    corrupted_files = []
    valid_files = []
    
    for i, filename in enumerate(image_files):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%)")
        
        image_path = os.path.join(image_dir, filename)
        is_valid, error = check_image_file(image_path)
        
        if is_valid:
            valid_files.append(filename)
        else:
            corrupted_files.append((filename, error))
    
    print(f"\n=== Inspection Results ===")
    print(f"Valid files: {len(valid_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\nCorrupted files (first 10):")
        for filename, error in corrupted_files[:10]:
            print(f"  {filename}: {error}")
    
    # Remove corrupted files from JSON if provided
    if json_file and corrupted_files:
        remove_corrupted_from_json(json_file, corrupted_files)
    
    return corrupted_files, valid_files

def remove_corrupted_from_json(json_file, corrupted_files):
    """Remove corrupted files from JSON"""
    
    print(f"\nRemoving corrupted files from JSON...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    corrupted_filenames = [f[0] for f in corrupted_files]
    original_count = len(data)
    
    # Remove data referencing corrupted files
    filtered_data = []
    for item in data:
        if 'images' in item and item['images']:
            # Extract filename from image path
            image_path = item['images'][0]
            filename = os.path.basename(image_path)
            
            if filename not in corrupted_filenames:
                filtered_data.append(item)
    
    # Save results
    output_file = json_file.replace('.json', '_cleaned.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"Original: {original_count} -> After cleaning: {len(filtered_data)}")
    print(f"Removed data: {original_count - len(filtered_data)}")
    print(f"Cleaned file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Find and check corrupted image files")
    parser.add_argument('--image_dir', type=str, required=True, help='Image directory')
    parser.add_argument('--json_file', type=str, help='JSON file (optional)')
    args = parser.parse_args()
    
    check_corrupted_images(args.image_dir, args.json_file)

if __name__ == "__main__":
    main() 
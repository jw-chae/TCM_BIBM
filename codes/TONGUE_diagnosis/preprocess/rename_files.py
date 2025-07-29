#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to rename files by replacing spaces with underscores
"""
import os
import argparse
import re

def rename_files_with_spaces(directory):
    """Rename files by replacing spaces with underscores"""
    
    print(f"Directory: {directory}")
    
    # Get file list
    files = os.listdir(directory)
    
    renamed_count = 0
    for filename in files:
        if ' ' in filename:  # Only files with spaces
            new_filename = filename.replace(' ', '_')
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"Error: Failed to rename {filename} - {e}")
    
    print(f"Total {renamed_count} files renamed")

def main():
    parser = argparse.ArgumentParser(description="Rename files by replacing spaces with underscores")
    parser.add_argument('--directory', type=str, required=True, help='Directory containing files')
    args = parser.parse_args()
    
    rename_files_with_spaces(args.directory)

if __name__ == "__main__":
    main() 
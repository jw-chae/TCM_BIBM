#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명의 공백을 언더스코어로 변경하는 스크립트
"""
import os
import argparse
import re

def rename_files_with_spaces(directory):
    """파일명의 공백을 언더스코어로 변경"""
    
    print(f"디렉토리: {directory}")
    
    # 파일 목록 가져오기
    files = os.listdir(directory)
    
    renamed_count = 0
    for filename in files:
        if ' ' in filename:  # 공백이 포함된 파일만
            new_filename = filename.replace(' ', '_')
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"변경: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"오류: {filename} 변경 실패 - {e}")
    
    print(f"총 {renamed_count}개 파일명 변경 완료")

def main():
    parser = argparse.ArgumentParser(description="파일명의 공백을 언더스코어로 변경")
    parser.add_argument('--directory', type=str, required=True, help='파일이 있는 디렉토리')
    args = parser.parse_args()
    
    rename_files_with_spaces(args.directory)

if __name__ == "__main__":
    main() 
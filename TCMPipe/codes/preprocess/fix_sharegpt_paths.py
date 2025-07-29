#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShareGPT 파일의 이미지 경로를 절대 경로로 수정하는 스크립트
"""
import json
import os
import argparse

def fix_sharegpt_paths(input_file, output_file, image_dir):
    """ShareGPT 파일의 이미지 경로를 절대 경로로 수정"""
    
    # 입력 파일 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 이미지 디렉토리의 절대 경로
    abs_image_dir = os.path.abspath(image_dir)
    
    print(f"입력 파일: {input_file}")
    print(f"이미지 디렉토리: {abs_image_dir}")
    print(f"데이터 개수: {len(data)}")
    
    # 경로 수정
    fixed_count = 0
    for item in data:
        if 'images' in item and item['images']:
            for i, image_path in enumerate(item['images']):
                # 상대 경로를 절대 경로로 변환
                if image_path.startswith('25.1.8之前所有with上中医三院/'):
                    filename = image_path.replace('25.1.8之前所有with上中医三院/', '')
                    abs_path = os.path.join(abs_image_dir, filename)
                    item['images'][i] = abs_path
                    fixed_count += 1
    
    # 수정된 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"수정된 이미지 경로: {fixed_count}개")
    print(f"결과 파일: {output_file}")
    
    # 샘플 확인
    print("\n수정된 경로 샘플 (처음 3개):")
    for i, item in enumerate(data[:3]):
        if 'images' in item and item['images']:
            print(f"{i+1}. {item['images'][0]}")

def main():
    parser = argparse.ArgumentParser(description="ShareGPT 파일의 이미지 경로를 절대 경로로 수정")
    parser.add_argument('--input_file', type=str, required=True, help='입력 ShareGPT 파일')
    parser.add_argument('--output_file', type=str, required=True, help='출력 ShareGPT 파일')
    parser.add_argument('--image_dir', type=str, required=True, help='이미지 디렉토리 경로')
    args = parser.parse_args()
    
    fix_sharegpt_paths(args.input_file, args.output_file, args.image_dir)

if __name__ == "__main__":
    main() 
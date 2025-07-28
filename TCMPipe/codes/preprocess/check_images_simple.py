#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단하고 정확한 이미지 검사 스크립트
"""
import os
import argparse
from PIL import Image
import json

def check_image_file_simple(image_path):
    """이미지 파일이 손상되었는지 간단히 확인"""
    try:
        with Image.open(image_path) as img:
            # 이미지 크기 확인
            width, height = img.size
            # 이미지 모드 확인
            mode = img.mode
            # 실제로 이미지 데이터 로드
            img.load()
        return True, None
    except Exception as e:
        return False, str(e)

def check_images_simple(image_dir, json_file=None):
    """손상된 이미지 파일들을 찾아서 검사 (간단한 방법)"""
    
    print(f"이미지 디렉토리: {image_dir}")
    
    # 이미지 파일 목록
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
    print(f"총 이미지 파일 개수: {len(image_files)}")
    
    corrupted_files = []
    valid_files = []
    
    for i, filename in enumerate(image_files):
        if i % 100 == 0:
            print(f"진행률: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%)")
        
        image_path = os.path.join(image_dir, filename)
        is_valid, error = check_image_file_simple(image_path)
        
        if is_valid:
            valid_files.append(filename)
        else:
            corrupted_files.append((filename, error))
    
    print(f"\n=== 검사 결과 ===")
    print(f"정상 파일: {len(valid_files)}개")
    print(f"손상 파일: {len(corrupted_files)}개")
    
    if corrupted_files:
        print(f"\n손상된 파일들 (처음 10개):")
        for filename, error in corrupted_files[:10]:
            print(f"  {filename}: {error}")
    
    # JSON 파일이 있다면 손상된 파일을 JSON에서 제거
    if json_file and corrupted_files:
        remove_corrupted_from_json_simple(json_file, corrupted_files)
    
    return corrupted_files, valid_files

def remove_corrupted_from_json_simple(json_file, corrupted_files):
    """손상된 파일들을 JSON에서 제거"""
    
    print(f"\nJSON 파일에서 손상된 파일들 제거 중...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    corrupted_filenames = [f[0] for f in corrupted_files]
    original_count = len(data)
    
    # 손상된 파일을 참조하는 데이터 제거
    filtered_data = []
    for item in data:
        if 'images' in item and item['images']:
            # 이미지 경로에서 파일명만 추출
            image_path = item['images'][0]
            filename = os.path.basename(image_path)
            
            if filename not in corrupted_filenames:
                filtered_data.append(item)
    
    # 결과 저장
    output_file = json_file.replace('.json', '_cleaned.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"원본: {original_count}개 -> 정리 후: {len(filtered_data)}개")
    print(f"제거된 데이터: {original_count - len(filtered_data)}개")
    print(f"정리된 파일: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="간단한 이미지 검사")
    parser.add_argument('--image_dir', type=str, required=True, help='이미지 디렉토리')
    parser.add_argument('--json_file', type=str, help='JSON 파일 (선택사항)')
    args = parser.parse_args()
    
    check_images_simple(args.image_dir, args.json_file)

if __name__ == "__main__":
    main() 
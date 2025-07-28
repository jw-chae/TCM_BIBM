#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON에 있지만 실제 이미지 파일이 없는 데이터를 추출하는 스크립트
"""
import json
import os
import argparse

def extract_missing_images(json_file, image_dir, output_file):
    """JSON에 있지만 실제 이미지 파일이 없는 데이터 추출"""
    
    # JSON 데이터 로드
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 실제 이미지 파일 목록
    existing_files = set(os.listdir(image_dir))
    
    print(f"JSON 데이터 개수: {len(data)}")
    print(f"이미지 폴더 파일 개수: {len(existing_files)}")
    
    # 이미지가 없는 데이터 추출
    missing_data = []
    for item in data:
        if item['image'] not in existing_files:
            missing_data.append(item)
    
    print(f"이미지가 없는 데이터 개수: {len(missing_data)}")
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(missing_data, f, ensure_ascii=False, indent=2)
    
    print(f"결과가 {output_file}에 저장되었습니다.")
    
    # 샘플 출력
    if missing_data:
        print("\n이미지가 없는 데이터 샘플 (처음 5개):")
        for i, item in enumerate(missing_data[:5]):
            print(f"{i+1}. {item['image']} - {item['output'][:50]}...")

def main():
    parser = argparse.ArgumentParser(description="이미지가 없는 JSON 데이터 추출")
    parser.add_argument('--json_file', type=str, required=True, help='원본 JSON 파일 경로')
    parser.add_argument('--image_dir', type=str, required=True, help='이미지 폴더 경로')
    parser.add_argument('--output_file', type=str, required=True, help='출력 JSON 파일 경로')
    args = parser.parse_args()
    
    extract_missing_images(args.json_file, args.image_dir, args.output_file)

if __name__ == "__main__":
    main() 
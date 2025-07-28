#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미지 매칭 문제 진단 스크립트
"""
import json
import os
import argparse

def check_image_matching(json_file, image_dir):
    """JSON의 이미지 이름과 실제 파일 이름 매칭 확인"""
    
    # JSON 데이터 로드
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 실제 이미지 파일 목록
    existing_files = set(os.listdir(image_dir))
    
    print(f"JSON 데이터 개수: {len(data)}")
    print(f"이미지 폴더 파일 개수: {len(existing_files)}")
    print()
    
    # JSON에서 사용하는 이미지 이름들
    json_images = set()
    for item in data:
        json_images.add(item['image'])
    
    print(f"JSON에서 참조하는 고유 이미지 이름 개수: {len(json_images)}")
    print()
    
    # 매칭 확인
    matched = 0
    unmatched_json = []
    unmatched_files = []
    
    for item in data:
        if item['image'] in existing_files:
            matched += 1
        else:
            unmatched_json.append(item['image'])
    
    # 실제 파일 중 JSON에 없는 것들
    for filename in existing_files:
        if filename not in json_images:
            unmatched_files.append(filename)
    
    print(f"매칭된 개수: {matched}")
    print(f"JSON에 있지만 파일이 없는 개수: {len(unmatched_json)}")
    print(f"파일은 있지만 JSON에 없는 개수: {len(unmatched_files)}")
    print()
    
    # 샘플 출력
    if unmatched_json:
        print("JSON에 있지만 파일이 없는 이미지 이름 (처음 10개):")
        for img in unmatched_json[:10]:
            print(f"  - {img}")
        print()
    
    if unmatched_files:
        print("파일은 있지만 JSON에 없는 이미지 이름 (처음 10개):")
        for img in unmatched_files[:10]:
            print(f"  - {img}")
        print()
    
    # JSON의 이미지 이름 패턴 분석
    print("JSON 이미지 이름 패턴 분석:")
    patterns = {}
    for img in json_images:
        ext = os.path.splitext(img)[1].lower()
        patterns[ext] = patterns.get(ext, 0) + 1
    
    for ext, count in patterns.items():
        print(f"  {ext}: {count}개")
    
    print()
    print("실제 파일 확장자 패턴 분석:")
    file_patterns = {}
    for img in existing_files:
        ext = os.path.splitext(img)[1].lower()
        file_patterns[ext] = file_patterns.get(ext, 0) + 1
    
    for ext, count in file_patterns.items():
        print(f"  {ext}: {count}개")

def main():
    parser = argparse.ArgumentParser(description="이미지 매칭 문제 진단")
    parser.add_argument('--json_file', type=str, required=True, help='JSON 파일 경로')
    parser.add_argument('--image_dir', type=str, required=True, help='이미지 폴더 경로')
    args = parser.parse_args()
    
    check_image_matching(args.json_file, args.image_dir)

if __name__ == "__main__":
    main() 
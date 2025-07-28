#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
매칭 결과를 검증하는 스크립트
"""
import json
import os
import re
from collections import Counter

def normalize_filename(filename):
    """파일명의 날짜 형식을 정규화"""
    pattern = r'(\d{4})-(\d{1,2})-(\d{1,2})'
    match = re.search(pattern, filename)
    if match:
        year = match.group(1)[2:]  # 2024 -> 24
        month = str(int(match.group(2)))  # 12 -> 12, 01 -> 1
        day = str(int(match.group(3)))    # 01 -> 1, 08 -> 8
        normalized = filename.replace(match.group(0), f"{year}.{month}.{day}")
        return normalized
    return filename

def verify_matching(json_file, image_dir):
    """매칭 결과를 검증"""
    
    # JSON 데이터 로드
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 실제 이미지 파일 목록
    existing_files = set(os.listdir(image_dir))
    
    print(f"JSON 데이터 개수: {len(data)}")
    print(f"이미지 폴더 파일 개수: {len(existing_files)}")
    print()
    
    # 매칭 결과 추적
    matched_files = set()  # 실제로 사용된 파일들
    unmatched_json = []
    unmatched_files = []
    
    for item in data:
        original_image = item['image']
        normalized_image = normalize_filename(original_image)
        no_space_image = original_image.replace(' ', '')
        no_space_normalized = normalized_image.replace(' ', '')
        
        # 매칭되는 파일 찾기
        matched_file = None
        if original_image in existing_files:
            matched_file = original_image
        elif normalized_image in existing_files:
            matched_file = normalized_image
        elif no_space_image in existing_files:
            matched_file = no_space_image
        elif no_space_normalized in existing_files:
            matched_file = no_space_normalized
        
        if matched_file:
            matched_files.add(matched_file)
        else:
            unmatched_json.append(original_image)
    
    # 실제 파일 중 매칭되지 않은 것들
    for filename in existing_files:
        if filename not in matched_files:
            unmatched_files.append(filename)
    
    print(f"매칭된 JSON 데이터 개수: {len(data) - len(unmatched_json)}")
    print(f"매칭 안된 JSON 데이터 개수: {len(unmatched_json)}")
    print(f"매칭된 이미지 파일 개수: {len(matched_files)}")
    print(f"매칭 안된 이미지 파일 개수: {len(unmatched_files)}")
    print()
    
    # 중복 매칭 확인
    json_images = [item['image'] for item in data]
    json_image_counter = Counter(json_images)
    duplicates = {img: count for img, count in json_image_counter.items() if count > 1}
    
    if duplicates:
        print(f"JSON에서 중복된 이미지 이름: {len(duplicates)}개")
        print("중복 예시 (처음 5개):")
        for img, count in list(duplicates.items())[:5]:
            print(f"  {img}: {count}회")
        print()
    
    # 매칭 안된 JSON 데이터 샘플
    if unmatched_json:
        print("매칭 안된 JSON 데이터 샘플 (처음 10개):")
        for i, img in enumerate(unmatched_json[:10]):
            print(f"  {i+1}. {img}")
        print()
    
    # 매칭 안된 이미지 파일 샘플
    if unmatched_files:
        print("매칭 안된 이미지 파일 샘플 (처음 10개):")
        for i, img in enumerate(unmatched_files[:10]):
            print(f"  {i+1}. {img}")
        print()
    
    # 매칭 통계
    print("=== 매칭 통계 ===")
    print(f"JSON 데이터: {len(data)}개")
    print(f"이미지 파일: {len(existing_files)}개")
    print(f"매칭된 JSON: {len(data) - len(unmatched_json)}개")
    print(f"매칭된 이미지: {len(matched_files)}개")
    print(f"매칭률: {(len(data) - len(unmatched_json)) / len(data) * 100:.1f}%")
    print(f"이미지 활용률: {len(matched_files) / len(existing_files) * 100:.1f}%")

if __name__ == "__main__":
    verify_matching("../../dataset/25.1.8之前所有with上中医三院.json", 
                   "../../dataset/25.1.8之前所有with上中医三院") 
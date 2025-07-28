#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
매칭 안되는 데이터들을 분석하는 스크립트
"""
import json
import os
import argparse
import re
from collections import Counter

def normalize_filename(filename):
    """파일명의 날짜 형식을 정규화"""
    pattern = r'(\d{4})-(\d{1,2})-(\d{1,2})'
    match = re.search(pattern, filename)
    if match:
        year = match.group(1)[2:]  # 2024 -> 24
        month = match.group(2)
        day = match.group(3)
        normalized = filename.replace(match.group(0), f"{year}.{month}.{day}")
        return normalized
    return filename

def analyze_unmatched_data(json_file, image_dir, output_file):
    """매칭 안되는 데이터들을 분석"""
    
    # JSON 데이터 로드
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 실제 이미지 파일 목록
    existing_files = set(os.listdir(image_dir))
    
    print(f"JSON 데이터 개수: {len(data)}")
    print(f"이미지 폴더 파일 개수: {len(existing_files)}")
    
    # 매칭 안되는 데이터 찾기
    unmatched_data = []
    
    for item in data:
        original_image = item['image']
        normalized_image = normalize_filename(original_image)
        
        # 공백 제거 버전도 시도
        no_space_image = original_image.replace(' ', '')
        no_space_normalized = normalized_image.replace(' ', '')
        
        if (original_image in existing_files or 
            normalized_image in existing_files or
            no_space_image in existing_files or
            no_space_normalized in existing_files):
            continue  # 매칭됨
        else:
            unmatched_data.append(item)
    
    print(f"매칭 안되는 데이터 개수: {len(unmatched_data)}")
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unmatched_data, f, ensure_ascii=False, indent=2)
    
    print(f"매칭 안되는 데이터가 {output_file}에 저장되었습니다.")
    
    # 패턴 분석
    print("\n=== 매칭 안되는 데이터 분석 ===")
    
    # 파일명 패턴 분석
    filename_patterns = []
    for item in unmatched_data:
        filename = item['image']
        # 날짜 패턴 추출
        date_pattern = re.search(r'(\d{4}-\d{1,2}-\d{1,2})', filename)
        if date_pattern:
            filename_patterns.append(f"날짜형식: {date_pattern.group(1)}")
        else:
            filename_patterns.append("날짜형식 없음")
    
    pattern_counter = Counter(filename_patterns)
    print("\n날짜 패턴 분석:")
    for pattern, count in pattern_counter.most_common():
        print(f"  {pattern}: {count}개")
    
    # 확장자 분석
    ext_counter = Counter()
    for item in unmatched_data:
        ext = os.path.splitext(item['image'])[1].lower()
        ext_counter[ext] += 1
    
    print("\n확장자 분석:")
    for ext, count in ext_counter.most_common():
        print(f"  {ext}: {count}개")
    
    # 샘플 출력
    print(f"\n매칭 안되는 데이터 샘플 (처음 20개):")
    for i, item in enumerate(unmatched_data[:20]):
        print(f"{i+1:2d}. {item['image']} - {item['output'][:50]}...")
    
    # 실제 파일과 유사한 이름이 있는지 확인
    print(f"\n=== 유사한 파일명 확인 (처음 10개) ===")
    for i, item in enumerate(unmatched_data[:10]):
        filename = item['image']
        # 파일명에서 날짜 부분 제거하고 이름만 추출
        name_part = re.sub(r'\d{4}-\d{1,2}-\d{1,2}\s*', '', filename)
        name_part = re.sub(r'\d+\.\d+\.\d+\s*', '', name_part)
        name_part = re.sub(r'\s+\d+\.(jpg|jpeg|png)', '', name_part)
        
        # 유사한 파일 찾기
        similar_files = []
        for existing_file in existing_files:
            if name_part in existing_file:
                similar_files.append(existing_file)
        
        if similar_files:
            print(f"{i+1}. {filename}")
            print(f"   유사한 파일들: {similar_files[:3]}")  # 최대 3개만 표시
        else:
            print(f"{i+1}. {filename} - 유사한 파일 없음")

def main():
    parser = argparse.ArgumentParser(description="매칭 안되는 데이터 분석")
    parser.add_argument('--json_file', type=str, required=True, help='원본 JSON 파일 경로')
    parser.add_argument('--image_dir', type=str, required=True, help='이미지 폴더 경로')
    parser.add_argument('--output_file', type=str, required=True, help='출력 JSON 파일 경로')
    args = parser.parse_args()
    
    analyze_unmatched_data(args.json_file, args.image_dir, args.output_file)

if __name__ == "__main__":
    main() 
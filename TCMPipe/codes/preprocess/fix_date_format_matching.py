#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
날짜 형식 차이로 인한 매칭 문제를 해결하는 스크립트
"""
import json
import os
import argparse
import re

def normalize_filename(filename):
    """파일명의 날짜 형식을 정규화"""
    # 2024-11-28 형식을 24.11.28 형식으로 변환
    pattern = r'(\d{4})-(\d{1,2})-(\d{1,2})'
    match = re.search(pattern, filename)
    if match:
        year = match.group(1)[2:]  # 2024 -> 24
        month = match.group(2)
        day = match.group(3)
        normalized = filename.replace(match.group(0), f"{year}.{month}.{day}")
        return normalized
    return filename

def find_matching_files(json_file, image_dir, output_file):
    """날짜 형식 차이를 고려하여 매칭되는 파일 찾기"""
    
    # JSON 데이터 로드
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 실제 이미지 파일 목록
    existing_files = set(os.listdir(image_dir))
    
    print(f"JSON 데이터 개수: {len(data)}")
    print(f"이미지 폴더 파일 개수: {len(existing_files)}")
    
    # 매칭 결과
    matched_data = []
    unmatched_data = []
    fixed_matches = []
    
    for item in data:
        original_image = item['image']
        normalized_image = normalize_filename(original_image)
        
        # 공백 제거 버전도 시도
        no_space_image = original_image.replace(' ', '')
        no_space_normalized = normalized_image.replace(' ', '')
        
        if original_image in existing_files:
            # 정확히 매칭
            matched_data.append(item)
        elif normalized_image in existing_files:
            # 날짜 형식 변환 후 매칭
            fixed_matches.append({
                'original': original_image,
                'normalized': normalized_image,
                'type': 'date_format'
            })
            item['image'] = normalized_image
            matched_data.append(item)
        elif no_space_image in existing_files:
            # 공백 제거 후 매칭
            fixed_matches.append({
                'original': original_image,
                'normalized': no_space_image,
                'type': 'no_space'
            })
            item['image'] = no_space_image
            matched_data.append(item)
        elif no_space_normalized in existing_files:
            # 날짜 형식 변환 + 공백 제거 후 매칭
            fixed_matches.append({
                'original': original_image,
                'normalized': no_space_normalized,
                'type': 'date_format_no_space'
            })
            item['image'] = no_space_normalized
            matched_data.append(item)
        else:
            # 여전히 매칭 안됨
            unmatched_data.append(item)
    
    print(f"정확히 매칭된 개수: {len(matched_data) - len(fixed_matches)}")
    print(f"날짜 형식 변환 후 매칭된 개수: {len([m for m in fixed_matches if m['type'] == 'date_format'])}")
    print(f"공백 제거 후 매칭된 개수: {len([m for m in fixed_matches if m['type'] == 'no_space'])}")
    print(f"날짜 형식 변환 + 공백 제거 후 매칭된 개수: {len([m for m in fixed_matches if m['type'] == 'date_format_no_space'])}")
    print(f"여전히 매칭 안된 개수: {len(unmatched_data)}")
    
    # 매칭된 데이터 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matched_data, f, ensure_ascii=False, indent=2)
    
    print(f"매칭된 데이터가 {output_file}에 저장되었습니다.")
    
    # 날짜 형식 변환 예시 출력
    if fixed_matches:
        print("\n매칭 수정 예시 (처음 10개):")
        for i, match in enumerate(fixed_matches[:10]):
            print(f"{i+1}. {match['original']} -> {match['normalized']} ({match['type']})")

def main():
    parser = argparse.ArgumentParser(description="날짜 형식 차이를 고려한 이미지 매칭")
    parser.add_argument('--json_file', type=str, required=True, help='원본 JSON 파일 경로')
    parser.add_argument('--image_dir', type=str, required=True, help='이미지 폴더 경로')
    parser.add_argument('--output_file', type=str, required=True, help='출력 JSON 파일 경로')
    args = parser.parse_args()
    
    find_matching_files(args.json_file, args.image_dir, args.output_file)

if __name__ == "__main__":
    main() 
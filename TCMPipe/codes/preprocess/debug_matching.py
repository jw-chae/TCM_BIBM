#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
매칭 로직 디버깅 스크립트
"""
import json
import os
import re

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

def debug_matching(json_file, image_dir):
    """매칭 로직 디버깅"""
    
    # JSON 데이터 로드
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 실제 이미지 파일 목록
    existing_files = set(os.listdir(image_dir))
    
    # 테스트할 파일들
    test_files = [
        "2024-12-01 姜冬梅 1733014964720.jpg",
        "2024-12-01 张栩珩 1733016857002.jpg",
        "2024-12-01 张语庭 1733022784235.jpg"
    ]
    
    print("=== 매칭 로직 디버깅 ===")
    
    for test_file in test_files:
        print(f"\n테스트 파일: {test_file}")
        
        # 1단계: 정규화
        normalized = normalize_filename(test_file)
        print(f"  정규화: {normalized}")
        
        # 2단계: 공백 제거
        no_space = test_file.replace(' ', '')
        no_space_normalized = normalized.replace(' ', '')
        print(f"  공백제거: {no_space}")
        print(f"  정규화+공백제거: {no_space_normalized}")
        
        # 3단계: 매칭 확인
        print(f"  원본 파일 존재: {test_file in existing_files}")
        print(f"  정규화 파일 존재: {normalized in existing_files}")
        print(f"  공백제거 파일 존재: {no_space in existing_files}")
        print(f"  정규화+공백제거 파일 존재: {no_space_normalized in existing_files}")
        
        # 4단계: 유사한 파일 찾기
        similar = []
        for existing in existing_files:
            if "姜冬梅" in existing or "张栩珩" in existing or "张语庭" in existing:
                if any(name in existing for name in ["姜冬梅", "张栩珩", "张语庭"]):
                    similar.append(existing)
        
        if similar:
            print(f"  유사한 파일들: {similar}")
        
        # 5단계: 실제 매칭 로직 테스트
        if (test_file in existing_files or 
            normalized in existing_files or
            no_space in existing_files or
            no_space_normalized in existing_files):
            print("  ✅ 매칭 성공!")
        else:
            print("  ❌ 매칭 실패!")

if __name__ == "__main__":
    debug_matching("../../dataset/25.1.8之前所有with上中医三院.json", 
                  "../../dataset/25.1.8之前所有with上中医三院") 
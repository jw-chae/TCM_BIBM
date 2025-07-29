#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
중복을 제거하고 이미지 활용률을 100%로 만드는 데이터셋 생성
"""
import json
import os
import argparse
import random
import re
from collections import defaultdict

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

def create_unique_dataset(json_file, image_dir, output_file, train_ratio=0.9, seed=42):
    """중복을 제거하고 이미지 활용률을 100%로 만드는 데이터셋 생성"""
    
    # JSON 데이터 로드
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 실제 이미지 파일 목록
    existing_files = set(os.listdir(image_dir))
    
    print(f"원본 JSON 데이터 개수: {len(data)}")
    print(f"이미지 폴더 파일 개수: {len(existing_files)}")
    
    # 이미지별로 JSON 데이터 그룹화
    image_to_jsons = defaultdict(list)
    
    for item in data:
        original_image = item['image']
        normalized_image = normalize_filename(original_image)
        no_space_image = original_image.replace(' ', '')
        no_space_normalized = normalized_image.replace(' ', '')
        
        # 매칭되는 실제 파일 찾기
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
            # 매칭된 파일명으로 업데이트
            item['image'] = matched_file
            image_to_jsons[matched_file].append(item)
    
    print(f"매칭된 이미지 개수: {len(image_to_jsons)}")
    
    # 각 이미지당 하나의 JSON 데이터만 선택 (중복 제거)
    unique_data = []
    for image_file, json_items in image_to_jsons.items():
        # 첫 번째 항목 선택 (또는 랜덤 선택)
        selected_item = json_items[0]
        unique_data.append(selected_item)
    
    print(f"중복 제거 후 데이터 개수: {len(unique_data)}")
    
    # 랜덤 셔플
    random.seed(seed)
    random.shuffle(unique_data)
    
    # train/val 분할
    n_total = len(unique_data)
    n_train = int(n_total * train_ratio)
    train_data = unique_data[:n_train]
    val_data = unique_data[n_train:]
    
    # 결과 저장
    result = {
        'train': train_data,
        'val': val_data,
        'total': n_total,
        'train_count': len(train_data),
        'val_count': len(val_data),
        'image_utilization': len(unique_data) / len(existing_files) * 100
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 최종 결과 ===")
    print(f"총 데이터: {n_total}개")
    print(f"Train set: {len(train_data)}개")
    print(f"Val set: {len(val_data)}개")
    print(f"이미지 활용률: {result['image_utilization']:.1f}%")
    print(f"결과가 {output_file}에 저장되었습니다.")
    
    # ShareGPT 형식으로도 저장
    train_sharegpt = []
    val_sharegpt = []
    
    for item in train_data:
        user_content = "<image>" + item['prompt'].replace('<image>', '').strip()
        messages = [
            {"content": user_content, "role": "user"},
            {"content": item['output'], "role": "assistant"}
        ]
        image_path = os.path.join(image_dir, os.path.basename(item['image']))
        rel_image_path = os.path.relpath(image_path, os.path.dirname(output_file))
        images = [rel_image_path]
        train_sharegpt.append({"messages": messages, "images": images})
    
    for item in val_data:
        user_content = "<image>" + item['prompt'].replace('<image>', '').strip()
        messages = [
            {"content": user_content, "role": "user"},
            {"content": item['output'], "role": "assistant"}
        ]
        image_path = os.path.join(image_dir, os.path.basename(item['image']))
        rel_image_path = os.path.relpath(image_path, os.path.dirname(output_file))
        images = [rel_image_path]
        val_sharegpt.append({"messages": messages, "images": images})
    
    # ShareGPT 형식 파일 저장
    base_name = os.path.splitext(output_file)[0]
    train_file = f"{base_name}_train_sharegpt.json"
    val_file = f"{base_name}_val_sharegpt.json"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_sharegpt, f, ensure_ascii=False, indent=2)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_sharegpt, f, ensure_ascii=False, indent=2)
    
    print(f"ShareGPT 형식 파일도 생성되었습니다:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")

def main():
    parser = argparse.ArgumentParser(description="중복 제거 및 이미지 활용률 100% 데이터셋 생성")
    parser.add_argument('--json_file', type=str, required=True, help='원본 JSON 파일 경로')
    parser.add_argument('--image_dir', type=str, required=True, help='이미지 폴더 경로')
    parser.add_argument('--output_file', type=str, required=True, help='출력 JSON 파일 경로')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='train 비율 (기본값: 0.9)')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드 (기본값: 42)')
    args = parser.parse_args()
    
    create_unique_dataset(args.json_file, args.image_dir, args.output_file, 
                         args.train_ratio, args.seed)

if __name__ == "__main__":
    main() 
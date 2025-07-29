#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL json 데이터를 8:2로 랜덤 분할하여 각각 ShareGPT/vllm 포맷(messages+images, 멀티턴 지원)으로 변환

jsonl이 아니라 전체를 리스트로 묶어 하나의 json 파일로 저장합니다.

사용법 예시:
python split_and_convert_to_qwen25vl_jsonl.py --input_json ../dataset/25.1.10-25.6.3.json --image_dir ../dataset/25.1.10-25.6.3 --output_train_json ../dataset/train.sharegpt.json --output_val_json ../dataset/val.sharegpt.json

- input_json: 원본 json 파일 경로
- image_dir: 이미지가 들어있는 폴더 경로
- output_train_json: train set json 파일 경로
- output_val_json: val set json 파일 경로

Qwen2.5-VL 포맷:
한 줄에 하나의 대화(chat) 리스트가 들어감
"""
import json
import argparse
import os
import random
import re

def convert_and_write(data, image_dir, output_json):
    out_list = []
    for item in data:
        messages = []
        images = []
        user_content = item['prompt']
        user_image_count = user_content.count('<image>')
        if user_image_count == 0:
            user_content = '<image>' + user_content
            user_image_count = 1
        image_path = os.path.join(image_dir, item['image'])
        rel_image_path = os.path.relpath(image_path, os.path.dirname(output_json))
        images.extend([rel_image_path] * user_image_count)
        messages.append({"content": user_content, "role": "user"})
        messages.append({"content": item['output'], "role": "assistant"})
        out_list.append({"messages": messages, "images": images})
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    print(f"{output_json} 파일이 생성되었습니다. (샘플 수: {len(data)})")

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL json 8:2 분할 및 ShareGPT/vllm 포맷 변환기 (json 리스트 버전)")
    parser.add_argument('--input_json', type=str, required=True, help='원본 json 파일 경로')
    parser.add_argument('--image_dir', type=str, required=True, help='이미지 폴더 경로')
    parser.add_argument('--output_train_json', type=str, required=True, help='train set json 파일 경로')
    parser.add_argument('--output_val_json', type=str, required=True, help='val set json 파일 경로')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드 (재현성)')
    args = parser.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)
    n_total = len(data)
    n_train = int(n_total * 0.8)
    train_data = data[:n_train]
    val_data = data[n_train:]

    convert_and_write(train_data, args.image_dir, args.output_train_json)
    convert_and_write(val_data, args.image_dir, args.output_val_json)
    print(f"총 {n_total}개 중 {len(train_data)}개는 train, {len(val_data)}개는 val로 분할 완료.")

if __name__ == "__main__":
    main() 
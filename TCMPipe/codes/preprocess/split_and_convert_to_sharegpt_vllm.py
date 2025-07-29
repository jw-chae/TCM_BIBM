#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL json 데이터를 8:2로 랜덤 분할하여 각각 ShareGPT/vllm 포맷(messages+images)으로 변환

사용법 예시:
python split_and_convert_to_sharegpt_vllm.py --input_json ../dataset/25.1.10-25.6.3.json --image_dir ../dataset/25.1.10-25.6.3 --output_train_jsonl ../dataset/train.sharegpt.jsonl --output_val_jsonl ../dataset/val.sharegpt.jsonl
"""
import json
import argparse
import os
import random

def convert_and_write(data, image_dir, output_jsonl):
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in data:
            user_content = "<image>" + item['prompt'].replace('<image>', '').strip()
            messages = [
                {"content": user_content, "role": "user"},
                {"content": item['output'], "role": "assistant"}
            ]
            image_path = os.path.join(image_dir, item['image'])
            rel_image_path = os.path.relpath(image_path, os.path.dirname(output_jsonl))
            images = [rel_image_path]
            out = {"messages": messages, "images": images}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"{output_jsonl} 파일이 생성되었습니다. (샘플 수: {len(data)})")

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL json 8:2 분할 및 ShareGPT/vllm 포맷 변환기")
    parser.add_argument('--input_json', type=str, required=True, help='원본 json 파일 경로')
    parser.add_argument('--image_dir', type=str, required=True, help='이미지 폴더 경로')
    parser.add_argument('--output_train_jsonl', type=str, required=True, help='train set jsonl 파일 경로')
    parser.add_argument('--output_val_jsonl', type=str, required=True, help='val set jsonl 파일 경로')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드 (재현성)')
    args = parser.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)
    n_total = len(data)
    n_train = int(n_total * 0.9)
    train_data = data[:n_train]
    val_data = data[n_train:]

    convert_and_write(train_data, args.image_dir, args.output_train_jsonl)
    convert_and_write(val_data, args.image_dir, args.output_val_jsonl)
    print(f"총 {n_total}개 중 {len(train_data)}개는 train, {len(val_data)}개는 val로 분할 완료.")

if __name__ == "__main__":
    main() 
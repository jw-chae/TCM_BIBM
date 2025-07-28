#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 증강 테스트 스크립트
"""

import json
import os
from augment_tongue_dataset import TongueImageAugmenter

def create_test_dataset():
    """테스트용 작은 데이터셋 생성"""
    test_data = [
        {
            "messages": [
                {
                    "content": "<image>根据图片判断舌诊内容",
                    "role": "user"
                },
                {
                    "content": "舌淡红，胖，边有齿印，苔薄白",
                    "role": "assistant"
                }
            ],
            "images": ["25.1.8之前所有with上中医三院/433-舌.jpg"]
        },
        {
            "messages": [
                {
                    "content": "<image>根据图片判断舌诊内容",
                    "role": "user"
                },
                {
                    "content": "舌暗红苔薄少",
                    "role": "assistant"
                }
            ],
            "images": ["25.1.8之前所有with上中医三院/145-舌.jpg"]
        }
    ]
    
    # 테스트 JSON 파일 저장
    test_json_path = "/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/test_dataset.json"
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    return test_json_path

def main():
    # 테스트 데이터셋 생성
    test_json_path = create_test_dataset()
    
    # 증강기 초기화 (증강 팩터를 2로 설정하여 빠른 테스트)
    augmenter = TongueImageAugmenter(augmentation_factor=2)
    
    # 테스트 실행
    image_dir = "/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/25.1.8之前所有with上中医三院"
    output_dir = "/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/test_augmented"
    
    print("테스트 데이터 증강 시작...")
    augmenter.augment_dataset(test_json_path, image_dir, output_dir)
    print("테스트 완료!")

if __name__ == "__main__":
    main() 
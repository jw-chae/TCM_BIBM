#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation2_ver2 import compute_category_similarity, load_json_config

def debug_sample_scores():
    """두 파일의 샘플 데이터 점수를 직접 비교합니다."""
    
    # 설정 파일 로드
    config = load_json_config('token_config.json')
    
    # 파일 경로
    file1 = "/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/shezhen_results/result_new_best.jsonl"
    file2 = "/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/shezhen_results/compare_alltrain_val64.jsonl"
    
    print("=== 샘플 데이터 점수 비교 ===\n")
    
    # 파일1에서 샘플 확인
    print("1. result_new_best.jsonl 샘플:")
    with open(file1, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # 처음 5개만
                break
            data = json.loads(line.strip())
            predict = data['predict']
            label = data['label']
            _, _, _, _, score = compute_category_similarity(predict, label, config)
            print(f"  예측: {predict}")
            print(f"  라벨: {label}")
            print(f"  점수: {score:.4f}")
            print()
    
    print("\n" + "="*50 + "\n")
    
    # 파일2에서 샘플 확인
    print("2. compare_alltrain_val64.jsonl 샘플:")
    with open(file2, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # 처음 5개만
                break
            data = json.loads(line.strip())
            predict = data['predict']
            label = data['label']
            _, _, _, _, score = compute_category_similarity(predict, label, config)
            print(f"  예측: {predict}")
            print(f"  라벨: {label}")
            print(f"  점수: {score:.4f}")
            print()
    
    # 전체 평균 점수 계산
    print("\n" + "="*50 + "\n")
    print("3. 전체 평균 점수 비교:")
    
    # 파일1 전체 평균
    scores1 = []
    with open(file1, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            predict = data['predict']
            label = data['label']
            _, _, _, _, score = compute_category_similarity(predict, label, config)
            scores1.append(score)
    
    avg1 = sum(scores1) / len(scores1)
    print(f"  result_new_best.jsonl: {avg1:.4f} ({len(scores1)}개)")
    
    # 파일2 전체 평균
    scores2 = []
    with open(file2, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            predict = data['predict']
            label = data['label']
            _, _, _, _, score = compute_category_similarity(predict, label, config)
            scores2.append(score)
    
    avg2 = sum(scores2) / len(scores2)
    print(f"  compare_alltrain_val64.jsonl: {avg2:.4f} ({len(scores2)}개)")
    
    print(f"  차이: {abs(avg1 - avg2):.4f}")

if __name__ == "__main__":
    debug_sample_scores() 
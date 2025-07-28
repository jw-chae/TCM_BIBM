#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

# evaluation2_ver2.py의 함수들을 import
sys.path.append(os.path.dirname(__file__))
from evaluation2_ver2 import load_json_config, compute_category_similarity

def load_jsonl(file_path):
    """JSONL 파일을 로드합니다."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def calculate_bleu_score(predictions, references):
    """BLEU 점수를 계산합니다 (Llama Factory 방식)."""
    smoothie = SmoothingFunction().method3
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = list(pred)
        ref_tokens = list(ref)
        try:
            bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu_score)
        except Exception as e:
            print(f"BLEU 계산 오류: {e}")
            bleu_scores.append(0.0)
    
    return np.mean(bleu_scores)

def calculate_our_metric(predictions, references, config):
    """우리 metric 점수를 계산합니다."""
    total_mean = 0.0
    for pred, ref in zip(predictions, references):
        pred_clean = pred.replace('舌诊结果: ', '').strip()
        ref_clean = ref.replace('舌诊结果: ', '').strip()
        _, _, _, _, s_mean = compute_category_similarity(pred_clean, ref_clean, config)
        total_mean += s_mean
    
    return total_mean / len(predictions) if predictions else 0.0

def main():
    input_dir = "../../shezhen_results"
    output_dir = "../../metric_results"
    config_path = "token_config.json"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # config 로드
    config = load_json_config(config_path)
    
    # 평가할 jsonl 파일들 (BLEU 결과 파일 제외)
    exclude_files = ['bleu_scores.jsonl', 'bleu_scores_char.jsonl', 'bleu_summary.txt', 'bleu_summary_char.txt']
    
    jsonl_files = []
    for f in os.listdir(input_dir):
        if f.endswith('.jsonl') and f not in exclude_files:
            jsonl_files.append(f)
    
    print("통합 결과 생성 중...")
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"평가할 파일 수: {len(jsonl_files)}")
    
    # 결과 저장용 딕셔너리
    results = {}
    
    for fname in sorted(jsonl_files):
        print(f"\n처리 중: {fname}")
        file_path = os.path.join(input_dir, fname)
        
        try:
            # 데이터 로드
            data = load_jsonl(file_path)
            if not data:
                print(f"  경고: {fname}에 데이터가 없습니다.")
                continue
            
            # predict와 label 분리
            predictions = [item.get('predict', '') for item in data]
            references = [item.get('label', '') for item in data]
            
            # BLEU 점수 계산
            bleu_score = calculate_bleu_score(predictions, references)
            
            # 우리 metric 점수 계산
            our_metric_score = calculate_our_metric(predictions, references, config)
            
            # 결과 저장
            results[fname] = {
                'our_metric': our_metric_score,
                'bleu': bleu_score,
                'sample_count': len(data)
            }
            
            print(f"  Our Metric: {our_metric_score:.4f}")
            print(f"  BLEU: {bleu_score:.4f}")
            print(f"  샘플 수: {len(data)}")
            
        except Exception as e:
            print(f"  오류: {fname} 처리 중 오류 발생 - {e}")
            continue
    
    # 통합 결과 파일 생성
    output_file = os.path.join(output_dir, "combined_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("통합 평가 결과\n")
        f.write("=" * 50 + "\n\n")
        
        for fname, result in sorted(results.items()):
            f.write(f"파일: {fname}\n")
            f.write(f"  Our Metric: {result['our_metric']:.4f}\n")
            f.write(f"  BLEU: {result['bleu']:.4f}\n")
            f.write(f"  샘플 수: {result['sample_count']}\n")
            f.write("-" * 30 + "\n")
    
    # 요약 파일 생성
    summary_file = os.path.join(output_dir, "results_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("평가 결과 요약\n")
        f.write("=" * 30 + "\n\n")
        
        # Our Metric 기준 정렬
        f.write("Our Metric 기준 (높은 순):\n")
        sorted_by_metric = sorted(results.items(), key=lambda x: x[1]['our_metric'], reverse=True)
        for fname, result in sorted_by_metric:
            f.write(f"{fname}: {result['our_metric']:.4f}\n")
        
        f.write("\nBLEU 기준 (높은 순):\n")
        sorted_by_bleu = sorted(results.items(), key=lambda x: x[1]['bleu'], reverse=True)
        for fname, result in sorted_by_bleu:
            f.write(f"{fname}: {result['bleu']:.4f}\n")
    
    print(f"\n통합 결과가 {output_file}에 저장되었습니다.")
    print(f"요약이 {summary_file}에 저장되었습니다.")

if __name__ == "__main__":
    main() 
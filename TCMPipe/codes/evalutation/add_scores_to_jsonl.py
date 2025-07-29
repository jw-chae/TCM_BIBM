#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

# evaluation2_ver2_enhanced.py의 함수들을 import
sys.path.append(os.path.dirname(__file__))
from evaluation2_ver2_enhanced import TongueConfig, TongueEvaluator

def calculate_bleu_score_single(prediction, reference):
    """단일 예측에 대한 BLEU 점수를 계산합니다."""
    smoothie = SmoothingFunction().method3
    pred_tokens = list(prediction)
    ref_tokens = list(reference)
    try:
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        return bleu_score
    except Exception as e:
        print(f"BLEU 계산 오류: {e}")
        return 0.0

def calculate_enhanced_metric_single(prediction, reference, evaluator):
    """단일 예측에 대한 Enhanced Metric 점수를 계산합니다."""
    pred_clean = prediction.replace('舌诊结果: ', '').strip()
    ref_clean = reference.replace('舌诊结果: ', '').strip()
    
    overall_score, category_scores = evaluator.evaluate(pred_clean, ref_clean)
    return overall_score, category_scores

def add_scores_to_jsonl(input_file, output_file, evaluator):
    """JSONL 파일의 각 행에 점수를 추가합니다."""
    print(f"처리 중: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    if not data:
        print(f"  경고: {input_file}에 데이터가 없습니다.")
        return
    
    enhanced_data = []
    total_enhanced_metric = 0.0
    total_bleu = 0.0
    category_totals = {'tongue': 0.0, 'coat': 0.0, 'location': 0.0, 'other': 0.0}
    
    for i, item in enumerate(data):
        prediction = item.get('predict', '')
        reference = item.get('label', '')
        
        # 점수 계산
        enhanced_score, category_scores = calculate_enhanced_metric_single(prediction, reference, evaluator)
        bleu_score = calculate_bleu_score_single(prediction, reference)
        
        # 원본 데이터에 점수 추가
        enhanced_item = item.copy()
        enhanced_item['enhanced_metric_score'] = round(enhanced_score, 4)
        enhanced_item['bleu_score'] = round(bleu_score, 4)
        enhanced_item['index'] = i
        
        # 카테고리별 점수 추가
        for cat, score in category_scores.items():
            if cat != 'overall':
                enhanced_item[f'{cat}_score'] = round(score, 4)
                category_totals[cat] += score
        
        enhanced_data.append(enhanced_item)
        
        total_enhanced_metric += enhanced_score
        total_bleu += bleu_score
    
    # 평균 점수 계산
    avg_enhanced_metric = total_enhanced_metric / len(data)
    avg_bleu = total_bleu / len(data)
    avg_category_scores = {cat: score / len(data) for cat, score in category_totals.items()}
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in enhanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"  완료: {output_file}")
    print(f"  평균 Enhanced Metric: {avg_enhanced_metric:.4f}")
    print(f"  평균 BLEU: {avg_bleu:.4f}")
    print(f"  카테고리별 평균:")
    for cat, score in avg_category_scores.items():
        print(f"    {cat}: {score:.4f}")
    print(f"  샘플 수: {len(data)}")
    
    return avg_enhanced_metric, avg_bleu, avg_category_scores, len(data)

def main():
    input_dir = "../../shezhen_results"
    output_dir = "../../metric_results"
    config_path = "token_config.json"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # config와 evaluator 로드
    config = TongueConfig(config_path)
    evaluator = TongueEvaluator(config)
    
    # 평가할 jsonl 파일들 (BLEU 결과 파일 제외)
    exclude_files = ['bleu_scores.jsonl', 'bleu_scores_char.jsonl']
    
    jsonl_files = []
    for f in os.listdir(input_dir):
        if f.endswith('.jsonl') and f not in exclude_files:
            jsonl_files.append(f)
    
    print("각 행에 Enhanced Metric 점수 추가 중...")
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"처리할 파일 수: {len(jsonl_files)}")
    
    # 결과 요약
    summary_results = {}
    
    for fname in sorted(jsonl_files):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname.replace('.jsonl', '_with_scores.jsonl'))
        
        try:
            avg_enhanced_metric, avg_bleu, avg_category_scores, sample_count = add_scores_to_jsonl(input_path, output_path, evaluator)
            
            summary_results[fname] = {
                'enhanced_metric': avg_enhanced_metric,
                'bleu': avg_bleu,
                'category_scores': avg_category_scores,
                'sample_count': sample_count
            }
            
        except Exception as e:
            print(f"  오류: {fname} 처리 중 오류 발생 - {e}")
            continue
    
    # 요약 파일 생성
    summary_file = os.path.join(output_dir, "enhanced_scores_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Enhanced Metric 점수 추가 완료 요약\n")
        f.write("=" * 50 + "\n\n")
        
        for fname, result in sorted(summary_results.items()):
            f.write(f"파일: {fname}\n")
            f.write(f"  평균 Enhanced Metric: {result['enhanced_metric']:.4f}\n")
            f.write(f"  평균 BLEU: {result['bleu']:.4f}\n")
            f.write(f"  카테고리별 평균:\n")
            for cat, score in result['category_scores'].items():
                f.write(f"    {cat}: {score:.4f}\n")
            f.write(f"  샘플 수: {result['sample_count']}\n")
            f.write("-" * 50 + "\n")
    
    print(f"\nEnhanced Metric 점수 추가 완료!")
    print(f"요약이 {summary_file}에 저장되었습니다.")

if __name__ == "__main__":
    main() 
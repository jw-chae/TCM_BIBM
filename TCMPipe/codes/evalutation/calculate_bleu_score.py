#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def calculate_bleu_scores(predictions, references):
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
    return bleu_scores

def main():
    input_file = "../../shezhen_results/result_new_best.jsonl"
    print("BLEU 점수 계산을 시작합니다... (Llama Factory 방식)")
    print(f"입력 파일: {input_file}")
    data = load_jsonl(input_file)
    print(f"총 {len(data)}개의 샘플을 로드했습니다.")
    predictions = [item['predict'] for item in data]
    references = [item['label'] for item in data]
    bleu_scores = calculate_bleu_scores(predictions, references)
    avg_bleu = np.mean(bleu_scores)
    std_bleu = np.std(bleu_scores)
    min_bleu = np.min(bleu_scores)
    max_bleu = np.max(bleu_scores)
    print("\n=== BLEU 점수 결과 ===")
    print(f"평균 BLEU: {avg_bleu:.4f}")
    print(f"표준편차: {std_bleu:.4f}")
    print(f"최소값: {min_bleu:.4f}")
    print(f"최대값: {max_bleu:.4f}")
    output_file = "../../shezhen_results/bleu_scores_char.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (pred, ref, score) in enumerate(zip(predictions, references, bleu_scores)):
            result = {
                'index': i,
                'predict': pred,
                'label': ref,
                'bleu_score': score
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"\nBLEU 점수가 {output_file}에 저장되었습니다.")
    summary_file = "../../shezhen_results/bleu_summary_char.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"BLEU 점수 요약 (Llama Factory 방식)\n")
        f.write(f"==============\n")
        f.write(f"평균 BLEU: {avg_bleu:.4f}\n")
        f.write(f"표준편차: {std_bleu:.4f}\n")
        f.write(f"최소값: {min_bleu:.4f}\n")
        f.write(f"최대값: {max_bleu:.4f}\n")
        f.write(f"총 샘플 수: {len(data)}\n")
    print(f"요약이 {summary_file}에 저장되었습니다.")

if __name__ == "__main__":
    main() 
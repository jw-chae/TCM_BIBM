#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from collections import Counter, defaultdict
from typing import Tuple, Set

def parse_tongue_features_with_unknown(text: str) -> Tuple[Tuple[str, ...], Set[str]]:
    """텍스트에서 알려진 토큰과 unknown token을 추출하는 함수"""
    if not text:
        return tuple(), set()
    
    # config 파일에서 불러온 dict 사용
    config_path = os.path.join(os.path.dirname(__file__), 'token_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    token_dicts = config['token_dicts']
    pattern_table = config['pattern_table']
    tokens_found = set()
    unknown_tokens = set()
    
    # 복합 패턴 우선 분해
    for pat, mapped in pattern_table.items():
        if pat in text:
            tokens_found.update(mapped)
    
    # 일반 토큰 추출
    for prefix, dict_list in token_dicts.items():
        for kw in dict_list:
            if kw in text:
                tokens_found.add(f"{prefix}_{kw}")
    
    # Unknown token 추출 (패턴이나 토큰으로 매칭되지 않은 부분)
    remaining_text = text
    for pat in pattern_table.keys():
        remaining_text = remaining_text.replace(pat, '')
    
    for prefix, dict_list in token_dicts.items():
        for kw in dict_list:
            remaining_text = remaining_text.replace(kw, '')
    
    # 남은 텍스트에서 의미있는 단어 추출
    words = re.findall(r'[\u4e00-\u9fff]+', remaining_text)
    for word in words:
        if len(word) >= 2:  # 2글자 이상만 unknown token으로 간주
            unknown_tokens.add(word)
    
    return tuple(tokens_found), unknown_tokens

def analyze_jsonl_file(file_path: str) -> dict:
    """JSONL 파일에서 unknown token을 분석하는 함수"""
    print(f"분석 중: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    total_samples = len(data)
    unknown_frequency = Counter()
    unknown_examples = defaultdict(list)
    
    for i, item in enumerate(data):
        # 예측과 라벨 텍스트 추출
        pred = item.get('predict', '').replace('舌诊结果: ', '').strip()
        label = item.get('label', '').replace('舌诊结果: ', '').strip()
        
        # 예측 텍스트에서 unknown token 추출
        _, pred_unknown = parse_tongue_features_with_unknown(pred)
        for token in pred_unknown:
            unknown_frequency[token] += 1
            unknown_examples[token].append({
                'type': 'predict',
                'index': i,
                'text': pred
            })
        
        # 라벨 텍스트에서 unknown token 추출
        _, label_unknown = parse_tongue_features_with_unknown(label)
        for token in label_unknown:
            unknown_frequency[token] += 1
            unknown_examples[token].append({
                'type': 'label',
                'index': i,
                'text': label
            })
    
    return {
        'file_path': file_path,
        'total_samples': total_samples,
        'unknown_frequency': dict(unknown_frequency.most_common()),
        'unknown_examples': {k: v[:5] for k, v in unknown_examples.items()},  # 각 토큰당 최대 5개 예시
        'total_unknown_types': len(unknown_frequency)
    }

def main():
    """메인 함수"""
    print("=== Unknown Token 확인 스크립트 ===\n")
    
    # 데이터 디렉토리 경로
    data_dir = "../../shezhen_results"
    
    if not os.path.exists(data_dir):
        print(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        return
    
    # JSONL 파일들 찾기
    jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print("JSONL 파일을 찾을 수 없습니다.")
        return
    
    print(f"발견된 JSONL 파일들: {jsonl_files}\n")
    
    # 각 파일 분석
    all_results = []
    global_unknown_frequency = Counter()
    
    for jsonl_file in jsonl_files:
        file_path = os.path.join(data_dir, jsonl_file)
        result = analyze_jsonl_file(file_path)
        all_results.append(result)
        
        # 전역 통계에 추가
        for token, freq in result['unknown_frequency'].items():
            global_unknown_frequency[token] += freq
    
    # 결과 출력
    print("\n" + "="*60)
    print("전체 Unknown Token 분석 결과")
    print("="*60)
    
    print(f"\n총 Unknown Token 종류: {len(global_unknown_frequency)}")
    print(f"총 Unknown Token 출현 횟수: {sum(global_unknown_frequency.values())}")
    
    print("\nUnknown Token 빈도순 (상위 20개):")
    for i, (token, freq) in enumerate(global_unknown_frequency.most_common(20), 1):
        print(f"{i:2d}. {token}: {freq}회")
    
    # 파일별 상세 결과
    print("\n" + "="*60)
    print("파일별 상세 결과")
    print("="*60)
    
    for result in all_results:
        print(f"\n파일: {os.path.basename(result['file_path'])}")
        print(f"샘플 수: {result['total_samples']}")
        print(f"Unknown Token 종류: {result['total_unknown_types']}")
        
        if result['unknown_frequency']:
            print("주요 Unknown Tokens:")
            for token, freq in list(result['unknown_frequency'].items())[:10]:
                print(f"  {token}: {freq}회")
    
    # 결과를 JSON 파일로 저장
    output_file = "unknown_tokens_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'global_statistics': {
                'total_unknown_types': len(global_unknown_frequency),
                'total_occurrences': sum(global_unknown_frequency.values()),
                'unknown_frequency': dict(global_unknown_frequency.most_common())
            },
            'file_results': all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n분석 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main() 
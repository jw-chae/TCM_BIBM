#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from collections import Counter
from typing import Tuple, Set

def parse_tongue_features_with_unknown(text: str) -> Tuple[Tuple[str, ...], Set[str]]:
    """
    텍스트에서 알려진 토큰과 unknown token을 추출하는 함수
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        (known_tokens, unknown_tokens): 알려진 토큰 튜플과 unknown token 집합
    """
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

def analyze_unknown_tokens(text: str) -> dict:
    """
    텍스트의 unknown token을 분석하는 함수
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        분석 결과 딕셔너리
    """
    known_tokens, unknown_tokens = parse_tongue_features_with_unknown(text)
    
    return {
        'original_text': text,
        'known_tokens': list(known_tokens),
        'unknown_tokens': list(unknown_tokens),
        'known_count': len(known_tokens),
        'unknown_count': len(unknown_tokens),
        'total_tokens': len(known_tokens) + len(unknown_tokens)
    }

def main():
    """테스트 실행"""
    print("=== Unknown Token 확인 코드 테스트 ===\n")
    
    # 테스트 케이스들
    test_cases = [
        "舌质淡红，舌苔薄白，舌体胖大",
        "舌质紫暗，舌苔黄腻，舌尖红",
        "舌质淡白，舌苔厚腻，舌边有齿痕",
        "舌质红绛，舌苔剥脱，舌面有裂纹",
        "舌质淡紫，舌苔白腻，舌体瘦小"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"테스트 케이스 {i}:")
        print(f"원본 텍스트: {text}")
        
        result = analyze_unknown_tokens(text)
        
        print(f"알려진 토큰 ({result['known_count']}개): {result['known_tokens']}")
        print(f"Unknown 토큰 ({result['unknown_count']}개): {result['unknown_tokens']}")
        print(f"총 토큰 수: {result['total_tokens']}")
        print("-" * 50)
    
    # 전체 unknown token 통계
    print("\n=== 전체 Unknown Token 통계 ===")
    all_unknown = Counter()
    
    for text in test_cases:
        _, unknown_tokens = parse_tongue_features_with_unknown(text)
        all_unknown.update(unknown_tokens)
    
    print(f"총 Unknown Token 종류: {len(all_unknown)}")
    print("Unknown Token 빈도순:")
    for token, freq in all_unknown.most_common():
        print(f"  {token}: {freq}회")

if __name__ == "__main__":
    main() 
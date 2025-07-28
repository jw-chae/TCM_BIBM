#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import argparse
from pathlib import Path

class OptimizedTongueEvaluator:
    def __init__(self, token_config_path: str):
        """최적화된 혀 진단 평가기 초기화"""
        with open(token_config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.category_map = self.config['category_map']
        self.token_dicts = self.config['token_dicts']
        self.pattern_table = self.config['pattern_table']
        self.synonyms = self.config.get('synonyms', {})
        self.weights = self.config.get('weights', {
            'tongue': 1.0, 'coat': 1.0, 'location': 1.0, 'other': 1.0
        })
        
        # 토큰 매핑 생성
        self.token_to_category = {}
        for category, tokens in self.token_dicts.items():
            for token in tokens:
                self.token_to_category[token] = category
        
        # 패턴 매핑 생성
        self.pattern_to_tokens = {}
        for pattern, tokens in self.pattern_table.items():
            self.pattern_to_tokens[pattern] = tokens
    
    def extract_tokens(self, text: str) -> Dict[str, Set[str]]:
        """텍스트에서 토큰 추출 (최적화된 버전)"""
        tokens = defaultdict(set)
        
        # 1. 패턴 매칭 (긴 패턴부터)
        sorted_patterns = sorted(self.pattern_to_tokens.keys(), key=len, reverse=True)
        remaining_text = text
        
        for pattern in sorted_patterns:
            if pattern in remaining_text:
                pattern_tokens = self.pattern_to_tokens[pattern]
                for token in pattern_tokens:
                    category, value = token.split('_', 1)
                    mapped_category = self.category_map.get(category, 'other')
                    tokens[mapped_category].add(value)
                
                # 매칭된 패턴 제거
                remaining_text = remaining_text.replace(pattern, '', 1)
        
        # 2. 개별 토큰 매칭
        for category, token_list in self.token_dicts.items():
            mapped_category = self.category_map.get(category, 'other')
            for token in token_list:
                if token in remaining_text:
                    tokens[mapped_category].add(token)
        
        return dict(tokens)
    
    def calculate_similarity(self, pred_tokens: Dict[str, Set[str]], 
                           label_tokens: Dict[str, Set[str]]) -> float:
        """동의어와 가중치를 고려한 유사도 계산"""
        total_score = 0.0
        total_weight = 0.0
        
        # 모든 카테고리 통합
        all_categories = set(pred_tokens.keys()) | set(label_tokens.keys())
        
        for category in all_categories:
            pred_set = pred_tokens.get(category, set())
            label_set = label_tokens.get(category, set())
            
            if not pred_set and not label_set:
                continue
            
            category_weight = self.weights.get(category, 1.0)
            
            # 정확 매칭
            exact_matches = pred_set & label_set
            exact_score = len(exact_matches) * 1.0
            
            # 동의어 매칭
            synonym_score = 0.0
            remaining_pred = pred_set - exact_matches
            remaining_label = label_set - exact_matches
            
            # 카테고리 내 동의어 매칭
            if category in self.synonyms:
                for pred_token in remaining_pred:
                    for label_token in remaining_label:
                        for synonym_pair, weight in self.synonyms[category].items():
                            token1, token2 = synonym_pair.split('|')
                            if (pred_token == token1 and label_token == token2) or \
                               (pred_token == token2 and label_token == token1):
                                synonym_score += weight
                                break
            
            # 교차 카테고리 동의어 매칭
            if 'CROSS' in self.synonyms:
                for pred_token in remaining_pred:
                    for label_token in remaining_label:
                        for synonym_pair, weight in self.synonyms['CROSS'].items():
                            token1, token2 = synonym_pair.split('|')
                            if (pred_token == token1 and label_token == token2) or \
                               (pred_token == token2 and label_token == token1):
                                synonym_score += weight * 0.5  # 교차 매칭은 가중치 감소
                                break
            
            # 부분 매칭 (문자 단위)
            partial_score = 0.0
            for pred_token in remaining_pred:
                for label_token in remaining_label:
                    if pred_token != label_token:
                        # 문자 단위 유사도 계산
                        char_similarity = self._calculate_char_similarity(pred_token, label_token)
                        if char_similarity > 0.5:  # 임계값
                            partial_score += char_similarity * 0.3  # 부분 매칭 가중치
            
            # 카테고리별 점수 계산
            category_score = exact_score + synonym_score + partial_score
            max_possible = max(len(pred_set), len(label_set))
            
            if max_possible > 0:
                normalized_score = category_score / max_possible
                total_score += normalized_score * category_weight
                total_weight += category_weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_char_similarity(self, token1: str, token2: str) -> float:
        """문자 단위 유사도 계산"""
        if not token1 or not token2:
            return 0.0
        
        # Jaccard 유사도
        chars1 = set(token1)
        chars2 = set(token2)
        
        intersection = len(chars1 & chars2)
        union = len(chars1 | chars2)
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_single(self, prediction: str, label: str) -> Dict[str, float]:
        """단일 예측-라벨 쌍 평가"""
        pred_tokens = self.extract_tokens(prediction)
        label_tokens = self.extract_tokens(label)
        
        similarity = self.calculate_similarity(pred_tokens, label_tokens)
        
        return {
            'similarity': similarity,
            'pred_tokens': pred_tokens,
            'label_tokens': label_tokens
        }
    
    def evaluate_batch(self, predictions: List[str], labels: List[str]) -> Dict[str, float]:
        """배치 평가"""
        if len(predictions) != len(labels):
            raise ValueError("예측과 라벨의 개수가 일치하지 않습니다.")
        
        similarities = []
        token_analysis = defaultdict(lambda: {'pred': 0, 'label': 0, 'match': 0})
        
        for pred, label in zip(predictions, labels):
            result = self.evaluate_single(pred, label)
            similarities.append(result['similarity'])
            
            # 토큰 분석
            for category in set(result['pred_tokens'].keys()) | set(result['label_tokens'].keys()):
                pred_count = len(result['pred_tokens'].get(category, set()))
                label_count = len(result['label_tokens'].get(category, set()))
                match_count = len(result['pred_tokens'].get(category, set()) & 
                                result['label_tokens'].get(category, set()))
                
                token_analysis[category]['pred'] += pred_count
                token_analysis[category]['label'] += label_count
                token_analysis[category]['match'] += match_count
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        return {
            'average_similarity': avg_similarity,
            'similarities': similarities,
            'token_analysis': dict(token_analysis)
        }

def load_jsonl(file_path: str) -> List[Dict]:
    """JSONL 파일 로드"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_results(results: Dict, output_path: str):
    """결과 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='최적화된 혀 진단 평가')
    parser.add_argument('--pred_file', required=True, help='예측 파일 경로')
    parser.add_argument('--label_file', required=True, help='라벨 파일 경로')
    parser.add_argument('--config_file', default='token_config_optimized.json', 
                       help='토큰 설정 파일 경로')
    parser.add_argument('--output_dir', default='metric_results', help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 데이터 로드
    pred_data = load_jsonl(args.pred_file)
    label_data = load_jsonl(args.label_file)
    
    # 예측과 라벨 추출
    predictions = [item.get('predict', '') for item in pred_data]
    labels = [item.get('label', '') for item in label_data]
    
    # 평가기 초기화
    evaluator = OptimizedTongueEvaluator(args.config_file)
    
    # 배치 평가 수행
    results = evaluator.evaluate_batch(predictions, labels)
    
    # 결과 저장
    output_path = os.path.join(args.output_dir, 'optimized_evaluation_results.json')
    save_results(results, output_path)
    
    print(f"평가 완료: {output_path}")
    print(f"평균 유사도: {results['average_similarity']:.4f}")
    
    # 토큰 분석 출력
    print("\n토큰 분석:")
    for category, stats in results['token_analysis'].items():
        precision = stats['match'] / stats['pred'] if stats['pred'] > 0 else 0
        recall = stats['match'] / stats['label'] if stats['label'] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{category}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

if __name__ == "__main__":
    main() 
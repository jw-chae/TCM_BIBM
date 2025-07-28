import json
import os
from evaluation2_ver2 import parse_tongue_features, compute_category_similarity, load_json_config

def debug_full_dataset():
    config = load_json_config('token_config.json')
    
    with open('../../dataset/result_new_best.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    print(f"전체 데이터 수: {len(data)}")
    
    # 점수별로 분류
    high_scores = []  # 0.8 이상
    medium_scores = []  # 0.5-0.8
    low_scores = []  # 0.5 미만
    
    for i, item in enumerate(data):
        pred = item.get('predict', '').strip()
        label = item.get('label', '').strip()
        
        s_tongue, s_coat, s_location, s_other, s_mean = compute_category_similarity(pred, label, config)
        
        if s_mean >= 0.8:
            high_scores.append((i, pred, label, s_mean))
        elif s_mean >= 0.5:
            medium_scores.append((i, pred, label, s_mean))
        else:
            low_scores.append((i, pred, label, s_mean))
    
    print(f"\n높은 점수 (≥0.8): {len(high_scores)}개")
    print(f"중간 점수 (0.5-0.8): {len(medium_scores)}개")
    print(f"낮은 점수 (<0.5): {len(low_scores)}개")
    
    print(f"\n=== 낮은 점수 샘플들 (상위 10개) ===")
    low_scores.sort(key=lambda x: x[3])  # 점수로 정렬
    for i, (idx, pred, label, score) in enumerate(low_scores[:10]):
        print(f"{i+1}. 점수: {score:.4f}")
        print(f"   예측: {pred}")
        print(f"   라벨: {label}")
        
        pred_tokens = parse_tongue_features(pred)
        label_tokens = parse_tongue_features(label)
        print(f"   예측 토큰: {list(pred_tokens)}")
        print(f"   라벨 토큰: {list(label_tokens)}")
        print()
    
    print(f"\n=== 높은 점수 샘플들 (상위 5개) ===")
    high_scores.sort(key=lambda x: x[3], reverse=True)  # 점수로 정렬
    for i, (idx, pred, label, score) in enumerate(high_scores[:5]):
        print(f"{i+1}. 점수: {score:.4f}")
        print(f"   예측: {pred}")
        print(f"   라벨: {label}")
        print()

if __name__ == "__main__":
    debug_full_dataset() 
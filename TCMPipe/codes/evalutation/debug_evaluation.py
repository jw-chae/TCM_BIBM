import json
import os
from evaluation2_ver2 import parse_tongue_features, compute_category_similarity, load_json_config

def debug_evaluation():
    config = load_json_config('token_config.json')
    
    # 샘플 데이터
    samples = [
        ("舌淡红，胖，边有齿印，苔薄白腻", "舌淡红，胖，边有齿印，苔薄白腻"),
        ("舌边尖红苔薄少", "舌尖红苔薄腻"),
        ("舌苔薄腻", "舌中根腻"),
        ("舌淡暗，苔厚白腻", "舌淡红，胖，边有齿印，苔薄黄腻"),
        ("舌苔薄白", "舌苔薄白")
    ]
    
    print("=== 토큰 매칭 디버깅 ===\n")
    
    for i, (pred, label) in enumerate(samples):
        print(f"샘플 {i+1}:")
        print(f"예측: {pred}")
        print(f"라벨: {label}")
        
        # 토큰 추출
        pred_tokens = parse_tongue_features(pred)
        label_tokens = parse_tongue_features(label)
        
        print(f"예측 토큰: {list(pred_tokens)}")
        print(f"라벨 토큰: {list(label_tokens)}")
        
        # 카테고리별 점수 계산
        s_tongue, s_coat, s_location, s_other, s_mean = compute_category_similarity(pred, label, config)
        
        print(f"혀질 점수: {s_tongue:.4f}")
        print(f"태질 점수: {s_coat:.4f}")
        print(f"위치 점수: {s_location:.4f}")
        print(f"기타 점수: {s_other:.4f}")
        print(f"평균 점수: {s_mean:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    debug_evaluation() 
import json
import os
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.optimize import linear_sum_assignment
import argparse
import networkx as nx

# =====================
# CONFIG LOADERS
# =====================
def load_json_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# =====================
# DATA STRUCTURES
# =====================
@dataclass
class ParsedRecord:
    image: str
    output: str
    diagnosis: str
    time: str

# =====================
# TOKEN PARSING
# =====================
@lru_cache(maxsize=4096)
def parse_tongue_features(text: str) -> Tuple[str, ...]:
    if not text:
        return tuple()
    # config 파일에서 불러온 dict 사용
    with open(os.path.join(os.path.dirname(__file__), 'token_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    token_dicts = config['token_dicts']
    pattern_table = config['pattern_table']
    tokens_found = set()
    # 복합 패턴 우선 분해
    for pat, mapped in pattern_table.items():
        if pat in text:
            tokens_found.update(mapped)
    # 일반 토큰 추출
    for prefix, dict_list in token_dicts.items():
        for kw in dict_list:
            if kw in text:
                tokens_found.add(f"{prefix}_{kw}")
    return tuple(tokens_found)

# =====================
# TOKEN SIMILARITY
# =====================
def parse_synonyms(synonyms_raw: Dict[str, Dict[str, float]]) -> Dict[str, Dict[Tuple[str, str], float]]:
    out = {}
    for cat, pairs in synonyms_raw.items():
        out[cat] = {}
        for k, v in pairs.items():
            a, b = k.split('|')
            out[cat][(a, b)] = v
    return out

def token_similarity(tk1: str, tk2: str, synonyms: Dict[str, Dict[Tuple[str, str], float]]) -> float:
    if '_' not in tk1 or '_' not in tk2:
        return 0.0
    prefix1, val1 = tk1.split('_', 1)
    prefix2, val2 = tk2.split('_', 1)
    if prefix1 == prefix2:
        syn_map = synonyms.get(prefix1, {})
        if (val1, val2) in syn_map:
            return syn_map[(val1, val2)]
        if (val2, val1) in syn_map:
            return syn_map[(val2, val1)]
        if val1 == val2:
            return 1.0
        return 0.0
    cross_map = synonyms.get('CROSS', {})
    if (tk1, tk2) in cross_map:
        return cross_map[(tk1, tk2)]
    if (tk2, tk1) in cross_map:
        return cross_map[(tk2, tk1)]
    return 0.0

# =====================
# CATEGORY SPLIT
# =====================
def classify_to_categories(token_list: List[str], category_map: Dict[str, str]) -> Dict[str, List[str]]:
    cats = {'tongue': [], 'coat': [], 'location': [], 'other': []}
    for tk in token_list:
        if '_' not in tk:
            cats['other'].append(tk)
            continue
        prefix, _ = tk.split('_', 1)
        cat = category_map.get(prefix, 'other')
        if cat not in cats:
            cats['other'].append(tk)
        else:
            cats[cat].append(tk)
    return cats

# =====================
# TOKEN LIST SIMILARITY (벡터화)
# =====================
def compute_token_list_similarity(pred: List[str], label: List[str], synonyms: Dict[str, Dict[Tuple[str, str], float]]) -> float:
    if not pred and not label:
        return 1.0
    if not pred or not label:
        return 0.0
    
    pred_arr = np.array(pred)
    label_arr = np.array(label)
    sim_matrix = np.vectorize(lambda x, y: token_similarity(x, y, synonyms))(pred_arr[:, None], label_arr)
    
    if np.max(sim_matrix) == 0:
        return 0.0
    
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_sim = sim_matrix[row_ind, col_ind].sum()
    
    # 부분 매칭 보너스: 일부 토큰이라도 매칭되면 기본 점수 부여
    matched_tokens = len([i for i in range(len(row_ind)) if sim_matrix[row_ind[i], col_ind[i]] > 0])
    total_tokens = max(len(pred), len(label))
    
    # 기본 매칭 점수
    base_score = total_sim / total_tokens if total_tokens > 0 else 0.0
    
    # 완벽한 매칭인 경우 1.0점 반환
    if matched_tokens == total_tokens and matched_tokens > 0:
        return 1.0
    
    # 부분 매칭 보너스: 매칭된 토큰이 있으면 최소 점수 보장
    if matched_tokens > 0:
        bonus_score = 0.3 * (matched_tokens / total_tokens)
        return max(base_score, bonus_score)
    
    return base_score

# =====================
# CATEGORY-BASED EVAL
# =====================
def get_category_weights(config: dict) -> dict:
    # config에 weights가 있으면 사용, 없으면 기본값(동일 가중치)
    default = {"tongue": 1.0, "coat": 1.0, "location": 1.0, "other": 1.0}
    return config.get("weights", default)

def compute_category_similarity(pred_str: str, label_str: str, config: Dict[str, Any]) -> Tuple[float, float, float, float]:
    synonyms = parse_synonyms(config['synonyms'])
    category_map = config['category_map']
    weights = get_category_weights(config)
    pred_tokens = parse_tongue_features(pred_str)
    label_tokens = parse_tongue_features(label_str)
    cats_pred = classify_to_categories(list(pred_tokens), category_map)
    cats_label = classify_to_categories(list(label_tokens), category_map)
    score_tongue = compute_token_list_similarity(cats_pred['tongue'], cats_label['tongue'], synonyms)
    score_coat = compute_token_list_similarity(cats_pred['coat'], cats_label['coat'], synonyms)
    score_location = compute_token_list_similarity(cats_pred.get('location', []), cats_label.get('location', []), synonyms)
    # other: 둘 다 비어있지 않으면 기존대로, 한 쪽이라도 비어있으면 0.0
    if cats_pred['other'] and cats_label['other']:
        score_other = compute_token_list_similarity(cats_pred['other'], cats_label['other'], synonyms)
    else:
        score_other = 0.0
    # 완벽한 매칭인 경우 1.0점 반환
    if pred_str.strip() == label_str.strip():
        return 1.0, 1.0, 1.0, 1.0, 1.0
    
    # 가중 평균
    total_weight = weights['tongue'] + weights['coat'] + weights['location'] + weights['other']
    mean_score = (
        score_tongue * weights['tongue'] +
        score_coat * weights['coat'] +
        score_location * weights['location'] +
        score_other * weights['other']
    ) / total_weight
    return score_tongue, score_coat, score_location, score_other, mean_score

# =====================
# BATCH EVAL
# =====================
def batch_eval_and_save(input_dir: str, output_dir: str, config_path: str):
    os.makedirs(output_dir, exist_ok=True)
    config = load_json_config(config_path)
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    logger = logging.getLogger("eval_batch")
    for fname in jsonl_files:
        path = os.path.join(input_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            comparison_data = [json.loads(line) for line in f if line.strip()]
        if not comparison_data:
            logger.warning(f"No data in {fname}")
            continue
        total_mean = 0.0
        for item in comparison_data:
            pred = item.get('predict', '').replace('舌诊结果: ', '').strip()
            label = item.get('label', '').replace('舌诊结果: ', '').strip()
            _, _, _, _, s_mean = compute_category_similarity(pred, label, config)
            total_mean += s_mean
        avg = total_mean / len(comparison_data)
        out_name = fname.rsplit('.jsonl', 1)[0] + '.txt'
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, 'w', encoding='utf-8') as fout:
            fout.write(f"{avg:.4f}\n")
        logger.info(f"{fname}: 평균 점수 = {avg:.4f}")

def compute_similarity_score_graph(G, predict_str: str, label_str: str, config: dict) -> float:
    source_node = "OUT_" + predict_str
    target_node = "OUT_" + label_str
    try:
        length = nx.shortest_path_length(G, source=source_node, target=target_node)
        if length > 0:
            return 1.0 / length
        else:
            return 0.0
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # 그래프에 없으면 토큰 기반 유사도 fallback
        _, _, _, _, mean_score = compute_category_similarity(predict_str, label_str, config)
        return mean_score

def build_simple_graph_from_jsonl(jsonl_path: str) -> nx.Graph:
    G = nx.Graph()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            for key in ['predict', 'label']:
                node = 'OUT_' + item[key]
                if not G.has_node(node):
                    G.add_node(node)
    return G

def main():
    parser = argparse.ArgumentParser(description="Tongue Evaluation CLI")
    parser.add_argument('--input', type=str, required=True, help='Input jsonl file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output file or directory')
    parser.add_argument('--config', type=str, required=True, help='Config json path')
    parser.add_argument('--mode', type=str, default='category', choices=['category', 'graph'], help='Evaluation mode')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.mode == 'category':
        if os.path.isdir(args.input):
            batch_eval_and_save(args.input, args.output, args.config)
        else:
            # 단일 파일 평가
            config = load_json_config(args.config)
            with open(args.input, 'r', encoding='utf-8') as f:
                comparison_data = [json.loads(line) for line in f if line.strip()]
            total_mean = 0.0
            total_tongue = 0.0
            total_coat = 0.0
            total_location = 0.0
            total_other = 0.0
            
            for item in comparison_data:
                pred = item.get('predict', '').replace('舌诊结果: ', '').strip()
                label = item.get('label', '').replace('舌诊结果: ', '').strip()
                s_tongue, s_coat, s_location, s_other, s_mean = compute_category_similarity(pred, label, config)
                total_mean += s_mean
                total_tongue += s_tongue
                total_coat += s_coat
                total_location += s_location
                total_other += s_other
            
            avg = total_mean / len(comparison_data)
            avg_tongue = total_tongue / len(comparison_data)
            avg_coat = total_coat / len(comparison_data)
            avg_location = total_location / len(comparison_data)
            avg_other = total_other / len(comparison_data)
            
            with open(args.output, 'w', encoding='utf-8') as fout:
                fout.write(f"전체 평균 점수: {avg:.4f}\n")
                fout.write(f"혀질(Tongue) 점수: {avg_tongue:.4f}\n")
                fout.write(f"태질(Coat) 점수: {avg_coat:.4f}\n")
                fout.write(f"위치(Location) 점수: {avg_location:.4f}\n")
                fout.write(f"기타(Other) 점수: {avg_other:.4f}\n")
                fout.write(f"데이터 수: {len(comparison_data)}\n")
            
            print(f"[INFO] {os.path.basename(args.input)}: 평균 점수 = {avg:.4f}")
            print(f"[INFO] 혀질(Tongue) 점수 = {avg_tongue:.4f}")
            print(f"[INFO] 태질(Coat) 점수 = {avg_coat:.4f}")
            print(f"[INFO] 위치(Location) 점수 = {avg_location:.4f}")
            print(f"[INFO] 기타(Other) 점수 = {avg_other:.4f}")
            print(f"[INFO] 데이터 수 = {len(comparison_data)}")
    elif args.mode == 'graph':
        config = load_json_config(args.config)
        if os.path.isdir(args.input):
            print("[WARN] graph mode는 단일 파일만 지원합니다.")
            return
        G = build_simple_graph_from_jsonl(args.input)
        with open(args.input, 'r', encoding='utf-8') as f:
            comparison_data = [json.loads(line) for line in f if line.strip()]
        total_sim = 0.0
        for item in comparison_data:
            pred = item.get('predict', '').replace('舌诊结果: ', '').strip()
            label = item.get('label', '').replace('舌诊结果: ', '').strip()
            sim = compute_similarity_score_graph(G, pred, label, config)
            total_sim += sim
        avg = total_sim / len(comparison_data)
        with open(args.output, 'w', encoding='utf-8') as fout:
            fout.write(f"{avg:.4f}\n")
        print(f"[INFO][GRAPH] {os.path.basename(args.input)}: 평균 점수 = {avg:.4f}")

if __name__ == "__main__":
    main() 
import json
import os
import sys
import webbrowser
from collections import Counter, defaultdict
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import re
import glob

###############################################################################
# (1) Tongue feature parsing function
###############################################################################
def parse_tongue_features(tongue_str):
    """
    Same logic as before, extracting tokens with prefixes like COLOR_, SHAPE_, 
    THICK_, COATCOLOR_, NATURE_.
    """
    if not tongue_str:
        return []
    tongue_color_dict = ["淡红","暗红","红","紫","暗","淡暗","淡","暗紫","紫暗","红紫","红暗","略暗","瘀点"]
    tongue_shape_dict = ["胖","瘦","齿痕","裂纹","肿胀","中裂","边暗","有裂纹","边有齿印","胖淡","瘦尖","胖舌","瘦舌"]
    coating_color_dict = ["白","黄","黑","灰","薄白","薄黄","厚白","厚黄","花剥","紫痕","白腻","黄腻","黑腻","灰腻"]
    coating_thickness_dict = ["薄","厚","少","花剥","薄少","薄腻","厚腻"]
    coating_nature_dict = ["腻","滑","燥","干","水滑","黏","花剥","滑腻"]

    tokens_found = set()

    # Force-split cases like "薄少" if needed
    if "薄少" in tongue_str:
        tokens_found.add("THICK_薄")
        tokens_found.add("THICK_少")

    def try_extract_keywords(input_str, dict_list, prefix):
        for kw in dict_list:
            if kw in input_str:
                tokens_found.add(f"{prefix}_{kw}")

    try_extract_keywords(tongue_str, tongue_color_dict,   "COLOR")
    try_extract_keywords(tongue_str, tongue_shape_dict,   "SHAPE")
    try_extract_keywords(tongue_str, coating_color_dict,  "COATCOLOR")
    try_extract_keywords(tongue_str, coating_thickness_dict, "THICK")
    try_extract_keywords(tongue_str, coating_nature_dict, "NATURE")

    return list(tokens_found)

###############################################################################
# (2) Load JSON files
###############################################################################
def load_json_files(paths):
    all_records = []
    for jp in paths:
        if os.path.exists(jp):
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_records.extend(data)
                else:
                    all_records.append(data)
        else:
            print(f"[DEBUG] File not found: {jp}")
    return all_records

###############################################################################
# (3) Add comparison_data (predict/label) to the graph
###############################################################################
def add_evaluation_outputs_to_graph(G, comparison_data):
    for item in comparison_data:
        predict_str = item.get("predict", "").strip()
        label_str   = item.get("label", "").strip()

        for out_text in [predict_str, label_str]:
            if not out_text:
                continue
            out_node = f"OUT_{out_text}"
            if not G.has_node(out_node):
                G.add_node(out_node, label=out_text, node_type="output")

            tokens = parse_tongue_features(out_text)
            for tk in tokens:
                tk_node = f"TOKEN_{tk}"
                if not G.has_node(tk_node):
                    G.add_node(tk_node, label=tk, node_type="tongue_feature")
                if not G.has_edge(out_node, tk_node):
                    G.add_edge(out_node, tk_node, relation="has_feature", weight=1)
                else:
                    G[out_node][tk_node]['weight'] += 1

###############################################################################
# (4) Build the KG while optionally including comparison_data
###############################################################################
def build_kg_with_tokens(json_paths, comparison_data=None):
    all_records = load_json_files(json_paths)
    if not all_records:
        print("[ERROR] Could not load data from the provided JSON files.")
        return None, []

    G = nx.Graph()
    parsed_data = []

    for record in all_records:
        image_path = record.get("image", "")
        output_text = record.get("output", "")
        diagnosis_val = record.get("diagnosis", "")
        time_info = record.get("time", "")

        if not isinstance(output_text, str):
            output_text = str(output_text) if output_text else ""
        if not isinstance(diagnosis_val, str):
            diagnosis_val = str(diagnosis_val) if diagnosis_val else ""

        parsed_data.append({
            "image": image_path,
            "output": output_text.strip(),
            "diagnosis": diagnosis_val.strip(),
            "time": time_info.strip() if time_info else ""
        })

    # (A) For each original output -> "OUT_..." + token nodes
    for item in parsed_data:
        out_str = item["output"]
        diag_str = item["diagnosis"]
        out_node_id = f"OUT_{out_str}"
        diag_node_id = f"DIAG_{diag_str}" if diag_str else None

        # Output node
        if out_str and not G.has_node(out_node_id):
            G.add_node(out_node_id, label=out_str, node_type="output")

        # Connect tokens
        tokens = parse_tongue_features(out_str)
        for tk in tokens:
            tk_node_id = f"TOKEN_{tk}"
            if not G.has_node(tk_node_id):
                G.add_node(tk_node_id, label=tk, node_type="tongue_feature")
            if out_str:
                if not G.has_edge(out_node_id, tk_node_id):
                    G.add_edge(out_node_id, tk_node_id, relation="has_feature", weight=1)
                else:
                    G[out_node_id][tk_node_id]['weight'] += 1

        # Diagnosis node
        if diag_str and not G.has_node(diag_node_id):
            G.add_node(diag_node_id, label=diag_str, node_type="diagnosis")

        if diag_str:
            if not G.has_edge(out_node_id, diag_node_id):
                G.add_edge(out_node_id, diag_node_id, relation="associated_diagnosis", weight=1)
            else:
                G[out_node_id][diag_node_id]['weight'] += 1

    # (B) Also add predict/label from comparison_data to the graph
    if comparison_data:
        add_evaluation_outputs_to_graph(G, comparison_data)

    return G, parsed_data

###############################################################################
# (5) Visualize the KG
###############################################################################
def visualize_kg(G, output_html="tongue_diagnosis_kg_tokens.html"):
    try:
        from pyvis.network import Network
    except ImportError:
        print("[WARN] The 'pyvis' package is not installed. Please run 'pip install pyvis'.")
        return

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        notebook=False,
    )
    color_map = {
        "output": "red",
        "tongue_feature": "orange",
        "diagnosis": "blue",
        "image": "green",
        "time": "gray"
    }

    for node_id, node_data in G.nodes(data=True):
        ntype = node_data.get('node_type', "")
        label = node_data.get('label', node_id)
        node_color = color_map.get(ntype, "white")
        net.add_node(
            node_id,
            label=label,
            title=f"{ntype}: {label}",
            color=node_color
        )

    for u, v, edge_data in G.edges(data=True):
        relation = edge_data.get('relation', "")
        weight = edge_data.get('weight', 1)
        net.add_edge(
            u, v,
            label=relation,
            value=weight
        )

    net.save_graph(output_html)
    print(f"[INFO] '{output_html}' was created.")
    abs_path = os.path.abspath(output_html)
    try:
        webbrowser.open_new_tab("file://" + abs_path)
    except:
        print("[WARN] Failed to automatically open in the web browser. Please open manually:", abs_path)

###############################################################################
# (6) (OLD) compute_similarity_score + (OLD) evaluate_llm
#     Using the shortest path in the graph (similar to the original)
###############################################################################
def compute_similarity_score(G, predict_str, label_str):
    if predict_str == label_str:
        return 1.0
    source_node = "OUT_" + predict_str
    target_node = "OUT_" + label_str
    try:
        length = nx.shortest_path_length(G, source=source_node, target=target_node)
        if length > 0:
            return 1.0 / length
        else:
            return 0.0
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 0.0

def evaluate_llm_shortest_path(G, comparison_data):
    """
    (Original method) uses the graph shortest path for predict/label similarity, 
    then applies the Hungarian algorithm for a global matching.
    """
    if not comparison_data:
        print("[ERROR] The comparison dataset is empty.")
        return

    predictions = [item["predict"].replace("舌诊结果: ", "").strip() for item in comparison_data]
    labels = [item["label"].replace("舌诊结果: ", "").strip() for item in comparison_data]

    unique_predictions = list(set(predictions))
    unique_labels = list(set(labels))

    cost_matrix = np.zeros((len(unique_predictions), len(unique_labels)))

    for i, pred in enumerate(unique_predictions):
        for j, lab in enumerate(unique_labels):
            sim = compute_similarity_score(G, pred, lab)
            cost = (1.0 / sim) if sim > 0 else 1e6
            cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    total_similarity = 0.0
    assigned_pairs = []
    for i, j in zip(row_ind, col_ind):
        pred_str = unique_predictions[i]
        lab_str  = unique_labels[j]
        sim = compute_similarity_score(G, pred_str, lab_str)
        assigned_pairs.append((pred_str, lab_str, sim))
        total_similarity += sim

    avg_similarity = total_similarity / len(assigned_pairs) if assigned_pairs else 0.0

    print("[evaluate_llm_shortest_path] ========================================")
    print(f"[INFO] Total similarity score: {total_similarity:.4f}")
    print(f"[INFO] Average similarity score: {avg_similarity:.4f}")
    print("[INFO] Matched pairs and similarity:")
    for pred_str, lab_str, sim in assigned_pairs:
        print(f"  Predict: {pred_str} <-> Label: {lab_str} | Similarity: {sim:.4f}")

###############################################################################
# (7) Newly added: category-based Hungarian matching
###############################################################################

# 7-1) (Optional) synonyms for partial similarity
COLOR_SYNONYMS = {
    ("暗红", "红"): 0.8,
    ("淡红", "红"): 0.7,
    # ... add more as needed
}

def token_similarity(tk1, tk2):
    """
    Partial similarity for tokens within the same category.
    e.g., tk1="COLOR_暗红", tk2="COLOR_红", etc.
    """
    if "_" not in tk1 or "_" not in tk2:
        return 0.0
    prefix1, val1 = tk1.split("_", 1)
    prefix2, val2 = tk2.split("_", 1)
    if prefix1 != prefix2:
        return 0.0  # Different categories => 0 similarity

    # Choose synonyms map depending on the prefix
    synonyms_map = None
    if prefix1 == "COLOR":
        synonyms_map = COLOR_SYNONYMS
    # elif prefix1 == "SHAPE": ...
    # elif prefix1 == "THICK": ...
    # etc...

    if synonyms_map is not None:
        if (val1, val2) in synonyms_map:
            return synonyms_map[(val1, val2)]
        if (val2, val1) in synonyms_map:
            return synonyms_map[(val2, val1)]

    # Exact match
    if val1 == val2:
        return 1.0
    return 0.0

def compute_token_list_similarity(tokens_pred, tokens_label):
    """
    Compute maximum similarity between two lists of tokens using Hungarian matching.
    """
    if not tokens_pred and not tokens_label:
        return 1.0
    if not tokens_pred or not tokens_label:
        return 0.0

    psize = len(tokens_pred)
    lsize = len(tokens_label)

    sim_matrix = np.zeros((psize, lsize))
    for i, ptk in enumerate(tokens_pred):
        for j, ltk in enumerate(tokens_label):
            sim_matrix[i][j] = token_similarity(ptk, ltk)

    # Hungarian algorithm uses "min cost", so cost = 1 - similarity
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    total_sim = 0.0
    for r, c in zip(row_ind, col_ind):
        total_sim += sim_matrix[r, c]

    denom = max(psize, lsize)
    final_score = total_sim / denom if denom > 0 else 0.0
    return final_score

# 7-2) Category classification (舌, 苔, 其他)
def classify_to_three_categories(token_list):
    """
    Receive a list of tokens (prefix_value) and split them into 
    (舌, 苔, 其他) = (tongue, coat, other).
    For example: if prefix = COLOR or SHAPE => 'tongue', 
                 if prefix = COATCOLOR or THICK or NATURE => 'coat',
                 otherwise => 'other'.
    """
    tongue_group = []
    coat_group = []
    other_group = []

    for tk in token_list:
        if "_" not in tk:
            other_group.append(tk)
            continue

        prefix, val = tk.split("_", 1)
        # Example: "COLOR" + "暗红" => 'tongue'
        if prefix in ["COLOR", "SHAPE"]:
            tongue_group.append(tk)
        elif prefix in ["COATCOLOR", "THICK", "NATURE"]:
            coat_group.append(tk)
        else:
            other_group.append(tk)

    return tongue_group, coat_group, other_group

def compute_category_similarity(predict_str, label_str):
    """
    1) Parse tokens from predict_str / label_str
    2) Classify them into (tongue, coat, other)
    3) Perform Hungarian similarity for each category
    4) Return (score_tongue, score_coat, score_other, mean_score)
    """
    tokens_pred = parse_tongue_features(predict_str)
    tokens_label = parse_tongue_features(label_str)

    pred_tongue, pred_coat, pred_other = classify_to_three_categories(tokens_pred)
    lab_tongue, lab_coat, lab_other   = classify_to_three_categories(tokens_label)

    score_tongue = compute_token_list_similarity(pred_tongue, lab_tongue)
    score_coat   = compute_token_list_similarity(pred_coat, lab_coat)
    score_other  = compute_token_list_similarity(pred_other, lab_other)

    # Simple average (use different weights if needed)
    mean_score = (score_tongue + score_coat + score_other) / 3.0

    return score_tongue, score_coat, score_other, mean_score

def evaluate_llm_category_based(comparison_data):
    """
    Client request: separate evaluation into '舌', '苔', '其他' categories, 
    each with Hungarian matching. 
    Here we do a 1:1 comparison for each (predict, label) pair and compute
    the average score.
    """
    if not comparison_data:
        print("[ERROR] The comparison dataset is empty.")
        return

    total_mean_score = 0.0
    for idx, item in enumerate(comparison_data, 1):
        pred_str = item.get("predict", "").replace("舌诊结果: ", "").strip()
        lab_str  = item.get("label", "").replace("舌诊结果: ", "").strip()
        s_tongue, s_coat, s_other, s_mean = compute_category_similarity(pred_str, lab_str)

        total_mean_score += s_mean
        print(f"[{idx}] PRED: {pred_str}")
        print(f"     LABEL: {lab_str}")
        print(f"     => 舌={s_tongue:.3f}, 苔={s_coat:.3f}, 其他={s_other:.3f}, Final={s_mean:.3f}\n")

    avg_of_mean = total_mean_score / len(comparison_data)
    print("[evaluate_llm_category_based] ========================================")
    print(f"[INFO] Out of {len(comparison_data)} pairs, average score: {avg_of_mean:.4f}")

###############################################################################
# (8) main
###############################################################################
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')

    # 1) JSON paths (그래프 빌드는 생략)
    json_paths = []

    # 2) Load comparison data (jsonl)
    comparison_path = "Project_Tsinghua_Paper/TCMPipe/results/compare_alltrain_val16.jsonl"
    comparison_data = []
    if os.path.exists(comparison_path):
        with open(comparison_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    comparison_data.append(json.loads(line))
    else:
        print(f"[ERROR] File not found: {comparison_path}")
        return

    # 3) (그래프 빌드 및 시각화 생략)
    # 4) (OLD) Evaluate via shortest path in the graph (생략)

    # 5-B) (NEW) Category-based Hungarian matching
    if comparison_data:
        evaluate_llm_category_based(comparison_data)
    else:
        print("[ERROR] No comparison data loaded.")

def batch_eval_and_save(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_files = glob.glob(os.path.join(input_dir, '*.jsonl'))
    for jsonl_path in jsonl_files:
        comparison_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    comparison_data.append(json.loads(line))
        if not comparison_data:
            print(f"[WARN] No data in {jsonl_path}")
            continue
        # 점수 계산
        total_mean_score = 0.0
        for item in comparison_data:
            pred_str = item.get("predict", "").replace("舌诊结果: ", "").strip()
            lab_str  = item.get("label", "").replace("舌诊结果: ", "").strip()
            _, _, _, s_mean = compute_category_similarity(pred_str, lab_str)
            total_mean_score += s_mean
        avg_of_mean = total_mean_score / len(comparison_data)
        # 결과 저장
        base = os.path.basename(jsonl_path)
        out_path = os.path.join(output_dir, base.replace('.jsonl', '.txt'))
        with open(out_path, 'w', encoding='utf-8') as fout:
            fout.write(f"{avg_of_mean:.4f}\n")
        print(f"[INFO] {base}: 평균 점수 = {avg_of_mean:.4f}")

if __name__ == "__main__":
    # 기존 main은 유지, 아래 코드로 batch 평가 가능
    batch_eval_and_save(
        input_dir="Project_Tsinghua_Paper/TCMPipe/shezhen_results",
        output_dir="Project_Tsinghua_Paper/TCMPipe/metric_results"
    )

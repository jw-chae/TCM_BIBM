import json
import os

# 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, '../../dataset')
result_dir = os.path.join(base_dir, '../../shezhen_results')

# 파일 경로
pred_file = os.path.join(dataset_dir, 'val.qwen2_5vl_output.jsonl')
label_file = os.path.join(result_dir, 'label.txt')
out_file = os.path.join(result_dir, 'compare_qwen2_5vl_label.jsonl')

# 예측값 로드
qwen2_5vl_preds = []
with open(pred_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            qwen2_5vl_preds.append(item['qwen2_5vl_output'].strip().replace('。', '').replace('.', ''))

# 라벨값 로드
labels = []
with open(label_file, 'r', encoding='utf-8') as f:
    for line in f:
        labels.append(line.strip().replace('。', '').replace('.', ''))

assert len(qwen2_5vl_preds) == len(labels), f"예측({len(qwen2_5vl_preds)})과 라벨({len(labels)}) 개수 불일치"

# jsonl 생성
with open(out_file, 'w', encoding='utf-8') as fout:
    for pred, label in zip(qwen2_5vl_preds, labels):
        fout.write(json.dumps({'predict': pred, 'label': label}, ensure_ascii=False) + '\n')

print(f"[완료] {out_file} 생성. 총 {len(qwen2_5vl_preds)}개") 
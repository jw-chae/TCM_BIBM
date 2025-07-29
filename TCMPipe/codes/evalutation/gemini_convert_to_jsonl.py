#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
val.gemini_compare.jsonl(텍스트) → {predict, label} jsonl 변환 스크립트

- 입력: [idx/total] <경로/파일명> → <예측>
- 정답: 25.1.10-25.6.3.json (image, output)
- 출력: {"predict": <예측>, "label": <정답>} jsonl
"""
import re
import os
import json
import argparse

PRED_LINE_RE = re.compile(r"^\[\d+/\d+\]\s+(.+?)\s+→\s+(.+)$")

def parse_pred_file(pred_path):
    mapping = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = PRED_LINE_RE.match(line)
            if not m:
                continue
            img_path, pred = m.groups()
            fname = os.path.basename(img_path.strip()).strip()
            mapping[fname] = pred.strip()
    return mapping

def parse_gt_file(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {os.path.basename(item["image"]).strip(): item["output"].strip() for item in data}

def convert(pred_path, gt_path, out_path):
    pred_map = parse_pred_file(pred_path)
    gt_map = parse_gt_file(gt_path)
    cnt = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for fname, pred in pred_map.items():
            label = gt_map.get(fname)
            if label is None:
                continue
            obj = {"predict": pred, "label": label}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            cnt += 1
    print(f"[INFO] 변환 완료: {cnt}개 → {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="예측 텍스트 파일")
    parser.add_argument("--gt", required=True, help="정답 json 파일")
    parser.add_argument("--out", required=True, help="출력 jsonl 파일")
    args = parser.parse_args()
    convert(args.pred, args.gt, args.out)

if __name__ == "__main__":
    main() 
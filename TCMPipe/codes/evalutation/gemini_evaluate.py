#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gemini 결과 평가 스크립트
=========================

예측 파일(`val.gemini_output copy.jsonl`)과
정답 파일(`25.1.10-25.6.3.json`)을 비교하여 평균 점수를 계산한다.

• 예측 파일 형식 (plain text, 각 행):
  [idx/total] <상대경로 이미지> → <예측 텍스트>

• 정답 파일 형식 (JSON array):
  { "image": "<파일명>", "output": "<정답 텍스트>", ... }

평가는 `evaluation2_ver2.compute_category_similarity` 를 그대로 사용한다.
"""

import re
import json
import os
import argparse
import logging
from typing import Dict, Tuple

# 같은 디렉터리의 기존 평가 로직 재사용
from evaluation2_ver2 import compute_category_similarity, load_json_config

LOGGER = logging.getLogger("gemini_eval")

# -----------------------------
# 파싱 유틸
# -----------------------------
PRED_LINE_RE = re.compile(r"^\[\d+/\d+\]\s+(.+?)\s+→\s+(.+)$")

def parse_prediction_file(pred_path: str) -> Dict[str, str]:
    """예측 파일을 읽어 {basename: prediction} 딕셔너리 반환 (경로 무시, 파일명만)"""
    mapping: Dict[str, str] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = PRED_LINE_RE.match(line)
            if not m:
                LOGGER.warning("형식 무시: %s", line[:120])
                continue
            img_path, pred_txt = m.groups()
            # 폴더 경로 제거, 파일명만 추출, 공백도 제거
            basename = os.path.basename(img_path.strip()).strip()
            mapping[basename] = pred_txt.strip()
    return mapping


def parse_ground_truth(gt_path: str) -> Dict[str, str]:
    """정답 파일(JSON array) 읽어 {basename: label_output} 반환"""
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {os.path.basename(item["image"]): item["output"].strip() for item in data}

# -----------------------------
# 메인 평가
# -----------------------------

def evaluate(pred_map: Dict[str, str], gt_map: Dict[str, str], config_path: str) -> Tuple[float, int]:
    """두 맵을 비교하여 평균 점수 및 비교된 샘플 수 반환"""
    config = load_json_config(config_path)
    total = 0.0
    cnt = 0
    for img, pred in pred_map.items():
        label = gt_map.get(img)
        if label is None:
            LOGGER.debug("정답 없음, 스킵: %s", img)
            continue
        *_ignored, mean_score = compute_category_similarity(pred, label, config)
        total += mean_score
        cnt += 1
    avg = total / cnt if cnt else 0.0
    return avg, cnt

def convert_to_jsonl(pred_path: str, gt_path: str, out_path: str):
    """
    Gemini 예측 파일(pred_path)과 정답(gt_path)로부터
    {"predict": <예측>, "label": <정답>} 형식의 JSONL 파일을 생성한다.
    """
    pred_map = parse_prediction_file(pred_path)
    gt_map = parse_ground_truth(gt_path)
    cnt = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for img, pred in pred_map.items():
            # 파일명만 추출해서 매칭 (혹시 key에 경로가 남아있어도)
            img_key = os.path.basename(img.strip())
            label = gt_map.get(img_key)
            if label is None:
                continue
            obj = {"predict": pred, "label": label}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            cnt += 1
    print(f"[INFO] 변환 완료: {cnt}개 → {out_path}")

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Gemini tongue diagnosis evaluation")
    parser.add_argument("--pred", required=True, help="예측 결과 파일 (val.gemini_output copy.jsonl)")
    parser.add_argument("--gt", required=True, help="정답 JSON 파일 (25.1.10-25.6.3.json)")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "token_config.json"), help="token_config.json 경로")
    parser.add_argument("--out", help="평균 점수를 기록할 txt 파일 (선택)")
    parser.add_argument("--convert-jsonl", help="변환된 JSONL 파일 경로(선택). 지정 시 변환만 수행")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    LOGGER.info("예측 파일 읽는 중: %s", args.pred)
    pred_map = parse_prediction_file(args.pred)
    LOGGER.info("정답 파일 읽는 중: %s", args.gt)
    gt_map = parse_ground_truth(args.gt)

    avg, cnt = evaluate(pred_map, gt_map, args.config)
    LOGGER.info("평가 완료 – 비교 샘플 %d개, 평균 점수 %.4f", cnt, avg)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(f"{avg:.4f}\n")
        LOGGER.info("결과를 %s 에 저장", args.out)
    else:
        print(f"AVERAGE_SCORE\t{avg:.4f}")

    if args.convert_jsonl:
        convert_to_jsonl(args.pred, args.gt, args.convert_jsonl)
        return

if __name__ == "__main__":
    main() 
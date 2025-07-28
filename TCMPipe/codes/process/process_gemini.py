import json
import os
import time
from typing import Dict
from google import genai

Part = genai.types.Part  # Part 클래스 참조

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

INPUT_PATH  = os.path.join(DATASET_DIR, "val.sharegpt.json")
OUTPUT_PATH = os.path.join(DATASET_DIR, "val.gemini_output.jsonl")
IMG_BASE    = DATASET_DIR

PROMPT = (
    "You are an expert in Chinese medicine tongue diagnosis. "
    "Please analyze the tongue photos provided based on Chinese medicine tongue diagnosis and output the results in a single sentence, as in the example:"
    "(舌尖红苔薄腻, 舌淡红苔薄白, 舌胖苔薄白 etc)."
)
RATE_LIMIT_SECONDS = 4

# API 키 설정: 환경변수 없으면 기본값 사용
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBzrd3FQDcSx7potqx77VgdZacPXuElae8")

client = genai.Client(api_key=API_KEY)

def get_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        ext, "application/octet-stream"
    )

def gemini_infer(image_path: str) -> str:
    try:
        mime_type = get_mime_type(image_path)
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                Part.from_text(text=PROMPT),
                Part.from_bytes(data=img_bytes, mime_type=mime_type),
            ],
        )
        return resp.text.strip()
    except Exception as e:
        print(f"[API ERROR] {image_path}: {e}")
        return "[API_ERROR]"

def process_sample(sample: Dict, idx: int, total: int, out_f) -> None:
    rel = sample["images"][0]
    abs_path = os.path.join(IMG_BASE, rel)

    result = (
        gemini_infer(abs_path) if os.path.exists(abs_path) else "[FILE_NOT_FOUND]"
    )

    json.dump({"image": rel, "gemini_output": result}, out_f, ensure_ascii=False)
    out_f.write("\n")
    out_f.flush()  # 즉시 디스크에 기록

    print(f"[{idx}/{total}] {rel} → {result}")

def main() -> None:
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for idx, sample in enumerate(data, 1):
            process_sample(sample, idx, len(data), out_f)
            time.sleep(RATE_LIMIT_SECONDS)

    print(f"\n✅  완료 → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

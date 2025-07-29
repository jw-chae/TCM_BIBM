import os
import json
import time
import replicate

# 환경변수에서 Replicate API 토큰을 읽어옴
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN 환경변수를 먼저 설정하세요!")

# Replicate 토큰 환경변수 설정 (라이브러리 내부에서도 필요)
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

INPUT_PATH = '../../dataset/val.sharegpt.json'
OUTPUT_PATH = '../../dataset/val.llama4scout_output.jsonl'
IMG_BASE = '../../dataset/'
PROMPT = (
    "You are an expert in Chinese medicine tongue diagnosis. "
    "Please analyze the tongue photos provided based on Chinese medicine tongue diagnosis and output the results in a single sentence, as in the example:"
    "(舌尖红苔薄腻, 舌淡红苔薄白, 舌胖苔薄白 etc)."
)
RATE_LIMIT_SECONDS = 4

MODEL_REF = "meta/llama-4-maverick-instruct"

# Llama-4-Scout-Instruct로 멀티모달 이미지+텍스트 프롬프트 전송
def llama4scout_infer(image_path: str, prompt: str) -> str:
    try:
        with open(image_path, "rb") as img_f:
            output = replicate.run(
                MODEL_REF,
                input={
                    "image": img_f,
                    "prompt": prompt,
                    "max_new_tokens": 64,
                    "temperature": 0.2
                }
            )
        # Replicate는 generator일 수 있음
        if hasattr(output, '__iter__') and not isinstance(output, str):
            return "".join(list(output)).strip()
        return str(output).strip()
    except Exception as e:
        print(f"[ERROR] {image_path}: {e}")
        return "[API_ERROR]"

def main():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    for idx, sample in enumerate(data, 1):
        rel = sample["images"][0]
        abs_path = os.path.join(IMG_BASE, rel)
        if not os.path.exists(abs_path):
            result = "[FILE_NOT_FOUND]"
        else:
            result = llama4scout_infer(abs_path, PROMPT)
        results.append({"image": rel, "llama4scout_output": result})
        print(f"[{idx}/{len(data)}] {rel} → {result}")
        time.sleep(RATE_LIMIT_SECONDS)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✅  완료 → {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 
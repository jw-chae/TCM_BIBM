import os
import json
import time
import base64
from openai import OpenAI

# 환경변수에서 XAI API 키를 읽어옴
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise RuntimeError("XAI_API_KEY 환경변수를 먼저 설정하세요!")

client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

INPUT_PATH = '../../dataset/val.sharegpt.json'
OUTPUT_PATH = '../../dataset/val.grok_output.jsonl'
IMG_BASE = '../../dataset/'
PROMPT = (
    "You are an expert in Chinese medicine tongue diagnosis. "
    "Please analyze the tongue photos provided based on Chinese medicine tongue diagnosis and output the results in a single sentence, as in the example:"
    "(舌尖红苔薄腻, 舌淡红苔薄白, 舌胖苔薄白 etc)."
)
RATE_LIMIT_SECONDS = 4

# Grok API로 멀티모달 이미지+텍스트 프롬프트 전송
def grok_infer(image_path: str, prompt: str) -> str:
    with open(image_path, "rb") as img_f:
        img_bytes = img_f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    img_url = f"data:image/jpeg;base64,{img_b64}"
    try:
        response = client.chat.completions.create(
            model="grok-2-vision-1212",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url}}
                ]}
            ],
            max_tokens=64,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
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
            result = grok_infer(abs_path, PROMPT)
        results.append({"image": rel, "grok_output": result})
        print(f"[{idx}/{len(data)}] {rel} → {result}")
        time.sleep(RATE_LIMIT_SECONDS)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✅  완료 → {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 
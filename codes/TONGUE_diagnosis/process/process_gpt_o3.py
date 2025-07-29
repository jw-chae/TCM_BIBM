import os
import json
import time
import openai
import base64
from typing import List

# Read OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY environment variable first!")
openai.api_key = OPENAI_API_KEY

INPUT_PATH = '../../dataset/val.sharegpt_gt.json'
OUTPUT_PATH = '../../dataset/val.gpt_o3_output_gt.json'
IMG_BASE = '../../dataset/'
PROMPT = (
    "You are an expert in Chinese medicine tongue diagnosis. "
    "Please analyze the tongue photos provided based on Chinese medicine tongue diagnosis and output the results in a single sentence, as in the example:"
    "(舌尖红苔薄腻, 舌淡红苔薄白, 舌胖苔薄白 etc).")
RATE_LIMIT_SECONDS = 4

# Send multimodal image+text prompt to GPT-4o (or 3.5/4)
# (As of June 2024, only gpt-4o supports images in OpenAI API)
def gpt_o3_infer(image_path: str, prompt: str) -> str:
    with open(image_path, "rb") as img_f:
        img_bytes = img_f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    img_url = f"data:image/jpeg;base64,{img_b64}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url}}
                ]}
            ],
            max_tokens=64,
            temperature=0.2,
        )
        return response["choices"][0]["message"]["content"].strip()
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
            result = gpt_o3_infer(abs_path, PROMPT)
        results.append({"image": rel, "gpt_output": result})
        print(f"[{idx}/{len(data)}] {rel} → {result}")
        time.sleep(RATE_LIMIT_SECONDS)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅  Complete → {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 
import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# CLI argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Qwen2.5-VL-32B multimodal inference automation')
    parser.add_argument('--input', type=str, required=True, help='Input jsonl file (image, prompt)')
    parser.add_argument('--output', type=str, required=True, help='Output jsonl file')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum tokens to generate')
    parser.add_argument('--flash_attention', action='store_true', help='Use flash_attention_2')
    return parser.parse_args()

def load_image(image_path):
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[Warning] Image load failed: {image_path} ({e})")
        return None

def main():
    args = parse_args()

    # Load model/processor
    if args.flash_attention:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct",
            torch_dtype="auto",
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")

    # Load input data
    data = []
    with open(args.input, 'r', encoding='utf-8') as fin:
        for line in fin:
            if line.strip():
                item = json.loads(line)
                data.append(item)

    results = []
    for item in tqdm(data, desc='Qwen2.5-VL inference'):
        image_path = item['image']
        prompt = item['prompt']
        image = load_image(image_path)
        if image is None:
            output_text = "[Image load failed]"
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        result = {
            "qwen2_5vl_output": output_text,
            "image": image_path,
            "prompt": prompt
        }
        results.append(result)

    with open(args.output, 'w', encoding='utf-8') as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"[Complete] {args.output} saved. Total {len(results)} items")

if __name__ == '__main__':
    main() 
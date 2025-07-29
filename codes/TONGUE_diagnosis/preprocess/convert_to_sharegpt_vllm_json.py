import json
import os
import argparse

def convert_and_write_json(data, image_dir, output_json):
    """Convert data to ShareGPT/vllm format JSON"""
    out_list = []
    for item in data:
        user_content = "<image>" + item['prompt'].replace('<image>', '').strip()
        messages = [
            {"content": user_content, "role": "user"},
            {"content": item['output'], "role": "assistant"}
        ]
        image_path = os.path.join(image_dir, os.path.basename(item['image']))
        rel_image_path = os.path.relpath(image_path, os.path.dirname(output_json))
        images = [rel_image_path]
        out = {"messages": messages, "images": images}
        out_list.append(out)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    print(f"{output_json} file created. (Sample count: {len(data)})")

def main():
    parser = argparse.ArgumentParser(description="ShareGPT/vllm format JSON converter (full dataset without train/val split)")
    parser.add_argument('--input_json', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--image_dir', type=str, required=True, help='Image folder path')
    parser.add_argument('--output_json', type=str, required=True, help='Output JSON file path')
    args = parser.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    convert_and_write_json(data, args.image_dir, args.output_json)

if __name__ == "__main__":
    main() 
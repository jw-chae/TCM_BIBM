import json
import os
import argparse

def convert_and_write_json(data, image_dir, output_json):
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
    print(f"{output_json} 파일이 생성되었습니다. (샘플 수: {len(data)})")

def main():
    parser = argparse.ArgumentParser(description="ShareGPT/vllm 포맷 json 변환기 (train/val 분할 없이 전체)")
    parser.add_argument('--input_json', type=str, required=True, help='원본 json 파일 경로')
    parser.add_argument('--image_dir', type=str, required=True, help='이미지 폴더 경로')
    parser.add_argument('--output_json', type=str, required=True, help='출력 json 파일 경로')
    args = parser.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    convert_and_write_json(data, args.image_dir, args.output_json)

if __name__ == "__main__":
    main() 
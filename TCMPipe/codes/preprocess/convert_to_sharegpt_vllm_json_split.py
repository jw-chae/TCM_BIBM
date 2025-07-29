import json
import os
import argparse
import random
import re

def normalize_filename(filename):
    """파일명의 날짜 형식을 정규화"""
    # 2024-12-01 형식을 24.12.1 형식으로 변환 (앞의 0 제거)
    pattern = r'(\d{4})-(\d{1,2})-(\d{1,2})'
    match = re.search(pattern, filename)
    if match:
        year = match.group(1)[2:]  # 2024 -> 24
        month = str(int(match.group(2)))  # 12 -> 12, 01 -> 1
        day = str(int(match.group(3)))    # 01 -> 1, 08 -> 8
        normalized = filename.replace(match.group(0), f"{year}.{month}.{day}")
        return normalized
    return filename

def filter_existing_images(data, image_dir):
    """이미지 파일이 실제로 존재하는 데이터만 필터링 (개선된 매칭)"""
    existing_files = set(os.listdir(image_dir))
    filtered_data = []
    missing_count = 0
    
    for item in data:
        original_image = item['image']
        normalized_image = normalize_filename(original_image)
        
        # 공백 제거 버전도 시도
        no_space_image = original_image.replace(' ', '')
        no_space_normalized = normalized_image.replace(' ', '')
        
        if (original_image in existing_files or 
            normalized_image in existing_files or
            no_space_image in existing_files or
            no_space_normalized in existing_files):
            
            # 매칭된 파일명으로 업데이트
            if original_image in existing_files:
                pass  # 그대로 유지
            elif normalized_image in existing_files:
                item['image'] = normalized_image
            elif no_space_image in existing_files:
                item['image'] = no_space_image
            elif no_space_normalized in existing_files:
                item['image'] = no_space_normalized
                
            filtered_data.append(item)
        else:
            missing_count += 1
    
    print(f"총 {len(data)}개 중 {len(filtered_data)}개는 이미지 존재, {missing_count}개는 이미지 없음")
    return filtered_data

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
    parser = argparse.ArgumentParser(description="ShareGPT/vllm 포맷 json 9:1 분할 변환기 (개선된 이미지 매칭)")
    parser.add_argument('--input_json', type=str, required=True, help='원본 json 파일 경로')
    parser.add_argument('--image_dir', type=str, required=True, help='이미지 폴더 경로')
    parser.add_argument('--output_train_json', type=str, required=True, help='train set json 파일 경로')
    parser.add_argument('--output_val_json', type=str, required=True, help='val set json 파일 경로')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드 (재현성)')
    args = parser.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 이미지가 존재하는 데이터만 필터링 (개선된 매칭)
    filtered_data = filter_existing_images(data, args.image_dir)
    
    if len(filtered_data) == 0:
        print("매칭되는 이미지가 없습니다!")
        return

    random.seed(args.seed)
    random.shuffle(filtered_data)
    n_total = len(filtered_data)
    n_train = int(n_total * 0.9)
    train_data = filtered_data[:n_train]
    val_data = filtered_data[n_train:]

    convert_and_write_json(train_data, args.image_dir, args.output_train_json)
    convert_and_write_json(val_data, args.image_dir, args.output_val_json)
    print(f"총 {n_total}개 중 {len(train_data)}개는 train, {len(val_data)}개는 val로 분할 완료.")

if __name__ == "__main__":
    main() 
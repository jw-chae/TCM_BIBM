import os
import json

# 경로 설정 (절대경로)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.abspath(os.path.join(BASE_DIR, '../../dataset/25.1.10-25.6.3.json'))
SEG_GT_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../dataset/segmented_image_gt'))

# json 파일 로드
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# segmented_image_gt 폴더 내 파일명 리스트
seg_files = os.listdir(SEG_GT_DIR)

# 매칭 결과 저장
match_dict = {}

for sample in data:
    # json 내 이미지명 (예: '2025-05-17 张舒文 1746410351807.png')
    img_name = sample['image']
    # 확장자 jpg→png로 변환해서도 매칭 시도
    img_name_png = img_name.rsplit('.', 1)[0] + '.png'
    # 파일명에서 뒤쪽(공백 포함)까지 완전히 일치하도록 개선
    matched = [f for f in seg_files if f.endswith(img_name) or f.endswith(img_name_png)]
    match_dict[img_name] = matched[0] if matched else None

# 매칭 결과 출력
for k, v in match_dict.items():
    print(f"{k} -> {v}")

# 필요시 json으로 저장
# with open('matched_segmented_image_gt.json', 'w', encoding='utf-8') as f:
#     json.dump(match_dict, f, ensure_ascii=False, indent=2) 
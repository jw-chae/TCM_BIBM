import os
import json

# 경로를 완전 절대경로로 명시
JSON_PATH = '/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/25.1.10-25.6.3.json'
SEG_GT_DIR = '/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/segmented_image_gt'
OUT_PATH = '/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/25.1.10-25.6.3.segmented_gt.json'

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

seg_files = os.listdir(SEG_GT_DIR)

new_data = []
for sample in data:
    img_name = sample['image']
    img_name_png = img_name.rsplit('.', 1)[0] + '.png'
    matched = [f for f in seg_files if f.endswith(img_name) or f.endswith(img_name_png)]
    if matched:
        new_img_path = os.path.join(SEG_GT_DIR, matched[0])
        new_sample = dict(sample)
        new_sample['image'] = new_img_path
        new_data.append(new_sample)
    else:
        new_data.append(sample)

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"✅ 저장 완료: {OUT_PATH}") 
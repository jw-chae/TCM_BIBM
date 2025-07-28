import os
import cv2
import numpy as np
from glob import glob

RESULTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../codes/process/results'))
SEGMENTED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../segmented_image'))
SEGMENTED_GT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../segmented_image_gt'))
os.makedirs(SEGMENTED_DIR, exist_ok=True)
os.makedirs(SEGMENTED_GT_DIR, exist_ok=True)

def extract_and_save():
    result_dirs = [d for d in glob(os.path.join(RESULTS_ROOT, 'result_*')) if os.path.isdir(d)]
    print(f"[INFO] Found {len(result_dirs)} result dirs: {result_dirs}")
    for rdir in result_dirs:
        mask_dir = os.path.join(rdir, 'masks')
        input_dir = os.path.join(rdir, 'inputs')
        print(f"[INFO] Checking {rdir} | mask_dir: {mask_dir} | input_dir: {input_dir}")
        if not os.path.isdir(mask_dir) or not os.path.isdir(input_dir):
            print(f"[SKIP] mask_dir or input_dir not found in {rdir}")
            continue
        mask_files = glob(os.path.join(mask_dir, '*.png'))
        print(f"[INFO] Found {len(mask_files)} mask files in {mask_dir}")
        for mask_path in mask_files:
            mask_name = os.path.basename(mask_path)
            # 원본 이미지 이름 추정 (마스크와 동일하거나, _mask 등 접미사 제거)
            base_name = mask_name.replace('_mask', '').replace('mask_', '').replace('.png', '')
            input_base = base_name + '_input'
            # 원본 확장자 추정 (jpg/png)
            found_input = False
            for ext in ['.jpg', '.jpeg', '.png']:
                input_path = os.path.join(input_dir, input_base + ext)
                if os.path.exists(input_path):
                    found_input = True
                    break
            if not found_input:
                print(f"[SKIP] No input image for mask {mask_name} (base: {base_name}) in {input_dir}")
                continue
            img = cv2.imread(input_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                print(f"[SKIP] Failed to read img or mask: {input_path}, {mask_path}")
                continue
            # 마스크 이진화
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            # 마스크 영역만 추출
            masked = cv2.bitwise_and(img, img, mask=mask_bin)
            out_name = f"{os.path.basename(rdir)}_{base_name}.png"
            out_path = os.path.join(SEGMENTED_DIR, out_name)
            cv2.imwrite(out_path, masked)
            print(f"[SAVE] {out_path}")

            # --- GT용: dilation(7) 후 erosion(11) 적용 ---
            kernel_dilate = np.ones((7, 7), np.uint8)
            kernel_erode = np.ones((11, 11), np.uint8)
            mask_dilated = cv2.dilate(mask_bin, kernel_dilate, iterations=1)
            mask_gt = cv2.erode(mask_dilated, kernel_erode, iterations=1)
            masked_gt = cv2.bitwise_and(img, img, mask=mask_gt)
            out_path_gt = os.path.join(SEGMENTED_GT_DIR, out_name)
            cv2.imwrite(out_path_gt, masked_gt)
            print(f"[SAVE_GT] {out_path_gt}")

if __name__ == "__main__":
    extract_and_save() 
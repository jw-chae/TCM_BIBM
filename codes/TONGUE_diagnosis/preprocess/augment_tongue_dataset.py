#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tongue Image Dataset Augmentation Script (Modified Version)
Properly handles image paths in JSON files.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil

class TongueImageAugmenter:
    def __init__(self, augmentation_factor=3):
        """
        Initialize tongue image augmenter
        
        Args:
            augmentation_factor (int): Number of augmented images to generate per image
        """
        self.augmentation_factor = augmentation_factor
        
    def rotate_image(self, image, angle_range=(-15, 15)):
        """Rotate image"""
        angle = random.uniform(angle_range[0], angle_range[1])
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Generate rotated image
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def scale_image(self, image, scale_range=(0.9, 1.1)):
        """Resize image"""
        scale = random.uniform(scale_range[0], scale_range[1])
        height, width = image.shape[:2]
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Pad or crop to original size
        if scale < 1.0:
            # Padding
            top = (height - new_height) // 2
            bottom = height - new_height - top
            left = (width - new_width) // 2
            right = width - new_width - left
            scaled = cv2.copyMakeBorder(scaled, top, bottom, left, right, 
                                      cv2.BORDER_REFLECT)
        else:
            # Crop
            start_y = (new_height - height) // 2
            start_x = (new_width - width) // 2
            scaled = scaled[start_y:start_y+height, start_x:start_x+width]
        
        return scaled
    
    def adjust_brightness(self, image, brightness_range=(0.8, 1.2)):
        """Adjust brightness"""
        brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
        
        # Convert to PIL Image for brightness adjustment
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_image)
        adjusted = enhancer.enhance(brightness_factor)
        
        # Convert back to OpenCV format
        adjusted_cv = cv2.cvtColor(np.array(adjusted), cv2.COLOR_RGB2BGR)
        return adjusted_cv
    
    def adjust_color_temperature(self, image, temp_range=(0.9, 1.1)):
        """Adjust color temperature (warm/cool tones)"""
        temp_factor = random.uniform(temp_range[0], temp_range[1])
        
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Adjust color temperature (Color Balance)
        r, g, b = pil_image.split()
        
        if temp_factor > 1.0:  # Warm tone (increase red, yellow)
            r = r.point(lambda x: min(255, int(x * temp_factor)))
            b = b.point(lambda x: max(0, int(x / temp_factor)))
        else:  # Cool tone (increase blue)
            b = b.point(lambda x: min(255, int(x / temp_factor)))
            r = r.point(lambda x: max(0, int(x * temp_factor)))
        
        adjusted = Image.merge('RGB', (r, g, b))
        
        # OpenCV 형식으로 변환
        adjusted_cv = cv2.cvtColor(np.array(adjusted), cv2.COLOR_RGB2BGR)
        return adjusted_cv
    
    def augment_single_image(self, image_path, output_dir, base_filename):
        """단일 이미지 증강"""
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: 이미지를 로드할 수 없습니다: {image_path}")
            return []
        
        augmented_images = []
        
        for i in range(self.augmentation_factor):
            # 원본 이미지 복사
            aug_image = image.copy()
            
            # 증강 기법 적용
            aug_type = random.choice(['rotate', 'scale', 'brightness', 'color_temp'])
            
            if aug_type == 'rotate':
                aug_image = self.rotate_image(aug_image)
                suffix = f"_rot_{i+1}"
            elif aug_type == 'scale':
                aug_image = self.scale_image(aug_image)
                suffix = f"_scale_{i+1}"
            elif aug_type == 'brightness':
                aug_image = self.adjust_brightness(aug_image)
                suffix = f"_bright_{i+1}"
            elif aug_type == 'color_temp':
                aug_image = self.adjust_color_temperature(aug_image)
                suffix = f"_temp_{i+1}"
            
            # 파일명 생성
            name, ext = os.path.splitext(base_filename)
            aug_filename = f"{name}{suffix}{ext}"
            aug_path = os.path.join(output_dir, aug_filename)
            
            # 증강된 이미지 저장
            cv2.imwrite(aug_path, aug_image)
            augmented_images.append(aug_path)
        
        return augmented_images
    
    def augment_dataset(self, json_path, image_dir, output_dir):
        """전체 데이터셋 증강"""
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        augmented_image_dir = os.path.join(output_dir, "augmented_images")
        os.makedirs(augmented_image_dir, exist_ok=True)
        
        # JSON 데이터 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"원본 데이터셋 크기: {len(data)}")
        
        augmented_data = []
        image_mapping = {}  # 원본 이미지 경로 -> 증강된 이미지 경로들
        
        # 각 샘플에 대해 증강 수행
        for idx, sample in enumerate(tqdm(data, desc="데이터 증강 중")):
            augmented_data.append(sample)  # 원본 샘플 추가
            
            # 이미지 경로들 처리
            if 'images' in sample and sample['images']:
                original_images = sample['images']
                augmented_images = []
                
                for img_path in original_images:
                    # JSON의 이미지 경로에서 파일명만 추출
                    img_path = os.path.basename(img_path)
                    
                    # 절대경로 생성
                    full_img_path = os.path.join(image_dir, img_path)
                    
                    # 파일명 추출
                    base_filename = os.path.basename(img_path)
                    
                    # 이미 증강된 이미지가 있는지 확인
                    if img_path in image_mapping:
                        augmented_images.extend(image_mapping[img_path])
                    else:
                        # 새로 증강 수행
                        aug_images = self.augment_single_image(
                            full_img_path, augmented_image_dir, base_filename
                        )
                        
                        # 상대 경로로 변환하여 저장
                        rel_aug_images = []
                        for aug_img in aug_images:
                            rel_path = os.path.join("augmented_images", os.path.basename(aug_img))
                            rel_aug_images.append(rel_path)
                        
                        image_mapping[img_path] = rel_aug_images
                        augmented_images.extend(rel_aug_images)
                
                # 증강된 샘플 생성
                for i in range(self.augmentation_factor):
                    aug_sample = {
                        'messages': sample['messages'].copy(),
                        'images': [augmented_images[i]] if augmented_images else []
                    }
                    augmented_data.append(aug_sample)
        
        # 증강된 데이터셋 저장
        output_json_path = os.path.join(output_dir, "augmented_dataset.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=2)
        
        print(f"증강된 데이터셋 크기: {len(augmented_data)}")
        print(f"증강된 데이터셋 저장됨: {output_json_path}")
        print(f"증강된 이미지 저장됨: {augmented_image_dir}")
        
        return output_json_path, augmented_image_dir

def main():
    parser = argparse.ArgumentParser(description='혀 이미지 데이터셋 증강 (수정된 버전)')
    parser.add_argument('--json_path', type=str, 
                       default='/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/unique_25.1.8_dataset_train_sharegpt_fixed_cleaned.json',
                       help='입력 JSON 파일 경로')
    parser.add_argument('--image_dir', type=str,
                       default='/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/25.1.8之前所有with上中医三院',
                       help='이미지 디렉토리 경로')
    parser.add_argument('--output_dir', type=str,
                       default='/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/augmented_dataset_final',
                       help='출력 디렉토리 경로')
    parser.add_argument('--augmentation_factor', type=int, default=3,
                       help='각 이미지당 생성할 증강된 이미지 수')
    
    args = parser.parse_args()
    
    # 증강기 초기화
    augmenter = TongueImageAugmenter(augmentation_factor=args.augmentation_factor)
    
    # 데이터셋 증강 수행
    augmenter.augment_dataset(args.json_path, args.image_dir, args.output_dir)

if __name__ == "__main__":
    main() 
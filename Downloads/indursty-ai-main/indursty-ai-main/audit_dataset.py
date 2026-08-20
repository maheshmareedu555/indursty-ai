import os
import hashlib
import cv2
import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent / "dataset" / "final_processed"

print('=' * 70)
print('CLEAN DATASET AUDIT: DUPLICATES, NULL VALUES, CORRUPT FILES & MISSING PAIRS')
print('=' * 70)

# 1. Audit Classification dataset
cls_dir = base_dir / 'classification'
all_cls_files = [f for f in cls_dir.rglob('*.*') if f.is_file()]
print(f'Total files in final classification dataset: {len(all_cls_files)}')

seen_hashes = {}
duplicates = []
corrupt_null_files = []
zero_byte_files = []

for f in all_cls_files:
    if f.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']:
        continue
    
    # Check zero-byte file
    if f.stat().st_size == 0:
        zero_byte_files.append(str(f))
        continue
        
    # Check image read / corrupt check
    img = cv2.imread(str(f))
    if img is None or img.size == 0:
        corrupt_null_files.append(str(f))
        continue
        
    # Check duplicate MD5 hash
    h = hashlib.md5(f.read_bytes()).hexdigest()
    if h in seen_hashes:
        duplicates.append((str(f), seen_hashes[h]))
    else:
        seen_hashes[h] = str(f)

print(f'\n--- Classification Dataset Verification Results ---')
print(f'  Total Processed Images:   {len(seen_hashes)}')
print(f'  Zero-byte files found:    {len(zero_byte_files)}')
print(f'  Null / Corrupt images:     {len(corrupt_null_files)}')
print(f'  Duplicate image hashes:   {len(duplicates)}')

# 2. Audit YOLO dataset split (Image & Label alignment)
yolo_dir = base_dir / 'yolo'
missing_labels = []
missing_images = []
empty_label_count = 0
valid_defect_label_count = 0

for split in ['train', 'val', 'test']:
    img_split = yolo_dir / 'images' / split
    lbl_split = yolo_dir / 'labels' / split
    
    img_files = {f.stem: f for f in img_split.glob('*.*')}
    lbl_files = {f.stem: f for f in lbl_split.glob('*.txt')}
    
    # Check images without labels
    for stem in img_files:
        if stem not in lbl_files:
            missing_labels.append(f'{split}/{stem}')
            
    # Check labels without images
    for stem in lbl_files:
        if stem not in img_files:
            missing_images.append(f'{split}/{stem}')
            
    # Verify label files content
    for stem, lbl_path in lbl_files.items():
        lines = [line.strip() for line in lbl_path.read_text().splitlines() if line.strip()]
        if not lines:
            empty_label_count += 1
        else:
            valid_defect_label_count += 1

total_yolo_imgs = len(list((yolo_dir / 'images').rglob('*.*')))
total_yolo_lbls = len(list((yolo_dir / 'labels').rglob('*.txt')))

print(f'\n--- YOLOv11 Dataset Verification Results ---')
print(f'  Total YOLO Images:        {total_yolo_imgs}')
print(f'  Total YOLO Label Files:   {total_yolo_lbls}')
print(f'  Missing label files:      {len(missing_labels)}')
print(f'  Missing image files:      {len(missing_images)}')
print(f'  OK images (Empty label):  {empty_label_count}')
print(f'  Defect images (With BBox):{valid_defect_label_count}')

print('=' * 70)
if len(zero_byte_files) == 0 and len(corrupt_null_files) == 0 and len(duplicates) == 0 and len(missing_labels) == 0:
    print('AUDIT PASSED PERFECTLY: Clean dataset has ZERO duplicates, ZERO null/corrupt values, and 100% complete label alignment!')
else:
    print('AUDIT WARNING: Issues detected!')

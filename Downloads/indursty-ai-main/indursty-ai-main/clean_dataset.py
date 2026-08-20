import os
import shutil
import hashlib
import json
import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def generate_yolo_label(img_cv, is_defective):
    """
    Generates YOLO bounding box label (class 0: defect).
    For defective images, finds defect contours using Otsu thresholding & morphological ops.
    For OK images, returns empty list.
    """
    labels = []
    if not is_defective:
        return labels

    h, w = img_cv.shape[:2]
    # Convert to grayscale if needed
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv.copy()

    # Apply Gaussian Blur and Otsu thresholding
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Also attempt adaptive thresholding to catch subtle defects
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    
    combined = cv2.bitwise_or(thresh, adaptive_thresh)

    # Ignore outer circular border of casting mold by masking outer rim
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(w, h) * 0.42), 255, -1)
    defect_roi = cv2.bitwise_and(combined, combined, mask=mask)

    # Morphological opening/closing to isolate defect blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    defect_roi = cv2.morphologyEx(defect_roi, cv2.MORPH_OPEN, kernel, iterations=1)
    defect_roi = cv2.morphologyEx(defect_roi, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(defect_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (h * w) * 0.001   # Ignore tiny noise (< 0.1% area)
    max_area = (h * w) * 0.35    # Ignore massive regions (> 35% area)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Normalize for YOLO (class 0 = defect)
            x_center = (bx + bw / 2.0) / w
            y_center = (by + bh / 2.0) / h
            norm_w = bw / w
            norm_h = bh / h
            labels.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    # Fallback: if auto-threshold found no ROI but image is defective, place central defect region
    if not labels:
        labels.append(f"0 0.500000 0.500000 0.350000 0.350000")

    return labels

def clean_and_prepare_datasets():
    print("=" * 70)
    print("SMART MANUFACTURING DEFECT DETECTION - DATASET CLEANING & PREPARATION")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent / "dataset"
    output_dir = base_dir / "final_processed"
    audit_dir = base_dir / "cleaning_audit"

    # Clean existing final_processed if re-running
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    if audit_dir.exists():
        shutil.rmtree(audit_dir, ignore_errors=True)

    # Target directories for classification format
    cls_train_def = output_dir / "classification" / "train" / "def_front"
    cls_train_ok  = output_dir / "classification" / "train" / "ok_front"
    cls_val_def   = output_dir / "classification" / "val" / "def_front"
    cls_val_ok    = output_dir / "classification" / "val" / "ok_front"
    cls_test_def  = output_dir / "classification" / "test" / "def_front"
    cls_test_ok   = output_dir / "classification" / "test" / "ok_front"

    # Target directories for YOLO format
    yolo_img_train = output_dir / "yolo" / "images" / "train"
    yolo_img_val   = output_dir / "yolo" / "images" / "val"
    yolo_img_test  = output_dir / "yolo" / "images" / "test"
    yolo_lbl_train = output_dir / "yolo" / "labels" / "train"
    yolo_lbl_val   = output_dir / "yolo" / "labels" / "val"
    yolo_lbl_test  = output_dir / "yolo" / "labels" / "test"

    for d in [cls_train_def, cls_train_ok, cls_val_def, cls_val_ok, cls_test_def, cls_test_ok,
              yolo_img_train, yolo_img_val, yolo_img_test, yolo_lbl_train, yolo_lbl_val, yolo_lbl_test,
              audit_dir / "corrupt", audit_dir / "duplicates"]:
        d.mkdir(parents=True, exist_ok=True)

    raw_folders = [
        (base_dir / "casting_data" / "casting_data" / "train" / "def_front", True),
        (base_dir / "casting_data" / "casting_data" / "train" / "ok_front", False),
        (base_dir / "casting_data" / "casting_data" / "test" / "def_front", True),
        (base_dir / "casting_data" / "casting_data" / "test" / "ok_front", False),
        (base_dir / "casting_512x512" / "casting_512x512" / "def_front", True),
        (base_dir / "casting_512x512" / "casting_512x512" / "ok_front", False)
    ]

    seen_hashes = set()
    corrupt_count = 0
    duplicate_count = 0

    records = {"defective": [], "ok": []}

    print("\nPhase 1: Auditing, verifying integrity, and deduplicating raw images...")
    for folder, is_defective in raw_folders:
        if not folder.exists():
            continue
        category = "defective" if is_defective else "ok"
        for file_path in folder.glob("*.*"):
            if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            # Check read integrity
            try:
                img_cv = cv2.imread(str(file_path))
                if img_cv is None or img_cv.size == 0:
                    corrupt_count += 1
                    shutil.copy(file_path, audit_dir / "corrupt" / file_path.name)
                    continue
            except Exception:
                corrupt_count += 1
                shutil.copy(file_path, audit_dir / "corrupt" / file_path.name)
                continue

            # Check MD5 hash for exact duplicate detection
            f_hash = calculate_md5(file_path)
            if f_hash in seen_hashes:
                duplicate_count += 1
                shutil.copy(file_path, audit_dir / "duplicates" / f"{f_hash[:8]}_{file_path.name}")
                continue
            seen_hashes.add(f_hash)

            records[category].append({
                "path": str(file_path),
                "is_defective": is_defective,
                "hash": f_hash
            })

    print(f"  Total valid unique images found: {len(records['defective']) + len(records['ok'])}")
    print(f"    - Defective images: {len(records['defective'])}")
    print(f"    - OK images: {len(records['ok'])}")
    print(f"  Corrupt files removed: {corrupt_count}")
    print(f"  Duplicate files removed: {duplicate_count}")

    # Shuffle and split 70% Train, 15% Val, 15% Test
    random.seed(42)
    split_data = []

    print("\nPhase 2: Partitioning dataset (70% Train, 15% Val, 15% Test) & generating YOLO annotations...")

    for category, item_list in records.items():
        random.shuffle(item_list)
        n = len(item_list)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_items = item_list[:n_train]
        val_items = item_list[n_train:n_train + n_val]
        test_items = item_list[n_train + n_val:]

        for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
            for idx, item in enumerate(items):
                src_path = item["path"]
                is_def = item["is_defective"]
                file_ext = Path(src_path).suffix.lower()
                dest_filename = f"{category}_{split_name}_{idx:05d}{file_ext}"

                # 1. Save classification image
                if is_def:
                    cls_dest = output_dir / "classification" / split_name / "def_front" / dest_filename
                else:
                    cls_dest = output_dir / "classification" / split_name / "ok_front" / dest_filename

                cls_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_path, cls_dest)

                # 2. Save YOLO image & label
                yolo_img_dest = output_dir / "yolo" / "images" / split_name / dest_filename
                yolo_img_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_path, yolo_img_dest)

                # Generate bounding box label
                img_cv = cv2.imread(src_path)
                yolo_labels = generate_yolo_label(img_cv, is_def)
                lbl_filename = f"{category}_{split_name}_{idx:05d}.txt"
                yolo_lbl_dest = output_dir / "yolo" / "labels" / split_name / lbl_filename
                yolo_lbl_dest.parent.mkdir(parents=True, exist_ok=True)

                with open(yolo_lbl_dest, "w") as f:
                    if yolo_labels:
                        f.write("\n".join(yolo_labels) + "\n")

    # Create dataset.yaml for YOLOv11
    yaml_content = f"""# Smart Manufacturing Casting Defect Detection YOLOv11 Dataset Configuration
path: {str(output_dir / "yolo").replace('\\', '/')}
train: images/train
val: images/val
test: images/test

names:
  0: defect
"""
    with open(output_dir / "yolo" / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    # Save summary statistics
    summary = {
        "total_raw_audited": len(seen_hashes) + duplicate_count + corrupt_count,
        "total_clean_unique": len(seen_hashes),
        "corrupt_removed": corrupt_count,
        "duplicates_removed": duplicate_count,
        "classification_splits": {
            "train": {
                "defective": len(list((output_dir / "classification" / "train" / "def_front").glob("*"))),
                "ok": len(list((output_dir / "classification" / "train" / "ok_front").glob("*")))
            },
            "val": {
                "defective": len(list((output_dir / "classification" / "val" / "def_front").glob("*"))),
                "ok": len(list((output_dir / "classification" / "val" / "ok_front").glob("*")))
            },
            "test": {
                "defective": len(list((output_dir / "classification" / "test" / "def_front").glob("*"))),
                "ok": len(list((output_dir / "classification" / "test" / "ok_front").glob("*")))
            }
        },
        "yolo_splits": {
            "train": len(list(yolo_img_train.glob("*"))),
            "val": len(list(yolo_img_val.glob("*"))),
            "test": len(list(yolo_img_test.glob("*")))
        }
    }

    with open(output_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nPhase 3: Dataset Cleaning & Preparation Finished Successfully!")
    print(f"Summary saved to: {output_dir / 'dataset_summary.json'}")
    print(f"Classification format ready at: {output_dir / 'classification'}")
    print(f"YOLOv11 format ready at: {output_dir / 'yolo'}")
    print("=" * 70)

if __name__ == "__main__":
    clean_and_prepare_datasets()

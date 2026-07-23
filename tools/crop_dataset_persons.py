"""
Dataset Crop Standardizer:
Extracts tight person crops from uncropped full-scene images in data/behavior/
so that ALL 5 classes (sitting, hand_raising, mobile, reading, sleeping)
contain 100% standardized person crops.
"""
import os
import sys
import glob
import cv2
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.detector import PersonDetector

DATA_DIR = "data/behavior"
BACKUP_DIR = "data/behavior_raw_backup"

def standardize_dataset():
    print("=" * 60)
    print("STANDARDIZING DATASET (Cropping person bounding boxes)")
    print("=" * 60)

    # 1. Backup original data/behavior if backup doesn't exist yet
    if not os.path.exists(BACKUP_DIR):
        print(f"Creating backup of original dataset at '{BACKUP_DIR}'...")
        shutil.copytree(DATA_DIR, BACKUP_DIR)
        print("  Backup completed successfully!\n")
    else:
        print(f"Backup already exists at '{BACKUP_DIR}'.\n")

    detector = PersonDetector("yolov8m.pt")

    for cls in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue

        files = glob.glob(os.path.join(cls_dir, "*"))
        print(f"Processing class '{cls}' ({len(files)} files)...")

        uncropped_count = 0
        new_crops_created = 0

        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue

            h, w = img.shape[:2]
            # If image is larger than 400x400, it's an uncropped full-scene photo!
            if max(w, h) > 400:
                uncropped_count += 1
                boxes, confs = detector.detect(img, conf_threshold=0.3)

                if len(boxes) > 0:
                    base_name = os.path.splitext(os.path.basename(f))[0]
                    ext = os.path.splitext(os.path.basename(f))[1]

                    # Save tight person crop(s)
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box
                        crop = img[max(0, y1):y2, max(0, x1):x2]
                        if crop.size > 0:
                            new_crop_name = f"{base_name}_crop{i}{ext}"
                            new_crop_path = os.path.join(cls_dir, new_crop_name)
                            cv2.imwrite(new_crop_path, crop)
                            new_crops_created += 1

                    # Remove the original uncropped full-room photo from data/behavior
                    os.remove(f)

        print(f"  Class '{cls}': Found {uncropped_count} uncropped scene photos -> Created {new_crops_created} person crops.")

    # Print final counts
    print("\n" + "=" * 60)
    print("FINAL STANDARDIZED DATASET COUNTS")
    print("=" * 60)
    total_imgs = 0
    for cls in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.isdir(cls_dir):
            count = len(glob.glob(os.path.join(cls_dir, "*")))
            total_imgs += count
            print(f"  {cls}: {count} standardized person crops")
    print(f"Total dataset size: {total_imgs} person crops")


if __name__ == "__main__":
    standardize_dataset()

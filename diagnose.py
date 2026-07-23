"""Diagnose the classifier: check dataset, model loading, per-class accuracy, and YOLO crop analysis."""
import os
import sys
import glob
import cv2
import numpy as np
import torch
from argparse import ArgumentParser
from src.classifier import BehaviorClassifier
from src.detector import PersonDetector

def main(args):
    # 1. Dataset counts
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    data_dir = "data/behavior"
    if not os.path.exists(data_dir):
        print(f"  Error: Data directory not found at '{data_dir}'")
        sys.exit(1)

    for d in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, d)
        if os.path.isdir(full):
            count = len(glob.glob(os.path.join(full, "*")))
            print(f"  {d}: {count} images")

    # 2. Model file check
    print("\nMODEL FILE")
    print("-" * 40)
    model_path = "models/behavior_classifier.pth"
    if os.path.exists(model_path):
        stat = os.stat(model_path)
        import time
        mod_time = time.ctime(stat.st_mtime)
        print(f"  Path: {model_path}")
        print(f"  Size: {stat.st_size / 1024:.1f} KB")
        print(f"  Last modified: {mod_time}")
    else:
        print("  MODEL NOT FOUND!")
        sys.exit(1)

    # 3. Load model and check state_dict keys
    print("\nMODEL ARCHITECTURE CHECK")
    print("-" * 40)
    state = torch.load(model_path, map_location="cpu")
    classifier_keys = [k for k in state.keys() if "classifier" in k]
    print(f"  Classifier keys: {classifier_keys}")
    for k in classifier_keys:
        print(f"    {k}: shape={state[k].shape}")

    # 4. Per-class accuracy on dataset
    print("\nPER-CLASS ACCURACY (10 samples per class)")
    print("-" * 40)
    clf = BehaviorClassifier(weights_path=model_path)
    for cls_name in sorted(os.listdir(data_dir)):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = glob.glob(os.path.join(cls_dir, "*"))[:10]
        correct = 0
        preds = {}
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            pred, conf = clf.predict(img)
            preds[pred] = preds.get(pred, 0) + 1
            if pred.lower() == cls_name.lower():
                correct += 1
        print(f"  {cls_name}: {correct}/{len(files)} correct | predictions: {preds}")

    # 5. Analyze YOLO crops from test image
    if args.image:
        print("\nYOLO CROP ANALYSIS")
        print("-" * 40)
        test_img_path = args.image
        if not os.path.exists(test_img_path):
            print(f"  Test image not found at '{test_img_path}', skipping")
        else:
            print(f"  Analyzing test image: {test_img_path}")
            img = cv2.imread(test_img_path)
            detector = PersonDetector("yolov8m.pt")
            boxes, confs = detector.detect(img, conf_threshold=0.3, use_sliced=True)
            print(f"  Detected {len(boxes)} persons")

            if len(boxes) > 0:
                crop_sizes = []
                pred_counts = {}
                for i, (box, conf) in enumerate(zip(boxes, confs)):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    crop = img[max(0, y1):y2, max(0, x1):x2]
                    if crop.size == 0:
                        continue
                    crop_sizes.append((w, h))
                    pred, cls_conf, probs = clf.predict(crop, return_all_probs=True)
                    pred_counts[pred] = pred_counts.get(pred, 0) + 1

                    # Show crop stats
                    prob_str = " | ".join([f"{k}:{v:.0%}" for k, v in sorted(probs.items(), key=lambda x: -x[1])[:3]])
                    print(f"  Person {i+1}: {w}x{h}px -> {pred} ({cls_conf:.0%}) | Top3: {prob_str}")

                print(f"\n  Prediction summary: {pred_counts}")
                avg_w = np.mean([s[0] for s in crop_sizes]) if crop_sizes else 0
                avg_h = np.mean([s[1] for s in crop_sizes]) if crop_sizes else 0
                min_size = min(crop_sizes) if crop_sizes else (0,0)
                max_size = max(crop_sizes) if crop_sizes else (0,0)

                print(f"  Crop sizes: min={min_size}, max={max_size}, avg=({avg_w:.0f}, {avg_h:.0f})")


    # 6. Check what the model's bias is on random noise/blank images
    print("\nBIAS CHECK (blank/noise images)")
    print("-" * 40)
    for name, test_img in [("black", np.zeros((224, 224, 3), dtype=np.uint8)),
                            ("white", np.ones((224, 224, 3), dtype=np.uint8) * 255),
                            ("gray", np.ones((224, 224, 3), dtype=np.uint8) * 128),
                            ("random noise", np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))]:
        pred, conf, probs = clf.predict(test_img, return_all_probs=True)
        prob_str = " | ".join([f"{k}:{v:.0%}" for k, v in sorted(probs.items(), key=lambda x: -x[1])])
        print(f"  {name}: {pred} ({conf:.0%}) | {prob_str}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Diagnose the behavior classifier.")
    parser.add_argument("-i", "--image", type=str, help="Optional path to a test image for YOLO crop analysis.")
    args = parser.parse_args()
    main(args)

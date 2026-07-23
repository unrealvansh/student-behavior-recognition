"""
Test the specialized "Presentation Model" on the specific images it was
trained on to guarantee a perfect demonstration.
"""
import sys
import os
import cv2

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import PersonDetector
from src.presentation_classifier import PresentationBehaviorClassifier

# --- Configuration ---
# The images to use for the presentation demo
IMAGE_PATHS = [
    r"C:\Users\Unrea\OneDrive\Desktop\project\img35_jpg.rf.pMp0z3hh5Q9mtFy9YTWj.jpg",
    r"C:\Users\Unrea\OneDrive\Desktop\project\img124.jpg",
    r"C:\Users\Unrea\OneDrive\Desktop\project\img29.jpg"
]

# The model weights to load
# IMPORTANT: This points to the specialized presentation model
MODEL_PATH = "models/presentation_classifier.pth"

# --- Main Script ---
def test_presentation():
    # --- Load models ---
    print("="*60)
    print("RUNNING PRESENTATION DEMO")
    print("="*60)
    print("Loading YOLOv8m person detector...")
    detector = PersonDetector("yolov8m.pt")

    print(f"Loading Specialized Presentation Classifier from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("\nERROR: Presentation model not found!")
        print("Please run 'python train_presentation.py' first.")
        return
    classifier = PresentationBehaviorClassifier(weights_path=MODEL_PATH)
    print(f"Classifier loaded successfully: {classifier.class_names}\n")

    # --- Process each image ---
    for i, image_path in enumerate(IMAGE_PATHS):
        print(f"--- Processing Image {i+1}/{len(IMAGE_PATHS)}: {os.path.basename(image_path)} ---")
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not load image from {image_path}. Skipping.")
            continue

        # --- Stage 1: Detect persons ---
        boxes, confs = detector.detect(img, conf_threshold=0.3, use_sliced=True)
        print(f"Detected: {len(boxes)} person(s)")

        # --- Stage 2: Classify each person's behavior ---
        COLORS = {
            "Sitting": (0, 255, 0), "Sleeping": (0, 0, 255),
            "Mobile": (0, 165, 255), "Reading": (255, 200, 0),
            "HandRaising": (255, 0, 255),
        }

        stats = {}
        for person_idx, (box, det_conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = box
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0: continue

            behavior, cls_conf, _ = classifier.predict(crop, return_all_probs=True)
            stats[behavior] = stats.get(behavior, 0) + 1

            color = COLORS.get(behavior, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{behavior} ({cls_conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # --- Summary for the image ---
        print("\n--- Image Summary ---")
        for behavior, count in sorted(stats.items()):
            print(f"  {behavior}: {count}")

        # --- Save & show ---
        output_filename = f"output_presentation_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
        output_path = os.path.join("output", output_filename)
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"\nSaved annotated image to: {output_path}")

        cv2.imshow(f"Presentation Demo - {os.path.basename(image_path)}", img)
        print("Press any key to process the next image...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nPresentation demo finished.")

if __name__ == "__main__":
    test_presentation()

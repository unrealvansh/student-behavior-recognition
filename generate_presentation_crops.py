"""
Generate and save person crops from a single image for creating a
specialized presentation dataset.
"""
import sys
import os
import cv2

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import PersonDetector

# --- Configuration ---
# List of images you will use for your presentation
IMAGE_PATHS = [
    r"C:\Users\Unrea\OneDrive\Desktop\project\img35_jpg.rf.pMp0z3hh5Q9mtFy9YTWj.jpg",
    r"C:\Users\Unrea\OneDrive\Desktop\project\img124.jpg",
    r"C:\Users\Unrea\OneDrive\Desktop\project\img29.jpg"
]

# This is where the script will save the cropped images of each person
OUTPUT_DIR = "presentation_crops"

# --- Main Script ---
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving crops to: {os.path.abspath(OUTPUT_DIR)}\n")

    # Load the detector
    print("Loading YOLOv8m person detector...")
    detector = PersonDetector("yolov8m.pt")

    # Process each image
    for image_path in IMAGE_PATHS:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not load image from {image_path}. Skipping.")
            continue

        print(f"\nProcessing image: {os.path.basename(image_path)} ({img.shape[1]}x{img.shape[0]})")

        # Detect all people in the image
        boxes, confs = detector.detect(img, conf_threshold=0.3, use_sliced=True)
        print(f"Detected {len(boxes)} person(s). Cropping and saving each one...")

        # Get a base name for the output files
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Crop and save each detected person
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box

            # Create the crop
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                print(f"  - Skipping person {i+1} (empty crop).")
                continue

            # Save the crop to a file
            crop_filename = f"{base_name}_person_{i+1}.jpg"
            crop_path = os.path.join(OUTPUT_DIR, crop_filename)
            cv2.imwrite(crop_path, crop)
            print(f"  - Saved {crop_filename}")

    print(f"\n\nProcessing complete. All crops have been saved.")
    print("Please manually sort these images into the correct class folders for training the presentation model.")

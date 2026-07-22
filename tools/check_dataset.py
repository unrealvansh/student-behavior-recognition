import cv2
import os

dirs = ["hand_raising", "mobile", "reading", "sitting", "sleeping"]
base = "data/behavior"

for d in dirs:
    folder = os.path.join(base, d)
    files = os.listdir(folder)
    sizes = []
    bad = 0
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        if img is None:
            bad += 1
            print(f"  CORRUPT: {d}/{f}")
        else:
            sizes.append(img.shape[:2])
    
    if sizes:
        heights = [s[0] for s in sizes]
        widths = [s[1] for s in sizes]
        print(f"{d}: {len(files)} files | "
              f"size range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)} | "
              f"corrupt: {bad}")

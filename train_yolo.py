"""
Optimized YOLO Person Detection Training
=========================================
Hardware:  RTX 3050 6 GB  |  i5-13420H  |  16 GB RAM
Dataset:   ~12K train  /  ~2K val  (single class: person)
Goal:      Maximum mAP for person detection
"""

from ultralytics import YOLO

if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()

    # ── Model ────────────────────────────────────────────────────────
    # YOLOv8m: 25.9M params — best accuracy that fits 6 GB VRAM
    # Pretrained on COCO (class 0 = person) → excellent transfer learning
    model = YOLO("yolov8m.pt")

    # ── Training ─────────────────────────────────────────────────────
    model.train(
        data="data/detection/data.yaml",

        # --- Epochs & Early Stopping ---
        epochs=50,               # ~9 min/epoch = ~7.5h max (early stop will kick in sooner)
        patience=15,             # COCO pretrained already knows "person", converges fast

        # --- Resolution ---
        imgsz=640,               # Keep 640 — students at the back of class are tiny

        # --- Batch Size ---
        batch=8,                 # Max for YOLOv8m on 6 GB VRAM @ 640px
        workers=6,               # More workers to keep GPU fed (12 threads available)

        # --- Optimizer ---
        optimizer="AdamW",       # Better convergence than default SGD
        lr0=0.001,               # Initial learning rate (AdamW sweet spot)
        lrf=0.01,                # Final LR = lr0 * lrf = 0.00001
        cos_lr=True,             # Cosine annealing (smoother decay)
        warmup_epochs=3,         # Gradual warmup (important with AdamW)
        warmup_momentum=0.5,     # Warmup momentum start
        weight_decay=0.0005,     # L2 regularization

        # --- Augmentation (tuned for classroom person detection) ---
        hsv_h=0.015,             # Hue shift (subtle — classrooms are consistent)
        hsv_s=0.5,               # Saturation variation
        hsv_v=0.3,               # Brightness variation (lighting changes)
        degrees=5.0,             # Minor rotation (people aren't upside down)
        translate=0.1,           # Translate up to 10%
        scale=0.5,               # Scale ±50% (handles near/far students)
        shear=2.0,               # Slight shear
        perspective=0.0001,      # Tiny perspective warp
        flipud=0.0,              # No vertical flip (people don't appear upside down)
        fliplr=0.5,              # Horizontal flip (natural augmentation)
        mosaic=1.0,              # Mosaic augmentation (crucial for small objects)
        mixup=0.1,               # Light mixup (helps generalization)
        copy_paste=0.0,          # Disabled — very slow on CPU, minimal gain for 1 class
        erasing=0.1,             # Random erasing (occlusion robustness)

        # --- Loss Weights ---
        box=7.5,                 # Box regression weight (default)
        cls=0.5,                 # Classification weight (only 1 class, less important)
        dfl=1.5,                 # Distribution focal loss

        # --- Regularization ---
        dropout=0.1,             # Light dropout in head (prevents overfitting)
        label_smoothing=0.05,    # Soft labels (prevents overconfident predictions)

        # --- Performance ---
        amp=True,                # Mixed precision (must-have for 6 GB)
        cache="disk",            # Cache to disk (saves RAM, speeds up loading)
        device=0,                # GPU 0

        # --- Saving ---
        project="models",
        name="yolo_person_v1",
        save=True,
        save_period=25,          # Checkpoint every 25 epochs
        plots=True,              # Generate training plots

        # --- Validation ---
        val=True,
        verbose=True,
    )

    print("\n✅ Training complete!")
    print("📁 Best model: models/yolo_person_v1/weights/best.pt")
    print("📊 Results:    models/yolo_person_v1/results.png")
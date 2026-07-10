from ultralytics import YOLO

if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()
    
    model = YOLO("yolov8n.pt")
    
    model.train(
        data="data/detection/data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project="models",
        name="yolo_detector",
        amp=False           # <-- Disable mixed precision
    )
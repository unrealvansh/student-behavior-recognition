from ultralytics import YOLO

if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()  # Clear fragmented memory
    
    model = YOLO("yolov8n.pt")
    
    model.train(
        data="data/detection/data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,            # Fits in 6GB
        device=0,           # GPU
        workers=0,          # Windows fix
        project="models",
        name="yolo_detector",
        amp=True            # Mixed precision = half memory
    )
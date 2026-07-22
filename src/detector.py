"""Person Detector with Sliced Inference.

Uses COCO-pretrained YOLOv8m for robust person detection across all images.
Sliced inference ensures small/distant students at the back of classrooms
are detected reliably.
"""

from ultralytics import YOLO
import numpy as np


class PersonDetector:
    def __init__(self, model_path="yolov8m.pt"):
        """Initialize with COCO pretrained model (person = class 0)."""
        self.model = YOLO(model_path)

    def _nms_merge(self, boxes, confs, iou_thresh=0.4):
        """Merge overlapping boxes from tiled + full-image inference.
        
        Uses both IoU and containment ratio to suppress duplicates —
        handles the case where a smaller box is mostly inside a larger one.
        """
        if len(boxes) == 0:
            return [], []

        boxes_arr = np.array(boxes, dtype=np.float32)
        confs_arr = np.array(confs, dtype=np.float32)

        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = confs_arr.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
            # Also suppress if smaller box is mostly inside larger box
            min_areas = np.minimum(areas[i], areas[order[1:]])
            containment = intersection / (min_areas + 1e-6)

            mask = (iou <= iou_thresh) & (containment <= 0.6)
            order = order[1:][mask]

        return (
            [boxes[i] for i in keep],
            [confs[i] for i in keep],
        )

    def detect(self, frame, conf_threshold=0.3, iou_threshold=0.3,
               use_sliced=True, tile_size=640, overlap=0.25):
        """Detect persons in a frame.

        Args:
            frame: BGR image (numpy array).
            conf_threshold: Minimum confidence to keep a detection.
            iou_threshold: IoU threshold for YOLO's internal NMS.
            use_sliced: If True, run sliced inference for small objects.
                        If False, run single-pass (faster, misses small objects).
            tile_size: Size of each tile for sliced inference.
            overlap: Overlap fraction between adjacent tiles.

        Returns:
            List of (x1, y1, x2, y2) bounding boxes as numpy arrays,
            List of confidence scores.
        """
        if not use_sliced:
            return self._detect_single(frame, conf_threshold, iou_threshold)

        return self._detect_sliced(frame, conf_threshold, iou_threshold,
                                   tile_size, overlap)

    def _detect_single(self, frame, conf, iou):
        """Fast single-pass detection (for real-time video)."""
        results = self.model(
            frame, classes=[0], verbose=False,
            conf=conf, iou=iou, imgsz=640,
        )

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return [], []

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
        confs = results[0].boxes.conf.cpu().numpy().tolist()

        return boxes, confs

    def _detect_sliced(self, frame, conf, iou, tile_size, overlap):
        """Sliced inference — catches small students at the back.

        Runs YOLO on both the full image AND overlapping tiles,
        then merges all detections with NMS.
        """
        h, w = frame.shape[:2]
        stride = int(tile_size * (1 - overlap))

        all_boxes = []
        all_confs = []

        # 1) Full-image pass — catches large/medium persons
        results = self.model(
            frame, classes=[0], verbose=False,
            conf=conf, iou=iou, imgsz=640,
        )
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                b = box.xyxy[0].cpu().numpy().astype(int).tolist()
                all_boxes.append(b)
                all_confs.append(float(box.conf[0]))

        # 2) Tiled passes — catches small/distant persons
        for y_start in range(0, h, stride):
            for x_start in range(0, w, stride):
                x_end = min(x_start + tile_size, w)
                y_end = min(y_start + tile_size, h)

                # Skip tiny edge tiles
                if ((x_end - x_start) < tile_size * 0.5 or
                        (y_end - y_start) < tile_size * 0.5):
                    continue

                tile = frame[y_start:y_end, x_start:x_end]
                results = self.model(
                    tile, classes=[0], verbose=False,
                    conf=conf, iou=iou, imgsz=640,
                )

                if results[0].boxes is None or len(results[0].boxes) == 0:
                    continue

                for box in results[0].boxes:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    global_box = [
                        b[0] + x_start, b[1] + y_start,
                        b[2] + x_start, b[3] + y_start,
                    ]
                    all_boxes.append(global_box)
                    all_confs.append(float(box.conf[0]))

        # 3) Merge duplicates from overlapping tiles
        all_boxes, all_confs = self._nms_merge(all_boxes, all_confs,
                                                iou_thresh=0.4)

        return all_boxes, all_confs
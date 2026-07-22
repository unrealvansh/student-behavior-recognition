"""Smart 3-Step Pipeline: FIND → TRACK → UNDERSTAND.

Orchestrates the full student behavior recognition workflow:
  Step 1 (FIND + TRACK): YOLOv8 + ByteTrack  →  persistent student IDs
  Step 2 (UNDERSTAND):   MobileNetV2 CNN      →  behavior classification
  Step 3 (SMOOTH):       Label history voting  →  stable predictions

Designed for real-time webcam inference on standard hardware (i5 / 8 GB RAM).
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import time

from src.tracker import StudentTracker
from src.classifier import BehaviorClassifier


# ── Behavior definitions (from PPTX Slide 8) ─────────────────────────────
# Classifier output  →  Dashboard category
BEHAVIOR_MAP = {
    "Sitting":      {"label": "Sitting/Standing", "color": (160, 160, 160), "css": "gray-text",   "alert": False},
    "Sleeping":     {"label": "Sleeping",         "color": (79, 79, 237),   "css": "red-text",    "alert": True},
    "Mobile":       {"label": "Using Mobile",     "color": (0, 215, 255),   "css": "yellow-text", "alert": True},
    "Reading":      {"label": "Reading/Writing",  "color": (87, 201, 0),    "css": "green-text",  "alert": False},
    "HandRaising":  {"label": "Hand Raising",     "color": (255, 144, 30),  "css": "blue-text",   "alert": False},
    "Unknown":      {"label": "Detecting...",     "color": (160, 160, 160), "css": "gray-text",   "alert": False},
}

# BGR colors for bounding box outlines (matching PPTX slide 1 & 8 scheme)
BOX_COLORS = {
    "Sitting":     (160, 160, 160),   # Gray
    "Sleeping":    (70, 70, 230),      # Red
    "Mobile":      (0, 200, 255),      # Yellow
    "Reading":     (80, 200, 0),       # Green
    "HandRaising": (255, 140, 30),     # Blue
    "Unknown":     (160, 160, 160),
}


class SmartPipeline:
    """End-to-end student behavior recognition pipeline.

    Usage:
        pipeline = SmartPipeline()
        annotated_frame, stats, detections = pipeline.process_frame(frame)
    """

    def __init__(self, classifier_weights="models/behavior_classifier.pth",
                 tracker_model="yolov8m.pt",
                 label_window=15,
                 conf_threshold=0.3):
        """
        Args:
            classifier_weights: Path to trained MobileNetV2 .pth file.
            tracker_model: Path to YOLO model for person detection/tracking.
            label_window: Number of past frames to consider when smoothing
                          behavior labels (majority vote). Higher = more stable
                          but slower to react. 15 ≈ 0.5 sec at 30 fps.
            conf_threshold: Minimum YOLO confidence to accept a detection.
        """
        # Step 1: Person detector + tracker (YOLOv8 + ByteTrack)
        self.tracker = StudentTracker(model_path=tracker_model)
        self.conf_threshold = conf_threshold

        # Step 2: Behavior classifier (MobileNetV2)
        self.classifier = BehaviorClassifier(weights_path=classifier_weights)

        # Step 3: Temporal label smoothing (per-student rolling mode)
        self.label_window = label_window
        self._label_history = defaultdict(lambda: deque(maxlen=label_window))

        # Stats accumulator
        self._total_detections = 0
        self._total_alerts = 0
        
        # Cooldown state (seconds) to prevent spamming the dashboard
        self.alert_cooldown = 10.0
        self._last_log_time = {}

    def process_frame(self, frame):
        """Run the full 3-step pipeline on a single video frame.

        Args:
            frame: BGR numpy array from cv2.VideoCapture.

        Returns:
            annotated_frame: Frame with bounding boxes and labels drawn.
            stats: dict with counts per behavior + totals.
            new_detections: list of per-student detection dicts for the
                            "Recent Detections" table.
        """
        annotated = frame.copy()

        # ── Step 1: FIND + TRACK ──────────────────────────────────────────
        boxes, ids, confs = self.tracker.track(frame, self.conf_threshold)

        if len(boxes) == 0:
            stats = self._empty_stats()
            return annotated, stats, []

        # ── Step 2: UNDERSTAND (classify each tracked person) ─────────────
        raw_labels = []
        raw_confs = []
        for box in boxes:
            x1, y1, x2, y2 = box
            # Clamp to frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                raw_labels.append("Unknown")
                raw_confs.append(0.0)
                continue

            label, conf = self.classifier.predict(crop)
            raw_labels.append(label)
            raw_confs.append(conf)

        # ── Step 3: SMOOTH (temporal label voting per student ID) ─────────
        smoothed_labels = []
        for sid, label in zip(ids, raw_labels):
            self._label_history[sid].append(label)
            # Majority vote over recent history
            history = self._label_history[sid]
            vote = max(set(history), key=lambda x: list(history).count(x))
            smoothed_labels.append(vote)

        # ── Draw annotations on frame ─────────────────────────────────────
        new_detections = []
        for box, sid, label, conf in zip(boxes, ids, smoothed_labels, raw_confs):
            x1, y1, x2, y2 = box
            info = BEHAVIOR_MAP.get(label, BEHAVIOR_MAP["Unknown"])
            color = BOX_COLORS.get(label, (160, 160, 160))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            text = f"#{sid} {info['label']}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # ── Alert Cooldown Logic ──────────────────────────────────────────
            current_time = time.time()
            last_time = self._last_log_time.get(sid, 0)
            
            if current_time - last_time > self.alert_cooldown:
                # Build detection record for the dashboard table
                new_detections.append({
                    "time": datetime.now().strftime("%I:%M %p"),
                    "student": f"Student #{sid}",
                    "behaviour": info["label"],
                    "color": info["css"],
                })
                
                # Update cooldown timestamp
                self._last_log_time[sid] = current_time
                self._total_detections += 1

                # Count alerts only when the cooldown allows a log
                if info["alert"]:
                    self._total_alerts += 1

        # ── Aggregate stats ───────────────────────────────────────────────
        counts = defaultdict(int)
        for label in smoothed_labels:
            counts[label] += 1

        stats = {
            "sitting":     counts.get("Sitting", 0),
            "sleeping":    counts.get("Sleeping", 0),
            "phone":       counts.get("Mobile", 0),
            "reading":     counts.get("Reading", 0),
            "hand":        counts.get("HandRaising", 0),
            "total":       len(boxes),
            "detections":  self._total_detections,
            "alerts":      self._total_alerts,
            "status":      "Running",
        }

        return annotated, stats, new_detections

    def _empty_stats(self):
        """Return zeroed stats when no students are detected."""
        return {
            "sitting": 0, "sleeping": 0, "phone": 0,
            "reading": 0, "hand": 0,
            "total": 0,
            "detections": self._total_detections,
            "alerts": self._total_alerts,
            "status": "Running",
        }

    def reset(self):
        """Reset all state (label history, counters)."""
        self._label_history.clear()
        self._last_log_time.clear()
        self._total_detections = 0
        self._total_alerts = 0

    # ── Cleanup old student IDs that haven't been seen recently ───────────
    def cleanup_stale_ids(self, active_ids):
        """Remove label history and cooldown state for IDs that disappeared."""
        stale = [sid for sid in self._label_history if sid not in active_ids]
        for sid in stale:
            del self._label_history[sid]
            if sid in self._last_log_time:
                del self._last_log_time[sid]

from ultralytics import YOLO
import numpy as np
import os

MODEL_PATH = "runs/detect/models/orbitguard_detector/weights/best.pt"
detector = YOLO(MODEL_PATH)
def run_detction(image_path: str):
    results = detector.predict(source=image_path, conf=0.15)
    result = results[0]
    boxes = result.boxes
    debris_count = len(boxes)
    areas = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        areas.append(float(x2 - x1) * (y2 - y1))
        avg_area = float(np.mean(areas)) if areas else 0.0
        max_area = float(np.max(areas)) if areas else 0.0

        return debris_count, avg_area, max_area



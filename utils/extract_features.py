from ultralytics import YOLO
import numpy as np
import pandas as pd
import random
import os
os.makedirs("data/processed", exist_ok=True)

model = YOLO("runs/detect/models/orbitguard_detector/weights/best.pt")
def extract_features(image_folder):
    results = model(image_folder, conf=0.25)
    data = []
    for r in results:
        boxes = r.boxes
        count = len(boxes)
        if count > 0:
            areas = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                area = (x2 - x1) * (y2 - y1)
                areas.append(area)
                avg_area = float(np.mean(areas))
                max_area = float(np.max(areas))
        else:
            avg_area = 0
            max_area = 0

        sample = {
            "debris_count":count,
            "avg_area":avg_area,
            "max_area":max_area,
            "orbital_altitude_km":random.uniform(300, 1500),
            "velocity_kms":random.uniform(2, 14)
        }          
        data.append(sample)

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = extract_features("data/raw/valid/images")
    df.to_csv("data/processed/risk_features.csv", index=False)
    print("Feature extracted and saved")
from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data/raw/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        project="models",
        name="orbitguard_detector"
    )


if __name__ == "__main__":
    train()
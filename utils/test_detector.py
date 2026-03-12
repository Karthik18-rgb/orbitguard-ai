from ultralytics import YOLO
model = YOLO("runs/detect/models/orbitguard_detector/weights/best.pt")
results = model(
    "data/raw/valid/images",
    save=True,
    conf=0.25
)

print("Inference complete")
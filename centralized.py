# centralized.py
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov12n.pt")  # small version
    results = model.train(data="datasets/coco128/coco128.yaml", epochs=3, imgsz=320, device=0)
    print("Centralized training finished. Metrics:")
    print(results)

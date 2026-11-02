"""Train the cow_standing / cow_lying detector. Run on a laptop or Google Colab (the Pi 4 is too slow to train).

    pip install ultralytics
    python train_vision.py path/to/data.yaml

data.yaml comes with the Roboflow export (format "YOLOv11" or "YOLOv8"). When it finishes, the trained model is
runs/detect/train/weights/best.pt (the exact folder is printed at the end). Copy best.pt to the Pi's
~/cowtrol_cbt/vision/ and convert it there:  yolo export model=vision/best.pt format=ncnn
"""
import sys

from ultralytics import YOLO

data = sys.argv[1] if len(sys.argv) > 1 else "data.yaml"
model = YOLO("yolo11n.pt")  # start from the stock nano model; downloads itself the first time
model.train(data=data, epochs=100, imgsz=640, patience=20)
metrics = model.val()
print(f"\nmAP50: {metrics.box.map50:.3f}  precision: {metrics.box.mp:.3f}  recall: {metrics.box.mr:.3f}")
print(f"Trained model: {model.trainer.best}")

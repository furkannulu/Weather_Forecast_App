from  ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

results = model.train(data='./Dataset', epochs=20, imgsz=64)

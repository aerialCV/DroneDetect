import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch


if __name__ == '__main__':
    #model = YOLO("./ultralytics/cfg/models/v8/yolov8s.yaml")
    model = YOLO("./ultralytics/cfg/models/v8/yolov8s-drone-detect.yaml")

    model.train(data='VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=2,
                batch=32,
                close_mosaic=0,
                workers=8, 
                optimizer='SGD', 
                resume=False,
                show_labels=False,
                show_conf=False,
                project='runs/train',
                name='exp',
                )


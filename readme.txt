Pedestrian Detection using YOLOv8n and YOLOv11n

This project implements a full pipeline for training and evaluating pedestrian detection using two models:
- **YOLOv8n** (lightweight)
- **YOLOv11n** (enhanced/custom version)

It compares both models in terms of mAP, Precision, Recall, and inference time using a dataset in VOC format.

---

- `yolo_full_pipeline.py` : Main Python script with detailed comments. Trains, evaluates, and performs inference.
- `data.yaml`             : Dataset config file for training in VOC format.
- `train/`, `test/`       : Your labeled training and test sets.
- `yolov8n.pt`, `yolov11n.pt` : Pretrained model weights (download or place in working directory).

---

How to Run

1. **Set up your environment**:
pip install ultralytics pandas matplotlib

2.Set your dataset path in the script:
dataset_path = "/your/full/path/to/pedestrian_dataset"

3.Run training and inference:

python yolo_full_pipeline.py

Results will be saved under:
runs/train/YOLOv8n/
runs/train/YOLOv11n/
runs/train/predict/


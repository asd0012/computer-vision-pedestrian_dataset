import os
import time
import torch
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt

# === Step 1: Setup ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_path = '/csse/users/tth92/Desktop/Computer Vision/project/pedestrian_dataset' 
results_log = []

# === Step 2: Define train & eval function ===
def train_and_evaluate(model_path, version_name, save_dir):
    print(f"Loading model: {version_name}")
    model = YOLO(model_path)

    print(f"Training {version_name}...")
    start_time = time.time()
    model.train(data=f'{dataset_path}/data.yaml', epochs=50, imgsz=640, project=save_dir, name=version_name)
    train_time = time.time() - start_time

    print(f"Evaluating {version_name}...")
    metrics = model.val()
    metrics_dict = metrics.results_dict

    print(f"Running inference...")
    model.predict(source=f'{dataset_path}/test/images', save=True, project=f'{save_dir}/predict', name=version_name)

    # Save PR curve
    if hasattr(metrics, 'plot_pr'):
        metrics.plot_pr()
        plt.savefig(f'{save_dir}/{version_name}_pr_curve.png')
    if hasattr(metrics, 'plot_f1'):
        metrics.plot_f1()
        plt.savefig(f'{save_dir}/{version_name}_f1_curve.png')

    # Record result
    result = {
        'Model': version_name,
        'mAP@0.5': metrics_dict.get('metrics/mAP50(B)', None),
        'mAP@0.5:0.95': metrics_dict.get('metrics/mAP50-95(B)', None),
        'Precision': metrics_dict.get('metrics/precision(B)', None),
        'Recall': metrics_dict.get('metrics/recall(B)', None),
        'Train Time (s)': round(train_time, 2)
    }
    results_log.append(result)

# === Step 3: Run both models ===
os.makedirs('runs/summary', exist_ok=True)
train_and_evaluate('yolov8n.pt', 'YOLOv8n', 'runs/train')
train_and_evaluate('yolov11n.pt', 'YOLOv11n', 'runs/train')

# === Step 4: Save comparison result to CSV ===
df = pd.DataFrame(results_log)
df.to_csv('runs/summary/model_comparison.csv', index=False)
print("Training & evaluation complete. Results saved.")

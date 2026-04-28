# Pedestrian Detection with YOLOv8n and YOLOv11n

**Computer vision project comparing YOLOv8n and YOLOv11n for pedestrian detection in driving-related collision avoidance scenarios.**

This project trains and evaluates lightweight YOLO-based object detection models for detecting pedestrians in road-scene images. It was developed as a University of Canterbury computer vision project focused on comparing model accuracy, visual detection quality, and practical suitability for mid-range pedestrian detection.

> Status: coursework / portfolio project. The repository contains a training and evaluation pipeline, dataset configuration, model weights, and saved experiment outputs.

---

## Table of Contents

* [Overview](#overview)
* [Project Goal](#project-goal)
* [Models Compared](#models-compared)
* [Dataset](#dataset)
* [Results Summary](#results-summary)
* [Repository Structure](#repository-structure)
* [Technology Stack](#technology-stack)
* [How to Run](#how-to-run)
* [Outputs](#outputs)
* [Key Findings](#key-findings)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)
* [Related Report](#related-report)

---

## Overview

Pedestrian detection is an important computer vision task for driver-assistance and collision-avoidance systems. This project compares two YOLO models on a pedestrian detection dataset:

* **YOLOv8n** — lightweight YOLO model with strong speed / accuracy balance.
* **YOLOv11n** — newer YOLO model used for comparison against YOLOv8n.

The project uses an Ultralytics-based Python pipeline to train, validate, run inference, and export model comparison results.

---

## Project Goal

The main goal is to compare YOLOv8n and YOLOv11n for pedestrian detection, especially for road-scene images related to collision-avoidance scenarios.

The comparison focuses on:

* mAP@0.5
* mAP@0.5:0.95
* Precision
* Recall
* PR curves
* F1 curves
* Visual detection results

The project also briefly discusses MOG2 as a traditional computer vision baseline, but the implemented training pipeline focuses on YOLO-based detection.

---

## Models Compared

| Model    | Purpose                                                                                               |
| -------- | ----------------------------------------------------------------------------------------------------- |
| YOLOv8n  | Lightweight baseline model for real-time pedestrian detection.                                        |
| YOLOv11n | Newer YOLO model used to compare accuracy and detection quality.                                      |
| MOG2     | Traditional foreground-background method discussed as a baseline concept, not a class-aware detector. |

---

## Dataset

The dataset is configured using a YOLO-style `data.yaml` file with one target class:

```yaml
nc: 1
names: ['pedestrian']
```

Expected dataset layout:

```text
pedestrian_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The dataset was prepared from a pedestrian detection dataset and converted / configured for YOLO training.

---

## Results Summary

The project report recorded the following comparison after training and evaluation:

| Model    | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
| -------- | ------: | -----------: | --------: | -----: |
| YOLOv8n  |    0.92 |         0.71 |      0.93 |   0.83 |
| YOLOv11n |    0.94 |         0.74 |      0.95 |   0.85 |

YOLOv11n achieved slightly stronger results across the recorded metrics, while YOLOv8n remains a strong lightweight option for real-time or resource-constrained scenarios.

---

## Repository Structure

```text
computer-vision-pedestrian_dataset/
├── yolo_full_pipeline.py        # Main training, evaluation, and inference pipeline
├── data.yaml                    # Dataset configuration
├── train/                       # Training images and labels
├── test/                        # Test / validation images and labels
├── runs/train/                  # Saved training and prediction outputs
├── yolo11n.pt                   # YOLO model weight file
├── yolov11n.pt                  # YOLOv11n model weight file
├── yolov8n.pt                   # YOLOv8n model weight file
├── Computer_Vision Paper.pdf    # Project report
└── README.dataset.txt / README.roboflow.txt
```

---

## Technology Stack

* Python
* Ultralytics YOLO
* PyTorch
* OpenCV
* Pandas
* Matplotlib
* CUDA-enabled GPU environment for training

The project report used an NVIDIA RTX 4060 environment with Python 3.12, PyTorch, OpenCV, and Ultralytics YOLO.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/asd0012/computer-vision-pedestrian_dataset.git
cd computer-vision-pedestrian_dataset
```

### 2. Install dependencies

```bash
pip install ultralytics torch pandas matplotlib opencv-python
```

### 3. Check dataset paths

Update `data.yaml` and `dataset_path` in `yolo_full_pipeline.py` so they point to your local dataset location.

Example:

```python
dataset_path = "/your/full/path/to/pedestrian_dataset"
```

Also ensure `data.yaml` points to the correct train / validation / test folders.

### 4. Run the pipeline

```bash
python yolo_full_pipeline.py
```

The script trains and evaluates YOLOv8n and YOLOv11n, runs inference on test images, and saves a model comparison CSV.

---

## Outputs

Expected outputs include:

```text
runs/train/YOLOv8n/
runs/train/YOLOv11n/
runs/train/predict/
runs/summary/model_comparison.csv
```

The output folders may include:

* training curves
* validation metrics
* predicted bounding-box images
* PR / F1 curve outputs
* model comparison CSV

---

## Key Findings

* YOLOv8n performed strongly as a lightweight model and is suitable for real-time detection scenarios where speed and resource use matter.
* YOLOv11n achieved slightly better accuracy metrics in this experiment.
* YOLOv11n appeared stronger in more difficult visual cases such as smaller or partially occluded pedestrians.
* Traditional MOG2 can highlight foreground motion but does not provide class-aware pedestrian detection or bounding-box classification.

---

## Limitations

This project is a coursework / research prototype rather than a production ADAS system.

Known limitations:

* The dataset is limited compared with real-world driving conditions.
* The experiment focuses on offline image-based evaluation rather than live video deployment.
* Distance-specific evaluation for every pedestrian instance was not fully automated.
* Inference speed and deployment performance may differ on embedded hardware.
* MOG2 was discussed as a baseline method but not evaluated using the same detection metrics as YOLO models.

---

## Future Improvements

Possible next steps:

* Add clearer train / validation / test separation and reproducible configuration files.
* Add a `requirements.txt` file.
* Add command-line arguments for model path, dataset path, epochs, and image size.
* Add screenshots of sample predictions directly in this README.
* Evaluate inference speed on lower-power devices.
* Test on more diverse weather, lighting, occlusion, and distance conditions.
* Add a small demo video or animated GIF showing prediction outputs.

---

## Related Report

A short project report is included in the repository as:

```text
Computer_Vision Paper.pdf
```

The report discusses the motivation, model comparison, evaluation metrics, training environment, results, and conclusions for the pedestrian detection experiment.

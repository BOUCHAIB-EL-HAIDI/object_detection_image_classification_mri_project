# 🧠 MRI Brain Tumor — Classification & Object Detection

> **⚠️ Disclaimer:** This project is an educational prototype. It is not a medical diagnostic tool, and its results must not be used for clinical decision-making.

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Data Preparation](#data-preparation)
7. [Models and Methodology](#models-and-methodology)
8. [Evaluation Results](#evaluation-results)
9. [Usage](#usage)
10. [Airflow Pipeline](#airflow-pipeline)
11. [Streamlit Interface](#streamlit-interface)
12. [Docker](#docker)
13. [Limitations](#limitations)
14. [Future Improvements](#future-improvements)

---

## Project Overview

This project explores two Deep Learning tasks using brain MRI images:

| Task | Objective | Model |
|---|---|---|
| **Classification** | Classify an MRI image into one of four categories | MobileNetV2 |
| **Object Detection** | Locate tumor regions using bounding boxes | YOLOv8n |

The four classification categories are:

- Glioma
- Meningioma
- No Tumor
- Pituitary

### Technologies

- Python
- TensorFlow / Keras
- PyTorch
- Ultralytics YOLOv8
- MobileNetV2
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Albumentations

Planned integration technologies:

- Apache Airflow
- Streamlit
- Docker

---

## Architecture

The project contains two model pipelines that share the prepared MRI dataset.

```text
                 Raw MRI Dataset
                        |
                 Data Preparation
                        |
              Train / Validation / Test
                        |
             +----------+----------+
             |                     |
       Classification         Detection
             |                     |
         MobileNetV2             YOLOv8n
             |                     |
       4 Classes               3 Classes
             |                     |
             +----------+----------+
                        |
                  Model Evaluation
                        |
             Metrics, Reports and Plots
                        |
              Application Integration
                (Planned / In Progress)
                        |
             +----------+----------+
             |                     |
          Streamlit             Airflow
             |                     |
             +----------+----------+
                        |
                      Docker
```

---

## Dataset

**Source:** [MRI for Brain Tumor with Bounding Boxes — Kaggle](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)

The dataset contains 5,249 MRI images across four categories.

### Classification and detection classes

| Class | Classification | Object Detection |
|---|---|---|
| Glioma | Yes | Yes |
| Meningioma | Yes | Yes |
| No Tumor | Yes | Excluded |
| Pituitary | Yes | Yes |

**Important — No Tumor:** This class is retained in the four-class classification task. It is excluded from object detection because it has no tumor region to localize. Its detection annotation files are empty.

### Dataset split

The original training data was divided to create a separate test set. The original validation set was retained.

| Split | Images |
|---|---:|
| Training | 4,026 |
| Validation | 512 |
| Test | 711 |
| **Total** | **5,249** |

The test set was separated from the original training data before model training and evaluation.

### Downloading the dataset

Install the Kaggle CLI:

```bash
pip install kaggle
```

Configure your Kaggle API credentials, then download the dataset:

```bash
kaggle datasets download -d ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes
```

Extract the archive into:

```text
data/raw/MRI_Dataset/
```

---

## Project Structure

The following represents the intended organization of the project. Some integration folders and files may still be under development.

```text
OBJECT DETECTION MRI DEEP LEARNING/
│
├── data/
│   ├── raw/
│   │   └── MRI_Dataset/
│   ├── processed/
│   └── processed_v2/
│       ├── classification/
│       └── detection/
│           ├── images/
│           ├── labels/
│           └── data.yaml
│
├── notebooks/
│
├── src/
│   ├── classification/
│   │   ├── model.py
│   │   ├── trainfeatureE.py
│   │   ├── trainfineT.py
│   │   ├── trainfineT_aug.py
│   │   ├── trainfineT_l2.py
│   │   └── evaluate.py
│   │
│   └── detection/
│       ├── check_dataset.py
│       ├── check_dataset_v2.py
│       ├── prepare_dataset_v2.py
│       ├── train.py
│       └── evaluate.py
│
├── models/
│   └── classification/
│       └── mobilenetv2_finetuned_aug_best.keras
│
├── results/
│   ├── classification/
│   │   └── final_evaluation/
│   ├── detection/
│   │   └── yolov8n_mri-2/
│   └── evaluation/
│       └── yolo_test/
│           ├── test_metrics.csv
│           ├── test_metrics.txt
│           └── confusion_matrix.png
│
├── dashboard/              # Planned: Streamlit
├── dags/                   # Planned: Airflow
├── tests/                  # Planned: automated tests
│
├── Dockerfile              # Planned
├── docker-compose.yml      # Planned
├── requirements.txt
└── README.md
```

**Note:** Large datasets, model weights and generated results do not need to be committed to Git. Their download and reproduction instructions should be documented.

---

## Installation

### Prerequisites

- Python 3.10 or later, subject to library compatibility
- Git
- NVIDIA GPU (optional)
- NVIDIA-compatible PyTorch installation for GPU training

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

### Main libraries

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Image classification |
| PyTorch | YOLO model execution |
| Ultralytics | YOLOv8 training and evaluation |
| NumPy | Numerical operations |
| OpenCV | Image processing |
| Pillow | Image loading and resizing |
| Scikit-learn | Classification metrics |
| Matplotlib | Visualization |
| PyYAML | YOLO configuration |

Install Apache Airflow separately in its Docker environment when implementing the orchestration pipeline.

---

## Data Preparation

The preprocessing stage prepares separate datasets for classification and object detection.

### Classification

- Four categories are retained.
- Images are resized to **384 × 384**.
- MobileNetV2 preprocessing is applied during model input preparation.
- Training, validation and test data are kept separate.

### Object Detection

- Three tumor classes are used.
- Images are resized to **512 × 512** while preserving their aspect ratio through letterboxing.
- Bounding-box coordinates are adjusted to match the resized images.
- Annotations use the YOLO format:

```text
class_id x_center y_center width height
```

Coordinates are normalized between 0 and 1.

The detection dataset configuration is stored in `data.yaml`.

---

## Models and Methodology

### 1. Classification — MobileNetV2

**Base model:** MobileNetV2 pretrained on ImageNet.

Transfer Learning is used to adapt the pretrained network to the MRI classification task.

The training approach includes:

1. Feature extraction: freeze the pretrained backbone and train the classification head.
2. Fine-tuning: unfreeze selected backbone layers and continue training with a lower learning rate.

The classification head uses:

```text
GlobalAveragePooling2D
        |
Dense(256, ReLU)
        |
Dropout(0.4)
        |
Dense(4, Softmax)
```

### 2. Overfitting Prevention

The following techniques were explored:

| Technique | Purpose |
|---|---|
| Data Augmentation | Increase training-data variation |
| Dropout | Reduce dependence on individual neurons |
| L2 Regularization | Penalize excessively large weights |
| Early Stopping | Stop training when validation performance stops improving |
| Model Checkpoint | Save the best-performing model |
| ReduceLROnPlateau | Reduce the learning rate when progress stagnates |

### 3. Object Detection — YOLOv8n

**Model:** YOLOv8n pretrained on COCO.

Training configuration:

| Parameter | Value |
|---|---:|
| Epochs | 50 |
| Patience | 10 |
| Image size | 512 × 512 |
| Batch size | 8 |
| Detection classes | 3 |
| Model checkpoint | `best.pt` |

The trained model is saved at:

```text
results/detection/yolov8n_mri-2/weights/best.pt
```

---

## Evaluation Results

Both models were evaluated using a separate test set.

### Classification — Test Results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Glioma | 0.955 | 0.977 | 0.966 | 173 |
| Meningioma | 0.976 | 0.931 | 0.953 | 217 |
| No Tumor | 0.971 | 0.953 | 0.962 | 107 |
| Pituitary | 0.964 | 1.000 | 0.982 | 214 |
| **Overall** | **0.967** | **0.966** | **0.966** | **711** |

**Test Accuracy: 96.62%**

### Object Detection — YOLOv8 Test Results

| Metric | Overall | Glioma | Meningioma | Pituitary |
|---|---:|---:|---:|---:|
| Precision | 0.919 | 0.889 | 0.954 | 0.913 |
| Recall | 0.877 | 0.758 | 0.982 | 0.892 |
| mAP@50 | 0.940 | 0.874 | 0.989 | 0.957 |
| mAP@50–95 | 0.735 | 0.643 | 0.838 | 0.723 |

**Interpretation:**

- Meningioma achieved the strongest detection results.
- Glioma had the lowest recall, meaning some annotated tumor regions were missed.
- The overall mAP@50 was 0.940.
- The overall mAP@50–95 was 0.735.

These results describe performance on this dataset and do not establish clinical reliability.

### Evaluation outputs

YOLO test evaluation outputs are stored in:

```text
results/evaluation/yolo_test/
```

The folder includes evaluation plots and, when generated by the evaluation script:

- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `PR_curve.png`
- `F1_curve.png`
- `test_metrics.csv`
- `test_metrics.txt`

Classification evaluation outputs are stored in:

```text
results/classification/final_evaluation/
```

---

## Usage

### 1. Dataset preparation

```bash
python src/detection/prepare_dataset_v2.py
python src/detection/check_dataset_v2.py
```

Run the classification and detection preprocessing scripts according to their actual configurations before training.

### 2. Classification training

```bash
python src/classification/trainfeatureE.py
python src/classification/trainfineT.py
python src/classification/trainfineT_aug.py
python src/classification/trainfineT_l2.py
```

### 3. Classification evaluation

```bash
python src/classification/evaluate.py
```

### 4. YOLOv8 training

```bash
python src/detection/train.py
```

### 5. YOLOv8 test evaluation

```bash
python src/detection/evaluate.py
```

The evaluation script loads the saved `best.pt` checkpoint and evaluates it on the test split. It does not retrain the model.

---

## Airflow Pipeline

**Status: Planned / In Progress**

Apache Airflow will be used to orchestrate the machine learning workflow through a Directed Acyclic Graph (DAG).

The planned workflow is:

```text
check_dataset
      |
preprocess_data
      |
  +---+---+
  |       |
train_   train_
classifier yolo
  |       |
evaluate evaluate
classifier yolo
  |       |
  +---+---+
      |
 save_metrics
```

The DAG will manage:

- Task dependencies
- Execution order
- Task failures and retries
- Training and evaluation execution
- Saving metrics and generated results

---

## Streamlit Interface

**Status: Planned / In Progress**

The planned Streamlit application will provide an interface for interacting with the trained models.

Expected features:

- Upload an MRI image.
- Display the predicted classification category.
- Display the probability of each classification category.
- Visualize YOLO bounding boxes.
- Display detection confidence scores.
- Consult saved evaluation metrics and results.

The application will load saved model checkpoints rather than retraining models during user interactions.

---

## Docker

**Status: Planned / In Progress**

Docker will be used to package the application and its dependencies.

The planned components are:

| Component | Purpose |
|---|---|
| Streamlit | User interface |
| Airflow | Workflow orchestration |
| Docker Compose | Manage application services |

Expected configuration files:

```text
Dockerfile
docker-compose.yml
```

Docker Compose will manage service configuration, ports, volumes and dependencies.

---

## Limitations

| Limitation | Description |
|---|---|
| Dataset | Results depend on the dataset's size, quality and distribution. |
| No Tumor detection | No Tumor is included in classification but excluded from detection. |
| Glioma detection | Glioma has lower recall than the other detection classes. |
| Generalization | Performance on this dataset does not guarantee performance on other MRI datasets. |
| Clinical use | The models are educational prototypes and are not clinically validated. |

---



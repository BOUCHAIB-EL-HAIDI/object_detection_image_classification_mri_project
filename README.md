# 🧠 MRI Brain Tumor — Classification & Object Detection

> **⚠️ Disclaimer:** This project is a **pedagogical prototype** and does **not** constitute a medical diagnostic tool. Results must not be used for clinical decision-making.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Pipeline Overview](#pipeline-overview)
7. [Step-by-Step Results](#step-by-step-results)
   - [Data Exploration](#step-1--data-exploration--preparation)
   - [Preprocessing & Augmentation](#step-2--preprocessing--data-augmentation)
   - [Transfer Learning — Classification](#step-3--classification-by-transfer-learning)
   - [Overfitting Prevention](#step-4--overfitting-prevention)
   - [YOLOv8 Detection](#step-5--object-detection-with-yolov8)
   - [Model Evaluation](#step-6--model-evaluation)
8. [Usage](#usage)
9. [Docker](#docker)
10. [Airflow Pipeline](#airflow-pipeline)
11. [Streamlit Interface](#streamlit-interface)
12. [Limitations](#limitations)
13. [Author](#author)

---

## Project Overview

This project develops a **Deep Learning prototype** for two complementary tasks on brain MRI images:

| Task | Goal | Model |
|---|---|---|
| **Classification** | Predict the tumor category (4 classes) | MobileNetV2 + Transfer Learning |
| **Detection** | Localize tumor regions with bounding boxes | YOLOv8n (Ultralytics) |

**Four categories:** `Glioma` · `Meningioma` · `No Tumor` · `Pituitary`

Key techniques applied:
- Transfer Learning & Fine-tuning (ImageNet → MRI domain)
- Data Augmentation (geometric + photometric)
- Overfitting prevention: Dropout, Early Stopping, L2 regularization, ReduceLROnPlateau
- YOLOv8 with YOLO-format bounding box annotations
- Evaluation: Accuracy, F1-score, Confusion Matrix, mAP50, IoU

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Raw MRI Dataset                         │
│          (Kaggle — 4 classes + bounding box XML)             │
└───────────────────────────┬──────────────────────────────────┘
                            │
                 ┌──────────▼──────────┐
                 │  Data Preparation   │
                 │  (train/val/test)   │
                 └──────┬──────┬───────┘
                        │      │
          ┌─────────────▼─┐  ┌─▼─────────────────┐
          │ Classification│  │   Detection        │
          │  Pipeline     │  │   Pipeline         │
          │               │  │                    │
          │ MobileNetV2   │  │  YOLOv8n           │
          │ + Fine-tuning │  │  (YOLO format)     │
          │ + Regulariz.  │  │  3 classes          │
          └───────┬───────┘  └──────────┬─────────┘
                  │                     │
          ┌───────▼─────────────────────▼─────────┐
          │          Evaluation & Results          │
          │  Accuracy / F1 / mAP50 / IoU / Plots  │
          └───────────────────┬───────────────────┘
                              │
          ┌───────────────────▼───────────────────┐
          │          Streamlit Interface           │
          │  Upload → Classify → Detect → Display │
          └───────────────────────────────────────┘
                              │
          ┌───────────────────▼───────────────────┐
          │     Airflow DAG (Orchestration)        │
          │  preprocess → train → evaluate → save  │
          └───────────────────────────────────────┘
                              │
          ┌───────────────────▼───────────────────┐
          │     Docker (Containerization)          │
          │     Streamlit + Airflow services       │
          └───────────────────────────────────────┘
```

---

## Dataset

**Source:** [MRI for Brain Tumor with Bounding Boxes](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)

| Class | Role in Classification | Role in Detection |
|---|---|---|
| Glioma | ✅ Yes | ✅ Yes |
| Meningioma | ✅ Yes | ✅ Yes |
| No Tumor | ✅ Yes | ❌ Excluded (no annotated region) |
| Pituitary | ✅ Yes | ✅ Yes |

> **Note on "No Tumor":** This class is kept for classification (4 classes). Since no bounding-box region is annotated for it, it is excluded from the detection task (3 classes: Glioma, Meningioma, Pituitary).

### Downloading the dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Configure your API key (~/.kaggle/kaggle.json)
kaggle datasets download -d ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes
unzip mri-for-brain-tumor-with-bounding-boxes.zip -d data/raw/
```

---

## Project Structure

```text
projet-mri-deep-learning/
│
├── data/
│   ├── raw/                        # Original Kaggle dataset
│   ├── processed/
│   │   ├── classification/         # train / val / test splits (4 classes)
│   │   └── detection/              # YOLO-format dataset (3 classes)
│   └── processed_v2/
│       └── detection/              # Refined detection dataset + data.yaml
│
├── notebooks/
│   ├── data_exploration.py         # EDA, class distribution, annotation checks
│   └── preprocessing_augmentation.py # Augmentation pipeline demo
│
├── src/
│   ├── classification/
│   │   ├── model.py                # MobileNetV2 architecture builder
│   │   ├── trainfeatureE.py        # Phase 1: feature extraction (frozen backbone)
│   │   ├── trainfineT.py           # Phase 2: fine-tuning (baseline)
│   │   ├── trainfineT_aug.py       # Fine-tuning with augmentation config
│   │   ├── trainfineT_l2.py        # Fine-tuning with L2 regularization
│   │   └── evaluate.py             # Classification evaluation (metrics + plots)
│   │
│   └── detection/
│       ├── check_dataset.py        # Annotation integrity checks (v1)
│       ├── check_dataset_v2.py     # Refined annotation checks (v2)
│       ├── prepare_dataset_v2.py   # Convert XML → YOLO format, split dataset
│       ├── train.py                # YOLOv8 training entry point
│       └── evaluate.py             # YOLOv8 evaluation on test set
│
├── models/
│   └── classification/
│       ├── mobilenetv2_best.keras              # Feature extraction checkpoint
│       ├── mobilenetv2_finetuned_best.keras    # Fine-tuned baseline
│       ├── mobilenetv2_finetuned_aug_best.keras # Fine-tuned with augmentation
│       └── mobilenetv2_finetuned_l2_best.keras  # Fine-tuned with L2 reg
│
├── results/
│   ├── classification/
│   │   ├── augmentation/           # Learning curves (aug config)
│   │   ├── l2/                     # Learning curves (L2 config)
│   │   └── final_evaluation/       # Confusion matrix + classification report
│   ├── detection/
│   │   └── yolov8n_mri-2/         # YOLOv8 training outputs (curves, weights)
│   └── evaluation/
│       └── yolo_test/              # YOLOv8 test set evaluation
│
├── dashboard/                      # Streamlit application
│   └── app.py
│
├── dags/                           # Apache Airflow DAGs
│   └── mri_pipeline.py
│
├── tests/                          # Unit & integration tests
│   ├── test_preprocessing.py
│   ├── test_annotations.py
│   └── test_inference.py
│
├── yolov8n.pt                      # YOLOv8n pretrained weights (ImageNet)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- Git
- (Optional) NVIDIA GPU with CUDA for faster training

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/projet-mri-deep-learning.git
cd projet-mri-deep-learning

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt` — Key Dependencies

```
tensorflow>=2.13
ultralytics>=8.0
streamlit>=1.28
apache-airflow>=2.7
opencv-python
matplotlib
seaborn
scikit-learn
pandas
numpy
kaggle
```

---

## Pipeline Overview

```
J1  ─── Data exploration & annotation verification
J2  ─── Preprocessing, splits, augmentation pipeline
J3  ─── Transfer Learning: MobileNetV2 feature extraction
J4  ─── Fine-tuning, regularization, classification evaluation
J5  ─── Annotation conversion (XML → YOLO), dataset prep
J6  ─── YOLOv8 training, evaluation, error analysis
J7  ─── Airflow DAG implementation
J8  ─── Streamlit interface development
J9  ─── Docker integration & tests
J10 ─── Documentation, final review & demo
```

---

## Step-by-Step Results

### Step 1 — Data Exploration & Preparation

- Verified all image files and bounding-box XML annotations across all 4 categories.
- Detected and removed corrupted files and inconsistent annotations (negative coordinates, out-of-bounds boxes).
- Confirmed annotation format: Pascal VOC XML → converted to normalized YOLO format `(class_id, cx, cy, w, h)`.
- Applied stratified splitting to ensure class balance across splits:

| Split | Proportion |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

> No data leakage: splits are performed at the image level before any augmentation.

---

### Step 2 — Preprocessing & Data Augmentation

| Operation | Train | Val / Test |
|---|---|---|
| Resize | 224×224 (classif.) / 512×512 (detection) | Same |
| Normalization | `[0, 1]` (MobileNetV2 preprocessing) | Same |
| Random Rotation | ±15° | ❌ |
| Random Zoom | 10% | ❌ |
| Horizontal Flip | Yes | ❌ |
| Contrast Adjustment | Moderate | ❌ |
| Bounding Box Adaptation | Applied to all geometric transforms | — |

> Validation and test sets use **deterministic** preprocessing only (resize + normalize).

---

### Step 3 — Classification by Transfer Learning

**Model:** MobileNetV2 pretrained on ImageNet

**Two-phase training:**

1. **Feature Extraction** — Backbone frozen, only the classification head trained.
2. **Fine-tuning** — Last N layers of the backbone unfrozen, trained with a lower learning rate.

**Architecture (head):**
```
GlobalAveragePooling2D
→ Dense(256, activation='relu') + L2
→ Dropout(0.4)
→ Dense(4, activation='softmax')
```

---

### Step 4 — Overfitting Prevention

Four configurations were compared:

| Config | Dropout | L2 | Augmentation | Val Accuracy |
|---|---|---|---|---|
| Baseline fine-tune | 0.4 | ❌ | Minimal | ~94% |
| + Augmentation | 0.4 | ❌ | Full | ~95% |
| + L2 regularization | 0.4 | 1e-4 | Full | ~96% |

**Techniques applied:**

| Technique | Configuration |
|---|---|
| Data Augmentation | rotation, zoom, flip, contrast |
| Dropout | rate = 0.4 |
| Early Stopping | patience = 5, monitor = `val_loss` |
| ModelCheckpoint | save best model on `val_accuracy` |
| ReduceLROnPlateau | factor = 0.5, patience = 3 |
| L2 Regularization | λ = 1e-4 |

---

### Step 5 — Object Detection with YOLOv8

**Model:** `YOLOv8n` (nano) — pretrained on COCO

**Training configuration:**

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Patience (early stop) | 10 |
| Image size | 512×512 |
| Batch size | 8 |
| Optimizer | Auto (AdamW) |
| IoU threshold | 0.7 |
| Classes | 3 (Glioma, Meningioma, Pituitary) |

**Augmentations (built-in Ultralytics):**

```
HSV shift, Mosaic (1.0), Horizontal flip (0.5),
Scale (0.5), Random Erasing (0.4), RandAugment
```

---

### Step 6 — Model Evaluation

#### 📊 Classification — Final Test Results

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Glioma | 0.955 | 0.977 | **0.966** | 173 |
| Meningioma | 0.976 | 0.931 | **0.953** | 217 |
| No Tumor | 0.971 | 0.953 | **0.962** | 107 |
| Pituitary | 0.964 | 1.000 | **0.982** | 214 |
| **Overall** | **0.967** | **0.966** | **0.966** | **711** |

> **Test Accuracy: 96.62%**

#### 🎯 Detection — YOLOv8 Test Results

| Metric | Overall | Glioma | Meningioma | Pituitary |
|---|---|---|---|---|
| Precision | **0.919** | 0.889 | 0.954 | 0.913 |
| Recall | **0.877** | 0.758 | 0.982 | 0.892 |
| mAP@50 | **0.940** | 0.874 | 0.989 | 0.957 |
| mAP@50-95 | **0.735** | 0.643 | 0.838 | 0.723 |

> Best performance on **Meningioma** (mAP50 = 0.989). **Glioma** shows lower recall (0.758), likely due to diffuse, irregular boundaries.

---

## Usage

### 1. Data Preparation

```bash
# Explore the dataset
python notebooks/data_exploration.py

# Prepare classification splits
python src/classification/model.py

# Convert annotations and prepare YOLO dataset
python src/detection/prepare_dataset_v2.py

# Verify YOLO annotations
python src/detection/check_dataset_v2.py
```

### 2. Classification Training

```bash
# Phase 1: Feature extraction
python src/classification/trainfeatureE.py

# Phase 2: Fine-tuning (baseline)
python src/classification/trainfineT.py

# Phase 2: Fine-tuning + augmentation
python src/classification/trainfineT_aug.py

# Phase 2: Fine-tuning + L2 regularization
python src/classification/trainfineT_l2.py

# Evaluate on test set
python src/classification/evaluate.py
```

### 3. Detection Training

```bash
# Train YOLOv8
python src/detection/train.py

# Evaluate on test set
python src/detection/evaluate.py
```

### 4. Launch Streamlit Interface

```bash
streamlit run dashboard/app.py
```

---

## Docker

### Build & Run

```bash
# Build and start all services
docker-compose up --build

# Services:
#   - Streamlit   → http://localhost:8501
#   - Airflow     → http://localhost:8080
```

### `docker-compose.yml` Services

```yaml
services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./results:/app/results

  airflow:
    image: apache/airflow:2.7.0
    ports:
      - "8080:8080"
    volumes:
      - ./dags:/opt/airflow/dags
```

> Model weights and results are mounted as volumes — no retraining needed on container restart.

---

## Airflow Pipeline

The `dags/mri_pipeline.py` DAG automates the full ML workflow:

```
[check_data] → [preprocess] → [train_classifier] → [train_yolo]
                                      ↓                   ↓
                              [eval_classifier]   [eval_yolo]
                                      └─────────┬─────────┘
                                          [save_metrics]
```

**Access Airflow UI:** `http://localhost:8080`  
Default credentials: `admin` / `admin`

---

## Streamlit Interface

**Access:** `http://localhost:8501`

**Features:**
- 📤 Upload a brain MRI image
- 🏷️ Display predicted category + class probabilities (bar chart)
- 📦 Visualize detection bounding boxes with confidence scores
- 📈 Browse training metrics and evaluation results
- 🔁 Models loaded once at startup — no retraining on each interaction

---

## Limitations

| Limitation | Description |
|---|---|
| Dataset size | Moderate size; results may not generalize to clinical-grade data |
| "No Tumor" detection | Excluded from detection (no annotated region available) |
| Glioma recall | Lower recall (0.758) due to irregular tumor boundaries |
| Hardware | Trained on CPU; GPU would significantly reduce training time |
| Scope | Prototype only — not validated for clinical use |

---

## Author

**Idriss Aharhar**  
Certification RNCP — Développeur·se en Intelligence Artificielle  
Project period: 19/10 → 30/10/2026

---

> *This prototype is for educational purposes only and must not be presented as a medical diagnostic tool.*


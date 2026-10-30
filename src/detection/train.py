
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[2]


DATA_YAML = (
    PROJECT_DIR / "data" / "processed_v2" / "detection" / "data.yaml"
)


RESULTS_DIR = PROJECT_DIR / "results" / "detection"

IMAGE_SIZE = 512
EPOCHS = 50
BATCH_SIZE = 8


def train_model():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=10,
        device=device,
        project=str(RESULTS_DIR),
        name="yolov8n_mri",
        pretrained=True,
        plots=True,
        save=True,
        seed=42
    )

    best_model = (
        RESULTS_DIR / "yolov8n_mri" / "weights" / "best.pt"
    )

    print("\nTraining completed.")
    print(f"Best model: {best_model}")


def main():
    train_model()


if __name__ == "__main__":
    main()

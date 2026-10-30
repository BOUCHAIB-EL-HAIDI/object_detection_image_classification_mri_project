
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "detection"
    / "yolov8n_mri-2"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    PROJECT_DIR
    / "data"
    / "processed_v2"
    / "detection"
    / "data.yaml"
)


def evaluate_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=512,
        batch=8,
        device=0,
        plots=True,
        project=str(PROJECT_DIR / "results" / "evaluation"),
        name="yolo_test"
    )

    print("\nYOLOv8 Test Results")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")


if __name__ == "__main__":
    evaluate_model()

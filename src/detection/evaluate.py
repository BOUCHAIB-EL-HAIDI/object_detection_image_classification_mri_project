
from pathlib import Path
import csv

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

OUTPUT_DIR = PROJECT_DIR / "results" / "evaluation"
CSV_PATH = OUTPUT_DIR / "yolo_test" / "test_metrics.csv"


def save_results_csv(results):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics = results.box

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Class",
            "Precision",
            "Recall",
            "mAP50",
            "mAP50-95"
        ])

        # Overall results
        writer.writerow([
            "All",
            metrics.mp,
            metrics.mr,
            metrics.map50,
            metrics.map
        ])

        # Results for each class
        for class_id, class_name in results.names.items():
            writer.writerow([
                class_name,
                metrics.p[class_id],
                metrics.r[class_id],
                metrics.ap50[class_id],
                metrics.ap[class_id]
            ])

    print(f"\nCSV saved: {CSV_PATH}")


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
        project=str(OUTPUT_DIR),
        name="yolo_test"
    )

    print("\nYOLOv8 Test Results")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

    save_results_csv(results)


if __name__ == "__main__":
    evaluate_model()

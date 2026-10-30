
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "classification"
MODEL_PATH = PROJECT_DIR / "models" / "classification" / "mobilenetv2_finetuned_aug_best.keras"
RESULTS_DIR = PROJECT_DIR / "results" / "classification" / "final_evaluation"

IMAGE_SIZE = 384
BATCH_SIZE = 16


def load_test_dataset():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    return test_ds, test_ds.class_names


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input":
                tf.keras.applications.mobilenet_v2.preprocess_input
        },
    )


def save_confusion_matrix(y_true, y_pred, class_names):
    matrix = confusion_matrix(
        y_true, y_pred, labels=range(len(class_names))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    display.plot(cmap="Blues", xticks_rotation=45, values_format="d")
    plt.title("Final MobileNetV2 - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.show()
    plt.close()


def evaluate_model(model, test_ds, class_names):
    loss, accuracy = model.evaluate(test_ds, verbose=1)

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    predictions = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    print("\nFINAL MODEL EVALUATION")
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    save_confusion_matrix(y_true, y_pred, class_names)


def main():
    test_ds, class_names = load_test_dataset()

    print("Test classes:", class_names)
    print("Test images:", len(test_ds.file_paths))

    model = load_model()
    evaluate_model(model, test_ds, class_names)

    print("\nEvaluation completed.")
    print("Results saved to:", RESULTS_DIR)

    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()

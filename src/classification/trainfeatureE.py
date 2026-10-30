
from pathlib import Path

import tensorflow as tf

from model import create_model


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "classification"
MODEL_DIR = PROJECT_DIR / "models" / "classification"

IMAGE_SIZE = 384
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "val",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    return train_ds, val_ds, test_ds


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def train_model(model, train_ds, val_ds):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    return history


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = load_datasets()

    model, base_model = create_model(
        model_name="MobileNetV2",
        image_size=IMAGE_SIZE,
    )

    compile_model(model)

    print("\nStarting MobileNetV2 training...")
    print("Backbone frozen:", not base_model.trainable)
    print("Classes:", train_ds.class_names if hasattr(train_ds, "class_names") else
          ["Glioma", "Meningioma", "No Tumor", "Pituitary"])

    train_model(model, train_ds, val_ds)

    model_path = MODEL_DIR / "mobilenetv2_best.keras"
    model.save(model_path)

    print("\nModel saved to:", model_path)


if __name__ == "__main__":
    main()


from pathlib import Path

import tensorflow as tf


# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "classification"
MODEL_DIR = PROJECT_DIR / "models" / "classification"

# Training configuration
IMAGE_SIZE = 384
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42
LEARNING_RATE = 0.00001
LAYERS_TO_UNFREEZE = 30


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

    return train_ds, val_ds


def load_trained_model():
    model_path = MODEL_DIR / "mobilenetv2_best.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Run trainfeatureE.py first."
        )

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input":
                tf.keras.applications.mobilenet_v2.preprocess_input
        },
    )

    base_model = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and layer.name.startswith("mobilenetv2")
        ),
        None,
    )

    if base_model is None:
        raise ValueError("Could not find the MobileNetV2 backbone.")

    return model, base_model


def configure_fine_tuning(base_model):
    # Unfreeze the backbone, then freeze all its layers.
    base_model.trainable = True

    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze only the last 30 layers.
    for layer in base_model.layers[-LAYERS_TO_UNFREEZE:]:
        if not isinstance(
            layer, tf.keras.layers.BatchNormalization
        ):
            layer.trainable = True


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def train_model(model, train_ds, val_ds):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(
            MODEL_DIR / "mobilenetv2_finetuned_best.keras"
        ),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr,
        ],
    )

    return history


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = load_datasets()

    model, base_model = load_trained_model()

    configure_fine_tuning(base_model)

    trainable_layers = sum(
        layer.trainable for layer in base_model.layers
    )

    print("\nStarting MobileNetV2 fine-tuning...")
    print("Backbone:", base_model.name)
    print("Trainable backbone layers:", trainable_layers)
    print("Learning rate:", LEARNING_RATE)
    print("Training images:", len(train_ds.file_paths))
    print("Validation images:", len(val_ds.file_paths))

    compile_model(model)

    history = train_model(model, train_ds, val_ds)

    print("\nFine-tuning completed.")
    print(
        "Best model saved to:",
        MODEL_DIR / "mobilenetv2_finetuned_best.keras",
    )

    return history


if __name__ == "__main__":
    main()

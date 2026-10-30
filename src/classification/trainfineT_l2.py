
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import tensorflow as tf


# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "classification"
MODEL_DIR = PROJECT_DIR / "models" / "classification"
RESULTS_DIR = PROJECT_DIR / "results" / "classification" / "l2"

L2_MODEL_PATH = (
    MODEL_DIR / "mobilenetv2_finetuned_l2_best.keras"
)

L2_STRENGTH = 0.0001

# Training configuration
IMAGE_SIZE = 384
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42
LEARNING_RATE = 0.00001
LAYERS_TO_UNFREEZE = 30



def create_data_augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


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


def add_l2_regularization(model):
    old_head = model.layers[-1]

    regularized_head = tf.keras.layers.Dense(
        units=old_head.units,
        activation=old_head.activation,
        kernel_regularizer=tf.keras.regularizers.l2(L2_STRENGTH),
        name="predictions_l2",
    )

    features = model.layers[-2].output
    outputs = regularized_head(features)

    new_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=outputs,
    )

    regularized_head.set_weights(old_head.get_weights())

    return new_model



def count_images(directory):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def load_trained_model():
    # Start from the same feature-extraction checkpoint
    # used by the original fine-tuning experiment.
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
    model = add_l2_regularization(model)

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
    base_model.trainable = True

    for layer in base_model.layers:
        layer.trainable = False

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
        filepath=str(L2_MODEL_PATH),
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


def save_training_history(history):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "training_history.csv"
    history_data = history.history

    columns = list(history_data.keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch"] + columns)

        for index in range(len(history_data["loss"])):
            row = [
                index + 1,
                *[
                    history_data[column][index]
                    for column in columns
                ],
            ]
            writer.writerow(row)

    print("Training history saved:", csv_path)


def save_learning_curves(history):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    history_data = history.history
    epochs = range(1, len(history_data["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy curve
    axes[0].plot(
        epochs,
        history_data["accuracy"],
        label="Training accuracy",
    )
    axes[0].plot(
        epochs,
        history_data["val_accuracy"],
        label="Validation accuracy",
    )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss curve
    axes[1].plot(
        epochs,
        history_data["loss"],
        label="Training loss",
    )
    axes[1].plot(
        epochs,
        history_data["val_loss"],
        label="Validation loss",
    )
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    plot_path = RESULTS_DIR / "learning_curves.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Learning curves saved:", plot_path)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = load_datasets()

    augmentation = create_data_augmentation()

    # Augment training images only.
    # MobileNetV2 preprocessing remains inside the model.
    train_ds = train_ds.map(
        lambda images, labels: (
            augmentation(images, training=True),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    model, base_model = load_trained_model()

    configure_fine_tuning(base_model)

    trainable_layers = sum(
        layer.trainable for layer in base_model.layers
    )

    print("\nStarting MobileNetV2 fine-tuning with augmentation...")
    print("Backbone:", base_model.name)
    print("Trainable backbone layers:", trainable_layers)
    print("Learning rate:", LEARNING_RATE)
    print("Training images:", count_images(DATA_DIR / "train"))
    print("Validation images:", count_images(DATA_DIR / "val"))

    compile_model(model)

    history = train_model(model, train_ds, val_ds)

    save_training_history(history)
    save_learning_curves(history)

    print("\nAugmented fine-tuning completed.")
    print("Best model saved to:", L2_MODEL_PATH)
    print("Results saved to:", RESULTS_DIR)

    return history


if __name__ == "__main__":
    main()

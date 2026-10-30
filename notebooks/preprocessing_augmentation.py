
from pathlib import Path
import random

import numpy as np
import yaml
from PIL import Image


PROJECT_DIR = Path(
    r"C:\Users\ycode\Desktop\OBJECT DETECTION MRI DEEP LEARNING"
)

DATA_DIR = PROJECT_DIR / "data" / "raw" / "MRI_Dataset"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

CLASSIFICATION_DIR = OUTPUT_DIR / "classification"
DETECTION_DIR = OUTPUT_DIR / "detection"

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
DETECTION_CLASSES = ["Glioma", "Meningioma", "Pituitary"]

DETECTION_CLASS_MAP = {
    0: 0,
    1: 1,
    3: 2,
}

CLASSIFICATION_IMAGE_SIZE = 384
DETECTION_IMAGE_SIZE = 512

TEST_RATIO = 0.15
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def get_images(folder):
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_train_test():
    rng = random.Random(RANDOM_SEED)
    test_images = set()

    for class_name in CLASSES:
        folder = DATA_DIR / "Train" / class_name / "images"

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        images = get_images(folder)
        test_count = round(len(images) * TEST_RATIO)

        selected = rng.sample(images, test_count)

        for image_path in selected:
            test_images.add(image_path.resolve())

        print(
            f"{class_name}: "
            f"{len(images) - test_count} train, "
            f"{test_count} test"
        )

    print(f"Total test images: {len(test_images)}")
    return test_images


def read_labels(label_path):
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label: {label_path}")

    boxes = []
    class_ids = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            raise ValueError(f"Invalid label: {label_path}")

        original_id = int(values[0])
        box = list(map(float, values[1:]))

        if original_id not in DETECTION_CLASS_MAP:
            raise ValueError(
                f"Unexpected class ID {original_id}: {label_path}"
            )

        if not np.isfinite(box).all():
            raise ValueError(f"Invalid coordinates: {label_path}")

        x, y, width, height = box

        if not all(0 <= value <= 1 for value in box):
            raise ValueError(
                f"Coordinates outside [0, 1]: {label_path}"
            )

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid box dimensions: {label_path}")

        if (
            x - width / 2 < -1e-6
            or x + width / 2 > 1 + 1e-6
            or y - height / 2 < -1e-6
            or y + height / 2 > 1 + 1e-6
        ):
            raise ValueError(f"Box outside image: {label_path}")

        boxes.append(box)
        class_ids.append(DETECTION_CLASS_MAP[original_id])

    return boxes, class_ids


def load_image(image_path, image_size):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        return np.array(image)


def get_detection_labels(image_path, class_name):
    if class_name == "No Tumor":
        return [], []

    label_path = (
        image_path.parent.parent
        / "labels"
        / f"{image_path.stem}.txt"
    )

    return read_labels(label_path)


def save_image(image, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path, quality=95)


def save_labels(boxes, class_ids, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for box, class_id in zip(boxes, class_ids):
            coordinates = " ".join(
                f"{value:.6f}" for value in box
            )
            file.write(f"{class_id} {coordinates}\n")


def process_image(image_path, class_name, split):
    # Classification image: 384 x 384
    classification_image = load_image(
        image_path,
        CLASSIFICATION_IMAGE_SIZE
    )

    # Detection image: 512 x 512
    detection_image = load_image(
        image_path,
        DETECTION_IMAGE_SIZE
    )

    boxes, class_ids = get_detection_labels(
        image_path, class_name
    )

    filename = f"{class_name}__{image_path.stem}.jpg"

    # Classification output path
    classification_path = (
        CLASSIFICATION_DIR
        / split.lower()
        / class_name
        / filename
    )

    # Detection output paths
    detection_image_path = (
        DETECTION_DIR
        / "images"
        / split.lower()
        / filename
    )

    detection_label_path = (
        DETECTION_DIR
        / "labels"
        / split.lower()
        / f"{Path(filename).stem}.txt"
    )

    save_image(classification_image, classification_path)
    save_image(detection_image, detection_image_path)
    save_labels(boxes, class_ids, detection_label_path)


def create_yolo_config():
    config = {
        "path": DETECTION_DIR.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(DETECTION_CLASSES)),
    }

    config_path = DETECTION_DIR / "data.yaml"

    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    print(f"YOLO configuration saved: {config_path}")


def process_dataset():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_DIR}. "
            "Rename or remove the existing step2 folder "
            "before running preprocessing again."
        )

    test_images = split_train_test()

    counts = {"Train": 0, "Val": 0, "Test": 0}

    for source_split in ["Train", "Val"]:
        for class_name in CLASSES:
            folder = (
                DATA_DIR
                / source_split
                / class_name
                / "images"
            )

            if not folder.exists():
                raise FileNotFoundError(f"Missing folder: {folder}")

            images = get_images(folder)

            for image_path in images:
                if source_split == "Val":
                    output_split = "Val"
                elif image_path.resolve() in test_images:
                    output_split = "Test"
                else:
                    output_split = "Train"

                process_image(
                    image_path,
                    class_name,
                    output_split,
                )

                counts[output_split] += 1

            print(
                f"{source_split} - {class_name}: "
                f"{len(images)} images"
            )

    create_yolo_config()

    classification_train = sum(
        len(get_images(CLASSIFICATION_DIR / "train" / class_name))
        for class_name in CLASSES
    )

    print("\nPreprocessing completed.")
    print(f"Training originals: {counts['Train']}")
    print(f"Test originals: {counts['Test']}")
    print(f"Validation originals: {counts['Val']}")
    print(f"Classification training images: {classification_train}")
    print(f"Classification image size: {CLASSIFICATION_IMAGE_SIZE}")
    print(f"Detection image size: {DETECTION_IMAGE_SIZE}")
    print(f"Output folder: {OUTPUT_DIR}")


def main():
    process_dataset()


if __name__ == "__main__":
    main()

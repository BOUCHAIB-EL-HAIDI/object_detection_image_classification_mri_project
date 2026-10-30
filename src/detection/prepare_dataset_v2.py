
from pathlib import Path
import random

import numpy as np
import yaml
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "raw" / "MRI_Dataset"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed_v2"
DETECTION_DIR = OUTPUT_DIR / "detection"

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
DETECTION_CLASSES = ["Glioma", "Meningioma", "Pituitary"]

DETECTION_CLASS_MAP = {0: 0, 1: 1, 3: 2}

IMAGE_SIZE = 512
TEST_RATIO = 0.15
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def get_images(folder):
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_train_test():
    rng = random.Random(RANDOM_SEED)
    test_images = set()

    for class_name in CLASSES:
        folder = DATA_DIR / "Train" / class_name / "images"
        images = get_images(folder)
        test_count = round(len(images) * TEST_RATIO)

        for image_path in rng.sample(images, test_count):
            test_images.add(image_path.resolve())

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
            raise ValueError(f"Unexpected class ID {original_id}: {label_path}")

        if not np.isfinite(box).all():
            raise ValueError(f"Invalid coordinates: {label_path}")

        x, y, width, height = box

        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"Invalid box center: {label_path}")

        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Invalid box size: {label_path}")

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


def letterbox_image(image_path):
    with Image.open(image_path) as original:
        image = original.convert("RGB")

    original_width, original_height = image.size

    scale = min(
        IMAGE_SIZE / original_width,
        IMAGE_SIZE / original_height
    )

    new_width = round(original_width * scale)
    new_height = round(original_height * scale)

    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    pad_left = (IMAGE_SIZE - new_width) // 2
    pad_top = (IMAGE_SIZE - new_height) // 2

    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (114, 114, 114))
    canvas.paste(image, (pad_left, pad_top))

    return np.array(canvas), original_width, original_height, scale, pad_left, pad_top


def transform_boxes(boxes, original_width, original_height, scale, pad_left, pad_top):
    transformed = []

    for x, y, width, height in boxes:
        x_center = (x * original_width * scale + pad_left) / IMAGE_SIZE
        y_center = (y * original_height * scale + pad_top) / IMAGE_SIZE
        box_width = width * original_width * scale / IMAGE_SIZE
        box_height = height * original_height * scale / IMAGE_SIZE

        transformed.append([
            x_center, y_center, box_width, box_height
        ])

    return transformed


def save_labels(boxes, class_ids, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for box, class_id in zip(boxes, class_ids):
            coordinates = " ".join(f"{value:.6f}" for value in box)
            file.write(f"{class_id} {coordinates}\n")


def process_image(image_path, class_name, split):
    label_path = (
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    )

    if class_name == "No Tumor":
        boxes, class_ids = [], []
    else:
        boxes, class_ids = read_labels(label_path)

    image, width, height, scale, pad_left, pad_top = letterbox_image(image_path)

    boxes = transform_boxes(
        boxes, width, height, scale, pad_left, pad_top
    )

    filename = f"{class_name}__{image_path.stem}.jpg"

    image_output = DETECTION_DIR / "images" / split.lower() / filename
    label_output = DETECTION_DIR / "labels" / split.lower() / f"{Path(filename).stem}.txt"

    image_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(image_output, quality=95)

    save_labels(boxes, class_ids, label_output)


def create_yolo_config():
    config = {
        "path": DETECTION_DIR.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(DETECTION_CLASSES)),
    }

    with open(DETECTION_DIR / "data.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def process_dataset():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. Rename it before running again."
        )

    test_images = split_train_test()

    for source_split in ["Train", "Val"]:
        for class_name in CLASSES:
            images_dir = DATA_DIR / source_split / class_name / "images"

            for image_path in get_images(images_dir):
                if source_split == "Val":
                    split = "Val"
                elif image_path.resolve() in test_images:
                    split = "Test"
                else:
                    split = "Train"

                process_image(image_path, class_name, split)

            print(f"{source_split} - {class_name}: processed")

    create_yolo_config()
    print(f"\nCorrected detection dataset saved to: {DETECTION_DIR}")


def main():
    process_dataset()


if __name__ == "__main__":
    main()

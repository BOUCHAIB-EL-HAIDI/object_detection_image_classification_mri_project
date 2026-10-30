
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed_v2" / "detection"

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary"]


def check_dataset():
    config_path = DATA_DIR / "data.yaml"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print("YOLO configuration:")
    print(config)

    total_images = 0
    total_labels = 0

    for split in ["train", "val", "test"]:
        images_dir = DATA_DIR / "images" / split
        labels_dir = DATA_DIR / "labels" / split

        images = list(images_dir.glob("*.jpg"))
        labels = list(labels_dir.glob("*.txt"))

        missing = [
            image.name for image in images
            if not (labels_dir / f"{image.stem}.txt").exists()
        ]

        print(f"\n{split.upper()}")
        print(f"Images: {len(images)}")
        print(f"Label files: {len(labels)}")
        print(f"Images without labels: {len(missing)}")

        total_images += len(images)
        total_labels += len(labels)

    print(f"\nTotal images: {total_images}")
    print(f"Total label files: {total_labels}")


def visualize_bbox():
    image_path = (
        DATA_DIR / "images" / "val" / "Pituitary__Tr-pi_0042.jpg"
    )
    label_path = (
        DATA_DIR / "labels" / "val" / "Pituitary__Tr-pi_0042.txt"
    )

    if not image_path.exists() or not label_path.exists():
        print("\nVisualization example not found.")
        return

    with Image.open(image_path) as image:
        width, height = image.size
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image)

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        class_id, x, y, box_width, box_height = map(
            float, line.split()
        )

        x_center = x * width
        y_center = y * height
        box_width *= width
        box_height *= height

        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2

        rectangle = patches.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rectangle)
        ax.text(
            x_min,
            y_min,
            CLASS_NAMES[int(class_id)],
            color="white",
            backgroundcolor="red"
        )

    ax.set_title("YOLO bounding-box verification")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    check_dataset()
    visualize_bbox()


if __name__ == "__main__":
    main()


from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle


PROJECT_DIR = Path(r"C:\Users\ycode\Desktop\OBJECT DETECTION MRI DEEP LEARNING")
DATA_DIR = PROJECT_DIR / "data" / "raw" / "MRI_Dataset"

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_image_files(images_dir):
    if not images_dir.exists():
        return []

    return [
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def check_image(image_path):
    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            width, height = image.size

        return True, width, height

    except Exception:
        return False, None, None


def validate_label_file(label_path):
    if not label_path.exists():
        return "missing", 0, ["Label file not found"]

    errors = []
    box_count = 0

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return "invalid", 0, [str(error)]

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        parts = line.split()

        try:
            if len(parts) != 5:
                raise ValueError("Expected 5 values")

            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            if class_id not in CLASS_TO_ID.values():
                raise ValueError("Invalid class ID")

            values = [x_center, y_center, width, height]

            if not all(pd.notna(value) and 0 <= value <= 1 for value in values):
                raise ValueError("Coordinates must be between 0 and 1")

            if width <= 0 or height <= 0:
                raise ValueError("Bounding-box width and height must be positive")

            if (
                x_center - width / 2 < -1e-6
                or x_center + width / 2 > 1 + 1e-6
                or y_center - height / 2 < -1e-6
                or y_center + height / 2 > 1 + 1e-6
            ):
                raise ValueError("Bounding box extends outside image")

            box_count += 1

        except (ValueError, OverflowError) as error:
            errors.append(f"Line {line_number}: {error}")

    status = "invalid" if errors else "valid"
    return status, box_count, errors


def explore_dataset():
    records = []

    for split in ["Train", "Val"]:
        for class_name in CLASSES:
            class_dir = DATA_DIR / split / class_name
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"

            for image_path in get_image_files(images_dir):
                label_path = labels_dir / f"{image_path.stem}.txt"

                image_ok, width, height = check_image(image_path)
                label_status, box_count, errors = validate_label_file(label_path)

                records.append({
                    "split": split,
                    "class": class_name,
                    "image": str(image_path),
                    "label": str(label_path),
                    "image_ok": image_ok,
                    "width": width,
                    "height": height,
                    "label_status": label_status,
                    "box_count": box_count,
                    "label_errors": errors
                })

    return pd.DataFrame(records)


def display_summary(df):
    print("\n========== DATASET SUMMARY ==========")
    print(f"Total images: {len(df)}")
    print(f"Corrupted images: {(~df['image_ok']).sum()}")
    print(f"Missing label files: {(df['label_status'] == 'missing').sum()}")
    print(f"Invalid label files: {(df['label_status'] == 'invalid').sum()}")

    print("\n========== IMAGES PER CLASS ==========")
    print(
        df.groupby(["split", "class"])
        .size()
        .unstack(fill_value=0)
    )

    print("\n========== IMAGE DIMENSIONS ==========")
    valid_images = df[df["image_ok"]]
    print(
        valid_images.groupby(["width", "height"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\n========== BOUNDING BOXES ==========")
    print(f"Total valid boxes: {df['box_count'].sum()}")

    print("\nBoxes per image class:")
    print(df.groupby("class")["box_count"].sum())

    print("\n========== ANNOTATION ERRORS ==========")
    for _, row in df[df["label_errors"].map(bool)].head(10).iterrows():
        print(f"\n{row['label']}")
        for error in row["label_errors"]:
            print(f"  - {error}")



def show_sample_images(df, number=8):
    samples = df[
        (df["split"] == "Train") & df["image_ok"]
    ].head(number)

    if samples.empty:
        print("No valid images available to display.")
        return

    columns = 4
    rows = (len(samples) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (_, row) in zip(axes, samples.iterrows()):
        with Image.open(row["image"]) as image:
            image = image.convert("RGB")
            ax.imshow(image)
            image_width, image_height = image.size

        label_path = Path(row["label"])

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                except ValueError:
                    continue

                if (
                    class_id not in CLASS_TO_ID.values()
                    or not all(pd.notna(v) and 0 <= v <= 1
                               for v in [x_center, y_center, width, height])
                    or width <= 0
                    or height <= 0
                ):
                    continue

                x = (x_center - width / 2) * image_width
                y = (y_center - height / 2) * image_height

                ax.add_patch(
                    Rectangle(
                        (x, y),
                        width * image_width,
                        height * image_height,
                        fill=False,
                        edgecolor="red",
                        linewidth=2
                    )
                )

                ax.text(
                    x,
                    max(0, y - 5),
                    CLASSES[class_id],
                    color="white",
                    backgroundcolor="red",
                    fontsize=8
                )

        ax.set_title(row["class"])
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    df = explore_dataset()

    if df.empty:
        print("No images found. Check the dataset folder structure.")
        return

    display_summary(df)
    show_sample_images(df)

    print("\nData exploration completed.")


if __name__ == "__main__":
    main()

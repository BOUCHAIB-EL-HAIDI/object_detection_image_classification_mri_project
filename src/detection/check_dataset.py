
from pathlib import Path
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "detection"


def check_dataset():
    config_path = DATA_DIR / "data.yaml"

    if not config_path.exists():
        print("ERROR: data.yaml is missing.")
        return

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print("YOLO configuration:")
    print(config)

    for split in ["train", "val", "test"]:
        images_dir = DATA_DIR / "images" / split
        labels_dir = DATA_DIR / "labels" / split

        if not images_dir.exists() or not labels_dir.exists():
            print(f"\nERROR: Missing folders for {split}.")
            continue

        images = list(images_dir.glob("*.jpg"))
        labels = list(labels_dir.glob("*.txt"))

        missing = [
            image.stem for image in images
            if not (labels_dir / f"{image.stem}.txt").exists()
        ]

        print(f"\n{split.upper()}")
        print(f"Images: {len(images)}")
        print(f"Label files: {len(labels)}")
        print(f"Images without label files: {len(missing)}")

        if missing:
            print("Examples:", missing[:5])


def main():
    check_dataset()


if __name__ == "__main__":
    main()

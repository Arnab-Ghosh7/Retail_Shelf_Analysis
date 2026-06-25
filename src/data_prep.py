import os
import io
from pathlib import Path
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def prepare_yolo_dataset_from_parquet(split="train", limit=100, output_dir="datasets/sku110k_subset"):
    """
    Downloads the first parquet file for the split, reads it with pandas,
    and extracts a subset of images and labels in YOLO format.
    """
    print(f"\n--- Preparing '{split}' split (limit={limit}) ---")

    if split == "train":
        filename = "data/train-00000-of-00019.parquet"
    elif split == "val":
        filename = "data/validation-00000-of-00002.parquet"
    else:
        raise ValueError(f"Invalid split: {split}")

    print(f"Downloading parquet file '{filename}' from benjamintli/sku110k on HF...")
    parquet_path = hf_hub_download(
        repo_id="benjamintli/sku110k",
        filename=filename,
        repo_type="dataset"
    )
    print(f"Parquet file downloaded to: {parquet_path}")

    print("Loading parquet data...")
    df = pd.read_parquet(parquet_path, columns=["image", "objects"])
    print(f"Total rows available in parquet file: {len(df)}")

    img_dir = Path(output_dir) / "images" / split
    lbl_dir = Path(output_dir) / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for idx in tqdm(range(min(limit, len(df)))):
        row = df.iloc[idx]
        image_dict = row["image"]
        objects_dict = row["objects"]

        img_bytes = image_dict.get("bytes")
        if img_bytes is None:
            continue

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            continue

        filename_prefix = f"sku_{split}_{idx:04d}"

        img_path = img_dir / f"{filename_prefix}.jpg"
        img.save(img_path)

        label_path = lbl_dir / f"{filename_prefix}.txt"

        w, h = img.size
        bboxes = objects_dict.get("bbox", [])

        with open(label_path, "w") as f:
            for bbox in bboxes:
                x_min, y_min, box_w, box_h = bbox

                x_min = max(0.0, x_min)
                y_min = max(0.0, y_min)
                box_w = min(box_w, w - x_min)
                box_h = min(box_h, h - y_min)

                if box_w <= 0 or box_h <= 0:
                    continue

                x_center = x_min + box_w / 2.0
                y_center = y_min + box_h / 2.0

                x_center_norm = x_center / w
                y_center_norm = y_center / h
                width_norm = box_w / w
                height_norm = box_h / h

                class_id = 0
                f.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        processed += 1

    print(f"Successfully processed {processed} images and labels for split '{split}'")

if __name__ == "__main__":
    prepare_yolo_dataset_from_parquet(split="train", limit=200)
    prepare_yolo_dataset_from_parquet(split="val", limit=50)

    yaml_content = """
path: sku110k_subset  # dataset root dir
train: images/train  # train images (relative to path)
val: images/val      # val images (relative to path)

# Classes
names:
  0: product
"""
    yaml_path = Path("datasets/sku110k_subset/sku110k_subset.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content.strip())
    print(f"\nCreated YOLO dataset config file at {yaml_path}")

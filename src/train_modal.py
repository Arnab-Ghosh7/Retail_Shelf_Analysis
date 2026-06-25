import os
import shutil
from pathlib import Path
import modal

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "python3-opencv")
    .pip_install(
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scikit-learn>=1.2.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0"
    )
    .add_local_dir("./datasets/sku110k_subset", "/dataset")
)

app = modal.App("sku110k-training")

volume = modal.Volume.from_name("sku110k-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={"/models": volume}
)
def train_yolo():
    """
    Function running on Modal GPU. Copies dataset to mutable dir, trains YOLOv8 model,
    and commits results to Modal Volume.
    """
    from ultralytics import YOLO

    print("Checking mounted dataset...")
    if not os.path.exists("/dataset"):
        raise RuntimeError("Dataset mount not found at /dataset")

    print("Files in mount:", os.listdir("/dataset"))

    mutable_dataset_dir = "/tmp/dataset"
    if os.path.exists(mutable_dataset_dir):
        shutil.rmtree(mutable_dataset_dir)

    print(f"Copying dataset to {mutable_dataset_dir} for training...")
    shutil.copytree("/dataset", mutable_dataset_dir)

    yaml_content = """
path: /tmp/dataset
train: images/train
val: images/val
names:
  0: product
"""
    with open(f"{mutable_dataset_dir}/sku110k_subset.yaml", "w") as f:
        f.write(yaml_content.strip())

    print("Dataset ready. Loading pre-trained YOLOv8n weights...")
    model = YOLO("yolov8n.pt")

    print("Starting YOLO fine-tuning on remote GPU...")
    model.train(
        data=f"{mutable_dataset_dir}/sku110k_subset.yaml",
        epochs=15,
        imgsz=640,
        batch=8,
        device=0,
        project="/tmp/runs",
        name="sku110k"
    )

    os.makedirs("/models", exist_ok=True)
    best_weight_src = "/tmp/runs/sku110k/weights/best.pt"
    results_png_src = "/tmp/runs/sku110k/results.png"

    if os.path.exists(best_weight_src):
        shutil.copy(best_weight_src, "/models/best.pt")
        print("Copied best.pt to volume /models/best.pt")
    else:
        print("WARNING: best.pt weights not found!")

    if os.path.exists(results_png_src):
        shutil.copy(results_png_src, "/models/results.png")
        print("Copied results.png to volume /models/results.png")

    print("Committing changes to Modal Volume...")
    volume.commit()
    print("Remote training complete!")

@app.function(
    image=image,
    volumes={"/models": volume}
)
def get_trained_files():
    """
    Reads the trained weights and graphs from the Modal Volume and returns them as bytes.
    """
    files = {}
    best_path = "/models/best.pt"
    results_path = "/models/results.png"

    volume.reload()

    if os.path.exists(best_path):
        print("Reading best.pt...")
        with open(best_path, "rb") as f:
            files["best.pt"] = f.read()

    if os.path.exists(results_path):
        print("Reading results.png...")
        with open(results_path, "rb") as f:
            files["results.png"] = f.read()

    return files

@app.local_entrypoint()
def main():
    """
    Local entry point. Run with: modal run src/train_modal.py
    Triggers remote training on GPU, then downloads weights.
    """
    print("Starting remote training task on Modal...")
    train_yolo.remote()

    print("Downloading model weights and training logs from Modal Volume...")
    files = get_trained_files.remote()

    local_model_dir = Path("models")
    local_model_dir.mkdir(exist_ok=True)

    if "best.pt" in files:
        with open(local_model_dir / "best.pt", "wb") as f:
            f.write(files["best.pt"])
        print(f"Successfully saved weights to: {local_model_dir / 'best.pt'}")
    else:
        print("Error: best.pt not found in returned files.")

    if "results.png" in files:
        with open(local_model_dir / "results.png", "wb") as f:
            f.write(files["results.png"])
        print(f"Successfully saved training graphs to: {local_model_dir / 'results.png'}")

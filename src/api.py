import os
import sys
import io
import requests
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import modal

# Setup path to import local modules from the mounted directory
sys.path.append("/root")

# Define the container image with required ML and web libraries and include the local src code
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "python3-opencv")
    .pip_install(
        "fastapi>=0.90.0",
        "python-multipart>=0.0.6",
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scikit-learn>=1.2.0",
        "pillow>=9.4.0",
        "requests>=2.28.0"
    )
    .add_local_dir("./src", "/root/src")
)

app = modal.App("sku110k-api")

# Volume for trained models (read-only in API but read/write for loading)
volume = modal.Volume.from_name("sku110k-models", create_if_missing=True)

# Volume to persist and serve generated visualization output images
outputs_volume = modal.Volume.from_name("sku110k-outputs", create_if_missing=True)

# Define FastAPI application
web_app = FastAPI(title="Retail Shelf Analysis API")

# Global pipeline instance (lazy loaded)
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        sys.path.append("/root/src")
        from model_pipeline import ShelfAnalysisPipeline
        # Load weights from Modal Volume. GPU usage is mandatory as specified.
        pipeline = ShelfAnalysisPipeline(yolo_model_path="/models/best.pt", use_gpu=True)
    return pipeline

@web_app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    """
    Predict endpoint for retail shelf image analysis.
    Accepts an uploaded image file OR an image URL.
    """
    # 1. Fetch/load the image
    img = None
    if file is not None:
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    elif image_url is not None and image_url.strip() != "":
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'file' (upload) or 'image_url' must be provided."
        )

    # 2. Run shelf analysis pipeline
    try:
        pipe = get_pipeline()
        # Process and save visualization inside the /outputs volume
        res = pipe.process_image(img, output_dir="/outputs", eps=0.35)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # 3. Commit output volume changes so the file is saved and accessible
    # Wait, in newer versions of modal, volume changes are committed automatically, 
    # but calling commit is safer if it's a Volume.
    try:
        outputs_volume.commit()
    except Exception as e:
        print(f"Volume commit warning: {e}")

    # 4. Format return path
    vis_filename = os.path.basename(res["visualization_path"])
    
    # We return the filename under the outputs path
    return {
        "objects": res["objects"],
        "visualization_path": f"/outputs/{vis_filename}"
    }

# Endpoint to serve generated output files directly from the outputs volume
@web_app.get("/outputs/{filename}")
async def get_output(filename: str):
    # Reload volume to ensure we see the newly created image
    try:
        outputs_volume.reload()
    except Exception as e:
        print(f"Volume reload warning: {e}")
        
    filepath = os.path.join("/outputs", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(filepath, media_type="image/jpeg")

# Define the Modal web app function.
# Specifies GPU requirement, imports the code mount, and attaches the volumes.
@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": volume, "/outputs": outputs_volume}
)
@modal.asgi_app()
def api():
    return web_app

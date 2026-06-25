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

sys.path.append("/root")

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

volume = modal.Volume.from_name("sku110k-models", create_if_missing=True)

outputs_volume = modal.Volume.from_name("sku110k-outputs", create_if_missing=True)

web_app = FastAPI(title="Retail Shelf Analysis API")

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        sys.path.append("/root/src")
        from model_pipeline import ShelfAnalysisPipeline
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

    try:
        pipe = get_pipeline()
        res = pipe.process_image(img, output_dir="/outputs", eps=0.35)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    try:
        outputs_volume.commit()
    except Exception as e:
        print(f"Volume commit warning: {e}")

    vis_filename = os.path.basename(res["visualization_path"])

    return {
        "objects": res["objects"],
        "visualization_path": f"/outputs/{vis_filename}"
    }

@web_app.get("/outputs/{filename}")
async def get_output(filename: str):
    try:
        outputs_volume.reload()
    except Exception as e:
        print(f"Volume reload warning: {e}")

    filepath = os.path.join("/outputs", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(filepath, media_type="image/jpeg")

@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": volume, "/outputs": outputs_volume}
)
@modal.asgi_app()
def api():
    return web_app

# Retail Shelf Analysis System


## Architecture Overview
```
[Input Image/URL] 
       │
       ▼
┌──────────────────────────────────────────┐
│ 1. Product Detection (YOLOv8 Nano)       │  <-- Deployed on Modal GPU (T4)
└──────────────────────────────────────────┘
       │ (Bounding Box Crops)
       ▼
┌──────────────────────────────────────────┐
│ 2. Feature Extraction (ResNet-50)       │  <-- Feature vector extraction (2048-dim)
└──────────────────────────────────────────┘
       │ (L2-Normalized Embeddings)
       ▼
┌──────────────────────────────────────────┐
│ 3. Brand Clustering (DBSCAN / Cosine)    │  <-- Visual grouping without prior templates
└──────────────────────────────────────────┘
       │
       ├──────────────────────────┐
       ▼                          ▼
┌──────────────┐           ┌────────────────────────────────┐
│ JSON Output  │           │ Output Image Visualizations    │
└──────────────┘           └────────────────────────────────┘
```
The system follows a modular, serverless architecture optimized for high performance and scalability:



1. **Product Detection:** A fine-tuned YOLOv8 Nano model detects products on high-density retail shelves.
2. **Product Cropping & Embedding Extraction:** Bounding box crops are resized and passed through a pre-trained ResNet-50 network (with the classification head removed) to extract a 2048-dimensional feature vector, which is L2-normalized.
3. **Brand Grouping:** The normalized visual embeddings are clustered using **DBSCAN** with a **Cosine Distance** metric. Visually similar products (i.e. those sharing the same brand design/packaging) are grouped into the same cluster.
4. **Visualization & Response:** The pipeline generates an annotated image with color-coded bounding boxes corresponding to brand groups and returns a structured JSON payload with coordinates, confidence scores, and group labels.

---

## Technical Specifications & Training Details

### 1. Dataset Selection & Preprocessing
*   **Dataset:** **SKU-110K** (a dense retail shelf dataset featuring closely-packed objects and high occlusion).
*   **Subset Strategy:** To train within reasonable time and compute budgets, a subset of **200 training images** and **50 validation images** was programmatically streamed from Hugging Face (`benjamintli/sku110k`) and converted to YOLO bounding box format (`class_id x_center y_center width height` normalized).

### 2. Model & Training Configuration
*   **Base Model:** YOLOv8 Nano (`yolov8n.pt`) chosen for its real-time latency profile and light parameter size.
*   **Compute:** NVIDIA Tesla T4 GPU on Modal.com.
*   **Hyperparameters:**
    *   Epochs: 15
    *   Batch Size: 8
    *   Image Size: 640x640
    *   Device: CUDA (Tesla T4)

### 3. Evaluation Metrics (Validation Split)
*   **Precision (Box P):** `0.797`
*   **Recall (R):** `0.711`
*   **mAP50:** `0.755`
*   **mAP50-95:** `0.403`
*   **Inference Latency (remote T4):** `74.2ms` per image

---

## File Structure

```
├── datasets/                 # Local directory for dataset (git-ignored)
├── models/                   # Local model weights and training logs (git-ignored)
├── src/
│   ├── api.py                # FastAPI app deployed on Modal
│   ├── data_prep.py          # Hugging Face dataset download & prep script
│   ├── model_pipeline.py     # YOLO Inference + ResNet Embedding + Clustering Pipeline
│   └── train_modal.py        # Modal remote GPU training script
├── streamlit_app.py          # Streamlit UI frontend
├── .env                      # Credentials file (git-ignored)
├── .gitignore                # Git exclude rules
├── requirements.txt          # Python library dependencies
└── README.md                 # Project documentation
```

---

## Deployed Endpoints

The API is fully deployed on **Modal.com** with GPU resources enabled:

*   **API Base URL:** `https://hp-1996--sku110k-api-api.modal.run` (or your customized Modal deploy URL)
*   **Inference Endpoint:** `POST /predict`
    *   *Input (Form Data / Multipart):*
        *   `file`: Image upload (Optional)
        *   `image_url`: Image URL string (Optional)
    *   *Output (JSON):*
        ```json
        {
          "objects": [
            {
              "bbox": [x1, y1, x2, y2],
              "confidence": 0.9124,
              "group_id": "brand_0"
            }
          ],
          "visualization_path": "/outputs/vis_xxxxxxxx.jpg"
        }
        ```
*   **File Serving Endpoint:** `GET /outputs/{filename}`
    *   Serves the generated visualization JPEG image.

---

## Installation & Running Locally

### Prerequisites
*   Python 3.11
*   A Modal.com account (free tier includes GPU credits)

### 1. Set Up Environment
Clone the repository and set up a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure Credentials
Create a `.env` file in the root folder containing your Modal token credentials:
```env
MODAL_TOKEN_ID=ak-xxxxxxxxxxxxxxxxxxxx
MODAL_TOKEN_SECRET=as-xxxxxxxxxxxxxxxxxxxx
```
Authenticate the Modal CLI locally:
```bash
.\venv\Scripts\modal token set --token-id <YOUR_TOKEN_ID> --token-secret <YOUR_TOKEN_SECRET>
```

### 3. (Optional) Run Dataset Preparation
Generate the SKU-110K subset locally:
```bash
python src/data_prep.py
```

### 4. (Optional) Fine-tune the YOLO Model
Start the fine-tuning on Modal's remote T4 GPU:
```bash
$env:PYTHONIOENCODING="utf-8" # Windows CMD/PowerShell UTF-8 support
modal run src/train_modal.py
```
This script will train the model and automatically download the weights to `models/best.pt`.

### 5. Deploy the API Backend
Deploy the FastAPI server to Modal:
```bash
modal deploy src/api.py
```
This will print your unique deployed URL.

### 6. Run the Streamlit Interface
Launch the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```
You can paste your deployed Modal API URL into the sidebar, upload any shelf image, and visualize the product detection and brand grouping results in real-time.

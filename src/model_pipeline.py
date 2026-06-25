import os
import uuid
from pathlib import Path
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageColor
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

class ShelfAnalysisPipeline:
    def __init__(self, yolo_model_path="models/best.pt", use_gpu=False):
        """
        Initializes the pipeline by loading the YOLOv8 model and pre-trained ResNet-50.
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Pipeline running on device: {self.device}")

        if os.path.exists(yolo_model_path):
            print(f"Loading custom fine-tuned YOLO model from {yolo_model_path}...")
            self.detector = YOLO(yolo_model_path)
        else:
            print(f"WARNING: Fine-tuned model not found at {yolo_model_path}. Falling back to pre-trained yolov8n.pt")
            self.detector = YOLO("yolov8n.pt")

        print("Loading pre-trained ResNet-50 model...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _extract_embedding(self, crop_img):
        """
        Extracts L2-normalized visual embedding for a cropped product image.
        """
        tensor = self.transform(crop_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(tensor)
            embedding = torch.squeeze(features).cpu().numpy()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def group_brands(self, crops, eps=0.35, min_samples=2):
        """
        Clusters product crops by brand visual similarity using ResNet embeddings and DBSCAN.
        eps: Maximum cosine distance threshold (1 - cosine_similarity).
             e.g., eps=0.35 means similarity >= 0.65.
        """
        if not crops:
            return []

        print(f"Extracting embeddings for {len(crops)} detected products...")
        embeddings = []
        for crop in crops:
            emb = self._extract_embedding(crop)
            embeddings.append(emb)

        embeddings = np.array(embeddings)

        print(f"Running DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
        labels = clustering.labels_

        brand_ids = []
        noise_counter = 0

        for idx, label in enumerate(labels):
            if label == -1:
                brand_ids.append(f"brand_unique_{noise_counter}")
                noise_counter += 1
            else:
                brand_ids.append(f"brand_{label}")

        return brand_ids

    def process_image(self, image_path_or_pil, output_dir="outputs", eps=0.35, min_samples=2):
        """
        Executes the full pipeline:
        Detection -> Brand Grouping -> Visualization -> JSON response
        """
        if isinstance(image_path_or_pil, (str, Path)):
            img = Image.open(image_path_or_pil).convert("RGB")
        else:
            img = image_path_or_pil.convert("RGB")

        w, h = img.size

        print("Running product detection...")
        results = self.detector.predict(img, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()

        if len(boxes) == 0:
            print("No products detected.")
            return {"objects": [], "visualization_path": ""}

        crops = []
        valid_indices = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))
            crops.append(crop)
            valid_indices.append(idx)

        boxes = boxes[valid_indices]
        confidences = confidences[valid_indices]

        brand_ids = self.group_brands(crops, eps=eps, min_samples=min_samples)

        unique_groups = sorted(list(set(brand_ids)))
        color_palette = [
            "#FF3366", "#33FF66", "#3366FF", "#FFFF33", "#FF33FF", "#33FFFF",
            "#FF9933", "#9933FF", "#33FF99", "#FF3399", "#99FF33", "#3399FF"
        ]
        group_colors = {}
        for idx, g_id in enumerate(unique_groups):
            group_colors[g_id] = color_palette[idx % len(color_palette)]

        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)

        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
        except:
            font = None

        objects_output = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = float(confidences[idx])
            g_id = brand_ids[idx]
            color = group_colors[g_id]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

            label_text = f"{g_id} ({conf:.2f})"
            if font:
                try:
                    text_bbox = draw.textbbox((x1, y1), label_text, font=font)
                    draw.rectangle(text_bbox, fill=color)
                    draw.text((x1, y1), label_text, fill="black", font=font)
                except AttributeError:
                    draw.text((x1, y1 - 10), label_text, fill=color)
            else:
                draw.text((x1, y1 - 10), label_text, fill=color)

            objects_output.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
                "group_id": g_id
            })

        os.makedirs(output_dir, exist_ok=True)
        filename = f"vis_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(output_dir, filename)
        draw_img.save(output_path)
        print(f"Saved visualization output to {output_path}")

        return {
            "objects": objects_output,
            "visualization_path": output_path
        }

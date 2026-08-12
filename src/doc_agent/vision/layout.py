"""Stage 2 — layout detection / segmentation"""
from __future__ import annotations
from ..contracts import Page, Region
try:
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
except ImportError:
    pass

def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text/table/figure/heading regions. IMPLEMENT."""
    # We will use keremberke/yolov8s-doclaynet which has all necessary classes.
    # Custom downloaded to bypass needing it bundled locally.
    model_path = hf_hub_download("keremberke/yolov8s-doclaynet", "best.pt")
    model = YOLO(model_path)
    
    regions = []
    
    for page in pages:
        # Run inference without excessive logging
        results = model(page.image_path, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_idx = int(box.cls[0].item())
                class_name = model.names[cls_idx].lower()
                
                # 1. Filter out the header completely (page numbers) per user request
                # 2. KEEP the footer so OCR can parse it before we filter the English out
                if class_name in ["page-header", "header"]:
                    continue
                    
                # Extract coordinates as (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                # Map YOLO doclaynet class to our 'kind' contract
                kind = "text"
                if "table" in class_name:
                    kind = "table"
                elif class_name in ["picture", "figure", "formula"]:
                    kind = "figure"
                elif class_name in ["section-header", "title", "heading"]:
                    kind = "heading"
                # Else: text, list-item, caption, footnote, page-footer -> defaults to "text"
                
                regions.append(
                    Region(
                        page_id=page.id,
                        bbox=bbox,
                        kind=kind
                    )
                )
                
    return regions

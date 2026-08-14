"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""
from __future__ import annotations
from pathlib import Path
import cv2
import pytesseract

from ..contracts import Region, Chunk
from ..logging_conf import get_logger

logger = get_logger(__name__)


class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained TrOCR/Donut/Tesseract."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg.get("ocr", {}) if cfg else {}
        project_root = Path(__file__).resolve().parents[3]

        # Directory containing the fine-tuned Tesseract traineddata
        self.tessdata_dir = Path(self.cfg.get("models_dir", project_root / "src" / "doc_agent" / "models"))
        self.lang = self.cfg.get("model", "ben_seerah")
        self.psm = int(self.cfg.get("psm", 6))  # PSM 6: Assume a single uniform block of text

        # Validate that the traineddata model file actually exists
        model_file = self.tessdata_dir / f"{self.lang}.traineddata"
        if not model_file.is_file():
            raise FileNotFoundError(
                f"Tesseract traineddata file not found at: {model_file}. "
                "Ensure fine-tuned .traineddata is placed in src/doc_agent/models/ or configured in cfg['ocr']."
            )

        self.image_dirs = [
            project_root / "data" / "raw" / "preprocessed",
            project_root / "data" / "raw" / "images",
        ]

    def _resolve_page_image(self, page_id: str) -> Path:
        """Locate the image corresponding to page_id across preprocessed and raw directories."""
        for dir_path in self.image_dirs:
            if not dir_path.exists():
                continue
            # Check for standard extension first
            exact_path = dir_path / f"{page_id}.jpg"
            if exact_path.is_file():
                return exact_path
            # Check for any extension
            matches = [f for f in dir_path.glob(f"{page_id}.*") if f.is_file()]
            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"Page image for '{page_id}' not found in any of the search directories: "
            f"{[str(d) for d in self.image_dirs]}"
        )

    def transcribe_region(self, region: Region) -> str:
        """Transcribe text from a specific page region crop."""
        img_path = self._resolve_page_image(region.page_id)
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"OpenCV failed to read page image at: {img_path}")

        x1, y1, x2, y2 = region.bbox
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            logger.warning(f"Empty crop for region {region.bbox} on page {region.page_id}")
            return ""

        # Run OCR with explicit tessdata path, language, and PSM mode
        custom_config = f'--tessdata-dir "{self.tessdata_dir}" -l {self.lang} --psm {self.psm}'
        text = pytesseract.image_to_string(cropped, config=custom_config)
        return text.strip()


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> text chunks. (calls Reader)."""
    reader = Reader(cfg)
    chunks: list[Chunk] = []

    project_root = Path(__file__).resolve().parents[3]
    pdf_files = list((project_root / "data" / "raw").glob("*.pdf"))
    default_doc_id = pdf_files[0].stem if pdf_files else "unknown_doc"

    for i, region in enumerate(regions):
        text = reader.transcribe_region(region)
        if text:
            chunks.append(
                Chunk(
                    id=f"{region.page_id}_chunk_{i}",
                    doc_id=default_doc_id,
                    text=text,
                    page_ids=[region.page_id],
                )
            )

    return chunks



"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import pytesseract

from ..contracts import Region, Chunk
from ..logging_conf import get_logger

logger = get_logger(__name__)


class Reader:
    """Tesseract OCR reader using a fine-tuned .traineddata model."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg.get("ocr", {}) if cfg else {}

        project_root = Path(__file__).resolve().parents[3]

        # Directory containing the fine-tuned Tesseract .traineddata file.
        self.tessdata_dir = Path(
            self.cfg.get(
                "models_dir",
                project_root / "src" / "doc_agent" / "models",
            )
        ).resolve()

        # Tesseract language/model name.
        self.lang = self.cfg.get("model", "ben_seerah")

        # PSM 6 = Assume a single uniform block of text.
        self.psm = int(self.cfg.get("psm", 6))

        # Verify that the requested traineddata file exists.
        model_file = self.tessdata_dir / f"{self.lang}.traineddata"

        if not model_file.is_file():
            raise FileNotFoundError(
                f"Tesseract traineddata file not found at: {model_file}. "
                "Ensure the fine-tuned .traineddata file is placed in "
                "src/doc_agent/models/ or configure cfg['ocr']['models_dir']."
            )

        # Tell Tesseract where the custom traineddata files live.
        #
        # This is important because the custom model is NOT installed in
        # Tesseract's system tessdata directory.
        os.environ["TESSDATA_PREFIX"] = str(self.tessdata_dir)

        self.image_dirs = [
            project_root / "data" / "raw" / "preprocessed",
            project_root / "data" / "raw" / "images",
        ]

        logger.info(
            "Initialized Tesseract OCR: model=%s, tessdata_dir=%s, psm=%d",
            self.lang,
            self.tessdata_dir,
            self.psm,
        )

    def _resolve_page_image(self, page_id: str) -> Path:
        """Locate the image corresponding to page_id."""

        for dir_path in self.image_dirs:
            if not dir_path.exists():
                continue

            # Prefer standard .jpg filename.
            exact_path = dir_path / f"{page_id}.jpg"

            if exact_path.is_file():
                return exact_path

            # Otherwise accept any extension.
            matches = [
                path
                for path in dir_path.glob(f"{page_id}.*")
                if path.is_file()
            ]

            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"Page image for '{page_id}' not found in any of the "
            f"search directories: {[str(d) for d in self.image_dirs]}"
        )

    def transcribe_region(self, region: Region) -> str:
        """Transcribe text from a specific page region."""

        img_path = self._resolve_page_image(region.page_id)

        img = cv2.imread(str(img_path))

        if img is None:
            raise RuntimeError(
                f"OpenCV failed to read page image at: {img_path}"
            )

        x1, y1, x2, y2 = region.bbox

        height, width = img.shape[:2]

        # Clamp bounding box to image boundaries.
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        cropped = img[y1:y2, x1:x2]

        if cropped.size == 0:
            logger.warning(
                "Empty crop for region %s on page %s",
                region.bbox,
                region.page_id,
            )
            return ""

        # Run Tesseract.
        #
        # TESSDATA_PREFIX above tells Tesseract where ben_seerah.traineddata
        # lives. Do NOT pass --tessdata-dir here because the project path
        # contains spaces on Windows.
        text = pytesseract.image_to_string(
            cropped,
            lang=self.lang,
            config=f"--psm {self.psm}",
        )

        return text.strip()


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Transcribe OCR regions into text chunks."""

    reader = Reader(cfg)
    chunks: list[Chunk] = []

    project_root = Path(__file__).resolve().parents[3]

    # Determine document ID from the first raw PDF, if available.
    pdf_files = list(
        (project_root / "data" / "raw").glob("*.pdf")
    )

    default_doc_id = (
        pdf_files[0].stem
        if pdf_files
        else "unknown_doc"
    )

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

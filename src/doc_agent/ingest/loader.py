"""Stage 1 — load scanned page-images"""
from __future__ import annotations
import pymupdf as fitz
from pathlib import Path

try:
    from ..contracts import Page  # noqa: F401
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from doc_agent.contracts import Page

def load_pages(cfg: dict) -> list[Page]:
    """Read data/raw/ -> list[Page]. IMPLEMENT."""
    ingest_dir = Path(__file__).parent
    pdf_files = list(ingest_dir.glob("*.pdf"))
    if not pdf_files:
        # fallback to data/raw just in case
        pdf_files = list(Path("data/raw").glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("No PDF found in ingest folder or data/raw")
            
    pdf_path = pdf_files[0]
    out_dir = Path("data/raw/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        # We can use dpi=150 to get a good quality rasterized image of the page
        pix = page.get_pixmap(dpi=150)
        image_path = out_dir / f"page_{i:04d}.jpg"
        
        pix.save(str(image_path))
            
        pages.append(
            Page(
                id=f"page_{i:04d}",
                image_path=str(image_path),
                doc_id=pdf_path.stem
            )
        )
            
    return pages


if __name__ == "__main__":
    print("Testing load_pages locally...")
    out_pages = load_pages({})
    print(f"Extracted {len(out_pages)} pages. Here are the first 5:")
    for p in out_pages[:5]:
        print(p)

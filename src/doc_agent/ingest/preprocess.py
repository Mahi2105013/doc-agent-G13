"""Stage 1 — deskew / denoise / binarize / augment"""
from __future__ import annotations
from pathlib import Path

try:
    from ..contracts import Page  # noqa: F401
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from doc_agent.contracts import Page

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "OpenCV is required for Stage 1 preprocessing. "
        "Install it with: pip install opencv-python"
    ) from e
import subprocess   


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Classical preprocessing. IMPLEMENT."""
    # Ensure all operations run relative to the project root, not the current working directory
    project_root = Path(__file__).resolve().parents[3]
    out_dir = project_root / "data" / "raw" / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # We will temporarily store cv2 outputs before passing to imagemagick
    temp_img_path = str(out_dir / "temp_denoised.jpg")
    
    processed_pages = []
    
    for page in pages:
        # Extract the page index from the id e.g. "page_0017"
        try:
            page_idx = int(page.id.replace("page_", ""))
        except ValueError:
            continue
            
        if 16 <= page_idx <= 555:
            # 1. Read the image
            img = cv2.imread(page.image_path)
            if img is None:
                continue
                
            # 2. Convert to Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Maintain the original image name
            image_name = Path(page.image_path).name
            final_out_path = out_dir / image_name
            
            # Save the grayscale image
            cv2.imwrite(str(final_out_path), gray)
            
            # 3. Construct updated Page object
            processed_pages.append(
                Page(
                    id=page.id,
                    image_path=str(final_out_path),
                    doc_id=page.doc_id
                )
            )
            
    return processed_pages

if __name__ == "__main__":
    from doc_agent.ingest.loader import load_pages
    print("Loading raw pages...")
    raw_pages = load_pages({})
    print(f"Loaded {len(raw_pages)} pages. Running preprocessing on the first 20 pages to test logic on pages 17-19...")
    final_pages = run(raw_pages[:20], {})
    print(f"Preprocessing returned {len(final_pages)} processed pages (should be 3 pages: 17, 18, 19).")

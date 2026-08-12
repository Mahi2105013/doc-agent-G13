# test_layout.py (in the doc-agent-G13 directory)
from src.doc_agent.vision.layout import detect
from src.doc_agent.contracts import Page

p = Page(id="test", image_path="/mnt/Inter OS/Academics/4-1/429/doc-agent-G13/test_img.jpg", doc_id="doc1")
regions = detect([p], {})
print("Detected regions:", len(regions))

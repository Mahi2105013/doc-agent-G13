"""Stage 4 — chunk text"""
from __future__ import annotations
from ..contracts import Chunk


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Re-chunk text using recursive character boundaries adhering to contracts.Chunk."""
    index_cfg = cfg.get("index", {}) if cfg else {}
    chunk_size = index_cfg.get("chunk_tokens") or index_cfg.get("chunk_size", 500)
    chunk_overlap = index_cfg.get("overlap") or index_cfg.get("chunk_overlap", 50)
    separators = ["\n\n", "\n", " ", ""]

    def _split_text(text: str, max_len: int, overlap: int) -> list[str]:
        if len(text) <= max_len:
            return [text] if text.strip() else []

        # Find best separator
        chosen_sep = separators[-1]
        for sep in separators:
            if sep in text:
                chosen_sep = sep
                break

        splits = text.split(chosen_sep) if chosen_sep else list(text)
        docs = []
        current_doc: list[str] = []
        current_len = 0

        for split_part in splits:
            part_len = len(split_part) + (len(chosen_sep) if current_doc else 0)
            if current_len + part_len > max_len and current_doc:
                doc_str = chosen_sep.join(current_doc)
                if doc_str.strip():
                    docs.append(doc_str)
                # Apply overlap
                while current_doc and current_len > overlap:
                    popped = current_doc.pop(0)
                    current_len -= len(popped) + len(chosen_sep)
            current_doc.append(split_part)
            current_len += part_len

        if current_doc:
            doc_str = chosen_sep.join(current_doc)
            if doc_str.strip():
                docs.append(doc_str)

        return docs

    new_chunks: list[Chunk] = []
    for chunk in chunks:
        text = chunk.text if hasattr(chunk, "text") and chunk.text else ""
        if not text.strip():
            continue

        fragments = _split_text(text, chunk_size, chunk_overlap)
        parent_id = chunk.id if hasattr(chunk, "id") and chunk.id else "chunk"
        doc_id = chunk.doc_id if hasattr(chunk, "doc_id") and chunk.doc_id else "unknown_doc"
        page_ids = list(chunk.page_ids) if hasattr(chunk, "page_ids") and chunk.page_ids else []

        for j, frag in enumerate(fragments):
            new_chunks.append(
                Chunk(
                    id=f"{parent_id}_c{j}" if len(fragments) > 1 else parent_id,
                    doc_id=doc_id,
                    text=frag,
                    page_ids=page_ids,
                    score=getattr(chunk, "score", 0.0),
                )
            )

    return new_chunks if new_chunks else chunks
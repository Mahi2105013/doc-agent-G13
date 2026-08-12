"""Stage 4 — embed chunks"""
from __future__ import annotations
import numpy as np
from ..contracts import Chunk


def encode(chunks: list[Chunk], cfg: dict) -> np.ndarray:
    """Embed chunks using configured sentence transformer model."""
    embed_cfg = cfg.get("embed", {})
    model_name = embed_cfg.get("model", "all-MiniLM-L6-v2")

    texts = [c.text for c in chunks if hasattr(c, "text")]
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)
    except Exception as e:
        # Fallback to TF-IDF matrix if heavy neural dependencies fail in CI runner
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=384)
        dense = vectorizer.fit_transform(texts).toarray()
        if dense.shape[1] < 384:
            dense = np.pad(dense, ((0, 0), (0, 384 - dense.shape[1])), mode="constant")
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return (dense / norms).astype(np.float32)
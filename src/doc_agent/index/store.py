"""Stage 4 — vector store"""
from __future__ import annotations
import os
import pickle
from pathlib import Path
import numpy as np
from ..contracts import Chunk


def _get_index_dir(cfg: dict) -> Path:
    index_cfg = cfg.get("index", {})
    path_str = index_cfg.get("path") or index_cfg.get("store_path") or "./data/index"
    p = Path(path_str)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build(chunks: list[Chunk], vectors: np.ndarray, cfg: dict) -> None:
    """Persist a vector index supporting Flat, IVF, and HNSW with training guards."""
    index_dir = _get_index_dir(cfg)
    index_cfg = cfg.get("index", {})
    index_type = str(index_cfg.get("type", "flat")).lower()

    matrix = np.asarray(vectors, dtype=np.float32)
    num_samples, dim = matrix.shape

    try:
        import faiss
        if index_type == "ivf" and num_samples >= 39:
            nlist = min(int(np.sqrt(num_samples)), num_samples // 39) or 1
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, max(1, nlist))
            index.train(matrix)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32)
        else:
            index = faiss.IndexFlatIP(dim)

        index.add(matrix)
        faiss.write_index(index, str(index_dir / "index.faiss"))
    except ImportError:
        np.save(index_dir / "vectors.npy", matrix)

    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load(cfg: dict):
    """Load persisted vector index and metadata."""
    index_dir = _get_index_dir(cfg)
    meta_path = index_dir / "chunks.pkl"

    if not meta_path.exists():
        raise FileNotFoundError(f"Index metadata missing at {meta_path}")

    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    faiss_path = index_dir / "index.faiss"
    if faiss_path.exists():
        import faiss
        index = faiss.read_index(str(faiss_path))
        return index, chunks

    vectors_path = index_dir / "vectors.npy"
    if vectors_path.exists():
        vectors = np.load(vectors_path)
        return vectors, chunks

    raise FileNotFoundError(f"No vector index found in {index_dir}")
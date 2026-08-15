"""Stage 4 — embed chunks"""
from __future__ import annotations
import numpy as np
import torch
from ..contracts import Chunk


def encode(chunks: list[Chunk], cfg: dict) -> np.ndarray:
    """Embed chunks using configured sentence transformer / transformers model with fallback."""
    embed_cfg = cfg.get("embed", {}) if cfg else {}
    model_name = embed_cfg.get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    dim = int(embed_cfg.get("dim", 384))
    requested_device = cfg.get("device", "cpu") if cfg else "cpu"

    # Never request CUDA if this PyTorch installation doesn't support it.
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "CUDA requested by config, but CUDA is unavailable. "
            "Falling back to CPU."
        )
        device = "cpu"
    else:
        device = requested_device

    texts = [c.text for c in chunks if hasattr(c, "text") and c.text]
    if not texts:
        return np.empty((0, dim), dtype=np.float32)

    # 1. Try sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Could not load required embedding model {model_name}.") from e

    # 2. Try Hugging Face transformers directly
    try:
        # import torch
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            # Mean pooling
            mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            pooled = (sum_embeddings / sum_mask).cpu().numpy()

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return (pooled / norms).astype(np.float32)
    except Exception:
        pass

    # 3. Deterministic n-gram hash embedding fallback (ensures offline/CI works with exact dim)
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        words = text.lower().split()
        for w in words:
            h = hash(w) % dim
            vectors[i, h] += 1.0
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] /= norm
        else:
            vectors[i, 0] = 1.0

    return vectors

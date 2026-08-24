"""Stage 5 — dense retrieval"""
from __future__ import annotations
import numpy as np
from ..contracts import Chunk  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)


class Retriever:
    """Dense retriever that loads the persisted FAISS / numpy index and embeds queries
    with the same model used at index-build time."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._index = None       # faiss index OR np.ndarray
        self._chunks: list[Chunk] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy-load the persisted index exactly once.
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from ..index.store import load as store_load
        self._index, self._chunks = store_load(self.cfg)
        self._loaded = True
        logger.info(
            "Retriever: loaded index with %d chunks (type=%s)",
            len(self._chunks),
            type(self._index).__name__,
        )

    # ------------------------------------------------------------------
    # Embed a single query string using the same model as index-build.
    # ------------------------------------------------------------------
    def _embed_query(self, query: str) -> np.ndarray:
        from ..index.embed import encode
        from ..contracts import Chunk as _Chunk

        q_chunk = _Chunk(id="q", doc_id="q", text=query, page_ids=[])
        vec = encode([q_chunk], self.cfg)   # shape (1, dim)
        return vec[0]                        # shape (dim,)

    # ------------------------------------------------------------------
    # Public: top-k dense retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Return the top-k most relevant chunks for *query*.

        Sets ``chunk.score`` (cosine similarity in [-1, 1], higher = better)
        on every returned chunk so that ``decide()`` can judge evidence strength
        via ``is_weak()`` / ``top_score()``.
        """
        self._ensure_loaded()

        retrieve_cfg = self.cfg.get("retrieve", {})
        if k is None:
            k = int(retrieve_cfg.get("k", 10))
        k = max(1, k)

        if not self._chunks:
            logger.warning("Retriever: index is empty – returning no results.")
            return []

        query_vec = self._embed_query(query)                # (dim,)

        # ----------------------------------------------------------------
        # FAISS index (IndexFlatIP, IndexHNSWFlat, IndexIVFFlat …)
        # ----------------------------------------------------------------
        if not isinstance(self._index, np.ndarray):
            try:
                import faiss  # type: ignore
                q = np.expand_dims(query_vec, 0).astype(np.float32)
                actual_k = min(k, len(self._chunks))
                distances, indices = self._index.search(q, actual_k)  # (1, k)
                scores  = distances[0]
                indices = indices[0]
            except Exception as exc:
                logger.error("FAISS search failed (%s); falling back to numpy.", exc)
                indices, scores = self._numpy_search(query_vec, k)
        # ----------------------------------------------------------------
        # Plain numpy matrix  (shape: n_chunks × dim)
        # ----------------------------------------------------------------
        else:
            indices, scores = self._numpy_search(query_vec, k)

        results: list[Chunk] = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx].model_copy(update={"score": float(score)})
            results.append(chunk)

        logger.info(
            "Retriever: query=%r, k=%d, top_score=%.4f, n_results=%d",
            query[:60],
            k,
            results[0].score if results else 0.0,
            len(results),
        )
        return results

    def _numpy_search(
        self, query_vec: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cosine-similarity search against the stored numpy matrix."""
        matrix = np.asarray(self._index, dtype=np.float32)   # (n, dim)
        # L2-normalise query (vectors in store are already normalised)
        norm = np.linalg.norm(query_vec)
        q = query_vec / (norm + 1e-10)
        sims = matrix @ q                                      # (n,)
        actual_k = min(k, len(sims))
        top_idx = np.argsort(sims)[::-1][:actual_k]
        return top_idx, sims[top_idx]


# ---------------------------------------------------------------------------
# Evidence-strength policy: read by agent.decide() for evidence-gated re-search
# ---------------------------------------------------------------------------

def top_score(chunks: list[Chunk]) -> float:
    """Strength of the current evidence = best chunk score (0.0 if empty)."""
    return max((c.score for c in chunks), default=0.0)


def is_weak(chunks: list[Chunk], cfg: dict) -> bool:
    """Weak evidence = best score below cfg['retrieve']['weak_threshold']."""
    return top_score(chunks) < cfg["retrieve"]["weak_threshold"]


def next_k(k: int, cfg: dict) -> int | None:
    """Widen the net: k + k_step, or None once it would exceed k_max (→ ABSTAIN)."""
    nk = k + cfg["retrieve"]["k_step"]
    return nk if nk <= cfg["retrieve"]["k_max"] else None


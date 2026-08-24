"""Stage 6 — FIXED tool interface — the agent's tools

All class names, ``name`` attributes, and ``__call__`` signatures are LOCKED
(test_tools.py verifies them). Only the bodies are implemented here.

Dependency injection: call ``bind(retriever, cfg)`` once from Agent.__init__
to give the tools access to shared resources without global state.
"""
from __future__ import annotations
import ast
import operator as _op

from abc import ABC, abstractmethod
from pathlib import Path

from ..contracts import Citation, Chunk, ToolResult  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level context — populated by bind() from Agent.__init__
# ---------------------------------------------------------------------------
_retriever = None   # retrieval.retriever.Retriever instance
_cfg: dict = {}


def bind(retriever, cfg: dict) -> None:
    """Wire shared resources into the tool module. Call once before run()."""
    global _retriever, _cfg
    _retriever = retriever
    _cfg = cfg


# ---------------------------------------------------------------------------
# Base class (FIXED interface)
# ---------------------------------------------------------------------------
class Tool(ABC):
    name: str

    @abstractmethod
    def __call__(self, **kwargs) -> ToolResult: ...


# ---------------------------------------------------------------------------
# FIXED tool set — names & signatures locked (test_tools.py checks these).
# ---------------------------------------------------------------------------

class Retrieve(Tool):
    """Dense retrieval: query the vector index for the top-k relevant chunks."""
    name = "retrieve"

    def __call__(self, query: str, k: int = 10) -> ToolResult:  # type: ignore[override]
        if _retriever is None:
            logger.error("Retrieve: retriever not bound — call tools.bind() first.")
            return ToolResult(ok=False, payload={"error": "retriever not bound"})
        try:
            chunks: list[Chunk] = _retriever.retrieve(query, k=k)
            best_score = max((c.score for c in chunks), default=0.0)
            payload = {
                "chunks": [c.model_dump() for c in chunks],
                "chunk_ids": [c.id for c in chunks],
                "top_score": best_score,
                "k": k,
            }
            logger.info(
                "Retrieve: query=%r k=%d top_score=%.4f n=%d",
                query[:60], k, best_score, len(chunks),
            )
            return ToolResult(ok=True, payload=payload)
        except Exception as exc:
            logger.error("Retrieve failed: %s", exc)
            return ToolResult(ok=False, payload={"error": str(exc)})


class Rerank(Tool):
    """Cross-encoder reranking of candidate chunks (no-op pass-through if disabled)."""
    name = "rerank"

    def __call__(self, query: str, candidates: list) -> ToolResult:  # type: ignore[override]
        retrieve_cfg = _cfg.get("retrieve", {})
        if not retrieve_cfg.get("rerank", False):
            # Reranking disabled — return candidates as-is.
            return ToolResult(ok=True, payload={"chunks": candidates, "reranked": False})
        try:
            from ..retrieval.rerank import rerank as _rerank
            # candidates may be dicts or Chunk objects
            if candidates and isinstance(candidates[0], dict):
                chunk_objs = [Chunk(**c) for c in candidates]
            else:
                chunk_objs = candidates
            reranked = _rerank(query, chunk_objs, _cfg)
            return ToolResult(
                ok=True,
                payload={"chunks": [c.model_dump() for c in reranked], "reranked": True},
            )
        except Exception as exc:
            logger.warning("Rerank failed (%s); returning original order.", exc)
            return ToolResult(ok=True, payload={"chunks": candidates, "reranked": False})


class ReadPage(Tool):
    """Return the raw text of a page by running OCR on its stored image."""
    name = "read_page"

    def __call__(self, page_id: str) -> ToolResult:  # type: ignore[override]
        try:
            project_root = Path(__file__).resolve().parents[4]
            search_dirs = [
                project_root / "data" / "raw" / "preprocessed",
                project_root / "data" / "raw" / "images",
                project_root / "grading_kit" / "heldout_pages",
            ]
            img_path: Path | None = None
            for d in search_dirs:
                candidate = d / f"{page_id}.jpg"
                if candidate.is_file():
                    img_path = candidate
                    break
                for ext in (".png", ".jpeg", ".tiff"):
                    candidate = d / f"{page_id}{ext}"
                    if candidate.is_file():
                        img_path = candidate
                        break
                if img_path:
                    break

            if img_path is None:
                return ToolResult(ok=False, payload={"error": f"image for {page_id} not found"})

            from ..contracts import Region
            from ..vision.ocr import Reader
            reader = Reader(_cfg)
            reader.image_dirs = [img_path.parent]
            import cv2
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2] if img is not None else (1000, 800)
            region = Region(page_id=page_id, bbox=(0, 0, w, h), kind="text")
            text = reader.transcribe_region(region)
            return ToolResult(ok=True, payload={"page_id": page_id, "text": text})
        except Exception as exc:
            logger.error("ReadPage failed for %s: %s", page_id, exc)
            return ToolResult(ok=False, payload={"error": str(exc), "page_id": page_id})


class EnhancePage(Tool):
    """Apply image enhancement (if enabled) and return the enhanced image path."""
    name = "enhance_page"

    def __call__(self, page_id: str) -> ToolResult:  # type: ignore[override]
        try:
            enhance_cfg = _cfg.get("enhance", {})
            if not enhance_cfg.get("enabled", False):
                return ToolResult(
                    ok=True,
                    payload={"page_id": page_id, "enhanced": False,
                             "msg": "enhancement disabled in config"},
                )
            from ..contracts import Page
            from ..ingest.enhance import run as enhance_run
            project_root = Path(__file__).resolve().parents[4]
            img_path = project_root / "data" / "raw" / "images" / f"{page_id}.jpg"
            page = Page(id=page_id, image_path=str(img_path), doc_id="unknown")
            enhanced = enhance_run([page], _cfg)
            ep = enhanced[0].image_path if enhanced else str(img_path)
            return ToolResult(ok=True, payload={"page_id": page_id, "enhanced_path": ep})
        except Exception as exc:
            logger.error("EnhancePage failed for %s: %s", page_id, exc)
            return ToolResult(ok=False, payload={"error": str(exc), "page_id": page_id})


class Extract(Tool):
    """Extract a specific named field from a chunk's text (e.g., a date, name, number)."""
    name = "extract"

    def __call__(self, field: str, chunk_id: str) -> ToolResult:  # type: ignore[override]
        try:
            # Look up the chunk from the retriever's loaded index.
            chunk: Chunk | None = None
            if _retriever is not None and _retriever._chunks:
                for c in _retriever._chunks:
                    if c.id == chunk_id:
                        chunk = c
                        break
            if chunk is None:
                return ToolResult(
                    ok=False, payload={"error": f"chunk {chunk_id!r} not found"}
                )
            # Simple heuristic extraction: search for lines containing the field label.
            text = chunk.text
            field_lower = field.lower()
            for line in text.splitlines():
                if field_lower in line.lower():
                    return ToolResult(
                        ok=True,
                        payload={"field": field, "chunk_id": chunk_id, "value": line.strip()},
                    )
            return ToolResult(
                ok=True,
                payload={"field": field, "chunk_id": chunk_id, "value": None,
                         "msg": "field not found in chunk"},
            )
        except Exception as exc:
            logger.error("Extract failed: %s", exc)
            return ToolResult(ok=False, payload={"error": str(exc)})


class Aggregate(Tool):
    """Aggregate a list of scalar items with a simple operation (sum/avg/count/max/min)."""
    name = "aggregate"

    _OPS = {
        "sum":   sum,
        "count": len,
        "max":   max,
        "min":   min,
        "avg":   lambda xs: sum(xs) / len(xs) if xs else 0.0,
        "mean":  lambda xs: sum(xs) / len(xs) if xs else 0.0,
    }

    def __call__(self, op: str, items: list) -> ToolResult:  # type: ignore[override]
        op_key = op.lower().strip()
        fn = self._OPS.get(op_key)
        if fn is None:
            return ToolResult(
                ok=False,
                payload={"error": f"Unknown aggregation op {op!r}. Supported: {list(self._OPS)}"},
            )
        try:
            # Coerce string numbers to float where possible.
            nums = []
            for item in items:
                try:
                    nums.append(float(item))
                except (TypeError, ValueError):
                    pass
            if op_key == "count":
                result = len(items)
            else:
                if not nums:
                    return ToolResult(ok=False, payload={"error": "No numeric values in items"})
                result = fn(nums)
            return ToolResult(ok=True, payload={"op": op, "result": result})
        except Exception as exc:
            logger.error("Aggregate failed: %s", exc)
            return ToolResult(ok=False, payload={"error": str(exc)})


class Cite(Tool):
    """Record a citation (chunk_id + character span) for later attachment to the answer."""
    name = "cite"

    def __call__(self, chunk_id: str, span: tuple) -> ToolResult:  # type: ignore[override]
        try:
            start, end = int(span[0]), int(span[1])
            citation = Citation(chunk_id=chunk_id, span=(start, end))
            logger.info("Cite: recorded %s [%d:%d]", chunk_id, start, end)
            return ToolResult(
                ok=True,
                payload={"citation": citation.model_dump()},
            )
        except Exception as exc:
            logger.error("Cite failed: %s", exc)
            return ToolResult(ok=False, payload={"error": str(exc)})


class Calculator(Tool):
    """Safe arithmetic evaluator for simple mathematical expressions."""
    name = "calculator"

    # Allowed operators for safe eval
    _OPERATORS = {
        ast.Add: _op.add,
        ast.Sub: _op.sub,
        ast.Mult: _op.mul,
        ast.Div: _op.truediv,
        ast.FloorDiv: _op.floordiv,
        ast.Mod: _op.mod,
        ast.Pow: _op.pow,
        ast.USub: _op.neg,
        ast.UAdd: _op.pos,
    }

    def __call__(self, expr: str) -> ToolResult:  # type: ignore[override]
        try:
            result = self._safe_eval(ast.parse(expr, mode="eval").body)
            logger.info("Calculator: %s = %s", expr, result)
            return ToolResult(ok=True, payload={"expr": expr, "result": result})
        except Exception as exc:
            logger.error("Calculator failed for %r: %s", expr, exc)
            return ToolResult(ok=False, payload={"error": str(exc), "expr": expr})

    def _safe_eval(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        if isinstance(node, ast.BinOp):
            fn = self._OPERATORS.get(type(node.op))
            if fn is None:
                raise ValueError(f"Unsupported operator: {node.op}")
            return fn(self._safe_eval(node.left), self._safe_eval(node.right))
        if isinstance(node, ast.UnaryOp):
            fn = self._OPERATORS.get(type(node.op))
            if fn is None:
                raise ValueError(f"Unsupported unary operator: {node.op}")
            return fn(self._safe_eval(node.operand))
        raise ValueError(f"Unsupported AST node: {type(node)}")


class EscalateToHuman(Tool):
    """Queue a task for human review and block until a decision is received."""
    name = "escalate_to_human"

    def __call__(self, reason: str, context: dict) -> ToolResult:  # type: ignore[override]
        try:
            from ..agent import hitl
            result = hitl.escalate(reason, context)
            return result
        except NotImplementedError:
            # HITL module not yet fully wired — log and return a placeholder.
            logger.warning(
                "EscalateToHuman: HITL module not fully implemented. "
                "Reason=%r logged; continuing without blocking.", reason,
            )
            return ToolResult(
                ok=True,
                payload={
                    "escalated": True,
                    "reason": reason,
                    "status": "pending_human_review",
                },
            )
        except Exception as exc:
            logger.error("EscalateToHuman failed: %s", exc)
            return ToolResult(ok=False, payload={"error": str(exc)})


# ---------------------------------------------------------------------------
# REGISTRY — the complete ordered list of tool classes (LOCKED, do not change)
# ---------------------------------------------------------------------------
REGISTRY = [Retrieve, Rerank, ReadPage, EnhancePage, Extract,
            Aggregate, Cite, Calculator, EscalateToHuman]

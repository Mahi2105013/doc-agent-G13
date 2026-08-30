"""Stage 8 — FastAPI service"""
from __future__ import annotations
from ..contracts import *  # noqa

from fastapi import FastAPI
from .. import config, pipeline

app = FastAPI(title="doc-agent")
_cfg = config.load()

@app.post("/answer")
def answer(q: str) -> dict:
    """Return grounded, cited answer."""
    try:
        ans = pipeline.answer(q, _cfg)
        return {
            "text": ans.text,
            "grounded": ans.grounded,
            "citations": [
                {"chunk_id": c.chunk_id, "span": c.span} 
                for c in ans.citations
            ] if ans.citations else [],
            "confidence": ans.confidence
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "text": "Sorry, an error occurred while processing your request.",
            "grounded": False,
            "citations": [],
            "confidence": 0.0
        }

@app.get("/health")
def health() -> dict:
    return {"ok": True}


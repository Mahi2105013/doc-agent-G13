#!/usr/bin/env bash
# A2 — build the vector index
# Runs the full ingest pipeline:
#   pages → preprocess → (enhance) → layout → OCR → chunk → embed → store
# Both ingest and index stages are wired into pipeline.build_knowledge_base(),
# so a single `make ingest` call covers everything.
set -euo pipefail

# ---------------------------------------------------------------------------
# Locate project root (the directory containing this script's parent folder)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Project root: ${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if ! command -v python &>/dev/null; then
    echo "ERROR: python not found in PATH." >&2
    exit 1
fi

CONFIG="${PROJECT_ROOT}/configs/config.yaml"
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config not found at ${CONFIG}" >&2
    exit 1
fi

# Check that at least one PDF exists for the loader
PDF_COUNT=$(find "${PROJECT_ROOT}/src/doc_agent/ingest" "${PROJECT_ROOT}/data/raw" -maxdepth 1 -name "*.pdf" 2>/dev/null | wc -l)
if [[ "${PDF_COUNT}" -eq 0 ]]; then
    echo "WARNING: No PDF found in src/doc_agent/ingest/ or data/raw/. The loader will fail." >&2
fi

# ---------------------------------------------------------------------------
# Ensure src/ is on PYTHONPATH so `from doc_agent...` imports resolve
# ---------------------------------------------------------------------------
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Run the pipeline (ingest + embed + store in one call)
# ---------------------------------------------------------------------------
echo "==> Starting ingest + index build..."
cd "${PROJECT_ROOT}"
python scripts/run_ingest.py

echo ""
echo "==> Index build complete."
echo "    Artifacts written to: ${PROJECT_ROOT}/data/index/"

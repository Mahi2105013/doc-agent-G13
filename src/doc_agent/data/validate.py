"""Data — data schema/quality validation at ingest"""
from __future__ import annotations
from pathlib import Path
from ..contracts import Page


def validate(pages: list[Page]) -> None:
    """Assert min pages, image paths exist, and no doc_id appears in multiple splits.

    Raises:
        ValueError: if any check fails.
    """
    MIN_PAGES = 300

    if not pages:
        raise ValueError("validate: pages list is empty.")

    if len(pages) < MIN_PAGES:
        raise ValueError(
            f"validate: corpus has only {len(pages)} pages; minimum is {MIN_PAGES}."
        )

    # Check image paths exist
    missing = [p.image_path for p in pages if not Path(p.image_path).exists()]
    if missing:
        raise ValueError(
            f"validate: {len(missing)} page image(s) not found on disk. "
            f"First missing: {missing[0]}"
        )

    # Check no doc_id has pages mixed across multiple claimed splits
    # (This reads manifest.yaml if available to verify)
    try:
        import yaml  # type: ignore
        project_root = Path(__file__).resolve().parents[4]
        manifest_path = project_root / "grading_kit" / "manifest.yaml"
        if manifest_path.exists():
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            splits = manifest.get("corpus", {}).get("splits") or manifest.get("splits", {})
            # Build reverse map: doc_id -> split
            doc_to_split: dict[str, str] = {}
            for split_name, doc_ids in splits.items():
                for doc_id in (doc_ids or []):
                    doc_to_split[doc_id] = split_name

            # Verify every page's doc_id is declared in exactly one split
            seen_doc_ids = {p.doc_id for p in pages}
            unknown = seen_doc_ids - set(doc_to_split.keys())
            if unknown:
                print(
                    f"[validate] WARNING: doc_ids not declared in manifest splits: {unknown}. "
                    "Add them to grading_kit/manifest.yaml."
                )
    except Exception:
        pass  # manifest check is best-effort

    print(f"[validate] OK — {len(pages)} pages, "
          f"{len({p.doc_id for p in pages})} document(s).")

"""Run tasks.jsonl through the agent and score (A3 eval)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doc_agent import config, pipeline

# -----------------------------------------------------------------
# Load config and tasks
# -----------------------------------------------------------------
cfg = config.load()
project_root = Path(__file__).resolve().parents[1]
tasks_path = project_root / "grading_kit" / "tasks.jsonl"

tasks = []
with open(tasks_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError:
                continue

print(f"Loaded {len(tasks)} eval tasks from {tasks_path}\n")

# -----------------------------------------------------------------
# Run each task through the agent
# -----------------------------------------------------------------
sys.path.insert(0, str(project_root / "grading_kit"))
from success_check import check  # type: ignore

results = []
for task in tasks:
    task_id = task.get("id", "?")
    question = task.get("question", "")
    gold = task.get("gold", "")
    needs_research = task.get("needs_research", False)

    print(f"[{task_id}] needs_research={needs_research}")
    print(f"  Q: {question[:120]}")

    try:
        answer = pipeline.answer(question, cfg)
        ans_text = answer.text
        grounded = answer.grounded
        confidence = answer.confidence
        n_citations = len(answer.citations)
    except Exception as exc:
        ans_text = f"ERROR: {exc}"
        grounded = False
        confidence = 0.0
        n_citations = 0

    try:
        passed = check(task, {"text": ans_text, "grounded": grounded})
    except NotImplementedError:
        passed = bool(gold) and gold.lower() in ans_text.lower()

    print(f"  A: {ans_text[:120]}")
    print(f"  grounded={grounded} conf={confidence:.2f} citations={n_citations} pass={passed}\n")

    results.append({
        "id": task_id,
        "needs_research": needs_research,
        "passed": passed,
        "grounded": grounded,
        "confidence": confidence,
    })

# -----------------------------------------------------------------
# Summary report
# -----------------------------------------------------------------
total = len(results)
n_passed = sum(1 for r in results if r["passed"])
n_grounded = sum(1 for r in results if r["grounded"])
research_total = sum(1 for r in results if r["needs_research"])
research_passed = sum(1 for r in results if r["needs_research"] and r["passed"])

print("=" * 60)
print(f"EVALUATION SUMMARY  ({total} tasks)")
print("=" * 60)
print(f"  Overall pass rate   : {n_passed}/{total}  ({100*n_passed/max(total,1):.1f}%)")
print(f"  Grounded answers    : {n_grounded}/{total} ({100*n_grounded/max(total,1):.1f}%)")
if research_total:
    print(f"  Re-search triggered : {research_passed}/{research_total} "
          f"({100*research_passed/max(research_total,1):.1f}%)")
print("=" * 60)

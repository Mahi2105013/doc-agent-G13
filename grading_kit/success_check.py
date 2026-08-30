"""Per-task verifier. FIXED signature."""
from __future__ import annotations


def check(task: dict, answer: dict) -> bool:
    """Return True if `answer` satisfies `task`.

    Rules:
    - verifiable tasks (task['verifiable']=True):  gold substring must appear in answer text.
    - judged tasks (task['judged']=True):           require answer to be grounded (no LLM judge yet).
    - all tasks:                                    abstained answers (grounded=False, empty text) fail.
    """
    ans_text: str = str(answer.get("text", "")).strip()
    grounded: bool = bool(answer.get("grounded", False))

    # Abstention check: empty or explicitly not grounded = fail
    if not ans_text or ans_text.lower() in {"i don't know", "abstain", ""}:
        return False

    if task.get("verifiable", False):
        gold = str(task.get("gold", "")).strip().lower()
        if not gold:
            return grounded  # no gold to check against, require groundedness
        return gold in ans_text.lower()

    if task.get("judged", False):
        # LLM judge not yet wired — accept any grounded, non-empty answer
        return grounded and bool(ans_text)

    # Default: require grounded answer
    return grounded

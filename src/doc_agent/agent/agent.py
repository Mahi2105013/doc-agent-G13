"""Stage 6 - FIXED loop - perceive -> decide -> act -> observe, with cross-cutting seams.
Implement decide() and synthesize() only. Security, grounding, PII, and tracing run via hooks at the
marked seams - do NOT inline them here."""
from __future__ import annotations
import json
import re

from ..contracts import Answer, Citation, Chunk, ToolResult  # noqa
from .. import hooks
from .memory import Memory
from ..logging_conf import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Injection-defence: strip adversarial prompt patterns from user query input.
# --------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore\s+(your\s+)?instructions",
    r"disregard\s+(your\s+)?instructions",
    r"you\s+are\s+now",
    r"forget\s+(your\s+)?instructions",
    r"act\s+as\s+(if\s+)?",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def _sanitize(text: str) -> str:
    """Remove prompt-injection patterns from any user-supplied string."""
    return _INJECTION_RE.sub("[REDACTED]", text)


class Agent:
    """FIXED loop. Implement decide() (the policy) and synthesize() only."""

    def __init__(self, cfg: dict, retriever) -> None:
        self.cfg = cfg
        self.agent_cfg = cfg.get("agent", {})
        self.retriever = retriever
        self.mem = Memory()

        # Instantiate one instance of every tool class and index by name.
        from . import tools as _tools_mod
        _tools_mod.bind(retriever, cfg)
        self._tools: dict[str, _tools_mod.Tool] = {
            cls.name: cls() for cls in _tools_mod.REGISTRY
        }

        # Lazy LLM client.
        from ..llm.client import LLM
        self._llm = LLM(cfg)

    # ======================================================================
    # FIXED run loop — do NOT modify.
    # ======================================================================
    def run(self, query_text: str) -> Answer:
        state = {"query": query_text, "obs": []}
        for _ in range(self.agent_cfg.get("max_steps", 8)):
            hooks.run(hooks.ON_STEP, {"state": state})
            action = self.decide(state)                       # IMPLEMENT (policy)
            if action["tool"] == "stop":
                break
            hooks.run(hooks.ON_TOOL_CALL, {"action": action})    # guardrails/injection/trace
            result = self.act(action)                         # runs the tool via REGISTRY
            state["obs"].append(result)
            self.mem.add(result)
        hooks.run(hooks.BEFORE_ANSWER, {"state": state})      # grounding gate / PII redact
        ans = self.synthesize(state)                          # IMPLEMENT (grounded answer)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})        # trace / metrics
        return ans

    # ======================================================================
    # decide() — evidence-gated re-search (MANDATORY A3 agentic behaviour)
    # ======================================================================
    def decide(self, state: dict) -> dict:
        """Evidence-gated re-search — the MANDATORY agentic behaviour (A3 gate, fail-closed).

        Algorithm:
          1. On first call, retrieve at k = cfg['retrieve']['k'].
          2. After each retrieve, check is_weak(chunks, cfg):
             - If NOT weak → emit {"tool": "stop", "abstain": False} to synthesize.
             - If weak and next_k is available → retrieve again at wider k.
             - If weak and next_k is None (hit k_max) → ABSTAIN.
          3. For non-retrieve steps the LLM policy selects the next tool.

        Emits obs {"top_score": ..., "k": ...} on each retrieval step so
        traces/run.jsonl captures the full trajectory.
        """
        from ..retrieval.retriever import is_weak, next_k
        from ..llm import prompts

        retrieve_cfg = self.cfg.get("retrieve", {})
        query: str = _sanitize(state["query"])

        # ----------------------------------------------------------------
        # Phase 1: determine current evidence strength from last observation.
        # ----------------------------------------------------------------
        last_retrieve_obs: dict | None = None
        for obs in reversed(state["obs"]):
            if isinstance(obs, ToolResult) and obs.ok:
                payload = obs.payload
                if "top_score" in payload and "k" in payload:
                    last_retrieve_obs = payload
                    break

        # ----------------------------------------------------------------
        # Phase 2: evidence-gated branching (the core agentic loop).
        # ----------------------------------------------------------------
        if last_retrieve_obs is None:
            # No retrieval yet → issue the first retrieve at default k.
            k0 = int(retrieve_cfg.get("k", 10))
            logger.info("decide: initial retrieve k=%d", k0)
            return {"tool": "retrieve", "args": {"query": query, "k": k0}}

        current_k: int = int(last_retrieve_obs["k"])
        top_sc: float = float(last_retrieve_obs["top_score"])
        chunks_raw: list[dict] = last_retrieve_obs.get("chunks", [])
        chunks: list[Chunk] = [Chunk(**c) for c in chunks_raw] if chunks_raw else []

        # Emit structured observation so traces carry the re-search signal.
        obs_summary = {"top_score": top_sc, "k": current_k}
        logger.info("decide: top_score=%.4f k=%d weak_threshold=%.2f",
                    top_sc, current_k,
                    retrieve_cfg.get("weak_threshold", 0.35))

        if not is_weak(chunks, self.cfg):
            # Evidence is strong enough -> stop and synthesize.
            logger.info("decide: evidence sufficient -> synthesize")
            return {"tool": "stop", "abstain": False, "_obs": obs_summary}

        # Evidence is weak -> try to widen.
        nk = next_k(current_k, self.cfg)
        if nk is None:
            # Hit k_max and still weak → ABSTAIN.
            logger.info(
                "decide: k_max reached and evidence still weak → ABSTAIN "
                "(top_score=%.4f < threshold=%.2f)",
                top_sc, retrieve_cfg.get("weak_threshold", 0.35),
            )
            return {
                "tool": "stop",
                "abstain": True,
                "reason": (
                    f"Insufficient evidence after widening k to {current_k}. "
                    f"Best relevance score: {top_sc:.3f} "
                    f"(threshold: {retrieve_cfg.get('weak_threshold', 0.35):.2f})"
                ),
                "_obs": obs_summary,
            }

        # Widen k and retrieve again.
        logger.info("decide: evidence weak (%.4f) → widen k %d → %d", top_sc, current_k, nk)

        # Optionally let the LLM reformulate the query for the wider retrieval.
        reformulated_query = self._maybe_reformulate(query, state)
        return {"tool": "retrieve", "args": {"query": reformulated_query, "k": nk},
                "_obs": obs_summary}

    # ------------------------------------------------------------------
    def _maybe_reformulate(self, query: str, state: dict) -> str:
        """Attempt a lightweight LLM-based query reformulation to improve recall.

        Falls back to the original query if the LLM is unavailable."""
        try:
            mem_summary = self._memory_summary(query)
            obs_summary = self._obs_summary(state)
            reformulation_prompt = (
                f"You are a query-reformulation assistant.\n"
                f"Original query: {query}\n"
                f"Previous observations: {obs_summary}\n"
                f"The original query retrieved insufficient evidence. "
                f"Produce ONE alternative query (same language, broader or synonymous terms) "
                f"that might find better evidence. Output the query only, no explanation."
            )
            new_query = self._llm.complete(reformulation_prompt, max_tokens=80).strip()
            if new_query and len(new_query) < 500:
                logger.info("decide: reformulated query=%r", new_query[:80])
                return new_query
        except Exception as exc:
            logger.debug("Query reformulation failed (%s); using original.", exc)
        return query

    # ======================================================================
    # act() — dispatch tool from REGISTRY
    # ======================================================================
    def act(self, action: dict) -> ToolResult:
        """Run the tool named in action['tool'] with action['args']."""
        tool_name: str = action.get("tool", "")
        args: dict = action.get("args", {})

        tool_instance = self._tools.get(tool_name)
        if tool_instance is None:
            logger.error("act: unknown tool %r — skipping.", tool_name)
            return ToolResult(ok=False, payload={"error": f"Unknown tool: {tool_name}"})

        try:
            logger.info("act: tool=%s args=%s", tool_name, list(args.keys()))
            return tool_instance(**args)
        except Exception as exc:
            logger.error("act: tool=%s raised %s", tool_name, exc)
            return ToolResult(ok=False, payload={"error": str(exc), "tool": tool_name})

    # ======================================================================
    # synthesize() — grounded, cited answer; abstain if unsupported
    # ======================================================================
    def synthesize(self, state: dict) -> Answer:
        """Build a grounded, cited answer from the accumulated observations.

        Abstains (grounded=False, confidence=0.0) if:
          - The last decide() set abstain=True, OR
          - The LLM responds with the ABSTAIN sentinel, OR
          - No retrieved chunks are found in the observations.
        """
        from ..llm import prompts

        query: str = _sanitize(state["query"])

        # ----------------------------------------------------------------
        # Check if decide() already flagged an abstain.
        # ----------------------------------------------------------------
        for obs in reversed(state["obs"]):
            if isinstance(obs, dict) and obs.get("abstain"):
                reason = obs.get("reason", "Insufficient evidence.")
                return Answer(
                    text=f"ABSTAIN: {reason}",
                    citations=[],
                    grounded=False,
                    confidence=0.0,
                )

        # ----------------------------------------------------------------
        # Collect all retrieved chunks from observations.
        # ----------------------------------------------------------------
        all_chunks: list[Chunk] = []
        citations: list[Citation] = []

        for obs in state["obs"]:
            if isinstance(obs, ToolResult) and obs.ok:
                payload = obs.payload
                # Chunks from Retrieve tool
                if "chunks" in payload:
                    for c_dict in payload["chunks"]:
                        if isinstance(c_dict, dict):
                            all_chunks.append(Chunk(**c_dict))
                        elif isinstance(c_dict, Chunk):
                            all_chunks.append(c_dict)
                # Citations from Cite tool
                if "citation" in payload:
                    cit_dict = payload["citation"]
                    if isinstance(cit_dict, dict):
                        citations.append(Citation(**cit_dict))

        if not all_chunks:
            return Answer(
                text=(
                    "ABSTAIN: The provided evidence does not contain sufficient "
                    "information to answer this question."
                ),
                citations=[],
                grounded=False,
                confidence=0.0,
            )

        # Deduplicate by chunk id, keeping highest-score version.
        seen: dict[str, Chunk] = {}
        for c in all_chunks:
            if c.id not in seen or c.score > seen[c.id].score:
                seen[c.id] = c
        unique_chunks = sorted(seen.values(), key=lambda c: c.score, reverse=True)

        # ----------------------------------------------------------------
        # Format evidence block for the prompt.
        # ----------------------------------------------------------------
        evidence_lines: list[str] = []
        for rank, c in enumerate(unique_chunks[:10], start=1):
            preview = c.text[:600].replace("\n", " ")
            evidence_lines.append(
                f"[{c.id}] (score={c.score:.3f}, pages={c.page_ids})\n{preview}"
            )
        evidence_str = "\n\n---\n".join(evidence_lines)

        # ----------------------------------------------------------------
        # Call LLM to synthesize the answer.
        # ----------------------------------------------------------------
        prompt = prompts.SYNTHESIZE.format(query=query, evidence=evidence_str)
        try:
            raw_answer: str = self._llm.complete(prompt)
        except Exception as exc:
            logger.error("synthesize: LLM call failed: %s", exc)
            # Fall back to extractive answer from top chunk.
            raw_answer = unique_chunks[0].text[:400] if unique_chunks else ""

        # ----------------------------------------------------------------
        # Parse ABSTAIN sentinel.
        # ----------------------------------------------------------------
        if raw_answer.strip().startswith("ABSTAIN:"):
            return Answer(
                text=raw_answer.strip(),
                citations=[],
                grounded=False,
                confidence=0.0,
            )

        # ----------------------------------------------------------------
        # Parse confidence score from last line  "Confidence: 0.87"
        # ----------------------------------------------------------------
        confidence = 0.5
        lines = raw_answer.strip().splitlines()
        if lines:
            conf_match = re.search(r"confidence[:\s]+([0-9.]+)", lines[-1], re.IGNORECASE)
            if conf_match:
                try:
                    confidence = max(0.0, min(1.0, float(conf_match.group(1))))
                    # Remove the confidence line from the answer text.
                    raw_answer = "\n".join(lines[:-1]).strip()
                except ValueError:
                    pass

        # ----------------------------------------------------------------
        # Extract inline citations from the answer text  e.g. [chunk_id]
        # ----------------------------------------------------------------
        inline_ids: list[str] = re.findall(r"\[([^\[\]]+)\]", raw_answer)
        for cid in inline_ids:
            # Verify the chunk id exists in our evidence.
            if cid in seen:
                text_len = len(seen[cid].text)
                citations.append(Citation(chunk_id=cid, span=(0, text_len)))

        # Deduplicate citations by chunk_id.
        seen_cit: set[str] = set()
        unique_citations: list[Citation] = []
        for cit in citations:
            if cit.chunk_id not in seen_cit:
                unique_citations.append(cit)
                seen_cit.add(cit.chunk_id)

        return Answer(
            text=raw_answer,
            citations=unique_citations,
            grounded=len(unique_citations) > 0,
            confidence=confidence,
        )

    # ======================================================================
    # Helpers
    # ======================================================================
    def _memory_summary(self, query: str) -> str:
        recalled = self.mem.recall(query)
        if not recalled:
            return "None"
        parts: list[str] = []
        for item in recalled[:3]:
            if isinstance(item, ToolResult):
                parts.append(f"ToolResult(ok={item.ok}, keys={list(item.payload.keys())})")
            else:
                parts.append(str(item)[:120])
        return "; ".join(parts)

    def _obs_summary(self, state: dict) -> str:
        obs = state.get("obs", [])
        if not obs:
            return "None"
        parts: list[str] = []
        for item in obs[-3:]:
            if isinstance(item, ToolResult):
                score = item.payload.get("top_score", "?")
                k = item.payload.get("k", "?")
                parts.append(f"retrieve(k={k}, top_score={score})")
            else:
                parts.append(str(item)[:80])
        return "; ".join(parts)

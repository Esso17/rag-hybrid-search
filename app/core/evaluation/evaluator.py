"""
RAG evaluation using LLM-as-judge.

Three metrics, each scored 0.0–1.0:
  faithfulness       — every claim in the answer is backed by the retrieved context
  answer_relevance   — the answer actually addresses the question asked
  context_relevance  — the retrieved chunks are useful for answering the question

Overall score = 0.4 * faithfulness + 0.4 * answer_relevance + 0.2 * context_relevance
"""

import json
import logging
import re

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r'"score"\s*:\s*([\d.]+)', re.IGNORECASE)
_REASON_RE = re.compile(r'"reason"\s*:\s*"([^"]*)"', re.IGNORECASE | re.DOTALL)


def _llm(prompt: str, client: httpx.Client) -> str:
    try:
        r = client.post(
            f"{settings.LLM_BASE_URL}/api/generate",
            json={
                "model": settings.LLM_MODEL,
                "prompt": prompt,
                "temperature": 0.1,
                "stream": False,
            },
            timeout=45.0,
        )
        return r.json().get("response", "")
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return ""


def _parse(raw: str, fallback_score: float = 0.5) -> tuple[float, str]:
    """Extract (score, reason) from LLM JSON output with multiple fallbacks."""
    raw = raw.strip()

    # Direct parse
    try:
        obj = json.loads(raw)
        score = min(1.0, max(0.0, float(obj.get("score", fallback_score))))
        reason = str(obj.get("reason", ""))
        return score, reason
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # First {...} block
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            score = min(1.0, max(0.0, float(obj.get("score", fallback_score))))
            reason = str(obj.get("reason", ""))
            return score, reason
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Regex fallback
    sm = _SCORE_RE.search(raw)
    rm = _REASON_RE.search(raw)
    score = min(1.0, max(0.0, float(sm.group(1)))) if sm else fallback_score
    reason = rm.group(1) if rm else ""
    return score, reason


def _faithfulness(
    query: str, answer: str, context: list[str], client: httpx.Client
) -> tuple[float, str]:
    """Score how well the answer is grounded in the retrieved context."""
    ctx = "\n---\n".join(context[:5])  # cap to avoid context overflow
    ans = answer[:1500]

    prompt = f"""You are a RAG evaluation judge. Assess whether the answer is grounded in the provided context.

Context:
{ctx}

Question: {query}

Answer: {ans}

Faithfulness measures whether every factual claim in the answer can be traced back to the context.
Score 1.0 if all claims are fully supported. Score 0.0 if the answer introduces facts not present in the context.

Return ONLY valid JSON — no explanation, no markdown:
{{"score": 0.85, "reason": "Most claims are supported, but X was not mentioned in context."}}

JSON:"""

    raw = _llm(prompt, client)
    return _parse(raw, fallback_score=0.5)


def _answer_relevance(query: str, answer: str, client: httpx.Client) -> tuple[float, str]:
    """Score whether the answer actually addresses the question."""
    ans = answer[:1500]

    prompt = f"""You are a RAG evaluation judge. Assess whether the answer addresses the question.

Question: {query}

Answer: {ans}

Answer relevance measures whether the answer directly responds to what was asked.
Score 1.0 if the answer is completely on-topic and thorough. Score 0.0 if it is off-topic or evasive.

Return ONLY valid JSON — no explanation, no markdown:
{{"score": 0.9, "reason": "The answer directly addresses the question and covers the main points."}}

JSON:"""

    raw = _llm(prompt, client)
    return _parse(raw, fallback_score=0.5)


def _context_relevance(query: str, context: list[str], client: httpx.Client) -> tuple[float, str]:
    """Score whether the retrieved chunks are actually useful for the query."""
    chunks_preview = "\n---\n".join(f"[{i+1}] {c[:200]}" for i, c in enumerate(context[:5]))

    prompt = f"""You are a RAG evaluation judge. Assess whether the retrieved chunks are relevant to the question.

Question: {query}

Retrieved chunks:
{chunks_preview}

Context relevance measures what fraction of the chunks contain information useful for answering the question.
Score 1.0 if all chunks are highly relevant. Score 0.0 if none are relevant.

Return ONLY valid JSON — no explanation, no markdown:
{{"score": 0.7, "reason": "Chunks 1 and 3 are relevant, but chunks 2 and 4 are off-topic."}}

JSON:"""

    raw = _llm(prompt, client)
    return _parse(raw, fallback_score=0.5)


def evaluate(
    query: str,
    answer: str,
    context_chunks: list[str],
    llm_client: httpx.Client,
) -> dict:
    """
    Run all three evaluation metrics and return combined results.

    Returns:
        {
            faithfulness:      float,
            answer_relevance:  float,
            context_relevance: float,
            overall_score:     float,
            details: {
                faithfulness_reason:      str,
                answer_relevance_reason:  str,
                context_relevance_reason: str,
            }
        }
    """
    faith_score, faith_reason = _faithfulness(query, answer, context_chunks, llm_client)
    rel_score, rel_reason = _answer_relevance(query, answer, llm_client)
    ctx_score, ctx_reason = _context_relevance(query, context_chunks, llm_client)

    overall = round(0.4 * faith_score + 0.4 * rel_score + 0.2 * ctx_score, 4)

    logger.info(
        f"Eval: faithfulness={faith_score:.2f} relevance={rel_score:.2f} "
        f"context={ctx_score:.2f} overall={overall:.2f}"
    )

    return {
        "faithfulness": round(faith_score, 4),
        "answer_relevance": round(rel_score, 4),
        "context_relevance": round(ctx_score, 4),
        "overall_score": overall,
        "details": {
            "faithfulness_reason": faith_reason,
            "answer_relevance_reason": rel_reason,
            "context_relevance_reason": ctx_reason,
        },
    }

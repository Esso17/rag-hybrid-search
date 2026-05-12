"""Self-check: evaluate whether an answer fully addresses the original query"""

import json
import logging
import re

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_PASS_THROUGH = {"complete": True, "confidence": 0.5, "gaps": []}


def _extract_json_object(text: str) -> dict | None:
    """Robustly extract a JSON object from LLM output."""
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Find first {...} block
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Attempt to reconstruct from key patterns (last resort)
    complete = re.search(r'"complete"\s*:\s*(true|false)', text, re.IGNORECASE)
    confidence = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    gaps_block = re.search(r'"gaps"\s*:\s*\[(.*?)\]', text, re.DOTALL)

    if complete:
        result = {
            "complete": complete.group(1).lower() == "true",
            "confidence": float(confidence.group(1)) if confidence else 0.5,
            "gaps": re.findall(r'"([^"]+)"', gaps_block.group(1)) if gaps_block else [],
        }
        return result

    return None


def self_check(query: str, answer: str, llm_client: httpx.Client) -> dict:
    """
    Ask the LLM to evaluate whether the answer completely addresses the query.

    Returns:
        complete   (bool)    — True if all aspects are addressed
        confidence (float)   — 0.0–1.0, how confident the answer is correct
        gaps       (list[str]) — specific missing topics (empty when complete=True)
    """
    # Truncate to avoid context overflow on small local models
    answer_excerpt = answer[:1500] if len(answer) > 1500 else answer

    prompt = f"""Evaluate whether this answer completely addresses the question.

Question: {query}

Answer: {answer_excerpt}

Return ONLY valid JSON — no explanation, no markdown, nothing else:
{{"complete": true, "confidence": 0.85, "gaps": []}}

Rules:
- complete: true only if every aspect of the question is covered
- confidence: 0.0 = not confident at all, 1.0 = completely confident
- gaps: list specific topics or aspects that are missing (empty list if complete=true)

JSON:"""

    try:
        response = llm_client.post(
            f"{settings.LLM_BASE_URL}/api/generate",
            json={
                "model": settings.LLM_MODEL,
                "prompt": prompt,
                "temperature": 0.1,
                "stream": False,
            },
            timeout=30.0,
        )
        raw = response.json().get("response", "")
        result = _extract_json_object(raw)
        if result is not None:
            return {
                "complete": bool(result.get("complete", True)),
                "confidence": min(1.0, max(0.0, float(result.get("confidence", 0.5)))),
                "gaps": [str(g).strip() for g in result.get("gaps", []) if str(g).strip()],
            }
    except Exception as e:
        logger.warning(f"Self-check failed, treating as complete: {e}")

    return _PASS_THROUGH

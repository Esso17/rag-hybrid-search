"""Decompose complex queries into focused sub-questions for targeted retrieval"""

import json
import logging
import re

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def _extract_json_list(text: str) -> list[str] | None:
    """Robustly extract a JSON array from LLM output (local models are unreliable)."""
    text = text.strip()

    # Direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except json.JSONDecodeError:
        pass

    # Extract first [...] block (handles preamble/postamble)
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            pass

    # Last resort: extract all quoted strings
    items = re.findall(r'"([^"]{10,})"', text)
    if items:
        return items

    return None


def decompose_query(query: str, llm_client: httpx.Client) -> list[str]:
    """
    Break a complex multi-part question into 2-4 focused sub-questions.

    Returns the original query wrapped in a list if decomposition fails or
    the question is already focused.  Caps at 4 sub-questions regardless.
    """
    prompt = f"""You are a search query optimizer for technical documentation.

Analyze the question below. If it covers multiple distinct concepts or asks
"how X works AND what are the benefits AND how to configure Y", split it into
2-4 focused sub-questions that can each be searched independently.
If it is already a single focused question, return it unchanged as one item.

Question: {query}

Return ONLY a valid JSON array of strings. No explanation, no markdown, nothing else.
["sub-question 1", "sub-question 2"]

JSON array:"""

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
        sub_questions = _extract_json_list(raw)
        if sub_questions:
            sub_questions = [q for q in sub_questions[:4] if len(q) > 5]
            if sub_questions:
                logger.info(f"Decomposed query into {len(sub_questions)} sub-questions")
                return sub_questions
    except Exception as e:
        logger.warning(f"Query decomposition failed, using original: {e}")

    return [query]

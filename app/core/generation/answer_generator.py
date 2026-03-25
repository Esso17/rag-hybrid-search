"""LLM answer generation with optional DevOps prompts"""

import logging
from typing import Callable, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def generate_answer(
    query: str,
    context_chunks: list[str],
    llm_client: httpx.Client,
    use_devops_prompts: bool = False,
    prompt_builder: Optional[Callable] = None,
) -> str:
    """
    Generate answer using local LLM with optional DevOps prompts

    Args:
        query: User query
        context_chunks: List of context chunks to use
        llm_client: HTTP client for LLM API
        use_devops_prompts: Whether to use DevOps-optimized prompts
        prompt_builder: Optional custom prompt builder function

    Returns:
        Generated answer string
    """
    if use_devops_prompts and prompt_builder:
        # Use specialized DevOps prompt
        prompt = prompt_builder(query, context_chunks)
    else:
        # Use generic prompt
        context = "\n\n".join(context_chunks)
        prompt = f"""Based on the following context, provide a comprehensive answer to the question.

Context:
{context}

Question: {query}

Answer:"""

    response = llm_client.post(
        f"{settings.LLM_BASE_URL}/api/generate",
        json={
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "temperature": settings.LLM_TEMPERATURE,
            "stream": False,
        },
        timeout=120.0,
    )

    result = response.json()
    return result.get("response", "Unable to generate answer")

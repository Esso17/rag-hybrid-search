"""Async batch embedding generation"""

import asyncio
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def embed_batch_async(texts: list[str], max_concurrent: int = 20) -> list[list[float]]:
    """
    Generate embeddings for multiple texts concurrently

    Args:
        texts: List of text strings to embed
        max_concurrent: Maximum concurrent requests

    Returns:
        List of embeddings (same length as texts)

    Note: Creates semaphore in current event loop to avoid loop binding issues
    """
    base_url = settings.LLM_BASE_URL
    model = settings.EMBEDDING_MODEL
    semaphore = asyncio.Semaphore(max_concurrent)

    async def embed_single(text: str, client: httpx.AsyncClient) -> list[float]:
        """Generate embedding for a single text"""
        async with semaphore:
            try:
                response = await client.post(
                    f"{base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("embedding", [])
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                raise

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [embed_single(text, client) for text in texts]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        results = []
        for i, embedding in enumerate(embeddings):
            if isinstance(embedding, Exception):
                logger.error(f"Failed to embed text {i}: {embedding}")
                results.append([])  # Empty embedding on error
            else:
                results.append(embedding)

        return results

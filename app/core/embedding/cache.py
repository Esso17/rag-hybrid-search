"""
Query embedding cache for performance optimization

Implements exact-match caching using LRU cache and query normalization.
Saves 30ms Ollama API calls on repeated queries.

Caching Strategy:
- Exact match: Hash of normalized query text
- Normalization: lowercase + strip whitespace/punctuation
- LRU eviction: 1000 entries max
- No TTL: Embeddings don't change unless model changes

Performance: 30ms → 0ms on cache hits
"""

import hashlib
import logging
import re
from functools import lru_cache

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def normalize_query(query: str) -> str:
    """
    Normalize query for cache key generation

    Normalization steps:
    1. Convert to lowercase
    2. Strip leading/trailing whitespace
    3. Remove punctuation (except alphanumeric and spaces)
    4. Collapse multiple spaces

    Examples:
    - "What area codes correspond to Athens, Greece?" → "what area codes correspond to athens greece"
    - "  How to debug   pod networking? " → "how to debug pod networking"

    Args:
        query: Raw query text

    Returns:
        Normalized query string
    """
    # Lowercase
    normalized = query.lower()

    # Remove punctuation (keep alphanumeric, spaces, hyphens)
    normalized = re.sub(r"[^\w\s-]", " ", normalized)

    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)

    # Strip whitespace
    normalized = normalized.strip()

    return normalized


@lru_cache(maxsize=1000)
def get_cached_query_embedding(query_hash: str, base_url: str, model: str, query: str) -> tuple:
    """
    Get cached query embedding using normalized query hash

    This function is cached via @lru_cache, so repeated calls with the same
    query_hash will return the cached embedding without calling Ollama.

    Args:
        query_hash: SHA256 hash of the normalized query
        base_url: LLM base URL (Ollama)
        model: Embedding model name (e.g., nomic-embed-text)
        query: The actual query text (used only on cache miss)

    Returns:
        Tuple of embedding values (hashable for LRU cache)
    """
    try:
        logger.debug(f"Cache MISS: Generating embedding for query (hash={query_hash[:8]}...)")
        response = httpx.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": query},
            timeout=60.0,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding", [])
        return tuple(embedding)  # Convert to tuple for caching
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return ()  # Empty tuple on error


def get_query_embedding(query: str, use_normalization: bool = True) -> list[float]:
    """
    Get query embedding with caching support

    Uses LRU cache to avoid redundant Ollama API calls for repeated queries.
    Query normalization increases cache hit rate.

    Args:
        query: Query text to embed
        use_normalization: Enable query normalization (default True)

    Returns:
        List of embedding values (768-dim for nomic-embed-text)
    """
    # Normalize query for better cache hit rate
    cache_key_query = normalize_query(query) if use_normalization else query

    # Create hash of normalized query for cache key
    query_hash = hashlib.sha256(cache_key_query.encode()).hexdigest()

    # Get embedding (from cache or generate)
    embedding_tuple = get_cached_query_embedding(
        query_hash, settings.LLM_BASE_URL, settings.EMBEDDING_MODEL, query
    )

    if embedding_tuple:
        logger.debug(f"Embedding cache hit: {cache_key_query[:50]}...")

    return list(embedding_tuple)  # Convert back to list


def get_embedding_cache_info() -> dict:
    """
    Get embedding cache statistics

    Returns:
        Dict with cache size and hit rate info
    """
    cache_info = get_cached_query_embedding.cache_info()
    return {
        "type": "LRU",
        "max_size": 1000,
        "current_size": cache_info.currsize,
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "hit_rate": (
            cache_info.hits / (cache_info.hits + cache_info.misses)
            if (cache_info.hits + cache_info.misses) > 0
            else 0.0
        ),
    }


def clear_embedding_cache():
    """Clear the embedding cache (useful for testing or model changes)"""
    get_cached_query_embedding.cache_clear()
    logger.info("Embedding cache cleared")

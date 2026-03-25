"""Score normalization and fusion for hybrid search with advanced heuristics"""

import re
from datetime import datetime
from typing import Optional

from app.config import settings


def reciprocal_rank_fusion(
    vector_scores: dict[int, float], bm25_scores: dict[int, float], k: int = 60
) -> dict[int, float]:
    """
    Reciprocal Rank Fusion (RRF) - often outperforms weighted average

    Formula: RRF(d) = sum(1 / (k + rank(d)))

    Args:
        vector_scores: Dictionary mapping chunk_index to vector similarity score
        bm25_scores: Dictionary mapping chunk_index to BM25 score
        k: Constant to prevent division issues (default: 60, per research)

    Returns:
        Dictionary mapping chunk_index to RRF score
    """
    # Create rankings (sorted by score, descending)
    vector_ranking = {
        idx: rank + 1
        for rank, (idx, _) in enumerate(
            sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)
        )
    }
    bm25_ranking = {
        idx: rank + 1
        for rank, (idx, _) in enumerate(
            sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        )
    }

    # Calculate RRF scores
    all_indices = set(vector_scores.keys()) | set(bm25_scores.keys())
    rrf_scores = {}

    for idx in all_indices:
        v_rank = vector_ranking.get(idx, len(vector_ranking) + k)
        b_rank = bm25_ranking.get(idx, len(bm25_ranking) + k)
        rrf_scores[idx] = (1.0 / (k + v_rank)) + (1.0 / (k + b_rank))

    return rrf_scores


def normalize_and_combine_scores(
    vector_scores: dict[int, float], bm25_scores: dict[int, float], alpha: float = None
) -> dict[int, float]:
    """
    Normalize and combine vector and BM25 scores using weighted average

    Args:
        vector_scores: Dictionary mapping chunk_index to vector similarity score
        bm25_scores: Dictionary mapping chunk_index to BM25 score
        alpha: Weight for vector scores (1-alpha for BM25). Defaults to settings value

    Returns:
        Dictionary mapping chunk_index to combined score
    """
    if alpha is None:
        alpha = settings.HYBRID_SEARCH_ALPHA

    # Normalize scores to 0-1 range
    max_vector_score = max(vector_scores.values()) if vector_scores else 1.0
    max_bm25_score = max(bm25_scores.values()) if bm25_scores else 1.0

    # Combine scores from both methods
    combined_scores = {}
    all_indices = set(vector_scores.keys()) | set(bm25_scores.keys())

    for idx in all_indices:
        v_score = (vector_scores.get(idx, 0) / max_vector_score) if max_vector_score > 0 else 0
        b_score = (bm25_scores.get(idx, 0) / max_bm25_score) if max_bm25_score > 0 else 0
        combined_scores[idx] = alpha * v_score + (1 - alpha) * b_score

    return combined_scores


def calculate_query_overlap(query: str, content: str) -> float:
    """
    Calculate query-document overlap score

    Args:
        query: Search query
        content: Document content

    Returns:
        Overlap score (0-1)
    """
    # Tokenize and normalize
    query_terms = set(re.findall(r"\w+", query.lower()))
    content_terms = set(re.findall(r"\w+", content.lower()))

    if not query_terms:
        return 0.0

    # Calculate Jaccard similarity
    intersection = len(query_terms & content_terms)
    union = len(query_terms | content_terms)

    return intersection / union if union > 0 else 0.0


def has_exact_match(query: str, content: str) -> bool:
    """
    Check if query has exact phrase match in content

    Args:
        query: Search query
        content: Document content

    Returns:
        True if exact match found
    """
    return query.lower() in content.lower()


def calculate_chunk_quality_score(content: str, min_length: int = 50) -> float:
    """
    Calculate quality score for a chunk

    Penalizes:
    - Very short chunks (likely incomplete)
    - Very repetitive content
    - Chunks with low character variety

    Args:
        content: Chunk content
        min_length: Minimum expected length

    Returns:
        Quality score (0-1)
    """
    if not content:
        return 0.0

    score = 1.0

    # Penalize very short chunks
    if len(content) < min_length:
        score *= len(content) / min_length

    # Penalize low character variety (repetitive content)
    unique_chars = len(set(content.lower()))
    char_variety = unique_chars / min(len(content), 100)  # Normalize
    score *= min(char_variety * 2, 1.0)  # Boost variety importance

    # Check for excessive repetition (simple heuristic)
    words = content.split()
    if len(words) > 5:
        unique_words = len(set(words))
        word_diversity = unique_words / len(words)
        score *= 0.7 + 0.3 * word_diversity  # Don't penalize too heavily

    return max(score, 0.1)  # Minimum score of 0.1


def apply_metadata_boost(
    scores: dict[int, float], results: list[dict], boost_config: Optional[dict] = None
) -> dict[int, float]:
    """
    Apply metadata-based boosting to scores

    Args:
        scores: Current scores by chunk_index
        results: List of result dictionaries with metadata
        boost_config: Optional boost configuration

    Returns:
        Boosted scores
    """
    # Set defaults for missing boost config keys
    default_config = {
        "recency_weight": 0.1,  # 10% boost for recent docs
        "source_quality": {},  # Custom source quality scores
        "exact_match_boost": 0.2,  # 20% boost for exact matches
    }

    if boost_config is None:
        boost_config = default_config
    else:
        # Merge with defaults for missing keys
        for key, value in default_config.items():
            if key not in boost_config:
                boost_config[key] = value

    boosted_scores = scores.copy()

    for result in results:
        chunk_idx = result.get("chunk_index")
        if chunk_idx not in boosted_scores:
            continue

        base_score = boosted_scores[chunk_idx]
        boost_multiplier = 1.0

        # Recency boost (if metadata has timestamp)
        metadata = result.get("metadata", {})
        if "timestamp" in metadata:
            try:
                doc_time = datetime.fromisoformat(metadata["timestamp"])
                age_days = (datetime.now() - doc_time).days
                # Decay function: newer docs get higher boost
                recency_boost = 1.0 / (1.0 + age_days / 30)  # 30-day half-life
                boost_multiplier += boost_config.get("recency_weight", 0.1) * recency_boost
            except (ValueError, TypeError):
                pass

        # Source quality boost
        source = metadata.get("source", "")
        source_quality_map = boost_config.get("source_quality", {})
        if source in source_quality_map:
            boost_multiplier += source_quality_map[source]

        boosted_scores[chunk_idx] = base_score * boost_multiplier

    return boosted_scores


def enhanced_fusion(
    vector_scores: dict[int, float],
    bm25_scores: dict[int, float],
    query: str,
    results: list[dict],
    method: str = "rrf",
    apply_heuristics: bool = True,
    boost_config: Optional[dict] = None,
) -> dict[int, float]:
    """
    Enhanced score fusion with multiple heuristics

    Args:
        vector_scores: Vector similarity scores
        bm25_scores: BM25 scores
        query: Original search query
        results: Search results with content and metadata
        method: Fusion method ("rrf" or "weighted")
        apply_heuristics: Whether to apply quality/overlap heuristics
        boost_config: Optional metadata boost configuration

    Returns:
        Final fused scores
    """
    # Base fusion
    if method == "rrf":
        combined_scores = reciprocal_rank_fusion(vector_scores, bm25_scores)
    else:
        combined_scores = normalize_and_combine_scores(vector_scores, bm25_scores)

    if not apply_heuristics:
        return combined_scores

    # Apply heuristics
    for result in results:
        chunk_idx = result.get("chunk_index")
        content = result.get("content", "")

        if chunk_idx not in combined_scores:
            continue

        base_score = combined_scores[chunk_idx]
        heuristic_multiplier = 1.0

        # Exact match boost
        if has_exact_match(query, content):
            heuristic_multiplier += 0.2  # 20% boost

        # Query overlap boost
        overlap = calculate_query_overlap(query, content)
        heuristic_multiplier += 0.15 * overlap  # Up to 15% boost

        # Quality penalty/boost
        quality = calculate_chunk_quality_score(content)
        heuristic_multiplier *= quality

        combined_scores[chunk_idx] = base_score * heuristic_multiplier

    # Apply metadata boosts
    combined_scores = apply_metadata_boost(combined_scores, results, boost_config)

    return combined_scores

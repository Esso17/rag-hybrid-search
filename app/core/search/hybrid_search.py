"""Hybrid search combining vector similarity and BM25 with enhanced fusion"""

import logging
from typing import Optional

from app.config import settings
from app.core.embedding.cache import get_query_embedding
from app.core.search.score_fusion import enhanced_fusion, normalize_and_combine_scores

logger = logging.getLogger(__name__)


def hybrid_search(
    query: str,
    vector_store,
    bm25_index,
    top_k: int = None,
    use_faiss: bool = False,
    fusion_method: str = "rrf",
    use_heuristics: bool = True,
    boost_config: Optional[dict] = None,
) -> list[dict]:
    """
    Perform hybrid search combining BM25 and vector search with enhanced fusion

    Args:
        query: Search query
        vector_store: Vector store instance (FAISS or in-memory)
        bm25_index: BM25 index instance
        top_k: Number of results to return
        use_faiss: Whether FAISS is being used
        fusion_method: Fusion method ("rrf" or "weighted")
        use_heuristics: Whether to apply quality/overlap heuristics
        boost_config: Optional metadata boost configuration

    Returns:
        List of search results with content, score, chunk_index, source
    """
    if top_k is None:
        top_k = settings.TOP_K_RESULTS

    # Check settings for fusion method preference
    fusion_method = getattr(settings, "FUSION_METHOD", fusion_method)
    use_heuristics = getattr(settings, "USE_SEARCH_HEURISTICS", use_heuristics)

    # Vector search
    query_embedding = get_query_embedding(query)

    if use_faiss:
        # FAISS search (Phase 3: 100-1000x faster)
        vector_results = vector_store.search(
            query_embedding, limit=top_k * 2, ef_search=settings.FAISS_EF_SEARCH
        )
    else:
        # In-memory search (fallback)
        vector_results = vector_store.search(query_embedding, limit=top_k * 2)

    vector_scores = {}
    for i, result in enumerate(vector_results):
        vector_scores[result["payload"].get("chunk_index", i)] = result["score"]

    # BM25 search
    bm25_results = bm25_index.search(query, top_k=top_k * 2)
    bm25_scores = {r["chunk_index"]: r["score"] for r in bm25_results}

    # Build result objects for heuristic analysis
    results_with_content = []
    all_indices = set(vector_scores.keys()) | set(bm25_scores.keys())

    for chunk_idx in all_indices:
        # Find content and metadata from vector results
        content = None
        metadata = {}
        for r in vector_results:
            if r["payload"].get("chunk_index") == chunk_idx:
                content = r["payload"].get("content")
                metadata = r["payload"].get("metadata", {})
                break

        # Fallback to BM25 index if not in vector results
        if content is None:
            content = bm25_index.chunks[chunk_idx] if chunk_idx < len(bm25_index.chunks) else ""

        results_with_content.append(
            {"chunk_index": chunk_idx, "content": content, "metadata": metadata}
        )

    # Apply enhanced fusion
    if use_heuristics:
        combined_scores = enhanced_fusion(
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            query=query,
            results=results_with_content,
            method=fusion_method,
            apply_heuristics=True,
            boost_config=boost_config,
        )
        logger.debug(f"Using enhanced fusion with {fusion_method} and heuristics")
    else:
        # Simple fusion
        if fusion_method == "rrf":
            from app.core.search.score_fusion import reciprocal_rank_fusion

            combined_scores = reciprocal_rank_fusion(vector_scores, bm25_scores)
        else:
            combined_scores = normalize_and_combine_scores(vector_scores, bm25_scores)
        logger.debug(f"Using simple {fusion_method} fusion")

    # Return top-k results
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    final_results = []
    for chunk_idx, score in sorted_results:
        # Find content from our pre-built list
        content = None
        metadata = {}
        for r in results_with_content:
            if r["chunk_index"] == chunk_idx:
                content = r["content"]
                metadata = r["metadata"]
                break

        final_results.append(
            {
                "content": content,
                "score": score,
                "chunk_index": chunk_idx,
                "source": "hybrid",
                "metadata": metadata,
            }
        )

    return final_results

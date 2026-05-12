"""Hybrid search: parallel BM25 + vector retrieval, RRF fusion, cross-encoder reranking."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from app.config import settings
from app.core.embedding.cache import get_query_embedding
from app.core.search.score_fusion import (
    enhanced_fusion,
    normalize_and_combine_scores,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)

# Shared executor for parallel retrieval (BM25 + vector run concurrently)
_retrieval_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval")


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
    if top_k is None:
        top_k = settings.TOP_K_RESULTS

    fusion_method = settings.FUSION_METHOD
    use_heuristics = settings.USE_SEARCH_HEURISTICS
    use_reranker = settings.USE_RERANKER

    query_embedding = get_query_embedding(query)
    fetch_k = top_k * 3 if use_reranker else top_k * 2

    # ── Parallel retrieval: BM25 + vector run simultaneously ─────────────────
    def _vector_search():
        kwargs = {"limit": fetch_k}
        if use_faiss:
            kwargs["ef_search"] = settings.FAISS_EF_SEARCH
        return vector_store.search(query_embedding, **kwargs)

    def _bm25_search():
        return bm25_index.search(query, top_k=fetch_k)

    futures = {
        _retrieval_executor.submit(_vector_search): "vector",
        _retrieval_executor.submit(_bm25_search): "bm25",
    }
    vector_results, bm25_results = [], []
    for future in as_completed(futures):
        name = futures[future]
        try:
            result = future.result()
            if name == "vector":
                vector_results = result
            else:
                bm25_results = result
        except Exception as e:
            logger.warning(f"{name} retrieval failed: {e}")

    # ── Build lookup maps — O(n) ──────────────────────────────────────────────
    content_to_meta: dict[str, dict] = {}
    vector_scores: dict[int, float] = {}
    vector_content: dict[int, str] = {}
    for i, r in enumerate(vector_results):
        payload = r["payload"]
        idx = payload.get("chunk_index", i)
        vector_scores[idx] = r["score"]
        content = payload.get("content", "")
        vector_content[idx] = content
        if content and content not in content_to_meta:
            meta = dict(payload.get("metadata") or {})
            meta.setdefault("title", payload.get("title"))
            meta.setdefault("document_id", payload.get("document_id"))
            content_to_meta[content] = meta

    bm25_scores: dict[int, float] = {r["chunk_index"]: r["score"] for r in bm25_results}
    bm25_meta: dict[int, dict] = {r["chunk_index"]: r.get("metadata", {}) for r in bm25_results}
    bm25_content: dict[int, str] = {r["chunk_index"]: r["content"] for r in bm25_results}

    # ── Unified result map — O(1) final lookup ────────────────────────────────
    result_map: dict[int, dict] = {}
    for idx in set(vector_scores) | set(bm25_scores):
        if idx in bm25_content:
            content = bm25_content[idx]
            metadata = bm25_meta[idx] or content_to_meta.get(content, {})
        else:
            content = vector_content.get(idx, "")
            metadata = content_to_meta.get(content, {})
        result_map[idx] = {"chunk_index": idx, "content": content, "metadata": metadata}

    # ── Score fusion ──────────────────────────────────────────────────────────
    if use_heuristics:
        combined = enhanced_fusion(
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            query=query,
            results=list(result_map.values()),
            method=fusion_method,
            apply_heuristics=True,
            boost_config=boost_config,
        )
    elif fusion_method == "rrf":
        combined = reciprocal_rank_fusion(vector_scores, bm25_scores)
    else:
        combined = normalize_and_combine_scores(vector_scores, bm25_scores)

    # ── Build pre-rerank candidates ───────────────────────────────────────────
    sorted_indices = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    candidates = [
        {
            "content": result_map[idx]["content"],
            "score": score,
            "chunk_index": idx,
            "source": "hybrid",
            "metadata": result_map[idx]["metadata"],
        }
        for idx, score in sorted_indices
        if idx in result_map
    ]

    # ── Cross-encoder reranking (optional) ───────────────────────────────────
    if use_reranker and candidates:
        from app.core.retrieval.reranker import rerank

        return rerank(query, candidates, top_k)

    return candidates[:top_k]

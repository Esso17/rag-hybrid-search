"""Cross-encoder reranker — second-stage precision boost after hybrid retrieval.

Model: cross-encoder/ms-marco-MiniLM-L6-v2
  • 22M params, ~67 MB on disk
  • CPU-only, no GPU required
  • ~50 ms to rerank 20 candidates
  • Proven +10–25% precision over raw embedding similarity
"""

import logging

logger = logging.getLogger(__name__)

_reranker = None


def _load_model():
    """Lazy-load the cross-encoder (avoids import cost at startup if disabled)."""
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", max_length=512)
        logger.info("Reranker loaded: cross-encoder/ms-marco-MiniLM-L6-v2")
        return model
    except ImportError:
        logger.warning("sentence-transformers not installed — reranker disabled")
        return None
    except Exception as e:
        logger.warning(f"Reranker load failed: {e} — reranker disabled")
        return None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = _load_model()
    return _reranker


def rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    """Rerank results using cross-encoder scores.

    Accepts and returns the same list[dict] format used throughout the pipeline:
      {"content": str, "score": float, "chunk_index": int, "source": str, "metadata": dict}

    Falls back to original order if model is unavailable.
    """
    if not results:
        return results

    model = get_reranker()
    if model is None:
        return results[:top_k]

    pairs = [(query, r["content"] or "") for r in results]
    try:
        scores = model.predict(pairs)
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)
            result["score"] = float(score)
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.debug(f"Reranked {len(results)} → {top_k} results")
    except Exception as e:
        logger.warning(f"Reranking failed, using original order: {e}")

    return results[:top_k]

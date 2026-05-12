"""Text retrieval module (BM25-based keyword search)"""

from app.core.retrieval.bm25 import BM25, BM25Index, get_bm25_index, get_bm25_inverted_index

INVERTED_BM25_AVAILABLE = True

__all__ = [
    "BM25",
    "BM25Index",
    "get_bm25_index",
    "get_bm25_inverted_index",
    "INVERTED_BM25_AVAILABLE",
]

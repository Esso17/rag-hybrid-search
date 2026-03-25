"""Text retrieval module (BM25-based keyword search)"""

from app.core.retrieval.bm25 import BM25, BM25Index, TechnicalTokenizer, get_bm25_index

try:
    from app.core.retrieval.bm25_inverted import BM25InvertedOptimized, get_bm25_inverted_index

    INVERTED_BM25_AVAILABLE = True
except ImportError:
    INVERTED_BM25_AVAILABLE = False
    BM25InvertedOptimized = None
    get_bm25_inverted_index = None

__all__ = [
    "TechnicalTokenizer",
    "BM25",
    "BM25Index",
    "get_bm25_index",
    "BM25InvertedOptimized",
    "get_bm25_inverted_index",
    "INVERTED_BM25_AVAILABLE",
]

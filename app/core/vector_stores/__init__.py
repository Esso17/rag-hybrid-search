"""Vector storage backends for RAG system"""

from app.core.vector_stores.in_memory import InMemoryVectorStore, get_in_memory_vector_store

try:
    from app.core.vector_stores.faiss import FAISS_AVAILABLE, get_faiss_vector_store
except ImportError:
    FAISS_AVAILABLE = False
    get_faiss_vector_store = None

__all__ = [
    "InMemoryVectorStore",
    "get_in_memory_vector_store",
    "get_faiss_vector_store",
    "FAISS_AVAILABLE",
]

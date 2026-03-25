"""Configuration settings for RAG system"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    # API
    API_TITLE: str = "RAG Hybrid Search API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # LLM (Local Ollama)
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "phi3.5:3.8b"  # Lightweight & fast (2.2GB, 3-4GB RAM)
    LLM_TEMPERATURE: float = 0.6  # Slightly lower for more focused responses

    # Embeddings
    EMBEDDING_MODEL: str = "nomic-embed-text"  # Via Ollama
    EMBEDDING_DIMENSION: int = 768  # nomic-embed-text produces 768-dim vectors

    # Search (Optimized for K8s/Cilium technical docs)
    HYBRID_SEARCH_ALPHA: float = 0.6  # Favor semantic search slightly for technical terms
    TOP_K_RESULTS: int = 7  # More context for complex queries

    # Document processing (Optimized for code-heavy docs)
    CHUNK_SIZE: int = 800  # Smaller chunks for precise retrieval
    CHUNK_OVERLAP: int = 200  # More overlap to preserve context
    USE_CODE_AWARE_SPLITTING: bool = True  # Preserve YAML/code blocks
    USE_ENHANCED_BM25: bool = True  # Technical tokenization

    # K8s/Cilium specific
    USE_DEVOPS_PROMPTS: bool = True  # Use specialized prompts for DevOps queries

    # Phase 3 Optimizations (100-1000x speedup)
    USE_FAISS: bool = False  # Use FAISS for fast vector search (requires: pip install faiss-cpu)
    USE_INVERTED_BM25: bool = False  # Use inverted index for BM25 (50-500x faster)
    FAISS_USE_HNSW: bool = True  # Use HNSW index (recommended for >1k docs)
    FAISS_M: int = 32  # HNSW connections per layer (16-64, higher=more accurate)
    FAISS_EF_CONSTRUCTION: int = 200  # HNSW build effort (100-500)
    FAISS_EF_SEARCH: int = 50  # HNSW search effort (16-512, higher=more accurate)

    # Caching (400x speedup on cache hits)
    USE_QUERY_RESPONSE_CACHE: bool = True  # Semantic cache for full query-response pairs
    CACHE_MAX_SIZE: int = 1000  # Maximum cached entries (LRU eviction)
    CACHE_SIMILARITY_THRESHOLD: float = 0.95  # Minimum cosine similarity for cache hit (0.95-0.99)
    CACHE_TTL_SECONDS: int = 3600  # Time-to-live: 1 hour (stale data prevention)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()

"""Configuration settings for RAG system"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API
    API_TITLE: str = "RAG Hybrid Search API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # LLM (Ollama, local)
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "phi3.5:3.8b"
    LLM_TEMPERATURE: float = 0.6

    # Embeddings
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIMENSION: int = 768

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    USE_CODE_AWARE_SPLITTING: bool = True

    # Search
    TOP_K_RESULTS: int = 7
    HYBRID_SEARCH_ALPHA: float = 0.6  # weight for vector vs BM25 in weighted fusion
    FUSION_METHOD: str = "rrf"  # "rrf" (recommended) | "weighted"
    USE_SEARCH_HEURISTICS: bool = True  # exact-match + overlap + quality boosts

    # FAISS HNSW (always on; M and ef_construction are set once at index creation)
    USE_FAISS: bool = True
    FAISS_USE_HNSW: bool = True
    FAISS_EF_SEARCH: int = 50  # search-time recall/speed trade-off

    # Prompts
    USE_DEVOPS_PROMPTS: bool = True

    # Retrieval enhancements
    USE_RERANKER: bool = False  # cross-encoder rerank (~50ms, +10–25% precision)
    USE_CONTEXTUAL_PREFIX: bool = True  # prepend [title] to chunks before embedding

    # Semantic response cache
    USE_QUERY_RESPONSE_CACHE: bool = True
    CACHE_MAX_SIZE: int = 1000
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    CACHE_TTL_SECONDS: int = 3600

    # Kubernetes documentation
    K8S_DOCS_BASE_URL: str = "https://kubernetes.io/docs"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


settings = Settings()

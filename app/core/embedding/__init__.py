"""Embedding module for generating text embeddings"""

from app.core.embedding.async_embedder import embed_batch_async
from app.core.embedding.cache import get_query_embedding
from app.core.embedding.client import OllamaEmbedding, get_embedding_client

__all__ = ["embed_batch_async", "get_query_embedding", "OllamaEmbedding", "get_embedding_client"]

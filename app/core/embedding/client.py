"""Embedding generation using Ollama"""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class OllamaEmbedding:
    """Ollama embedding client"""

    def __init__(
        self,
        base_url: str = settings.LLM_BASE_URL,
        model: str = settings.EMBEDDING_MODEL,
    ):
        self.base_url = base_url
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text"""
        try:
            response = self.client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    def close(self):
        """Close the HTTP client"""
        self.client.close()


# Singleton instance
_embedding_client = None


def get_embedding_client() -> OllamaEmbedding:
    """Get or create embedding client"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = OllamaEmbedding()
    return _embedding_client

"""
In-memory vector store as an alternative to Qdrant
Useful when Docker is not available
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    """Simple in-memory vector store using cosine similarity"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: dict[int, np.ndarray] = {}
        self.payloads: dict[int, dict] = {}
        self.next_id = 0

    def add_points(self, vectors: list[list[float]], payloads: list[dict]):
        """Add vectors and their metadata"""
        for vector, payload in zip(vectors, payloads):
            point_id = self.next_id
            self.vectors[point_id] = np.array(vector, dtype=np.float32)
            self.payloads[point_id] = payload
            self.next_id += 1

        logger.info(f"Added {len(vectors)} vectors to in-memory store")

    def search(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        """Search for similar vectors using cosine similarity"""
        if not self.vectors:
            return []

        query = np.array(query_vector, dtype=np.float32)

        # Compute cosine similarity for all vectors
        scores = []
        for point_id, vector in self.vectors.items():
            # Cosine similarity
            dot_product = np.dot(query, vector)
            norm_query = np.linalg.norm(query)
            norm_vector = np.linalg.norm(vector)

            if norm_query > 0 and norm_vector > 0:
                similarity = dot_product / (norm_query * norm_vector)
            else:
                similarity = 0.0

            scores.append(
                {
                    "id": point_id,
                    "score": float(similarity),
                    "payload": self.payloads[point_id],
                }
            )

        # Sort by score and return top-k
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:limit]

    def get_stats(self) -> dict:
        """Get store statistics"""
        return {
            "vector_count": len(self.vectors),
            "dimension": self.dimension,
            "total_points": self.next_id,
        }

    def clear(self):
        """Clear all data"""
        self.vectors.clear()
        self.payloads.clear()
        self.next_id = 0


# Singleton instance
_in_memory_vector_store = None


def get_in_memory_vector_store(dimension: int = 768) -> InMemoryVectorStore:
    """
    Get or create in-memory vector store (FALLBACK ONLY)

    This is used when Qdrant is not available.
    For production, use Qdrant instead (docker-compose up -d)
    """
    global _in_memory_vector_store
    if _in_memory_vector_store is None:
        _in_memory_vector_store = InMemoryVectorStore(dimension=dimension)
    return _in_memory_vector_store

"""
FAISS-based vector store for 100-1000x faster similarity search
Uses HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Install with: pip install faiss-cpu")


class FAISSVectorStore:
    """
    FAISS-based vector store with HNSW index

    Performance comparison (for 10k vectors):
    - Naive search (O(n)): ~100-200ms
    - FAISS HNSW: ~1-2ms
    Speedup: 100-200x faster

    At 100k vectors: 500-1000x faster
    """

    def __init__(
        self,
        dimension: int = 768,
        use_hnsw: bool = True,
        M: int = 32,  # noqa: N803
        ef_construction: int = 200,
    ):
        """
        Initialize FAISS vector store

        Args:
            dimension: Vector dimension (768 for nomic-embed-text)
            use_hnsw: Use HNSW index for fast approximate search (recommended)
            M: Number of connections per layer in HNSW (higher = more accurate but slower)
            ef_construction: Size of dynamic candidate list during index construction
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")

        self.dimension = dimension
        self.use_hnsw = use_hnsw
        self.payloads: dict[int, dict] = {}
        self.next_id = 0

        # Store parameters for clear() method
        self._M = M
        self._ef_construction = ef_construction

        # Create FAISS index
        if use_hnsw:
            # HNSW index for fast approximate search
            self.index = faiss.IndexHNSWFlat(dimension, M)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = 50  # Default search effort (adjustable)
            logger.info(
                f"FAISS HNSW index created: dimension={dimension}, M={M}, ef_construction={ef_construction}"
            )
        else:
            # Flat index for exact search (slower but 100% accurate)
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
            logger.info(f"FAISS Flat index created: dimension={dimension}")

        # ID mapping (FAISS uses sequential IDs, we need to map to our IDs)
        self.id_mapping: list[int] = []

    def add_points(self, vectors: list[list[float]], payloads: list[dict]):
        """Add vectors and their metadata"""
        if not vectors:
            return

        # Convert to numpy and normalize (for cosine similarity)
        vectors_np = np.array(vectors, dtype=np.float32)

        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vectors_normalized = vectors_np / norms

        # Add to FAISS index
        self.index.add(vectors_normalized)

        # Store payloads and update ID mapping
        for _i, payload in enumerate(payloads):
            point_id = self.next_id
            self.payloads[point_id] = payload
            self.id_mapping.append(point_id)
            self.next_id += 1

        logger.info(f"Added {len(vectors)} vectors to FAISS index (total: {self.index.ntotal})")

        # Auto-save every 100 vectors to persist data
        if self.index.ntotal % 100 < len(vectors):
            self.save()

    def search(
        self, query_vector: list[float], limit: int = 5, ef_search: Optional[int] = None
    ) -> list[dict]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding
            limit: Number of results to return
            ef_search: Search effort for HNSW (higher = more accurate but slower)
        """
        if self.index.ntotal == 0:
            return []

        # Normalize query vector
        query_np = np.array([query_vector], dtype=np.float32)
        norm = np.linalg.norm(query_np)
        if norm > 0:
            query_np = query_np / norm

        # Set search effort if specified (HNSW only)
        if ef_search is not None and self.use_hnsw:
            self.index.hnsw.efSearch = ef_search

        # Search
        k = min(limit, self.index.ntotal)
        distances, indices = self.index.search(query_np, k)

        # Convert results
        results = []
        for _i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.id_mapping):
                continue

            point_id = self.id_mapping[idx]
            results.append(
                {
                    "id": point_id,
                    "score": float(distance),  # Already normalized cosine similarity
                    "payload": self.payloads[point_id],
                }
            )

        return results

    def get_stats(self) -> dict:
        """Get store statistics"""
        stats = {
            "vector_count": self.index.ntotal,
            "dimension": self.dimension,
            "total_points": self.next_id,
            "index_type": "HNSW" if self.use_hnsw else "Flat",
        }

        if self.use_hnsw:
            stats["hnsw_M"] = self._M
            stats["hnsw_ef_search"] = self.index.hnsw.efSearch
            stats["hnsw_ef_construction"] = self._ef_construction

        return stats

    def clear(self):
        """Clear all data"""
        self.payloads.clear()
        self.id_mapping.clear()
        self.next_id = 0

        # Reset index - recreate with original parameters
        if self.use_hnsw:
            self.index = faiss.IndexHNSWFlat(self.dimension, self._M)
            self.index.hnsw.efConstruction = self._ef_construction
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

        logger.info("FAISS vector store cleared")

    def set_search_effort(self, ef_search: int):
        """
        Adjust search effort (HNSW only)

        Args:
            ef_search: Search effort (16-512)
                - Lower (16-32): Faster but less accurate
                - Default (50): Good balance
                - Higher (100-200): More accurate but slower
        """
        if self.use_hnsw:
            self.index.hnsw.efSearch = ef_search
            logger.info(f"FAISS HNSW efSearch set to {ef_search}")

    def save(self, data_dir: str = "/app/data"):
        """Save FAISS index and metadata to disk"""
        try:
            Path(data_dir).mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            index_path = os.path.join(data_dir, "faiss.index")
            faiss.write_index(self.index, index_path)

            # Save metadata
            metadata = {
                "payloads": self.payloads,
                "id_mapping": self.id_mapping,
                "next_id": self.next_id,
                "dimension": self.dimension,
                "use_hnsw": self.use_hnsw,
                "M": self._M,
                "ef_construction": self._ef_construction,
            }
            metadata_path = os.path.join(data_dir, "faiss_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            logger.info(f"FAISS index saved: {self.index.ntotal} vectors to {data_dir}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def load(self, data_dir: str = "/app/data") -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            index_path = os.path.join(data_dir, "faiss.index")
            metadata_path = os.path.join(data_dir, "faiss_metadata.json")

            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                logger.info("No saved FAISS index found")
                return False

            # Load FAISS index
            self.index = faiss.read_index(index_path)

            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)

            self.payloads = {int(k): v for k, v in metadata["payloads"].items()}
            self.id_mapping = metadata["id_mapping"]
            self.next_id = metadata["next_id"]

            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors from {data_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False


# Singleton instance
_faiss_vector_store = None


def get_faiss_vector_store(dimension: int = 768, use_hnsw: bool = True) -> FAISSVectorStore:
    """
    Get or create FAISS vector store

    Args:
        dimension: Vector dimension (768 for nomic-embed-text)
        use_hnsw: Use HNSW index for fast search (recommended for >1k vectors)
    """
    global _faiss_vector_store
    if _faiss_vector_store is None:
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS not available. Install with: pip install faiss-cpu\n"
                "Or for GPU support: pip install faiss-gpu"
            )
        _faiss_vector_store = FAISSVectorStore(dimension=dimension, use_hnsw=use_hnsw)
        # Try to load from disk
        _faiss_vector_store.load()
    return _faiss_vector_store

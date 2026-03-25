"""
Test core RAG modules with new structure
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEmbeddingModule:
    """Test embedding module"""

    def test_embedding_imports(self):
        """Test embedding module imports"""
        from app.core.embedding import embed_batch_async, get_embedding_client, get_query_embedding

        assert get_embedding_client is not None
        assert embed_batch_async is not None
        assert get_query_embedding is not None

    def test_embedding_client_creation(self):
        """Test embedding client can be created"""
        from app.core.embedding import get_embedding_client

        client = get_embedding_client()
        assert client is not None
        assert hasattr(client, "base_url")
        assert hasattr(client, "model")


class TestVectorStoresModule:
    """Test vector stores module"""

    def test_vector_stores_imports(self):
        """Test vector stores module imports"""
        from app.core.vector_stores import FAISS_AVAILABLE, get_in_memory_vector_store

        assert get_in_memory_vector_store is not None
        assert isinstance(FAISS_AVAILABLE, bool)

    def test_in_memory_store_creation(self):
        """Test in-memory store can be created"""
        from app.config import settings
        from app.core.vector_stores import get_in_memory_vector_store

        store = get_in_memory_vector_store(dimension=settings.EMBEDDING_DIMENSION)
        assert store is not None
        assert store.dimension == settings.EMBEDDING_DIMENSION
        assert hasattr(store, "add_points")
        assert hasattr(store, "search")

    def test_in_memory_store_operations(self):
        """Test basic in-memory store operations"""
        from app.config import settings
        from app.core.vector_stores import get_in_memory_vector_store

        dim = settings.EMBEDDING_DIMENSION
        store = get_in_memory_vector_store(dimension=dim)

        # Add some dummy vectors
        vectors = [[0.1] * dim, [0.2] * dim, [0.3] * dim]
        payloads = [
            {"content": "test1", "chunk_index": 0},
            {"content": "test2", "chunk_index": 1},
            {"content": "test3", "chunk_index": 2},
        ]
        store.add_points(vectors, payloads)

        # Search
        results = store.search([0.15] * dim, limit=2)
        assert len(results) == 2
        assert "payload" in results[0]
        assert "score" in results[0]


class TestRetrievalModule:
    """Test retrieval module (BM25)"""

    def test_retrieval_imports(self):
        """Test retrieval module imports"""
        from app.core.retrieval import BM25Index, TechnicalTokenizer, get_bm25_index

        assert get_bm25_index is not None
        assert TechnicalTokenizer is not None
        assert BM25Index is not None

    def test_bm25_index_creation(self):
        """Test BM25 index can be created"""
        from app.core.retrieval import get_bm25_index

        index = get_bm25_index()
        assert index is not None
        assert hasattr(index, "add_chunks")
        assert hasattr(index, "search")

    def test_technical_tokenizer(self):
        """Test technical tokenizer"""
        from app.core.retrieval import TechnicalTokenizer

        tokenizer = TechnicalTokenizer()

        # Test K8s terms
        tokens = tokenizer.tokenize("NetworkPolicy kubectl cilium-agent")
        assert "networkpolicy" in tokens or "network" in tokens
        assert "kubectl" in tokens
        assert "cilium" in tokens or "cilium-agent" in tokens


class TestSearchModule:
    """Test search module (enhanced fusion)"""

    def test_search_imports(self):
        """Test search module imports"""
        from app.core.search import (
            calculate_query_overlap,
            enhanced_fusion,
            has_exact_match,
            hybrid_search,
            reciprocal_rank_fusion,
        )

        assert hybrid_search is not None
        assert reciprocal_rank_fusion is not None
        assert enhanced_fusion is not None
        assert calculate_query_overlap is not None
        assert has_exact_match is not None

    def test_reciprocal_rank_fusion(self):
        """Test RRF score fusion"""
        from app.core.search import reciprocal_rank_fusion

        vector_scores = {0: 0.9, 1: 0.8, 2: 0.7}
        bm25_scores = {0: 5.0, 2: 4.0, 3: 3.0}

        rrf_scores = reciprocal_rank_fusion(vector_scores, bm25_scores)

        assert len(rrf_scores) == 4  # All unique indices
        assert 0 in rrf_scores  # Present in both
        assert 3 in rrf_scores  # Only in BM25
        assert all(score > 0 for score in rrf_scores.values())

    def test_query_overlap(self):
        """Test query-document overlap calculation"""
        from app.core.search import calculate_query_overlap

        overlap = calculate_query_overlap(
            "kubernetes pod networking", "Kubernetes uses CNI for pod networking and communication"
        )
        assert 0 <= overlap <= 1
        assert overlap > 0  # Should have some overlap

    def test_exact_match(self):
        """Test exact match detection"""
        from app.core.search import has_exact_match

        assert has_exact_match("kubernetes pod", "How to create a kubernetes pod in the cluster")
        assert not has_exact_match("kubernetes pod", "How to use K8s containers")


class TestIndexingModule:
    """Test indexing module"""

    def test_indexing_imports(self):
        """Test indexing module imports"""
        from app.core.indexing import (
            add_documents_batched,
            add_documents_parallel,
            process_document,
        )

        assert process_document is not None
        assert add_documents_parallel is not None
        assert add_documents_batched is not None


class TestGenerationModule:
    """Test generation module"""

    def test_generation_imports(self):
        """Test generation module imports"""
        from app.core.generation import generate_answer

        assert generate_answer is not None


class TestTextProcessingModule:
    """Test text processing module"""

    def test_text_processing_imports(self):
        """Test text processing module imports"""
        from app.core.text_processing import get_code_aware_splitter

        assert get_code_aware_splitter is not None


class TestRAGPipeline:
    """Test main RAG pipeline"""

    def test_rag_pipeline_creation(self):
        """Test RAG pipeline can be created"""
        from app.core.rag import get_rag_pipeline

        rag = get_rag_pipeline()
        assert rag is not None

    def test_rag_pipeline_has_enhanced_features(self):
        """Test RAG pipeline has enhanced fusion features"""
        from app.core.rag import get_rag_pipeline

        rag = get_rag_pipeline()

        # Check enhanced search methods
        assert hasattr(rag, "hybrid_search")
        assert hasattr(rag, "query")
        assert hasattr(rag, "add_document")
        assert hasattr(rag, "add_documents_parallel")
        assert hasattr(rag, "add_documents_batched")

    def test_rag_pipeline_attributes(self):
        """Test RAG pipeline has correct attributes"""
        from app.core.rag import get_rag_pipeline

        rag = get_rag_pipeline()

        assert hasattr(rag, "use_faiss")
        assert hasattr(rag, "use_code_aware")
        assert hasattr(rag, "use_devops_prompts")
        assert hasattr(rag, "text_splitter")
        assert hasattr(rag, "bm25_index")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

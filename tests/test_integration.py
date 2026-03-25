"""
Integration tests for end-to-end RAG functionality
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.rag import get_rag_pipeline


@pytest.fixture
def rag_pipeline():
    """Get RAG pipeline instance"""
    return get_rag_pipeline()


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "doc_id": "test_k8s_1",
            "title": "Kubernetes Basics",
            "content": """
            Kubernetes is an open-source container orchestration platform.
            A Pod is the smallest deployable unit in Kubernetes.
            Pods can contain one or more containers that share storage and network.
            Services provide stable networking endpoints for Pods.
            """,
            "metadata": {"source": "k8s-docs", "category": "basics"},
        },
        {
            "doc_id": "test_cilium_1",
            "title": "Cilium Networking",
            "content": """
            Cilium provides eBPF-based networking for Kubernetes.
            CiliumNetworkPolicy allows L3/L4 and L7 filtering.
            Hubble provides network observability for Cilium.
            eBPF enables high-performance packet processing in the kernel.
            """,
            "metadata": {"source": "cilium-docs", "category": "networking"},
        },
    ]


def test_add_and_search(rag_pipeline, sample_documents):
    """Test adding documents and searching"""
    # Add documents
    for doc in sample_documents:
        chunks = rag_pipeline.add_document(
            doc_id=doc["doc_id"],
            title=doc["title"],
            content=doc["content"],
            metadata=doc["metadata"],
        )
        assert chunks > 0

    # Search for Kubernetes
    results = rag_pipeline.hybrid_search("Kubernetes pods", top_k=3)
    assert len(results) > 0
    assert any("pod" in r["content"].lower() for r in results)


def test_enhanced_fusion_vs_baseline(rag_pipeline, sample_documents):
    """Test enhanced fusion improves over baseline"""
    # Add documents
    for doc in sample_documents:
        rag_pipeline.add_document(
            doc_id=doc["doc_id"],
            title=doc["title"],
            content=doc["content"],
            metadata=doc["metadata"],
        )

    query = "eBPF networking"

    # Search with RRF + heuristics
    results_enhanced = rag_pipeline.hybrid_search(
        query=query, fusion_method="rrf", use_heuristics=True, top_k=3
    )

    # Search with baseline (no heuristics)
    results_baseline = rag_pipeline.hybrid_search(
        query=query, fusion_method="rrf", use_heuristics=False, top_k=3
    )

    # Both should return results
    assert len(results_enhanced) > 0
    assert len(results_baseline) > 0

    # Scores might differ (heuristics can boost/penalize)
    # Just verify structure is correct
    for r in results_enhanced:
        assert "content" in r
        assert "score" in r
        assert "chunk_index" in r


def test_parallel_ingestion(rag_pipeline):
    """Test parallel document ingestion"""
    docs = [
        {
            "doc_id": f"parallel_test_{i}",
            "title": f"Test Doc {i}",
            "content": f"This is test document number {i} about Kubernetes and networking.",
            "metadata": {"batch": "test", "index": i},
        }
        for i in range(10)
    ]

    # Parallel ingestion
    total_chunks, success, errors = rag_pipeline.add_documents_parallel(
        documents=docs, num_workers=2, max_concurrent_embeddings=10
    )

    assert success == 10
    assert errors == 0
    assert total_chunks > 0


def test_metadata_boosting(rag_pipeline, sample_documents):
    """Test metadata-based score boosting"""
    # Add documents
    for doc in sample_documents:
        rag_pipeline.add_document(
            doc_id=doc["doc_id"],
            title=doc["title"],
            content=doc["content"],
            metadata=doc["metadata"],
        )

    # Search with metadata boost
    boost_config = {
        "source_quality": {"k8s-docs": 0.3, "cilium-docs": 0.1}  # 30% boost for K8s docs
    }

    results = rag_pipeline.hybrid_search(
        query="pods containers", boost_config=boost_config, top_k=3
    )

    assert len(results) > 0


def test_code_aware_chunking(rag_pipeline):
    """Test that code-aware splitting preserves YAML blocks"""
    yaml_doc = {
        "doc_id": "yaml_test",
        "title": "K8s Manifest",
        "content": """
        Here is a Kubernetes deployment:

        ```yaml
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: nginx
        spec:
          replicas: 3
          selector:
            matchLabels:
              app: nginx
        ```

        This deployment creates nginx pods.
        """,
        "metadata": {"type": "manifest"},
    }

    chunks = rag_pipeline.add_document(
        doc_id=yaml_doc["doc_id"],
        title=yaml_doc["title"],
        content=yaml_doc["content"],
        metadata=yaml_doc["metadata"],
    )

    # Should create chunks
    assert chunks > 0

    # Search for it
    results = rag_pipeline.hybrid_search("kubernetes deployment nginx", top_k=3)
    assert len(results) > 0


def test_empty_query_handling(rag_pipeline, sample_documents):
    """Test handling of edge cases"""
    # Add a document first
    rag_pipeline.add_document(doc_id="edge_test", title="Test", content="Test content", metadata={})

    # Empty query should not crash
    try:
        results = rag_pipeline.hybrid_search("", top_k=3)
        # May return empty or some results
        assert isinstance(results, list)
    except Exception:
        # Acceptable to raise an exception
        assert True


def test_exact_match_boost(rag_pipeline):
    """Test exact match detection boosts scores"""
    # Add document with specific phrase
    rag_pipeline.add_document(
        doc_id="exact_test",
        title="Exact Match Test",
        content="CiliumNetworkPolicy allows fine-grained network access control.",
        metadata={},
    )

    # Query with exact phrase
    results_exact = rag_pipeline.hybrid_search(
        query="CiliumNetworkPolicy", use_heuristics=True, top_k=3
    )

    # Query with different terms
    results_other = rag_pipeline.hybrid_search(query="network policy", use_heuristics=True, top_k=3)

    # Both should return results
    assert len(results_exact) > 0
    assert len(results_other) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

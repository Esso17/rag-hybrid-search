"""
Test API endpoints
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns correct info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data
    assert "features" in data


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "llm_available" in data
    assert "version" in data


def test_stats_endpoint():
    """Test stats endpoint"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "vector_backend" in data
    assert "vector_store" in data
    assert "bm25" in data
    assert "config" in data


def test_add_document():
    """Test adding a single document"""
    response = client.post(
        "/add-document",
        json={
            "title": "Test Document",
            "content": "This is a test document about Kubernetes networking and pod communication.",
            "metadata": {"source": "test"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert "chunk_count" in data
    assert data["status"] == "added"
    assert data["chunk_count"] > 0


def test_add_documents_batch():
    """Test batch document upload"""
    response = client.post(
        "/add-documents-batch",
        json={
            "documents": [
                {
                    "title": "Doc 1",
                    "content": "Kubernetes is a container orchestration platform for managing containerized applications.",
                    "metadata": {"source": "test1"},
                },
                {
                    "title": "Doc 2",
                    "content": "Cilium provides eBPF-based networking and security for Kubernetes clusters.",
                    "metadata": {"source": "test2"},
                },
            ],
            "num_workers": 2,
            "max_concurrent_embeddings": 10,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_documents"] == 2
    assert data["successful"] >= 0
    assert data["total_chunks"] >= 0
    assert "processing_time" in data


def test_search_basic():
    """Test basic search endpoint"""
    # First add a document
    client.post(
        "/add-document",
        json={
            "title": "Networking Guide",
            "content": "Kubernetes networking uses CNI plugins for pod-to-pod communication.",
            "metadata": {},
        },
    )

    # Then search
    response = client.post("/search", json={"query": "kubernetes networking", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert "total_results" in data


def test_search_enhanced():
    """Test enhanced search with fusion options"""
    response = client.post(
        "/search/enhanced",
        json={
            "query": "pod networking",
            "top_k": 5,
            "fusion_method": "rrf",
            "use_heuristics": True,
            "boost_config": {"exact_match_boost": 0.2},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data


def test_query_basic():
    """Test basic query endpoint (search + LLM)"""
    # Add a document first
    client.post(
        "/add-document",
        json={
            "title": "K8s Pods",
            "content": "A Pod is the smallest deployable unit in Kubernetes. It can contain one or more containers.",
            "metadata": {},
        },
    )

    # Note: This will fail if Ollama is not running, so we just check the endpoint exists
    response = client.post("/query", json={"query": "what is a pod?", "top_k": 3})
    # Could be 200 (success) or 500 (LLM not available)
    assert response.status_code in [200, 500]


def test_query_enhanced():
    """Test enhanced query endpoint"""
    response = client.post(
        "/query/enhanced",
        json={"query": "explain pods", "top_k": 3, "fusion_method": "rrf", "use_heuristics": True},
    )
    # Could be 200 (success) or 500 (LLM not available)
    assert response.status_code in [200, 500]


def test_upload_document(tmp_path):
    """Test file upload endpoint"""
    # Create a temporary text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test file for upload.")

    with open(test_file, "rb") as f:
        response = client.post(
            "/upload-document",
            files={"file": ("test.txt", f, "text/plain")},
            data={"title": "Uploaded Test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert "chunk_count" in data
    assert data["status"] == "uploaded"


def test_invalid_query():
    """Test that invalid queries are handled gracefully"""
    response = client.post("/search", json={"query": ""})  # Empty query
    # Should still return 200 with empty results or handle gracefully
    assert response.status_code in [200, 422, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

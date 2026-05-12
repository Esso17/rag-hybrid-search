"""
API endpoint tests using FastAPI TestClient.
No live Ollama or FAISS required — responses are validated structurally.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


# ── Infrastructure ─────────────────────────────────────────────────────────────


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "name" in data
    assert "endpoints" in data
    endpoints = data["endpoints"]
    assert "ingest" in endpoints
    assert "evaluate" in endpoints
    assert "feedback" not in endpoints


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "vector_store_connected" in data
    assert "llm_available" in data
    assert "version" in data


def test_stats():
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "vector_backend" in data
    assert "vector_store" in data
    assert "bm25" in data
    assert "cache" in data
    assert "config" in data
    cfg = data["config"]
    assert "fusion_method" in cfg
    assert "use_heuristics" in cfg


# ── Document ingestion ─────────────────────────────────────────────────────────


def test_ingest_single_doc():
    r = client.post(
        "/documents",
        json={
            "documents": [
                {
                    "title": "Test K8s Services",
                    "content": "A Kubernetes Service is an abstraction over a set of Pods providing stable networking.",
                    "metadata": {"source": "test"},
                }
            ]
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total_documents"] == 1
    assert data["successful"] == 1
    assert data["errors"] == 0
    assert data["total_chunks"] >= 1


def test_ingest_batch():
    r = client.post(
        "/documents",
        json={
            "documents": [
                {
                    "title": "Doc A",
                    "content": "Kubernetes NetworkPolicy controls ingress and egress traffic between pods.",
                },
                {
                    "title": "Doc B",
                    "content": "FAISS provides approximate nearest-neighbour search using HNSW graphs.",
                },
            ],
            "num_workers": 2,
            "max_concurrent_embeddings": 5,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total_documents"] == 2
    assert data["successful"] == 2
    assert data["total_chunks"] >= 2
    assert "processing_time" in data


def test_ingest_missing_content_field():
    r = client.post("/documents", json={"documents": [{"title": "No content"}]})
    assert r.status_code == 422


def test_upload_document(tmp_path):
    f = tmp_path / "k8s.txt"
    f.write_text("Kubernetes Deployments manage replica sets and rolling updates.")
    with open(f, "rb") as fp:
        r = client.post(
            "/documents/upload",
            files={"file": ("k8s.txt", fp, "text/plain")},
            data={"title": "K8s Deployments"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "uploaded"
    assert data["title"] == "K8s Deployments"
    assert data["chunk_count"] >= 1


# ── Search ─────────────────────────────────────────────────────────────────────


def test_search_default():
    r = client.post("/search", json={"query": "Kubernetes networking"})
    assert r.status_code == 200
    data = r.json()
    assert "query" in data
    assert "results" in data
    assert "total_results" in data


def test_search_with_fusion_options():
    r = client.post(
        "/search",
        json={
            "query": "Kubernetes ingress controller",
            "top_k": 3,
            "fusion_method": "weighted",
            "use_heuristics": False,
        },
    )
    assert r.status_code == 200
    assert r.json()["total_results"] <= 3


def test_search_result_schema():
    r = client.post("/search", json={"query": "pod networking", "top_k": 2})
    assert r.status_code == 200
    for result in r.json()["results"]:
        assert "content" in result
        assert "score" in result
        assert "document_id" in result
        assert "chunk_index" in result
        assert "source" in result


# ── Query (RAG) ────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_query_response_schema():
    r = client.post("/query", json={"query": "What is a Kubernetes Service?"})
    assert r.status_code == 200
    data = r.json()
    assert "query" in data
    assert "answer" in data
    assert "sources" in data
    assert "generation_time" in data
    assert "cache_hit" in data


@pytest.mark.slow
def test_query_cache_hit():
    q = {"query": "Explain Kubernetes StatefulSet test unique query string 42"}
    client.post("/query", json=q)  # prime cache
    r = client.post("/query", json=q)
    assert r.status_code == 200
    data = r.json()
    assert data["cache_hit"] is True
    assert data["generation_time"] < 0.1


@pytest.mark.slow
def test_query_fusion_options():
    r = client.post(
        "/query",
        json={
            "query": "How to debug a crashlooping pod?",
            "fusion_method": "rrf",
            "use_heuristics": True,
            "top_k": 3,
        },
    )
    assert r.status_code == 200
    assert "answer" in r.json()


# ── Compare ────────────────────────────────────────────────────────────────────


def test_query_compare_default():
    r = client.post("/query/compare", json={"query": "What is a Kubernetes Deployment?"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 3
    for result in data["results"]:
        assert "strategy_name" in result
        assert "answer" in result
        assert "latency_ms" in result
        assert "cache_hit" in result


def test_query_compare_custom():
    r = client.post(
        "/query/compare",
        json={
            "query": "Kubernetes pod lifecycle",
            "strategies": [
                {"name": "RRF", "use_hybrid": True, "fusion_method": "rrf", "use_heuristics": True},
                {"name": "Vector", "use_hybrid": False},
            ],
        },
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


# ── Agentic ────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_query_agentic_schema():
    r = client.post(
        "/query/agentic",
        json={
            "query": "How does Kubernetes handle rolling updates vs blue-green deployments?",
            "max_iterations": 1,
            "top_k": 3,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert isinstance(data["sub_questions"], list)
    assert "iterations" in data
    assert "final_complete" in data
    assert 0.0 <= data["final_confidence"] <= 1.0
    assert "generation_time" in data


def test_query_agentic_max_iterations_validation():
    r = client.post("/query/agentic", json={"query": "test", "max_iterations": 10})
    assert r.status_code == 422


# ── Evaluate ───────────────────────────────────────────────────────────────────


def test_evaluate_presupplied():
    r = client.post(
        "/evaluate",
        json={
            "query": "What is a Kubernetes Service?",
            "answer": "A Kubernetes Service provides stable networking to a set of Pods via label selectors.",
            "context": [
                "A Service is an abstraction that defines a logical set of Pods and a policy to access them.",
            ],
        },
    )
    assert r.status_code == 200
    data = r.json()
    m = data["metrics"]
    assert 0.0 <= m["faithfulness"] <= 1.0
    assert 0.0 <= m["answer_relevance"] <= 1.0
    assert 0.0 <= m["context_relevance"] <= 1.0
    assert 0.0 <= m["overall_score"] <= 1.0
    assert data["rag_time"] == 0.0
    assert "details" in data


def test_evaluate_end_to_end():
    r = client.post("/evaluate", json={"query": "What is FAISS HNSW?", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["rag_time"] > 0
    assert data["eval_time"] > 0
    assert "answer" in data


def test_evaluate_hallucination_detection():
    r = client.post(
        "/evaluate",
        json={
            "query": "What is a Kubernetes Pod?",
            "answer": "A Pod is a blockchain node using quantum networking on GPU clusters.",
            "context": [
                "A Pod is the smallest deployable unit in Kubernetes wrapping one or more containers."
            ],
        },
    )
    assert r.status_code == 200
    assert r.json()["metrics"]["faithfulness"] <= 0.5


# ── PageIndex pipeline ─────────────────────────────────────────────────────────


def test_ingest_pageindex_retrieval_method():
    r = client.post(
        "/documents",
        json={
            "documents": [
                {
                    "title": "K8s Services Guide",
                    "content": (
                        "# Kubernetes Services\n\n"
                        "A Service provides stable networking.\n\n"
                        "## ClusterIP\n\nDefault type, internal only.\n\n"
                        "## NodePort\n\nExposes on each node's IP.\n"
                    ),
                }
            ],
            "retrieval_method": "pageindex",
        },
    )
    assert r.status_code == 200
    data = r.json()
    # Response schema is always present; disk write may fail outside Docker (/app/data)
    assert data["total_documents"] == 1
    assert "successful" in data
    assert "errors" in data


def test_ingest_both_pipelines():
    r = client.post(
        "/documents",
        json={
            "documents": [
                {
                    "title": "Dual-indexed doc",
                    "content": "# Overview\n\nThis doc is indexed into both pipelines.",
                }
            ],
            "retrieval_method": "both",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total_documents"] == 1


@pytest.mark.slow
def test_query_pageindex_response_schema():
    r = client.post(
        "/query",
        json={"query": "What is a Kubernetes Service?", "retrieval_method": "pageindex"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["retrieval_method"] == "pageindex"
    assert "answer" in data
    assert "sources" in data
    assert "generation_time" in data
    for src in data["sources"]:
        assert src["source"] == "pageindex"
        assert "section_title" in src
        assert "node_id" in src


@pytest.mark.slow
def test_query_hybrid_response_schema():
    r = client.post(
        "/query",
        json={"query": "What is a ClusterIP service?", "retrieval_method": "hybrid"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["retrieval_method"] == "hybrid"
    for src in data["sources"]:
        assert src["source"] in ("hybrid", "vector", "bm25")


def test_query_invalid_retrieval_method():
    r = client.post(
        "/query",
        json={"query": "test", "retrieval_method": "unknown_method"},
    )
    assert r.status_code == 422


def test_benchmark_endpoint():
    r = client.post(
        "/benchmark",
        json={"queries": ["What is Kubernetes?", "How does NodePort work?"], "top_k": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert "summary" in data
    summary = data["summary"]
    assert "avg_latency_ms_hybrid" in summary
    assert "avg_latency_ms_pageindex" in summary
    assert summary["queries_run"] == 2
    for entry in data["results"]:
        assert "query" in entry
        assert "hybrid" in entry
        assert "pageindex" in entry
        assert "faster" in entry


# ── Removed endpoints must 404 ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/feedback"),
        ("GET", "/feedback"),
        ("POST", "/add-document"),
        ("POST", "/add-documents-batch"),
        ("POST", "/upload-document"),
        ("POST", "/search/enhanced"),
        ("POST", "/query/enhanced"),
        ("POST", "/pageindex/documents"),
        ("POST", "/pageindex/query"),
    ],
)
def test_removed_endpoints_return_404(method, path):
    if method.upper() == "GET":
        r = client.get(path)
    else:
        r = getattr(client, method.lower())(path, json={})
    assert r.status_code == 404

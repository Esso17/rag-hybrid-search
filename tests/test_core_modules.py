"""Unit tests for core modules — no live Ollama/FAISS required."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Config ─────────────────────────────────────────────────────────────────────


class TestConfig:
    def test_settings_load(self):
        from app.config import settings

        assert settings.EMBEDDING_DIMENSION == 768
        assert settings.FUSION_METHOD in ("rrf", "weighted")
        assert isinstance(settings.USE_RERANKER, bool)
        assert isinstance(settings.USE_CONTEXTUAL_PREFIX, bool)
        assert settings.CACHE_MAX_SIZE > 0
        assert 0.0 < settings.CACHE_SIMILARITY_THRESHOLD <= 1.0

    def test_no_dead_keys(self):
        """Removed config keys must not appear on settings."""
        from app.config import settings

        for dead in ("USE_ENHANCED_BM25", "USE_INVERTED_BM25", "FAISS_M", "FAISS_EF_CONSTRUCTION"):
            assert not hasattr(settings, dead), f"Dead key still present: {dead}"


# ── BM25 / Retrieval ───────────────────────────────────────────────────────────


class TestBM25:
    def test_imports(self):
        from app.core.retrieval import BM25, BM25Index, get_bm25_index

        assert BM25 is not None
        assert BM25Index is not None
        assert get_bm25_index is not None

    def test_tokenize_preserves_k8s_terms(self):
        from app.core.retrieval.bm25 import _tokenize

        tokens = _tokenize("NetworkPolicy kubectl StatefulSet")
        assert "networkpolicy" in tokens
        assert "kubectl" in tokens
        assert "statefulset" in tokens

    def test_tokenize_camelcase_split(self):
        from app.core.retrieval.bm25 import _tokenize

        tokens = _tokenize("HorizontalPodAutoscaler")
        assert "horizontal" in tokens or "horizontalpodautoscaler" in tokens

    def test_bm25_add_and_search(self):
        from app.core.retrieval.bm25 import BM25

        bm25 = BM25()
        bm25.add(["kubernetes pod networking service", "faiss vector search hnsw"])
        results = bm25.search("kubernetes service", top_k=2)
        assert len(results) > 0
        idxs = [idx for idx, _ in results]
        assert 0 in idxs  # first doc is more relevant

    def test_bm25_inverted_index_efficiency(self):
        """Inverted index: only candidate docs should be scored."""
        from app.core.retrieval.bm25 import BM25

        bm25 = BM25()
        bm25.add(["kubernetes deployment rollout"] * 50 + ["unrelated content xyz"] * 50)
        results = bm25.search("deployment rollout", top_k=5)
        # All results should be the k8s docs (indices 0-49)
        for idx, _ in results:
            assert idx < 50

    def test_bm25_index_add_and_search(self):
        from app.core.retrieval.bm25 import BM25Index

        index = BM25Index()
        index.add_chunks(
            ["Kubernetes Services provide stable networking", "Pods run containers"],
            metadatas=[{"title": "Services"}, {"title": "Pods"}],
        )
        results = index.search("Kubernetes Services networking", top_k=1)
        assert len(results) == 1
        assert results[0]["content"] == "Kubernetes Services provide stable networking"
        assert "score" in results[0]
        assert "metadata" in results[0]


# ── Vector Stores ──────────────────────────────────────────────────────────────


class TestVectorStores:
    def test_imports(self):
        from app.core.vector_stores import FAISS_AVAILABLE, get_in_memory_vector_store

        assert isinstance(FAISS_AVAILABLE, bool)
        assert get_in_memory_vector_store is not None

    def test_in_memory_add_and_search(self):
        from app.config import settings
        from app.core.vector_stores import get_in_memory_vector_store

        dim = settings.EMBEDDING_DIMENSION
        store = get_in_memory_vector_store(dimension=dim)
        vectors = [[float(i) / dim] * dim for i in range(3)]
        payloads = [{"content": f"doc{i}", "chunk_index": i} for i in range(3)]
        store.add_points(vectors, payloads)
        results = store.search(vectors[0], limit=2)
        assert len(results) == 2
        assert "payload" in results[0]
        assert "score" in results[0]


# ── Score Fusion ───────────────────────────────────────────────────────────────


class TestScoreFusion:
    def test_rrf_combines_both_sources(self):
        from app.core.search.score_fusion import reciprocal_rank_fusion

        v = {0: 0.9, 1: 0.8, 2: 0.7}
        b = {0: 5.0, 2: 4.0, 3: 3.0}
        scores = reciprocal_rank_fusion(v, b)
        assert len(scores) == 4
        assert all(s > 0 for s in scores.values())
        # doc 0 is in both → should rank highest
        assert scores[0] == max(scores.values())

    def test_rrf_union_of_keys(self):
        from app.core.search.score_fusion import reciprocal_rank_fusion

        scores = reciprocal_rank_fusion({1: 0.5}, {2: 0.5})
        assert 1 in scores and 2 in scores

    def test_query_overlap(self):
        from app.core.search.score_fusion import calculate_query_overlap

        overlap = calculate_query_overlap(
            "kubernetes pod networking",
            "Kubernetes uses CNI for pod networking and communication",
        )
        assert 0.0 <= overlap <= 1.0
        assert overlap > 0.0

    def test_exact_match(self):
        from app.core.search.score_fusion import has_exact_match

        assert has_exact_match("kubernetes pod", "How to create a kubernetes pod in the cluster")
        assert not has_exact_match("kubernetes pod", "How to use K8s containers")

    def test_normalize_and_combine(self):
        from app.core.search.score_fusion import normalize_and_combine_scores

        result = normalize_and_combine_scores({0: 0.9, 1: 0.5}, {0: 3.0, 2: 1.0})
        assert len(result) == 3
        assert all(0.0 <= v <= 1.0 for v in result.values())


# ── Text Processing ────────────────────────────────────────────────────────────


class TestTextProcessing:
    def test_code_aware_splitter(self):
        from app.core.text_processing import get_code_aware_splitter

        splitter = get_code_aware_splitter(chunk_size=200, chunk_overlap=20)
        text = (
            "Kubernetes Pods are the smallest deployable units. "
            "They wrap one or more containers that share storage and network.\n\n"
            "```yaml\napiVersion: v1\nkind: Pod\nmetadata:\n  name: my-pod\n```\n\n"
            "Services expose Pods via stable DNS names and IP addresses. "
            "ClusterIP, NodePort, and LoadBalancer are the three main types."
        )
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_splitter_preserves_code_blocks(self):
        from app.core.text_processing import get_code_aware_splitter

        splitter = get_code_aware_splitter(
            chunk_size=500, chunk_overlap=50, preserve_code_blocks=True
        )
        code = "```yaml\n" + "key: value\n" * 10 + "```"
        chunks = splitter.split_text(code)
        # At least one chunk should contain the yaml block intact
        assert any("yaml" in c for c in chunks)


# ── Embedding Cache ────────────────────────────────────────────────────────────


class TestEmbeddingCache:
    def test_normalize_query(self):
        from app.core.embedding.cache import normalize_query

        assert normalize_query("  Hello, World!  ") == "hello world"
        assert (
            normalize_query("kubectl get pods --all-namespaces")
            == "kubectl get pods --all-namespaces"
        )

    def test_same_query_same_hash(self):
        import hashlib

        from app.core.embedding.cache import normalize_query

        q1 = normalize_query("What is Kubernetes?")
        q2 = normalize_query("What is Kubernetes?")
        assert hashlib.sha256(q1.encode()).hexdigest() == hashlib.sha256(q2.encode()).hexdigest()

    def test_cache_info(self):
        from app.core.embedding.cache import get_embedding_cache_info

        info = get_embedding_cache_info()
        assert "max_size" in info
        assert "hit_rate" in info
        assert 0.0 <= info["hit_rate"] <= 1.0


# ── Query-Response Cache ───────────────────────────────────────────────────────


class TestQueryResponseCache:
    def test_cache_miss_on_empty(self):
        from app.core.cache.query_response_cache import QueryResponseCache

        cache = QueryResponseCache(dimension=4, max_size=10, similarity_threshold=0.95)
        assert cache.get([0.1, 0.2, 0.3, 0.4]) is None

    def test_cache_hit_exact(self):
        from app.core.cache.query_response_cache import QueryResponseCache

        cache = QueryResponseCache(dimension=4, max_size=10, similarity_threshold=0.90)
        emb = [1.0, 0.0, 0.0, 0.0]
        cache.put("test query", emb, "test answer")
        result = cache.get(emb)
        assert result is not None
        assert result["cache_hit"] is True
        assert result["response"] == "test answer"

    def test_cache_miss_dissimilar(self):
        from app.core.cache.query_response_cache import QueryResponseCache

        cache = QueryResponseCache(dimension=4, max_size=10, similarity_threshold=0.99)
        cache.put("q", [1.0, 0.0, 0.0, 0.0], "answer")
        assert cache.get([0.0, 1.0, 0.0, 0.0]) is None

    def test_cache_stats(self):
        from app.core.cache.query_response_cache import QueryResponseCache

        cache = QueryResponseCache(dimension=4, max_size=5)
        stats = cache.get_stats()
        assert "total_entries" in stats
        assert "semantic_enabled" in stats

    def test_lru_eviction(self):
        from app.core.cache.query_response_cache import QueryResponseCache

        cache = QueryResponseCache(dimension=4, max_size=3)
        for i in range(4):
            emb = [float(i), 0.0, 0.0, 0.0]
            cache.put(f"q{i}", emb, f"a{i}")
        assert len(cache.cache) == 3  # evicted oldest


# ── Evaluation ─────────────────────────────────────────────────────────────────


class TestEvaluation:
    def test_parse_fallback(self):
        """_parse must handle malformed LLM output gracefully."""
        from app.core.evaluation.evaluator import _parse

        score, reason = _parse('{"score": 0.85, "reason": "Good answer"}')
        assert score == pytest.approx(0.85)
        assert "Good" in reason

    def test_parse_regex_fallback(self):
        from app.core.evaluation.evaluator import _parse

        score, reason = _parse('score: 0.7 some text "reason": "ok"')
        assert 0.0 <= score <= 1.0

    def test_parse_out_of_range_clamped(self):
        from app.core.evaluation.evaluator import _parse

        score, _ = _parse('{"score": 1.5, "reason": ""}')
        assert score <= 1.0
        score, _ = _parse('{"score": -0.5, "reason": ""}')
        assert score >= 0.0


# ── RAG Pipeline ───────────────────────────────────────────────────────────────


class TestRAGPipeline:
    def test_pipeline_attributes(self):
        from app.core.rag import get_rag_pipeline

        rag = get_rag_pipeline()
        assert hasattr(rag, "hybrid_search")
        assert hasattr(rag, "query")
        assert hasattr(rag, "add_document")
        assert hasattr(rag, "add_documents_parallel")
        assert hasattr(rag, "vector_store")  # property
        assert hasattr(rag, "bm25_index")
        assert hasattr(rag, "use_faiss")
        assert hasattr(rag, "use_code_aware")
        assert hasattr(rag, "use_devops_prompts")

    def test_vector_store_property(self):
        from app.core.rag import get_rag_pipeline

        rag = get_rag_pipeline()
        store = rag.vector_store
        assert store is not None
        assert hasattr(store, "add_points")
        assert hasattr(store, "search")


# ── PageIndex Store ────────────────────────────────────────────────────────────


class TestPageIndexStore:
    MARKDOWN_DOC = """# Kubernetes Services

A Service provides stable networking.

## ClusterIP

Default type. Accessible only inside the cluster.

## NodePort

Exposes the Service on each node's IP at a static port.

### When to use NodePort

Use it for development or when you need direct node access.

## LoadBalancer

Provisions an external load balancer from the cloud provider.
"""

    def _fresh_store(self, tmp_path):
        from app.core.indexing.pageindex_store import PageIndexStore

        return PageIndexStore(data_dir=str(tmp_path))

    def test_add_document_returns_section_count(self, tmp_path):
        store = self._fresh_store(tmp_path)
        result = store.add_document("doc1", "K8s Services", self.MARKDOWN_DOC)
        assert result["doc_id"] == "doc1"
        assert result["sections"] >= 4  # root + 3 H2 + 1 H3

    def test_tree_builds_hierarchy(self, tmp_path):
        store = self._fresh_store(tmp_path)
        store.add_document("doc1", "K8s Services", self.MARKDOWN_DOC)
        doc = store.get_document_tree("doc1")
        tree = doc["tree"]
        assert len(tree) == 1  # single H1 root
        root = tree[0]
        assert root["title"] == "Kubernetes Services"
        assert root["node_id"] == "1"
        children = root["children"]
        assert len(children) == 3  # ClusterIP, NodePort, LoadBalancer
        assert children[1]["title"] == "NodePort"
        assert len(children[1]["children"]) == 1  # nested H3

    def test_node_ids_are_hierarchical(self, tmp_path):
        store = self._fresh_store(tmp_path)
        store.add_document("doc1", "K8s Services", self.MARKDOWN_DOC)
        doc = store.get_document_tree("doc1")
        root = doc["tree"][0]
        assert root["node_id"] == "1"
        assert root["children"][0]["node_id"] == "1.1"
        assert root["children"][1]["node_id"] == "1.2"
        assert root["children"][1]["children"][0]["node_id"] == "1.2.1"

    def test_get_section_content(self, tmp_path):
        store = self._fresh_store(tmp_path)
        store.add_document("doc1", "K8s Services", self.MARKDOWN_DOC)
        content = store.get_section_content("doc1", "1.2")
        assert content is not None
        assert "NodePort" in content

    def test_section_content_none_for_unknown_node(self, tmp_path):
        store = self._fresh_store(tmp_path)
        store.add_document("doc1", "K8s Services", self.MARKDOWN_DOC)
        assert store.get_section_content("doc1", "99.99") is None

    def test_section_content_none_for_unknown_doc(self, tmp_path):
        store = self._fresh_store(tmp_path)
        assert store.get_section_content("no_such_doc", "1") is None

    def test_flat_document_gets_fallback_root(self, tmp_path):
        store = self._fresh_store(tmp_path)
        store.add_document("doc2", "Plain Doc", "No headers here at all.")
        doc = store.get_document_tree("doc2")
        # The store prepends "# Plain Doc" so there is always a root
        assert len(doc["tree"]) == 1
        root = doc["tree"][0]
        assert root["node_id"] == "1"

    def test_save_and_load(self, tmp_path):
        from app.core.indexing.pageindex_store import PageIndexStore

        store = PageIndexStore(data_dir=str(tmp_path))
        store.add_document("doc1", "K8s Services", self.MARKDOWN_DOC)
        assert store.document_count() == 1

        store2 = PageIndexStore(data_dir=str(tmp_path))
        loaded = store2.load()
        assert loaded is True
        assert store2.document_count() == 1
        assert store2.get_document_tree("doc1") is not None

    def test_load_returns_false_when_no_file(self, tmp_path):
        from app.core.indexing.pageindex_store import PageIndexStore

        store = PageIndexStore(data_dir=str(tmp_path))
        assert store.load() is False

    def test_document_count(self, tmp_path):
        store = self._fresh_store(tmp_path)
        assert store.document_count() == 0
        store.add_document("a", "A", "# A\nContent A")
        store.add_document("b", "B", "# B\nContent B")
        assert store.document_count() == 2

    def test_get_all_trees(self, tmp_path):
        store = self._fresh_store(tmp_path)
        store.add_document("a", "A", "# A\nContent A")
        store.add_document("b", "B", "# B\nContent B")
        trees = store.get_all_trees()
        assert "a" in trees and "b" in trees


# ── PageIndex Retriever (parse-only) ──────────────────────────────────────────


class TestPageIndexRetrieverParsing:
    """Tests for the JSON parser — no LLM calls needed."""

    def _make_retriever(self):
        from unittest.mock import MagicMock

        from app.core.indexing.pageindex_store import PageIndexStore
        from app.core.retrieval.pageindex_retriever import PageIndexRetriever

        store = MagicMock(spec=PageIndexStore)
        return PageIndexRetriever(store=store, llm_url="http://localhost:11434", llm_model="test")

    def test_parse_clean_json(self):
        r = self._make_retriever()
        ids = r._parse_node_ids('{"relevant_nodes": ["1", "2.1", "3"]}')
        assert ids == ["1", "2.1", "3"]

    def test_parse_json_with_prose_prefix(self):
        r = self._make_retriever()
        ids = r._parse_node_ids('Here are the relevant sections: {"relevant_nodes": ["1.2"]}')
        assert ids == ["1.2"]

    def test_parse_empty_list(self):
        r = self._make_retriever()
        ids = r._parse_node_ids('{"relevant_nodes": []}')
        assert ids == []

    def test_parse_malformed_returns_empty(self):
        r = self._make_retriever()
        ids = r._parse_node_ids("I could not find relevant sections.")
        assert ids == []

    def test_serialise_tree_indentation(self):
        r = self._make_retriever()
        nodes = [
            {
                "node_id": "1",
                "title": "Intro",
                "line_start": 1,
                "line_end": 10,
                "children": [
                    {
                        "node_id": "1.1",
                        "title": "Background",
                        "line_start": 3,
                        "line_end": 10,
                        "children": [],
                    }
                ],
            }
        ]
        text = r._serialise_tree(nodes)
        assert "[1] Intro" in text
        assert "  [1.1] Background" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

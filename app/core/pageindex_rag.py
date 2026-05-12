"""
PageIndex RAG pipeline — vectorless, reasoning-based retrieval.

FULL PIPELINE COMPARISON
─────────────────────────────────────────────────────────────────────────────
 Hybrid RAG (existing)          │  PageIndex RAG (this file)
────────────────────────────────┼────────────────────────────────────────────
 Index time                     │  Index time
   chunk(800 chars, 200 overlap) │    parse markdown headers → section tree
   embed each chunk (nomic 768D) │    store lines + tree as JSON
   store in FAISS + BM25         │    no embeddings, no vector store
                                 │
 Query time                     │  Query time
   embed(query)                  │    serialise tree → compact TOC text
   FAISS ANN search              │    LLM call: "which sections answer X?"
   BM25 keyword search           │    parse node IDs from JSON response
   RRF fusion                    │    slice document lines by node ranges
   optional reranker             │    return section text as context
   LLM generation                │    LLM generation (same Ollama backend)
────────────────────────────────┼────────────────────────────────────────────
 Strength: speed, recall         │  Strength: relevance, structure awareness
 Weakness: similarity ≠ relevance│  Weakness: LLM call per doc at query time
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import time
from typing import Optional

import httpx

from app.config import settings
from app.core.indexing.pageindex_store import get_pageindex_store
from app.core.retrieval.pageindex_retriever import PageIndexRetriever

logger = logging.getLogger(__name__)


class PageIndexRAG:
    """Full RAG pipeline backed by PageIndex tree-navigation retrieval."""

    def __init__(self):
        self.store = get_pageindex_store()
        self.retriever = PageIndexRetriever(
            store=self.store,
            llm_url=settings.LLM_BASE_URL,
            llm_model=settings.LLM_MODEL,
        )
        self.llm_client = httpx.Client(timeout=120.0)

        if settings.USE_DEVOPS_PROMPTS:
            from app.prompts.devops_prompts import build_devops_prompt

            self.prompt_builder = build_devops_prompt
        else:
            self.prompt_builder = None

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_document(self, doc_id: str, title: str, content: str, metadata: dict = None) -> dict:
        """Index a single document into the PageIndex tree store."""
        return self.store.add_document(doc_id, title, content, metadata)

    def add_documents_parallel(self, documents: list[dict]) -> tuple[int, int, int]:
        """
        Bulk-index documents (sequential — tree building needs no I/O concurrency).
        Returns (total_sections, successful_docs, error_docs).
        """
        total_sections = 0
        successful = 0
        errors = 0

        for doc in documents:
            try:
                result = self.store.add_document(
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    content=doc["content"],
                    metadata=doc.get("metadata", {}),
                )
                total_sections += result.get("sections", 0)
                successful += 1
            except Exception as e:
                logger.error(f"PageIndex: failed to index '{doc.get('title')}': {e}")
                errors += 1

        return total_sections, successful, errors

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, query: str, top_k: Optional[int] = None) -> tuple[list[dict], str, dict]:
        """
        Full PageIndex pipeline:
          1. LLM scans each document's tree and picks relevant sections.
          2. Section text is sliced out by line range.
          3. Ollama generates an answer from the section content.

        Returns (search_results, answer, metadata).
        metadata keys: latency_ms, search_time_ms, cache_hit, retrieval_method
        """
        start_time = time.time()
        if top_k is None:
            top_k = settings.TOP_K_RESULTS

        search_start = time.time()
        search_results = self.retriever.retrieve(query, top_k=top_k)
        search_time_ms = (time.time() - search_start) * 1000

        context_chunks = [r["content"] for r in search_results]
        answer = self._generate_answer(query, context_chunks)

        latency_ms = (time.time() - start_time) * 1000
        return (
            search_results,
            answer,
            {
                "latency_ms": latency_ms,
                "search_time_ms": search_time_ms,
                "cache_hit": False,
                "retrieval_method": "pageindex",
            },
        )

    # ── Stats / lifecycle ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "documents": self.store.document_count(),
            "retrieval_method": "pageindex_tree_navigation",
        }

    def close(self):
        self.retriever.close()
        self.llm_client.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _generate_answer(self, query: str, context_chunks: list[str]) -> str:
        from app.core.generation.answer_generator import generate_answer

        return generate_answer(
            query=query,
            context_chunks=context_chunks,
            llm_client=self.llm_client,
            use_devops_prompts=settings.USE_DEVOPS_PROMPTS,
            prompt_builder=self.prompt_builder,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_pageindex_rag: Optional[PageIndexRAG] = None


def get_pageindex_rag() -> PageIndexRAG:
    global _pageindex_rag
    if _pageindex_rag is None:
        _pageindex_rag = PageIndexRAG()
    return _pageindex_rag

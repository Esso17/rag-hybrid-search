"""RAG pipeline with hybrid search, parallel processing, and K8s/Cilium optimizations"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import httpx

from app.config import settings
from app.core.cache import get_query_response_cache
from app.core.embedding import embed_batch_async, get_embedding_client, get_query_embedding
from app.core.retrieval import INVERTED_BM25_AVAILABLE, get_bm25_index, get_bm25_inverted_index
from app.core.vector_stores import (
    FAISS_AVAILABLE,
    get_faiss_vector_store,
    get_in_memory_vector_store,
)

logger = logging.getLogger(__name__)

# Thread pool for running async code from sync context
_rag_executor = ThreadPoolExecutor(max_workers=1)


class RAGPipeline:
    """Unified RAG pipeline with hybrid search"""

    def __init__(self):
        self.use_faiss = False  # FAISS for fast vector search
        self.in_memory_store = None  # In-memory fallback (testing)
        self.faiss_store = None  # FAISS vector store

        if settings.USE_FAISS and FAISS_AVAILABLE:
            try:
                self.faiss_store = get_faiss_vector_store(
                    dimension=settings.EMBEDDING_DIMENSION, use_hnsw=settings.FAISS_USE_HNSW
                )
                self.use_faiss = True
                logger.info(f"Using FAISS vector store (HNSW={settings.FAISS_USE_HNSW})")
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS: {e}")
                self.use_faiss = False
        elif settings.USE_FAISS and not FAISS_AVAILABLE:
            logger.warning("FAISS requested but not available. Install with: pip install faiss-cpu")

        if not self.use_faiss:
            self.in_memory_store = get_in_memory_vector_store(
                dimension=settings.EMBEDDING_DIMENSION
            )

        self.embedding_client = get_embedding_client()

        # Phase 3: Use inverted BM25 if enabled (50-500x faster)
        if settings.USE_INVERTED_BM25 and INVERTED_BM25_AVAILABLE:
            self.bm25_index = get_bm25_inverted_index()
            logger.info("Using BM25 with inverted index (50-500x faster)")
        else:
            self.bm25_index = get_bm25_index()

        # Text splitter - use code-aware if enabled
        self.use_code_aware = getattr(settings, "USE_CODE_AWARE_SPLITTING", True)
        if self.use_code_aware:
            try:
                from app.core.text_processing import get_code_aware_splitter

                self.text_splitter = get_code_aware_splitter(
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP,
                    preserve_code_blocks=True,
                )
                logger.info("Using code-aware text splitter")
            except ImportError:
                logger.warning("Code-aware splitter not available, using standard splitter")
                self.use_code_aware = False

        if not self.use_code_aware:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
            )
            logger.info("Using standard text splitter")

        # DevOps prompts - use if enabled
        self.use_devops_prompts = getattr(settings, "USE_DEVOPS_PROMPTS", True)
        if self.use_devops_prompts:
            try:
                from app.prompts.devops_prompts import build_devops_prompt

                self.prompt_builder = build_devops_prompt
                logger.info("Using DevOps-optimized prompts")
            except ImportError:
                logger.warning("DevOps prompts not available, using generic prompts")
                self.use_devops_prompts = False
                self.prompt_builder = None

        self._initialized = False
        self.llm_client = httpx.Client(timeout=120.0)

        # Query-response cache (semantic caching for 400x speedup on hits)
        self.use_cache = getattr(settings, "USE_QUERY_RESPONSE_CACHE", True)
        if self.use_cache:
            try:
                self.query_cache = get_query_response_cache()
                logger.info(
                    f"Query-response cache enabled: "
                    f"threshold={self.query_cache.similarity_threshold}, "
                    f"ttl={self.query_cache.ttl_seconds}s"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize query-response cache: {e}")
                self.use_cache = False
        else:
            self.query_cache = None

    def initialize(self):
        """Initialize vector database"""
        try:
            if self.use_faiss:
                logger.info("Using FAISS vector store (Phase 3 optimization)")
            else:
                logger.info("Using in-memory vector store")

            self._initialized = True
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise

    def add_document(self, doc_id: str, title: str, content: str, metadata: dict = None) -> int:
        """Add document to RAG system with optional code-aware chunking"""
        if not self._initialized:
            self.initialize()

        # Split document into chunks (code-aware if enabled)
        chunks = self.text_splitter.split_text(content)
        splitter_type = "code-aware" if self.use_code_aware else "standard"
        logger.info(f"Document '{title}' split into {len(chunks)} chunks ({splitter_type})")

        # Generate embeddings for chunks (async batch for better performance)
        def run_async_embeddings():
            """Helper to run async embeddings in a fresh event loop"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(embed_batch_async(chunks, max_concurrent=20))
            finally:
                loop.close()

        try:
            # Check if we're already in an event loop (e.g., FastAPI)
            asyncio.get_running_loop()
            # We're in a running loop, execute in thread pool
            future = _rag_executor.submit(run_async_embeddings)
            embeddings = future.result()
        except RuntimeError:
            # No running loop, execute directly
            embeddings = run_async_embeddings()

        # Store vectors based on selected backend
        payloads = []
        for idx, chunk in enumerate(chunks):
            payloads.append(
                {
                    "document_id": doc_id,
                    "chunk_index": idx,
                    "content": chunk,
                    "title": title,
                    "metadata": metadata or {},
                }
            )

        if self.use_faiss:
            self.faiss_store.add_points(embeddings, payloads)
        else:
            self.in_memory_store.add_points(embeddings, payloads)

        # Add to BM25 index
        self.bm25_index.add_chunks(chunks)

        return len(chunks)

    def add_documents_parallel(
        self,
        documents: list[dict],
        num_workers: int = 4,
        max_concurrent_embeddings: int = 20,
        progress_callback: Optional[Callable[[int, int, int, int], None]] = None,
        benchmark=None,
    ) -> tuple[int, int, int]:
        """
        Add multiple documents in parallel

        Args:
            documents: List of document dicts with doc_id, title, content, metadata
            num_workers: Number of parallel workers
            max_concurrent_embeddings: Max concurrent embedding requests per worker
            progress_callback: Optional callback(batch_num, total_batches, chunks, errors)
            benchmark: Optional benchmark object for timing

        Returns:
            Tuple of (total_chunks, successful_count, error_count)
        """
        if not self._initialized:
            self.initialize()

        from app.core.indexing.batch_indexer import add_documents_parallel as batch_add_parallel

        vector_store = self.faiss_store if self.use_faiss else self.in_memory_store

        return batch_add_parallel(
            documents=documents,
            text_splitter=self.text_splitter,
            vector_store=vector_store,
            bm25_index=self.bm25_index,
            use_faiss=self.use_faiss,
            num_workers=num_workers,
            max_concurrent_embeddings=max_concurrent_embeddings,
            progress_callback=progress_callback,
            benchmark=benchmark,
        )

    def add_documents_batched(
        self,
        documents: list[dict],
        batch_size: int = 10,
        num_workers: int = 4,
        max_concurrent_embeddings: int = 20,
        progress_callback: Optional[Callable[[int, int, int, int], None]] = None,
        benchmark=None,
    ) -> tuple[int, int, int]:
        """
        Add documents in batches with parallel processing for better progress tracking

        Args:
            documents: List of document dictionaries
            batch_size: Documents per batch
            num_workers: Number of parallel workers
            max_concurrent_embeddings: Max concurrent embedding requests
            progress_callback: Optional callback(batch_num, total_batches, chunks_so_far, errors_so_far)
            benchmark: Optional benchmark object

        Returns:
            Tuple of (total_chunks, successful_count, error_count)
        """
        if not self._initialized:
            self.initialize()

        from app.core.indexing.batch_indexer import add_documents_batched as batch_add_batched

        vector_store = self.faiss_store if self.use_faiss else self.in_memory_store

        return batch_add_batched(
            documents=documents,
            text_splitter=self.text_splitter,
            vector_store=vector_store,
            bm25_index=self.bm25_index,
            use_faiss=self.use_faiss,
            batch_size=batch_size,
            num_workers=num_workers,
            max_concurrent_embeddings=max_concurrent_embeddings,
            progress_callback=progress_callback,
            benchmark=benchmark,
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        fusion_method: str = "rrf",
        use_heuristics: bool = True,
        boost_config: Optional[dict] = None,
    ) -> list[dict]:
        """
        Perform hybrid search: combining BM25 + vector search with enhanced fusion

        Args:
            query: Search query
            top_k: Number of results to return
            fusion_method: "rrf" (recommended) or "weighted"
            use_heuristics: Enable quality/overlap/exact match heuristics
            boost_config: Optional metadata boost configuration

        Returns:
            List of search results
        """
        from app.core.search.hybrid_search import hybrid_search as perform_hybrid_search

        vector_store = self.faiss_store if self.use_faiss else self.in_memory_store

        return perform_hybrid_search(
            query=query,
            vector_store=vector_store,
            bm25_index=self.bm25_index,
            top_k=top_k,
            use_faiss=self.use_faiss,
            fusion_method=fusion_method,
            use_heuristics=use_heuristics,
            boost_config=boost_config,
        )

    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        """Generate answer using local LLM with optional DevOps prompts"""
        from app.core.generation.answer_generator import generate_answer as gen_answer

        return gen_answer(
            query=query,
            context_chunks=context_chunks,
            llm_client=self.llm_client,
            use_devops_prompts=self.use_devops_prompts,
            prompt_builder=self.prompt_builder if self.use_devops_prompts else None,
        )

    def query(
        self,
        query: str,
        use_hybrid: bool = True,
        top_k: int = None,
        fusion_method: str = "rrf",
        use_heuristics: bool = True,
        boost_config: Optional[dict] = None,
    ) -> tuple[list[dict], str, dict]:
        """
        Full RAG query: search and generate answer with semantic caching

        Args:
            query: Search query
            use_hybrid: Use hybrid search (True) or vector-only (False)
            top_k: Number of results
            fusion_method: "rrf" (recommended) or "weighted"
            use_heuristics: Enable quality/overlap heuristics
            boost_config: Optional metadata boost configuration

        Returns:
            Tuple of (search_results, generated_answer, metadata)
            metadata includes: cache_hit, latency_ms, search_time_ms
        """
        start_time = time.time()

        if top_k is None:
            top_k = settings.TOP_K_RESULTS

        # Phase 1: Check query-response cache (semantic similarity >0.95)
        query_embedding = get_query_embedding(query)
        cached_response = None

        if self.use_cache and self.query_cache:
            cached_response = self.query_cache.get(query_embedding)

            if cached_response:
                # Cache HIT: Skip entire pipeline (2050ms → <5ms)
                latency_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"Cache HIT: {latency_ms:.1f}ms (bypassed retrieval + generation), "
                    f"similarity={cached_response['similarity']:.4f}"
                )

                return (
                    [],
                    cached_response["response"],
                    {
                        "cache_hit": True,
                        "cache_similarity": cached_response["similarity"],
                        "latency_ms": latency_ms,
                        "search_time_ms": 0,
                    },
                )

        # Cache MISS: Run full pipeline
        search_start = time.time()

        # Search
        if use_hybrid:
            search_results = self.hybrid_search(
                query,
                top_k=top_k,
                fusion_method=fusion_method,
                use_heuristics=use_heuristics,
                boost_config=boost_config,
            )
        else:
            # Vector search only
            vector_store = self.faiss_store if self.use_faiss else self.in_memory_store

            if self.use_faiss:
                vector_results = vector_store.search(
                    query_embedding, limit=top_k, ef_search=settings.FAISS_EF_SEARCH
                )
            else:
                vector_results = vector_store.search(query_embedding, limit=top_k)

            search_results = [
                {
                    "content": r["payload"].get("content"),
                    "score": r["score"],
                    "chunk_index": r["payload"].get("chunk_index"),
                    "source": "vector",
                    "metadata": r["payload"].get("metadata", {}),
                }
                for r in vector_results
            ]

        search_time_ms = (time.time() - search_start) * 1000

        # Generate answer
        context_chunks = [r["content"] for r in search_results]
        answer = self.generate_answer(query, context_chunks)

        # Cache the query-response pair for future reuse
        if self.use_cache and self.query_cache:
            self.query_cache.put(query, query_embedding, answer)

        latency_ms = (time.time() - start_time) * 1000

        return (
            search_results,
            answer,
            {
                "cache_hit": False,
                "latency_ms": latency_ms,
                "search_time_ms": search_time_ms,
            },
        )

    def close(self):
        """Close connections"""
        self.embedding_client.close()
        self.llm_client.close()


# Singleton instance
_rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline with all optimizations enabled via config"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
        _rag_pipeline.initialize()
    return _rag_pipeline

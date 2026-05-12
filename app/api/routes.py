"""API routes for RAG system"""

import logging
import time
from pathlib import PurePosixPath
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.core.agentic_rag import AgenticRAGLoop
from app.core.evaluation import evaluate as run_evaluation
from app.core.pageindex_rag import get_pageindex_rag
from app.core.rag import get_rag_pipeline
from app.metrics import (
    agentic_iterations_histogram,
    agentic_queries_total,
    cache_hits_total,
    cache_misses_total,
    chunks_created_total,
    documents_ingested_total,
    evaluation_score_histogram,
    evaluation_total,
    generation_duration_seconds,
    search_duration_seconds,
)
from app.models.schemas import (
    AgenticRAGRequest,
    AgenticRAGResponse,
    BatchDocumentInput,
    BatchUploadResponse,
    BenchmarkEntry,
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkSummary,
    CompareRequest,
    CompareResponse,
    ComparisonResult,
    EnhancedQueryRequest,
    EvaluationMetrics,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    IterationLog,
    PageIndexResult,
    PipelineMetrics,
    QueryResponse,
    RAGResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _source_url(source: str, file_path: str) -> Optional[str]:
    if source.lower() != "kubernetes" or not file_path:
        return None
    path = PurePosixPath(file_path).with_suffix("")
    return f"{settings.K8S_DOCS_BASE_URL}/{path}/"


def _make_pi_result(r: dict, content_limit: int = 500) -> PageIndexResult:
    """Convert a PageIndex section dict to a PageIndexResult schema object."""
    return PageIndexResult(
        content=r["content"][:content_limit],
        score=r["score"],
        document_id=r["document_id"],
        title=r.get("title"),
        section_title=r.get("section_title", ""),
        node_id=r.get("node_id", ""),
    )


def _make_result(r: dict, content_limit: int = 200) -> SearchResult:
    meta = r.get("metadata") or {}
    source_name = meta.get("source", "")
    file_path = meta.get("file", "")
    return SearchResult(
        content=r["content"][:content_limit],
        score=r["score"],
        document_id=meta.get("document_id") or file_path or "unknown",
        chunk_index=r.get("chunk_index", 0),
        source=r.get("source", "hybrid"),
        title=meta.get("title") or meta.get("category") or None,
        url=_source_url(source_name, file_path),
        file_path=file_path or None,
    )


rag_pipeline = get_rag_pipeline()
pi_rag = get_pageindex_rag()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health"""
    vector_store_ok = False
    vector_store_type = "Unknown"
    llm_available = False

    try:
        if rag_pipeline.use_faiss:
            vector_store_ok = rag_pipeline.faiss_store.get_stats() is not None
            vector_store_type = "FAISS"
        else:
            vector_store_ok = rag_pipeline.in_memory_store.get_stats() is not None
            vector_store_type = "In-Memory"
    except Exception as e:
        logger.error(f"Vector store error: {e}")

    try:
        import httpx

        with httpx.Client() as client:
            llm_available = (
                client.get(f"{settings.LLM_BASE_URL}/api/tags", timeout=5.0).status_code == 200
            )
    except Exception as e:
        logger.error(f"LLM connection error: {e}")

    return HealthResponse(
        status="healthy" if vector_store_ok and llm_available else "degraded",
        vector_store_connected=vector_store_ok,
        vector_store_type=vector_store_type,
        llm_available=llm_available,
        llm_model=settings.LLM_MODEL,
        version=settings.API_VERSION,
    )


@router.post("/documents", response_model=BatchUploadResponse)
async def ingest_documents(batch: BatchDocumentInput):
    """
    Ingest documents into one or both retrieval pipelines.

    - `retrieval_method: "hybrid"` (default) — BM25 + vector index
    - `retrieval_method: "pageindex"` — hierarchical section tree (no embeddings)
    - `retrieval_method: "both"` — index into both simultaneously
    """
    try:
        start_time = time.time()
        doc_list = [
            {
                "doc_id": f"doc_{int(time.time() * 1000)}_{idx}",
                "title": doc.title,
                "content": doc.content,
                "metadata": doc.metadata or {},
            }
            for idx, doc in enumerate(batch.documents)
        ]

        total_chunks, successful, errors = 0, 0, 0
        method = batch.retrieval_method

        if method in ("hybrid", "both"):
            tc, sc, ec = rag_pipeline.add_documents_parallel(
                documents=doc_list,
                num_workers=batch.num_workers,
                max_concurrent_embeddings=batch.max_concurrent_embeddings,
            )
            total_chunks += tc
            successful = max(successful, sc)
            errors += ec

        if method in ("pageindex", "both"):
            tc, sc, ec = pi_rag.add_documents_parallel(doc_list)
            total_chunks += tc
            successful = max(successful, sc)
            errors += ec

        documents_ingested_total.inc(successful)
        chunks_created_total.inc(total_chunks)

        return BatchUploadResponse(
            total_documents=len(doc_list),
            successful=successful,
            errors=errors,
            total_chunks=total_chunks,
            processing_time=time.time() - start_time,
        )
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), title: str = Form(None)):
    """Upload a plain-text file as a document."""
    try:
        text = (await file.read()).decode("utf-8")
        doc_title = title or file.filename or "Untitled"
        doc_id = f"doc_{int(time.time() * 1000)}"
        chunk_count = rag_pipeline.add_document(
            doc_id=doc_id, title=doc_title, content=text, metadata={}
        )
        documents_ingested_total.inc()
        chunks_created_total.inc(chunk_count)
        return {
            "doc_id": doc_id,
            "title": doc_title,
            "chunk_count": chunk_count,
            "filename": file.filename,
            "status": "uploaded",
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=QueryResponse)
async def search(request: EnhancedQueryRequest):
    """
    Hybrid search (BM25 + vector) with RRF fusion.

    - `fusion_method`: `"rrf"` (default, recommended) or `"weighted"`
    - `use_heuristics`: enable exact-match + overlap + quality boosts (default true)
    - `boost_config`: optional per-metadata-field score multipliers
    """
    try:
        _t0 = time.perf_counter()
        results = rag_pipeline.hybrid_search(
            query=request.query,
            top_k=request.top_k or settings.TOP_K_RESULTS,
            fusion_method=request.fusion_method,
            use_heuristics=request.use_heuristics,
            boost_config=request.boost_config,
        )
        search_duration_seconds.labels(method="hybrid").observe(time.perf_counter() - _t0)

        return QueryResponse(
            query=request.query,
            results=[_make_result(r) for r in results],
            total_results=len(results),
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=RAGResponse)
async def query(request: EnhancedQueryRequest):
    """
    Full RAG: retrieve + LLM answer.

    - `retrieval_method: "hybrid"` (default) — BM25 + vector search, RRF fusion,
      semantic cache.  Use `fusion_method`, `use_heuristics`, `boost_config` to tune.
    - `retrieval_method: "pageindex"` — LLM reasons over the document section tree;
      no embeddings, no cache.  `fusion_method` / `use_heuristics` are ignored.
    """
    try:
        top_k = request.top_k or settings.TOP_K_RESULTS

        if request.retrieval_method == "pageindex":
            search_results, answer, metadata = pi_rag.query(
                query=request.query,
                top_k=top_k,
            )
            sources = [_make_pi_result(r) for r in search_results]
        else:
            search_results, answer, metadata = rag_pipeline.query(
                query=request.query,
                use_hybrid=request.use_hybrid,
                top_k=top_k,
                fusion_method=request.fusion_method,
                use_heuristics=request.use_heuristics,
                boost_config=request.boost_config,
            )
            sources = [_make_result(r) for r in search_results]

        if metadata.get("cache_hit"):
            cache_hits_total.inc()
        else:
            cache_misses_total.inc()
            generation_duration_seconds.observe(metadata["latency_ms"] / 1000)

        logger.info(
            f"Query: method={request.retrieval_method} "
            f"cache_hit={metadata.get('cache_hit', False)} "
            f"latency={metadata['latency_ms']:.1f}ms"
        )

        return RAGResponse(
            query=request.query,
            answer=answer,
            retrieval_method=request.retrieval_method,
            sources=sources,
            generation_time=metadata["latency_ms"] / 1000,
            cache_hit=metadata.get("cache_hit", False),
            cache_similarity=metadata.get("cache_similarity"),
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/compare", response_model=CompareResponse)
async def query_compare(request: CompareRequest):
    """
    A/B test different fusion strategies on the same query.
    Omit `strategies` to run the three defaults: RRF+heuristics, weighted, vector-only.
    """
    try:
        top_k = request.top_k or settings.TOP_K_RESULTS
        strategies = request.strategies or [
            {
                "name": "RRF + Heuristics (Recommended)",
                "use_hybrid": True,
                "fusion_method": "rrf",
                "use_heuristics": True,
            },
            {
                "name": "Weighted Average",
                "use_hybrid": True,
                "fusion_method": "weighted",
                "use_heuristics": False,
            },
            {"name": "Vector Only", "use_hybrid": False},
        ]

        results = []
        for strategy in strategies:
            t0 = time.time()
            search_results, answer, metadata = rag_pipeline.query(
                query=request.query,
                use_hybrid=strategy.get("use_hybrid", True),
                top_k=top_k,
                fusion_method=strategy.get("fusion_method", "rrf"),
                use_heuristics=strategy.get("use_heuristics", False),
                boost_config=strategy.get("boost_config"),
            )
            results.append(
                ComparisonResult(
                    strategy_name=strategy.get("name", "Unknown"),
                    answer=answer,
                    sources=[_make_result(r) for r in search_results],
                    latency_ms=(time.time() - t0) * 1000,
                    cache_hit=metadata.get("cache_hit", False),
                )
            )

        return CompareResponse(query=request.query, results=results)
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/agentic", response_model=AgenticRAGResponse)
async def query_agentic(request: AgenticRAGRequest):
    """
    Agentic RAG: decompose → retrieve → generate → self-check → gap-fill (up to `max_iterations`).
    Returns full iteration log with confidence score.
    """
    try:
        loop = AgenticRAGLoop(rag_pipeline, max_iterations=request.max_iterations)
        result = loop.run(query=request.query, top_k=request.top_k, use_hybrid=request.use_hybrid)

        agentic_queries_total.inc()
        agentic_iterations_histogram.observe(len(result["iterations"]))
        logger.info(
            f"Agentic: complete={result['final_complete']} "
            f"confidence={result['final_confidence']:.2f} "
            f"iterations={len(result['iterations'])} time={result['generation_time']:.2f}s"
        )

        return AgenticRAGResponse(
            query=request.query,
            answer=result["answer"],
            sources=[_make_result(r, content_limit=500) for r in result["sources"]],
            sub_questions=result["sub_questions"],
            iterations=[IterationLog(**it) for it in result["iterations"]],
            final_complete=result["final_complete"],
            final_confidence=result["final_confidence"],
            generation_time=result["generation_time"],
        )
    except Exception as e:
        logger.error(f"Agentic query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """
    Evaluate RAG output quality using LLM-as-judge (three metrics).

    - **faithfulness** — are all answer claims backed by the retrieved context? (detects hallucinations)
    - **answer_relevance** — does the answer actually address the question?
    - **context_relevance** — are the retrieved chunks useful for the question?
    - **overall_score** — weighted: `0.4 * faithfulness + 0.4 * answer_relevance + 0.2 * context_relevance`

    If `answer` is omitted the full RAG pipeline runs first, then the output is evaluated.
    Provide both `answer` and `context` to evaluate a pre-computed result without re-running retrieval.
    """
    import time as _time

    try:
        sources: list[SearchResult] = []
        rag_time = 0.0

        if request.answer and request.context is not None:
            # Evaluate pre-supplied answer+context directly
            answer = request.answer
            context_chunks = request.context
        else:
            # Run full RAG pipeline first
            _t0 = _time.perf_counter()
            search_results, answer, metadata = rag_pipeline.query(
                query=request.query,
                use_hybrid=True,
                top_k=request.top_k or settings.TOP_K_RESULTS,
            )
            rag_time = _time.perf_counter() - _t0
            context_chunks = [r["content"] for r in search_results]
            sources = [_make_result(r) for r in search_results]

        # Run evaluation
        _t1 = _time.perf_counter()
        result = run_evaluation(
            query=request.query,
            answer=answer,
            context_chunks=context_chunks,
            llm_client=rag_pipeline.llm_client,
        )
        eval_time = _time.perf_counter() - _t1

        evaluation_total.inc()
        evaluation_score_histogram.observe(result["overall_score"])
        logger.info(
            f"Evaluate: overall={result['overall_score']:.2f} "
            f"faith={result['faithfulness']:.2f} "
            f"rel={result['answer_relevance']:.2f} "
            f"ctx={result['context_relevance']:.2f} "
            f"rag={rag_time:.2f}s eval={eval_time:.2f}s"
        )

        return EvaluationResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            metrics=EvaluationMetrics(
                faithfulness=result["faithfulness"],
                answer_relevance=result["answer_relevance"],
                context_relevance=result["context_relevance"],
                overall_score=result["overall_score"],
            ),
            details=result["details"],
            rag_time=round(rag_time, 3),
            eval_time=round(eval_time, 3),
        )
    except Exception as e:
        logger.error(f"Evaluate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """System statistics: vector store, BM25 index, semantic cache."""
    try:
        if rag_pipeline.use_faiss:
            vector_stats = rag_pipeline.faiss_store.get_stats()
            backend = "FAISS"
        else:
            vector_stats = rag_pipeline.in_memory_store.get_stats()
            backend = "In-Memory"

        cache_stats = (
            rag_pipeline.query_cache.get_stats()
            if rag_pipeline.use_cache and rag_pipeline.query_cache
            else {"enabled": False}
        )

        return {
            "vector_backend": backend,
            "vector_store": vector_stats,
            "bm25": {"total_chunks": len(rag_pipeline.bm25_index.chunks)},
            "cache": cache_stats,
            "config": {
                "use_faiss": rag_pipeline.use_faiss,
                "use_code_aware_splitting": rag_pipeline.use_code_aware,
                "use_devops_prompts": rag_pipeline.use_devops_prompts,
                "use_query_response_cache": rag_pipeline.use_cache,
                "fusion_method": settings.FUSION_METHOD,
                "use_heuristics": settings.USE_SEARCH_HEURISTICS,
            },
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root():
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
        "endpoints": {
            "health": "GET  /health",
            "stats": "GET  /stats",
            "ingest": "POST /documents          (retrieval_method: hybrid|pageindex|both)",
            "upload": "POST /documents/upload",
            "search": "POST /search",
            "query": "POST /query              (retrieval_method: hybrid|pageindex)",
            "compare": "POST /query/compare",
            "agentic": "POST /query/agentic",
            "evaluate": "POST /evaluate",
            "benchmark": "POST /benchmark",
        },
    }


@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(request: BenchmarkRequest):
    """
    Run the same queries through both pipelines and compare side-by-side.

    Returns per-query latency and source counts for hybrid search and
    PageIndex, plus aggregate summary statistics.
    Add /evaluate calls after the benchmark if you want quality metrics too.
    """
    try:
        top_k = request.top_k or settings.TOP_K_RESULTS
        entries: list[BenchmarkEntry] = []

        for query in request.queries:
            # Hybrid
            t0 = time.time()
            h_results, h_answer, h_meta = rag_pipeline.query(
                query=query, use_hybrid=True, top_k=top_k
            )
            h_latency = (time.time() - t0) * 1000

            # PageIndex
            t1 = time.time()
            pi_results, pi_answer, pi_meta = pi_rag.query(query=query, top_k=top_k)
            pi_latency = (time.time() - t1) * 1000

            entries.append(
                BenchmarkEntry(
                    query=query,
                    hybrid=PipelineMetrics(
                        latency_ms=h_latency,
                        search_time_ms=h_meta.get("search_time_ms", 0),
                        num_sources=len(h_results),
                        answer=h_answer[:300],
                    ),
                    pageindex=PipelineMetrics(
                        latency_ms=pi_latency,
                        search_time_ms=pi_meta.get("search_time_ms", 0),
                        num_sources=len(pi_results),
                        answer=pi_answer[:300],
                    ),
                    faster="hybrid" if h_latency < pi_latency else "pageindex",
                )
            )

        n = len(entries)
        summary = BenchmarkSummary(
            avg_latency_ms_hybrid=sum(e.hybrid.latency_ms for e in entries) / n,
            avg_latency_ms_pageindex=sum(e.pageindex.latency_ms for e in entries) / n,
            avg_sources_hybrid=sum(e.hybrid.num_sources for e in entries) / n,
            avg_sources_pageindex=sum(e.pageindex.num_sources for e in entries) / n,
            queries_run=n,
        )
        return BenchmarkResponse(results=entries, summary=summary)
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

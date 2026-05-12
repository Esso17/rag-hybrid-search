"""Pydantic schemas for API request/response validation"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Optional metadata")


class SearchResult(BaseModel):
    content: str = Field(..., description="Result content (truncated to 200 chars)")
    score: float = Field(..., description="Relevance score")
    document_id: str = Field(..., description="Source document ID")
    chunk_index: int = Field(default=0, description="Chunk index within document")
    source: str = Field(default="hybrid", description="Search source: hybrid | vector | bm25")
    title: Optional[str] = Field(default=None, description="Document title")
    url: Optional[str] = Field(default=None, description="Source documentation URL")
    file_path: Optional[str] = Field(default=None, description="Relative file path")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(default=None, description="Number of results to return")
    use_hybrid: Optional[bool] = Field(default=True, description="Use hybrid search")


class QueryResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_results: int


class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchResult]
    generation_time: float = Field(..., description="Generation time in seconds")
    cache_hit: bool = Field(default=False)
    cache_similarity: Optional[float] = Field(default=None)


class HealthResponse(BaseModel):
    status: str
    vector_store_connected: bool
    vector_store_type: str
    llm_available: bool
    llm_model: str
    version: str


class IterationLog(BaseModel):
    iteration: int
    queries: list[str]
    chunks_used: int
    complete: bool
    confidence: float
    gaps: list[str]


class AgenticRAGRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(default=None, description="Number of results to return")
    max_iterations: int = Field(default=2, ge=1, le=3, description="Max agentic loop iterations")
    use_hybrid: bool = Field(default=True, description="Use hybrid search")


class BatchDocumentInput(BaseModel):
    documents: list[DocumentInput]
    num_workers: Optional[int] = 4
    max_concurrent_embeddings: Optional[int] = 20


class BatchUploadResponse(BaseModel):
    total_documents: int
    successful: int
    errors: int
    total_chunks: int
    processing_time: float


class EnhancedQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    use_hybrid: Optional[bool] = True
    fusion_method: Optional[str] = "rrf"
    use_heuristics: Optional[bool] = True
    boost_config: Optional[dict] = None


class CompareRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    strategies: Optional[list[dict]] = None


class ComparisonResult(BaseModel):
    strategy_name: str
    answer: str
    sources: list[SearchResult]
    latency_ms: float
    cache_hit: bool = False


class CompareResponse(BaseModel):
    query: str
    results: list[ComparisonResult]


class AgenticRAGResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchResult]
    sub_questions: list[str]
    iterations: list[IterationLog]
    final_complete: bool
    final_confidence: float
    generation_time: float


class EvaluationRequest(BaseModel):
    query: str = Field(..., description="The question to evaluate")
    answer: Optional[str] = Field(
        default=None, description="Pre-computed answer; if omitted the RAG pipeline runs first"
    )
    context: Optional[list[str]] = Field(
        default=None,
        description="Pre-computed context chunks; used only when answer is also provided",
    )
    top_k: Optional[int] = Field(
        default=None, description="top_k for retrieval when running the pipeline"
    )


class EvaluationMetrics(BaseModel):
    faithfulness: float = Field(..., description="0–1: answer claims grounded in context")
    answer_relevance: float = Field(..., description="0–1: answer addresses the question")
    context_relevance: float = Field(..., description="0–1: retrieved chunks are useful")
    overall_score: float = Field(..., description="Weighted: 0.4*faith + 0.4*rel + 0.2*ctx")


class EvaluationResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchResult]
    metrics: EvaluationMetrics
    details: dict = Field(
        default_factory=dict, description="Per-metric reasoning from the judge LLM"
    )
    rag_time: float = Field(..., description="RAG pipeline time (0 when answer was pre-supplied)")
    eval_time: float = Field(..., description="Evaluation time in seconds")

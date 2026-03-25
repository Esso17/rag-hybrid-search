"""Pydantic schemas for API request/response validation"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    """Input schema for adding a document"""

    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Optional metadata")


class SearchResult(BaseModel):
    """Single search result"""

    content: str = Field(..., description="Result content (truncated to 200 chars)")
    score: float = Field(..., description="Relevance score")
    document_id: str = Field(..., description="Source document ID")
    chunk_index: int = Field(default=0, description="Chunk index within document")
    source: str = Field(
        default="hybrid", description="Search source: 'hybrid', 'vector', or 'bm25'"
    )


class QueryRequest(BaseModel):
    """Input schema for search and query operations"""

    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(default=None, description="Number of results to return")
    use_hybrid: Optional[bool] = Field(
        default=True, description="Use hybrid search (vector + BM25)"
    )


class QueryResponse(BaseModel):
    """Response schema for search operations"""

    query: str = Field(..., description="Original query")
    results: list[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")


class RAGResponse(BaseModel):
    """Response schema for RAG query operations"""

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer from LLM")
    sources: list[SearchResult] = Field(..., description="Sources used for generation")
    generation_time: float = Field(..., description="Time taken to generate answer (seconds)")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint"""

    status: str = Field(..., description="Health status: 'healthy' or 'degraded'")
    vector_store_connected: bool = Field(..., description="Whether vector store is connected")
    vector_store_type: str = Field(..., description="Vector store type: 'In-Memory', 'FAISS', etc.")
    llm_available: bool = Field(..., description="Whether LLM is available")
    llm_model: str = Field(..., description="LLM model name")
    version: str = Field(..., description="API version")

"""Prometheus metrics for the RAG API."""

from prometheus_client import Counter, Histogram

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

cache_hits_total = Counter("cache_hits_total", "Semantic cache hits")
cache_misses_total = Counter("cache_misses_total", "Semantic cache misses")

search_duration_seconds = Histogram(
    "search_duration_seconds",
    "Hybrid search duration in seconds",
    ["method"],  # hybrid | vector | bm25
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

generation_duration_seconds = Histogram(
    "generation_duration_seconds",
    "LLM generation duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

documents_ingested_total = Counter("documents_ingested_total", "Documents ingested")
chunks_created_total = Counter("chunks_created_total", "Chunks created from documents")

agentic_queries_total = Counter("agentic_queries_total", "Agentic RAG queries run")
agentic_iterations_histogram = Histogram(
    "agentic_iterations",
    "Agentic iterations per query",
    buckets=[1.0, 2.0, 3.0],
)

evaluation_total = Counter("evaluation_total", "RAG evaluations run")
evaluation_score_histogram = Histogram(
    "evaluation_score",
    "Evaluation overall_score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

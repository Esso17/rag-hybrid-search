# RAG Hybrid Search

Q&A over K8s docs using two independent retrieval pipelines — hybrid (BM25 + FAISS HNSW) and PageIndex (LLM-guided tree navigation) — with cross-encoder reranking, semantic caching, and a local LLM via Ollama. Runs CPU-only — no GPU, no cloud, no per-query cost.

**Stack:** FastAPI · FAISS HNSW · BM25 · PageIndex · cross-encoder reranking · Ollama · Prometheus

---

## How it works

Two retrieval pipelines are available, selected per-request via `retrieval_method`:

```
Query
  │
  ├─ retrieval_method=hybrid ──────────────────────────────────────────────
  │    │
  │    ├─ Semantic cache check  ──hit──▶  Cached answer (< 5 ms)
  │    │         │ miss
  │    ├─ BM25 search  ┐
  │    ├─ Vector search┘ → RRF score fusion → Reranker (optional)
  │    │
  │    └─ LLM generation → Answer + sources
  │
  └─ retrieval_method=pageindex ───────────────────────────────────────────
       │  (no cache, no embeddings)
       ├─ Serialize document section tree → compact TOC text
       ├─ LLM: "which sections answer this query?" (one call per document)
       ├─ Slice document lines by returned node IDs
       │
       └─ LLM generation → Answer + sources
```

**No cache is checked on the PageIndex path.** Each PageIndex query pays the full per-document LLM cost every time. Use `retrieval_method="both"` at index time to load a document into both pipelines and benchmark them head-to-head.

**Agentic mode** (`POST /query/agentic`) adds an outer loop on top of hybrid retrieval: query decomposition → iterative retrieval → self-check → gap-filling, up to 3 iterations.

---

## Requirements

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- [Ollama](https://ollama.ai) installed and running on your host
- 4 GB RAM minimum (8 GB recommended)

---

## Quick start

```bash
# 1. Pull models (one-time, ~3 GB)
ollama pull phi3.5:3.8b
ollama pull nomic-embed-text

# 2. Start the API
docker compose up -d

# 3. Verify
curl http://localhost:8000/health
```

API is live at **http://localhost:8000** — interactive docs at **http://localhost:8000/docs**.

---

## Ingesting documents

### Upload a single file

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@my-doc.md" \
  -F "title=My Doc"
```

### Batch ingest a folder

```bash
# Via the ingest script (recommended for large folders)
pip install -r requirements.txt

python scripts/ingest_docs.py --source ./data/docs --name MyDocs

# Or via the API directly
curl -X POST http://localhost:8000/documents \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {"content": "...", "title": "Doc 1", "source": "local"}
    ],
    "retrieval_method": "hybrid"
  }'
```

`retrieval_method` accepts `"hybrid"` (BM25 + FAISS), `"pageindex"` (LLM tree navigation), or `"both"` (indexes into both simultaneously — required before using `/benchmark`).

The ingest script supports `--batch-size`, `--workers`, and `--concurrent` flags to control throughput. Use `--api` flag to send via HTTP instead of the direct pipeline.

---

## Querying

### Hybrid query (with semantic cache)

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How does X work?", "retrieval_method": "hybrid"}' | jq
```

Response:

```json
{
  "query": "How does X work?",
  "answer": "...",
  "retrieval_method": "hybrid",
  "sources": [{"title": "...", "content": "...", "score": 0.91, "source": "hybrid"}],
  "generation_time": 1.23,
  "cache_hit": false,
  "cache_similarity": null
}
```

### PageIndex query (structure-aware, no cache)

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How does X work?", "retrieval_method": "pageindex"}' | jq
```

PageIndex sources carry `section_title` and `node_id` instead of `chunk_index`. `cache_hit` is always `false` — the semantic cache is not used.

### Search only (no LLM)

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "pod scheduling", "top_k": 5}'
```

### Agentic RAG (complex multi-part questions)

Breaks the query into sub-questions, retrieves for each, self-checks the answer, and fills gaps iteratively.

```bash
curl -X POST http://localhost:8000/query/agentic \
  -H 'Content-Type: application/json' \
  -d '{"query": "How do I set up networking and storage for a stateful app?", "max_iterations": 3}'
```

Response includes `sub_questions`, `iterations` (per-iteration log), `final_confidence`, and `final_complete`.

### Compare strategies

A/B test different retrieval approaches on the same query.

```bash
curl -X POST http://localhost:8000/query/compare \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is a DaemonSet?", "strategies": ["hybrid", "vector", "bm25"]}'
```

### Evaluate answer quality (LLM-as-judge)

```bash
curl -X POST http://localhost:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is a ConfigMap?"}'
```

Returns three scores (0–1):

| Metric | What it measures |
|---|---|
| `faithfulness` | Are claims backed by retrieved context? (hallucination detection) |
| `answer_relevance` | Does the answer address the question? |
| `context_relevance` | Are the retrieved chunks useful? |
| `overall_score` | 0.4×faithfulness + 0.4×answer_relevance + 0.2×context_relevance |

---

## All endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API info + endpoint list |
| `GET` | `/health` | System health (LLM, vector store) |
| `GET` | `/stats` | Index size, cache stats, config |
| `POST` | `/documents` | Batch ingest — `retrieval_method`: `hybrid`, `pageindex`, or `both` |
| `POST` | `/documents/upload` | Single file upload (multipart) |
| `POST` | `/search` | Hybrid search — no LLM generation |
| `POST` | `/query` | Full RAG — `retrieval_method`: `hybrid` (cached) or `pageindex` (uncached) |
| `POST` | `/query/compare` | A/B test multiple strategies |
| `POST` | `/query/agentic` | Iterative agentic RAG (hybrid only) |
| `POST` | `/evaluate` | LLM-as-judge quality metrics |
| `POST` | `/benchmark` | Run both pipelines on the same query and compare scores |
| `GET` | `/metrics` | Prometheus metrics |

---

## Configuration

All settings are environment variables. Set them in `docker-compose.yml` or a `.env` file.

### LLM

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `phi3.5:3.8b` | Ollama model for generation |
| `LLM_BASE_URL` | `http://localhost:11434` | Ollama URL (`host.docker.internal` in Docker) |
| `LLM_TEMPERATURE` | `0.6` | Generation temperature (0–1) |

### Embeddings

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `EMBEDDING_DIMENSION` | `768` | Vector dimension (must match model) |

### Document processing

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `USE_CODE_AWARE_SPLITTING` | `true` | Preserve YAML/code block boundaries |
| `USE_CONTEXTUAL_PREFIX` | `true` | Prepend document title to chunks before embedding |

### Search & retrieval

| Variable | Default | Description |
|---|---|---|
| `TOP_K_RESULTS` | `7` | Results returned per query |
| `FUSION_METHOD` | `rrf` | `rrf` (reciprocal rank fusion) or `weighted` |
| `HYBRID_SEARCH_ALPHA` | `0.6` | Vector weight for weighted fusion (1-α = BM25 weight) |
| `USE_SEARCH_HEURISTICS` | `true` | Exact-match and overlap score boosts |
| `USE_FAISS` | `true` | FAISS vector index (disable for in-memory fallback) |
| `FAISS_USE_HNSW` | `true` | HNSW index (faster) vs flat (exact) |
| `FAISS_EF_SEARCH` | `50` | HNSW recall/speed trade-off at query time |

### Reranker

| Variable | Default | Description |
|---|---|---|
| `USE_RERANKER` | `false` | Cross-encoder reranking (adds ~50 ms, +10–25% precision) |

When enabled, the cross-encoder (`ms-marco-MiniLM-L6-v2`, 22 M params, ~67 MB) rescores candidates after the initial retrieval step. Downloaded automatically on first use.

### Semantic response cache (hybrid only)

| Variable | Default | Description |
|---|---|---|
| `USE_QUERY_RESPONSE_CACHE` | `true` | Enable semantic caching |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Min cosine similarity for a cache hit |
| `CACHE_TTL_SECONDS` | `3600` | Entry expiration (seconds) |
| `CACHE_MAX_SIZE` | `1000` | Max entries (LRU eviction after limit) |

Cache hits return in < 5 ms instead of ~2000 ms. **The semantic cache is only checked on the hybrid path.** PageIndex queries are never served from cache — each one runs a fresh LLM call per document. The cache persists to `data/query_response_cache.json` and is reloaded on restart.

### Prompts

| Variable | Default | Description |
|---|---|---|
| `USE_DEVOPS_PROMPTS` | `true` | DevOps/Kubernetes-optimised system prompt |

---

## Changing the LLM model

Any model available in Ollama works:

```bash
ollama pull mistral
```

Then set `LLM_MODEL=mistral` in `docker-compose.yml` and `docker compose up -d`.

---

## Prometheus metrics

Available at `GET /metrics`. Key metrics:

| Metric | Type | Description |
|---|---|---|
| `http_requests_total` | Counter | Requests by method, endpoint, status |
| `http_request_duration_seconds` | Histogram | Latency by endpoint |
| `cache_hits_total` / `cache_misses_total` | Counter | Cache effectiveness |
| `search_duration_seconds` | Histogram | Search time by strategy |
| `generation_duration_seconds` | Histogram | LLM generation time |
| `documents_ingested_total` | Counter | Documents ingested |
| `chunks_created_total` | Counter | Chunks created |
| `agentic_queries_total` | Counter | Agentic queries run |
| `agentic_iterations` | Histogram | Iterations per agentic query |
| `evaluation_total` | Counter | Evaluations run |
| `evaluation_score` | Histogram | Overall evaluation scores |

---

## Development

### Run without Docker

```bash
pip install -r requirements.txt
ollama serve  # in a separate terminal

uvicorn app.main:app --reload --port 8000
```

### Makefile targets

```bash
make install       # install Python deps
make models        # pull Ollama models
make up            # docker compose up -d
make down          # docker compose down
make logs-compose  # tail compose logs
make ingest        # ingest docs via API
make health        # quick health check
make stats         # index + cache stats
make lint          # ruff linter
make test          # unit + API tests (no live LLM)
make test-slow     # all tests including live Ollama
make test-unit     # unit tests only
make test-api      # API tests via TestClient
```

### Tests

```bash
# Fast tests (no Ollama required)
pytest tests/test_core_modules.py tests/test_api.py -v

# Include slow tests that need a live Ollama instance
pytest tests/ -v --slow
```

---

## License

MIT

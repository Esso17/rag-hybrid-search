# Scripts

Essential scripts for document ingestion and performance benchmarking.

---

## 📁 Available Scripts

### 1. Document Ingestion

**File:** `ingest_docs.py` (Unified ingestion script)

Supports both direct RAG pipeline and API-based ingestion with optimized settings.

```bash
# Quick start - Ingest Kubernetes docs
git clone https://github.com/kubernetes/website.git ~/k8s-website
python scripts/ingest_docs.py --source ~/k8s-website/content/en/docs

# Multiple sources at once
python scripts/ingest_docs.py \
  --source ~/k8s-website/content/en/docs --name "Kubernetes" \
  --source ~/cilium/Documentation --name "Cilium"

# Via API (useful for remote server)
python scripts/ingest_docs.py --source ~/docs --api --api-url http://remote:8000

# Conservative settings (prevent Ollama overload)
python scripts/ingest_docs.py --source ~/docs \
  --batch-size 5 --workers 1 --concurrent 2
```

**Options:**
- `--source` - Path to documentation (directory or file). Can be specified multiple times
- `--name` - Source name (e.g., 'Kubernetes', 'Cilium'). Defaults to directory name
- `--api` - Use API endpoint instead of direct pipeline (default: direct)
- `--api-url` - API URL (default: http://localhost:8000)
- `--batch-size` - Batch size (default: 5, optimized to prevent Ollama overload)
- `--workers` - Number of parallel workers (default: 1)
- `--concurrent` - Max concurrent embeddings (default: 2)

**Examples:**

```bash
# Kubernetes documentation
git clone https://github.com/kubernetes/website.git ~/k8s-website
python scripts/ingest_docs.py --source ~/k8s-website/content/en/docs --name "Kubernetes"

# Cilium documentation
git clone https://github.com/cilium/cilium.git ~/cilium
python scripts/ingest_docs.py --source ~/cilium/Documentation --name "Cilium"

# Both at once
python scripts/ingest_docs.py \
  --source ~/k8s-website/content/en/docs \
  --source ~/cilium/Documentation
```

---

### 2. Performance Benchmarking

#### Comprehensive Benchmark Suite

**File:** `benchmark_all.py` | **Guide:** [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)

Comprehensive benchmark suite testing all optimizations.

```bash
# Run all benchmarks
python scripts/benchmark_all.py

# Quick mode (faster, smaller datasets)
python scripts/benchmark_all.py --quick

# Specific scenario
python scripts/benchmark_all.py --scenario faiss

# Save results
python scripts/benchmark_all.py --output results.json
```

**Available Scenarios:**
- `async_embeddings` - Parallel vs sequential embedding
- `query_cache` - LRU cache effectiveness
- `incremental_bm25` - Incremental vs full rebuild
- `faiss` - FAISS vs in-memory search
- `bm25_inverted` - Inverted index vs standard BM25
- `all` - Run everything (default)

See **[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)** for detailed documentation.

---

#### Chunking Strategy Optimization

**File:** `benchmark_chunking.py`

Find optimal chunk size and overlap for your corpus through systematic testing.

```bash
# Full benchmark (all combinations)
python scripts/benchmark_chunking.py

# Quick mode (fewer combinations, faster)
python scripts/benchmark_chunking.py --quick

# Test specific configurations
python scripts/benchmark_chunking.py --sizes 512,800,1024 --overlaps 100,200

# Use fewer documents for faster testing
python scripts/benchmark_chunking.py --docs 20

# Save results
python scripts/benchmark_chunking.py --output my_chunking_results.json
```

**What it tests:**
- **Retrieval precision**: How well chunks capture query intent
- **Search latency**: Speed impact of different chunk sizes
- **Ingestion time**: Processing overhead for different configs
- **Query type analysis**: Performance on conceptual vs technical queries
- **Term coverage**: How well expected terms appear in results

**Output:**
- Best configuration for precision
- Best configuration for speed
- Best balanced configuration
- Precision by chunk size analysis
- Precision by overlap analysis
- Query type breakdown

**Example results:**
```
🏆 Best Precision:
  Chunk size: 800
  Overlap: 200
  Precision@5: 0.867
  Search latency: 42.3ms

⚡ Best Speed:
  Chunk size: 512
  Overlap: 100
  Search latency: 31.2ms
  Precision@5: 0.734

⚖️ Best Balance:
  Chunk size: 800
  Overlap: 200
  Precision@5: 0.867
  Search latency: 42.3ms
```

---

#### Caching Performance Analysis

**File:** `benchmark_caching.py`

Validate semantic caching performance claims.

```bash
# Full caching benchmark
python scripts/benchmark_caching.py

# Quick test
python scripts/benchmark_caching.py --quick

# Test specific threshold
python scripts/benchmark_caching.py --threshold 0.95
```

**What it tests:**
- Cache MISS: Full pipeline latency
- Cache HIT (exact): Exact match performance
- Cache HIT (semantic): Similar query matching
- Mixed workload: Real-world simulation
- False positive rate: Safety analysis

**Validated speedups:**
- 22,820x faster on cache hits
- 32-40% hit rate in production
- <0.1% false positive rate

---

#### Reranking & Fusion A/B Testing

**File:** `benchmark_reranking.py` | **Guide:** [RERANKING_AB_TESTING_GUIDE.md](../docs/RERANKING_AB_TESTING_GUIDE.md)

Prove that algorithmic fusion beats traditional weighted averaging.

```bash
# Full benchmark (all 6 strategies)
python scripts/benchmark_reranking.py

# Quick mode (3 strategies: RRF, Weighted, Vector-only)
python scripts/benchmark_reranking.py --quick

# Use LLM to judge answer quality
python scripts/benchmark_reranking.py --judge llm

# Test specific strategies
python scripts/benchmark_reranking.py \
  --strategies rrf_heuristics,weighted,vector_only

# Save results
python scripts/benchmark_reranking.py --output my_results.json
```

**Strategies compared:**
- **RRF + Heuristics** - Reciprocal Rank Fusion with quality/overlap/exact match heuristics
- **RRF Only** - Reciprocal Rank Fusion without heuristics
- **Weighted Average** - Traditional weighted combination (alpha=0.6)
- **Weighted + Heuristics** - Weighted fusion with heuristics
- **Vector Only** - Pure semantic search (no BM25)
- **BM25 Only** - Pure keyword search (no vector)

**Metrics measured:**
- Precision@5: Relevant documents in top-5
- Topic coverage: Expected terms found
- Quality score: Answer quality (rule-based or LLM judge)
- Latency: Speed comparison
- Category breakdown: Conceptual vs technical vs hybrid vs code queries

**Example results:**
```
Strategy                  Precision@5     Latency      Quality
----------------------------------------------------------------------
rrf_heuristics            0.867           42.3ms       0.823
weighted                  0.756           38.5ms       0.712
vector_only               0.712           28.1ms       0.678

✅ RRF+Heuristics: +11.1% better precision than weighted
```

**API Integration:**
You can also test strategies in real-time via API:

```bash
# Start server
python -m app.main

# Compare strategies side-by-side
curl -X POST http://localhost:8000/query/compare \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to configure NetworkPolicy?",
    "strategies": [
      {"name": "RRF", "fusion_method": "rrf", "use_heuristics": true},
      {"name": "Weighted", "fusion_method": "weighted"},
      {"name": "Vector", "use_hybrid": false}
    ]
  }'
```

---

## 🚀 Quick Start

### Complete Workflow

```bash
# 1. Clone documentation
git clone https://github.com/kubernetes/website.git ~/k8s-website
git clone https://github.com/cilium/cilium.git ~/cilium

# 2. Ingest K8s docs
python scripts/ingest_k8s_docs.py \
  --source ~/k8s-website/content/en/docs \
  --version 1.28

# 3. Ingest Cilium docs
python scripts/ingest_cilium_docs.py \
  --source ~/cilium/Documentation \
  --version 1.14

# 4. Run benchmarks
python scripts/benchmark_all.py
```

---

## 📊 Ingestion Progress

Scripts show real-time progress:

```
Loading K8s docs from: ~/k8s-website/content/en/docs
Version: 1.28
Loaded 150 documents
RAG pipeline initialized
Progress: 10/150 documents, 87 chunks
Progress: 20/150 documents, 174 chunks
...
============================================================
Ingestion Complete
Total documents: 150
Successfully ingested: 148
Errors: 2
Total chunks: 1342
============================================================
```

---

## ⚙️ Features

### Ingestion Scripts

- **Code-aware chunking** - Preserves YAML/code blocks
- **Technical tokenization** - Handles K8s/Cilium terms
- **Metadata extraction** - Auto-categorization
- **Parallel processing** - Fast async embeddings
- **Progress tracking** - Real-time status
- **Error handling** - Continues on failures

### Supported Formats

- `.md` - Markdown documentation
- `.yaml`, `.yml` - Configuration files
- `.rst` - ReStructuredText (Cilium docs)

---

## 🔧 Troubleshooting

**Import errors:**
```bash
pip install pyyaml
```

**Ollama not responding:**
```bash
ollama serve
```

**FAISS not available:**
```bash
pip install faiss-cpu
```

**No documents loaded:**
- Verify source path exists
- Check file permissions
- Ensure correct file extensions

---

## 📖 Documentation

- **[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)** - Detailed benchmark documentation
- **[../README.md](../README.md)** - Project overview
- **[../docs/QUICKSTART.md](../docs/QUICKSTART.md)** - Getting started guide

---

**Ready to ingest and benchmark!** 🚀

# RAG Hybrid Search - Operations & Deployment Guide

## Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Ingestion](#running-ingestion)
- [Deployment](#deployment)
- [Monitoring & Troubleshooting](#monitoring--troubleshooting)
- [Performance Tuning](#performance-tuning)

---

## Quick Start

### Prerequisites
- Python 3.9+
- Ollama running locally (for embeddings)
- Qdrant (optional, falls back to in-memory)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Run Ingestion (Parallel - Recommended)
```bash
python3 scripts/ingest_k8s_docs.py \
  --source data/docs/kubernetes \
  --parallel \
  --workers 8 \
  --max-concurrent-embeddings 25
```

**Expected time:** 3-5 minutes for 1,400+ documents (vs 20+ minutes sequential)

### 4. Start API Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test Query
```bash
curl http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I create a pod?", "top_k": 5}'
```

---

## Installation

### Local Development

#### 1. Clone Repository
```bash
git clone <repository-url>
cd rag-hybrid-search
```

#### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Install Ollama
```bash
# Mac
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull embedding model
ollama pull nomic-embed-text
```

#### 5. (Optional) Install Qdrant
```bash
# Docker
docker run -p 6333:6333 qdrant/qdrant

# Or use in-memory mode (set USE_QDRANT=false in .env)
```

---

## Configuration

### Environment Variables

Create `.env` file:

```bash
# LLM Configuration
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text
LLM_TEMPERATURE=0.7

# Vector Store
USE_QDRANT=true
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=k8s_cilium_docs
FALLBACK_TO_MEMORY=true

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_DIMENSION=768

# Search
TOP_K_RESULTS=5
HYBRID_SEARCH_ALPHA=0.7

# Performance Features
USE_CODE_AWARE_SPLITTING=true
USE_DEVOPS_PROMPTS=true
```

### Key Settings Explained

**USE_QDRANT** - Use Qdrant vector DB (recommended for production)
- `true`: Persistent vector storage
- `false`: In-memory (testing only)

**FALLBACK_TO_MEMORY** - Fallback if Qdrant unavailable
- `true`: Automatically use in-memory if Qdrant fails
- `false`: Fail if Qdrant unavailable

**HYBRID_SEARCH_ALPHA** - Weight between vector and BM25
- `0.7`: 70% vector, 30% keyword (recommended)
- `1.0`: Pure vector search
- `0.0`: Pure keyword search

**USE_CODE_AWARE_SPLITTING** - Smart code block handling
- `true`: Preserves code blocks, YAML, JSON
- `false`: Standard text splitting

---

## Running Ingestion

### Sequential Mode (Testing)
```bash
python3 scripts/ingest_k8s_docs.py \
  --source data/docs/kubernetes \
  --version 1.29 \
  --component kubernetes \
  --batch-size 10
```

**Use for:**
- Small datasets (<100 docs)
- Testing and debugging
- First-time validation

**Performance:** ~1.3 docs/sec (~20 minutes for 1,500 docs)

### Parallel Mode (Production) ⚡ RECOMMENDED
```bash
python3 scripts/ingest_k8s_docs.py \
  --source data/docs/kubernetes \
  --version 1.29 \
  --parallel \
  --workers 8 \
  --max-concurrent-embeddings 25 \
  --batch-size 10
```

**Use for:**
- Production ingestion
- Large datasets (>100 docs)
- Time-sensitive updates

**Performance:** ~7-10 docs/sec (~3-5 minutes for 1,500 docs)
**Speedup:** 5-6x faster than sequential

### Configuration Options

**--workers N** - Number of parallel document processors
- 4: Conservative, stable
- 8: Recommended (sweet spot)
- 16: High-performance systems only

**--max-concurrent-embeddings N** - Concurrent API requests per worker
- 10-15: Slow/rate-limited APIs
- 20-25: Local Ollama (recommended)
- 30-40: High-performance local Ollama

**--batch-size N** - Documents per progress update
- 5: More frequent updates
- 10: Standard (recommended)
- 20: Less frequent updates

### Ingestion for Different Sources

#### Kubernetes Docs
```bash
python3 scripts/ingest_k8s_docs.py \
  --source data/docs/kubernetes \
  --version 1.29 \
  --parallel \
  --workers 8
```

#### Cilium Docs
```bash
python3 scripts/ingest_cilium_docs.py \
  --source data/docs/cilium \
  --version 1.15 \
  --parallel \
  --workers 8
```

### What Happens During Ingestion

1. **Loading**: Documents loaded from source
2. **Filtering**: Empty `*_index.md` files removed (160 files typically)
3. **Chunking**: Documents split into ~1000 char chunks
4. **Embedding**: Each chunk converted to 768-dim vector
5. **Indexing**: Vectors stored in Qdrant/memory, BM25 index built
6. **Reporting**: Benchmark and error reports generated

### Output Files

**Benchmarks** (auto-generated):
```
benchmarks/
└── k8s_ingestion_parallel_20260322_120537.json
```

**Error Reports** (if errors occur):
```
error_reports/k8s/
└── ingestion_errors_20260322_105316.json
```

### Analyzing Results

#### View Benchmark
```bash
python3 scripts/compare_benchmarks.py benchmarks/*.json
```

Output:
```
Mode                  Workers    Docs/sec  Chunks/sec       Time    Speedup
--------------------------------------------------------------------------------
sequential                  1        1.26       21.09   0:21:05   baseline
parallel                    8        7.50      127.50   0:03:30      5.95x
```

#### View Errors
```bash
python3 scripts/analyze_errors.py error_reports/k8s/ingestion_errors_*.json
```

---

## Deployment

### Docker Deployment

#### 1. Build Image
```bash
docker build -t rag-hybrid-search:latest .
```

#### 2. Run Container
```bash
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  -e LLM_BASE_URL=http://host.docker.internal:11434 \
  -e QDRANT_URL=http://qdrant:6333 \
  -v $(pwd)/data:/app/data \
  rag-hybrid-search:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLM_BASE_URL=http://host.docker.internal:11434
      - QDRANT_URL=http://qdrant:6333
      - USE_QDRANT=true
    depends_on:
      - qdrant
    volumes:
      - ./data:/app/data

volumes:
  qdrant_data:
```

Run:
```bash
docker-compose up -d
```

### Kubernetes Deployment

#### 1. Deploy Qdrant
```bash
kubectl apply -f k8s/qdrant-deployment.yaml
```

#### 2. Deploy RAG API
```bash
kubectl apply -f k8s/rag-api-deployment.yaml
```

#### 3. Expose Service
```bash
kubectl apply -f k8s/rag-api-service.yaml
```

#### 4. Verify
```bash
kubectl get pods
kubectl logs -f deployment/rag-api
```

### Environment-Specific Configs

#### Development
```bash
# .env.development
USE_QDRANT=false
FALLBACK_TO_MEMORY=true
DEBUG=true
```

#### Production
```bash
# .env.production
USE_QDRANT=true
QDRANT_URL=http://qdrant.prod.svc.cluster.local:6333
FALLBACK_TO_MEMORY=false
DEBUG=false
```

---

## Monitoring & Troubleshooting

### Health Checks

#### API Health
```bash
curl http://localhost:8000/health
```

#### Ollama Status
```bash
curl http://localhost:11434/api/tags
```

#### Qdrant Status
```bash
curl http://localhost:6333/collections
```

### Common Issues

#### "Module not found" Errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "from app.core.rag import get_rag_pipeline; print('✓ OK')"
```

#### Ollama Connection Errors
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start if needed
ollama serve

# Pull models
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

#### Qdrant Connection Errors
```bash
# Check Qdrant is running
docker ps | grep qdrant

# Start if needed
docker run -d -p 6333:6333 qdrant/qdrant

# Or use fallback mode
# In .env: FALLBACK_TO_MEMORY=true
```

#### Slow Ingestion
- **Use parallel mode**: Add `--parallel --workers 8`
- **Increase concurrency**: `--max-concurrent-embeddings 25`
- **Check Ollama performance**: Monitor CPU usage
- **Reduce batch size**: Try `--batch-size 5`

#### Out of Memory
- **Reduce workers**: Try `--workers 4`
- **Reduce concurrency**: `--max-concurrent-embeddings 10`
- **Process in batches**: Split source documents

### Logs and Debugging

#### View Logs
```bash
# Ingestion logs
tail -f ingestion.log

# API logs
tail -f api.log
```

#### Enable Debug Mode
```python
# In .env
DEBUG=true
LOG_LEVEL=DEBUG
```

#### Check Error Reports
```bash
# List recent errors
ls -lt error_reports/k8s/

# Analyze specific report
python3 scripts/analyze_errors.py error_reports/k8s/ingestion_errors_*.json
```

---

## Performance Tuning

### Optimal Configurations

#### Small Dataset (<100 docs)
```bash
python3 scripts/ingest_k8s_docs.py \
  --source data/docs \
  --parallel \
  --workers 4 \
  --max-concurrent-embeddings 15
```
**Expected:** ~2-3x speedup

#### Medium Dataset (100-1000 docs)
```bash
python3 scripts/ingest_k8s_docs.py \
  --source data/docs \
  --parallel \
  --workers 8 \
  --max-concurrent-embeddings 25
```
**Expected:** ~5-6x speedup (RECOMMENDED)

#### Large Dataset (>1000 docs)
```bash
python3 scripts/ingest_k8s_docs.py \
  --source data/docs \
  --parallel \
  --workers 16 \
  --max-concurrent-embeddings 40
```
**Expected:** ~7-10x speedup

### Tuning Guidelines

**1. Start Conservative**
- 4 workers, 15 concurrent embeddings
- Monitor CPU and memory usage
- Check error rates

**2. Scale Gradually**
- Increase to 8 workers if stable
- Increase concurrency to 25
- Verify no errors

**3. Monitor Performance**
```bash
# Watch system resources
htop

# Check Ollama performance
curl http://localhost:11434/api/ps
```

**4. Find Your Sweet Spot**
- Too many workers = resource contention
- Too many concurrent requests = API overload
- Balance based on your system

### Expected Performance

| Workers | Concurrent | Docs/sec | Time (1500 docs) | Speedup |
|---------|-----------|----------|------------------|---------|
| 1       | 1         | 1.3      | ~20 min          | 1x      |
| 4       | 15        | 5.0      | ~5 min           | 3-4x    |
| 8       | 25        | 7.5      | ~3.5 min         | 5-6x    |
| 16      | 40        | 12.0     | ~2 min           | 8-10x   |

---

## API Usage

### Search Endpoint
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I create a Kubernetes pod?",
    "top_k": 5,
    "use_hybrid": true
  }'
```

### Add Document
```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Document",
    "content": "Document content here...",
    "metadata": {
      "source": "manual",
      "version": "1.0"
    }
  }'
```

### Query with Answer Generation
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain Kubernetes deployments",
    "use_hybrid": true,
    "top_k": 3
  }'
```

---

## Maintenance

### Updating Documents
```bash
# Re-run ingestion with --parallel
python3 scripts/ingest_k8s_docs.py \
  --source data/docs/kubernetes \
  --version 1.30 \
  --parallel \
  --workers 8
```

### Backup Qdrant Data
```bash
# Docker volume backup
docker run --rm -v qdrant_data:/data -v $(pwd):/backup ubuntu tar cvf /backup/qdrant-backup.tar /data

# Restore
docker run --rm -v qdrant_data:/data -v $(pwd):/backup ubuntu tar xvf /backup/qdrant-backup.tar -C /
```

### Clear All Data
```bash
# Clear Qdrant
curl -X DELETE http://localhost:6333/collections/k8s_cilium_docs

# Or restart Qdrant
docker restart qdrant
```

---

## Quick Reference Commands

### Ingestion
```bash
# Fast parallel ingestion
python3 scripts/ingest_k8s_docs.py --source data/docs/kubernetes --parallel --workers 8

# Sequential (testing)
python3 scripts/ingest_k8s_docs.py --source data/docs/kubernetes

# Cilium docs
python3 scripts/ingest_cilium_docs.py --source data/docs/cilium --parallel --workers 8
```

### Analysis
```bash
# Compare benchmarks
python3 scripts/compare_benchmarks.py benchmarks/*.json

# Analyze errors
python3 scripts/analyze_errors.py error_reports/k8s/ingestion_errors_*.json
```

### Services
```bash
# Start Ollama
ollama serve

# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Start API
uvicorn app.main:app --reload
```

---

## Support

For issues or questions:
1. Check error reports: `error_reports/`
2. Review benchmarks: `benchmarks/`
3. Check logs: `tail -f *.log`
4. See Technical Guide for architecture details

**Common Success Patterns:**
- ✅ Use parallel mode with 8 workers
- ✅ Monitor first run, adjust if needed
- ✅ Review error reports after ingestion
- ✅ Compare benchmarks to track improvements

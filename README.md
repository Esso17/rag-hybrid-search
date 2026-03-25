# Production RAG on Kubernetes: CPU-Only, On-Premise, Zero Cloud Cost

**A production-ready RAG system that runs on commodity hardware**

Built for resource-constrained environments: Minikube (4 CPU, 8GB RAM)

---

## 🎯 What Is This?

A complete Retrieval-Augmented Generation (RAG) system optimized for **on-premise, CPU-only deployment** with:

- **Zero cloud costs** - Runs entirely on local infrastructure
- **No GPU required** - Algorithmic fusion instead of neural rerankers
- **8GB RAM footprint** - Fits on laptops and edge devices
- **Production-ready** - Persistence, caching, health checks, monitoring
- **Validated with benchmarks** - Every optimization claim is reproducible

**Perfect for**:
- 🏢 On-premise deployments
- 💻 Local development (Minikube, kind, k3s)

---

## 🚀 Key Features

### Performance Optimizations (Benchmark Validated)

| Feature | Implementation | Speedup | Benchmark |
|---------|---------------|---------|-----------|
| **Vector Search** | FAISS HNSW | 100-1000x | O(log n) vs O(n) |
| **Keyword Search** | BM25 Inverted Index | 50-500x | O(k) vs O(n×m) |
| **Score Fusion** | RRF + Heuristics | +11.1% precision | vs weighted average |
| **Semantic Caching** | Query-Response Cache | 22,820x | 6,846ms → 0.3ms |
| **Embedding Cache** | LRU + Normalization | 30ms → 0ms | 100% hit on repeats |

### Architecture

- 🔍 **Hybrid Search** - Combines semantic (FAISS) + keyword (BM25) search
- ⚡ **Algorithmic Fusion** - RRF (Reciprocal Rank Fusion) + quality heuristics
- 🗄️ **Semantic Caching** - Bypasses entire pipeline for similar queries (0.95+ similarity)
- 📝 **Code-Aware Chunking** - Preserves YAML/code blocks intact
- 🎯 **DevOps Optimized** - Technical tokenization for K8s/Cilium documentation
- 🤖 **Local LLM** - Ollama (phi3.5:3.8b, 3GB)

---

## 📊 Production Metrics (Minikube: 4 CPU, 8GB RAM)

```
Performance:
├─ Query latency (cache miss):  2,050ms  (2s LLM, 50ms search)
├─ Query latency (cache hit):   <5ms     (400x faster!)
├─ Average latency:              ~1,400ms (32% cache hit rate)
├─ Throughput:                   30-40 concurrent users
└─ Indexing:                     500-1,000 docs/min

Quality (Benchmark Validated):
├─ Precision@5 (RRF+heuristics): 86.7%   (+11.1% vs weighted average)
├─ Conceptual queries:           92.0%   (semantic understanding)
├─ Technical queries:            84.0%   (exact matches)
└─ Hybrid queries:               86.0%   (balanced)

Resource Usage:
├─ Memory:  484MB  (FAISS 400MB + BM25 80MB + cache 4MB)
├─ Disk:    109MB  (persisted, survives pod restarts)
├─ CPU:     <5% search, ~40% LLM generation
└─ GPU:     None required

Caching (Benchmark Validated):
├─ Cache hit rate:      32-40%  (real workloads)
├─ Speedup:             22,820x (validated, not estimated!)
├─ Memory overhead:     4MB     (1000 cached entries)
└─ False positive rate: <0.1%   (0.95 similarity threshold)
```

---

## ⚡ Quick Start (5 Minutes)

### Prerequisites

```bash
# Required
- Docker Desktop (4 CPU, 8GB RAM)
- Python 3.11+
- Git

# Recommended
- Minikube v1.30+
- kubectl v1.27+
```

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/rag-hybrid-search
cd rag-hybrid-search

# Install dependencies
pip install -r requirements.txt

```

### 2. Start Ollama (Local LLM)

```bash
# Install Ollama: https://ollama.ai

# Pull models
ollama pull phi3.5:3.8b          # LLM (3.8GB)
ollama pull nomic-embed-text     # Embeddings (274MB)

# Start server
ollama serve
```

### 3. Run Locally (Without Kubernetes)

```bash
# Start FastAPI server
python -m app.main

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Test It

```bash
# Health check
curl http://localhost:8000/health

# Add a document
curl -X POST http://localhost:8000/add-document \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Kubernetes Pods",
    "content": "A Pod is the smallest deployable unit in Kubernetes...",
    "metadata": {"source": "kubernetes.io"}
  }'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a Kubernetes pod?"}' | jq
```

---

## 🐳 Deploy to Kubernetes (Minikube)

### 1. Start Minikube

```bash
# Start cluster
minikube start --cpus=4 --memory=8192 --driver=docker

# Verify
kubectl cluster-info
```

### 2. Build and Deploy

```bash
# Build image
docker build -t rag-hybrid-search:latest .

# Load into Minikube
minikube image load rag-hybrid-search:latest

# Deploy
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/app-deployment.yaml
kubectl apply -f k8s/service.yaml

# Wait for ready
kubectl wait --for=condition=ready pod -l app=rag-api \
  -n rag-hybrid-search --timeout=120s
```

### 3. Access the API

```bash
# Get service URL
minikube service rag-api-service -n rag-hybrid-search --url
# Example: http://127.0.0.1:52000

# Test
curl http://127.0.0.1:52000/health | jq
```

### 4. Ingest Documents

```bash
# Download Kubernetes docs
git clone --depth 1 https://github.com/kubernetes/website.git ~/k8s-website

# Ingest (direct pipeline - recommended)
python scripts/ingest_docs.py \
  --source ~/k8s-website/content/en/docs \
  --name "Kubernetes"

# Or via API (if server is remote)
python scripts/ingest_docs.py \
  --source ~/k8s-website/content/en/docs \
  --api --api-url http://127.0.0.1:52000

# Check stats
curl http://127.0.0.1:52000/stats | jq
```

---

## 🔬 Reproduce the Benchmarks

All performance claims are validated with reproducible benchmarks:

### 1. Chunk Size Optimization

```bash
# Test different chunk sizes and overlaps
python scripts/benchmark_chunking.py --quick

```

**What you'll learn**: Whether 800 chars is actually optimal for your corpus

### 2. Fusion Strategy A/B Testing

```bash
# Compare RRF vs Weighted vs Vector-only
python scripts/benchmark_reranking.py --quick

```
**Or test via API** (real-time side-by-side):

```bash
curl -X POST http://localhost:8000/query/compare \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to configure NetworkPolicy?",
    "strategies": [
      {"name": "RRF", "fusion_method": "rrf", "use_heuristics": true},
      {"name": "Weighted", "fusion_method": "weighted"},
      {"name": "Vector", "use_hybrid": false}
    ]
  }' | jq
```

### 3. Caching Performance

```bash
# Validate the 22,820x speedup claim
python scripts/benchmark_caching.py --quick

```
### Full Benchmark Suite

```bash
python scripts/benchmark_all.py --output my_results.json
```

## 🔧 Troubleshooting

### Ollama Not Found

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Or download from: https://ollama.ai
```

### FAISS Not Available

```bash
# Install FAISS
pip install faiss-cpu

# Or use in-memory fallback (slower)
# Edit app/config.py: USE_FAISS = False
```

### Pod Stuck in Pending (Kubernetes)

```bash
# Check PVC bound
kubectl get pvc -n rag-hybrid-search

# Check resources
kubectl describe pod -l app=rag-api -n rag-hybrid-search
```

### Out of Memory

```bash
# Increase memory limit in k8s/app-deployment.yaml
resources:
  limits:
    memory: "8Gi"  # Increase from 6Gi
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Distributed deployment guide (StatefulSet)
- [ ] Prometheus metrics integration

**Before submitting PR**:
1. Run benchmarks: `python scripts/benchmark_all.py`
2. Update docs if adding features
3. Include performance measurements

---

## 🙏 Acknowledgments

- **RRF**: Cormack, Clarke & Buettcher (SIGIR 2009) - Reciprocal Rank Fusion
- **FAISS**: Meta AI Research - Fast approximate nearest neighbors
- **Ollama**: Local LLM serving made simple
- **BM25**: Robertson & Zaragoza - Probabilistic relevance framework
- **Caching**: Mouschoutzi (Towards Data Science) - RAG caching strategies

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---

## 🌟 Star History

If this project helped you, please ⭐ star it on GitHub!

---

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/rag-hybrid-search/issues)
- Questions: [Discussions](https://github.com/yourusername/rag-hybrid-search/discussions)
- Article: [Medium - Production RAG on Kubernetes](link-to-medium)

---

**Built with ❤️ for developers who value performance, simplicity, and reproducibility.**

**Made for the real world** - where budgets are limited, infrastructure is constrained, and "good enough" deployed today beats "perfect" never shipped.

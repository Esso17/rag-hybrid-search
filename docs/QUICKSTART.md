# Quick Start Guide

Get up and running with RAG Hybrid Search in 5 minutes.

---

## Prerequisites

1. **Python 3.9+**
   ```bash
   python --version
   ```

2. **Ollama** - [Download here](https://ollama.ai)
   ```bash
   ollama --version
   ```

---

## Installation

```bash
# 1. Clone or navigate to the project
cd rag-hybrid-search

# 2. Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install FAISS for 100-1000x faster search
pip install faiss-cpu
```

---

## Setup Ollama

```bash
# Pull embedding model
ollama pull nomic-embed-text

# Pull LLM model
ollama pull llama3

# Start Ollama server (in a separate terminal)
ollama serve
```

---

## Your First Query

Create a test script `test.py`:

```python
from app.core.rag import get_rag_pipeline

# Initialize RAG
rag = get_rag_pipeline()

# Add a document
rag.add_document(
    doc_id="k8s-intro",
    title="Kubernetes Introduction",
    content="""
    Kubernetes is an open-source container orchestration platform.
    It manages containerized applications across a cluster of machines.
    Key concepts include Pods, Services, and Deployments.

    A Pod is the smallest deployable unit in Kubernetes.
    It can contain one or more containers that share storage and network.
    """,
    metadata={"source": "tutorial"}
)

# Query with hybrid search + enhanced fusion
results, answer = rag.query("What is a Kubernetes Pod?")

# Print results
print("=" * 60)
print("ANSWER:")
print("=" * 60)
print(answer)
print("\n" + "=" * 60)
print("TOP RESULTS:")
print("=" * 60)
for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result['score']:.3f}")
    print(f"   {result['content'][:150]}...")
```

Run it:

```bash
python test.py
```

---

## What Just Happened?

1. **Document Processing**:
   - Text split into chunks (code-aware)
   - Embeddings generated (via Ollama)
   - Stored in vector store (FAISS or in-memory)
   - BM25 index created

2. **Query Processing**:
   - Query embedded
   - Vector search (semantic similarity)
   - BM25 search (keyword matching)
   - RRF fusion + heuristics
   - Top results sent to LLM
   - Answer generated

**Total time**: ~5-10 seconds (first query loads models)

**Subsequent queries**: ~2-3 seconds

---

## Next Steps

### Add More Documents

```python
documents = [
    {
        "doc_id": "k8s-networking",
        "title": "Kubernetes Networking",
        "content": "...",
        "metadata": {"source": "official-docs"}
    },
    # ... more documents
]

# Parallel indexing for speed
total_chunks, success, errors = rag.add_documents_parallel(
    documents=documents,
    num_workers=4
)

print(f"Indexed {success} documents with {total_chunks} chunks")
```

### Customize Search

```python
# Use weighted average instead of RRF
results = rag.hybrid_search(
    query="kubernetes pods",
    fusion_method="weighted"
)

# Disable heuristics for baseline comparison
results = rag.hybrid_search(
    query="kubernetes pods",
    use_heuristics=False
)

# Add metadata boosting
boost_config = {
    "recency_weight": 0.2,
    "source_quality": {
        "official-docs": 0.3,
        "blog": 0.0
    }
}

results = rag.hybrid_search(
    query="kubernetes pods",
    boost_config=boost_config
)
```

### Compare Fusion Methods

```python
# Try the demo script
python examples/enhanced_fusion_demo.py
```

---

## Configuration

Edit `app/config.py` to customize:

```python
# Ollama settings
LLM_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# Search settings
FUSION_METHOD = "rrf"              # "rrf" or "weighted"
USE_SEARCH_HEURISTICS = True       # Enable heuristics
TOP_K_RESULTS = 5                  # Number of results

# Performance
USE_FAISS = True                   # Enable FAISS
USE_INVERTED_BM25 = True           # Enable inverted index
USE_CODE_AWARE_SPLITTING = True    # Preserve code blocks

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

---

## Troubleshooting

### Ollama not responding

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### Models not found

```bash
# List installed models
ollama list

# Pull missing models
ollama pull nomic-embed-text
ollama pull llama3
```

### FAISS not working

```bash
# Install FAISS
pip install faiss-cpu

# Verify installation
python -c "import faiss; print('FAISS OK')"
```

### Slow queries

1. Enable FAISS: `USE_FAISS = True` in config
2. Enable inverted BM25: `USE_INVERTED_BM25 = True`
3. Reduce results: `TOP_K_RESULTS = 3`
4. Check Ollama is using GPU (if available)

---

## Further Reading

- **[Main README](../README.md)** - Full feature list
- **[Configuration Guide](CONFIGURATION.md)** - Detailed settings
- **[Enhanced Fusion](../app/core/search/README.md)** - Deep dive into fusion
- **[Medium Article](../MEDIUM_ARTICLE.md)** - Project journey
- **[Code Structure](../REORGANIZATION.md)** - Module organization

---

**You're all set! Start building your RAG application.** 🚀

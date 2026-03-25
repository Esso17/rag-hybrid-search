# Data Ingestion Guide

Complete guide for ingesting documentation into the RAG Hybrid Search system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Ingestion Methods](#ingestion-methods)
3. [API Endpoints](#api-endpoints)
4. [Batch Processing](#batch-processing)
5. [Using Ingestion Scripts](#using-ingestion-scripts)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring & Troubleshooting](#monitoring--troubleshooting)

---

## Quick Start

### Prerequisites

- RAG API running (locally or in Kubernetes)
- Source documentation files (Markdown, text, etc.)
- API accessible at `http://localhost:8000` or your deployment URL

### Simple Single Document Upload

```bash
curl -X POST http://localhost:8000/add-document \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Getting Started with Kubernetes",
    "content": "Kubernetes is a container orchestration platform...",
    "metadata": {
      "source": "k8s-docs",
      "category": "getting-started",
      "version": "1.28"
    }
  }'
```

**Response:**
```json
{
  "doc_id": "doc_1774301752796",
  "title": "Getting Started with Kubernetes",
  "chunk_count": 3,
  "status": "added"
}
```

---

## Ingestion Methods

### Method 1: Single Document (API)

**Use when:**
- Uploading one document at a time
- Real-time updates
- Simple integration

**Endpoint:** `POST /add-document`

**Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/add-document",
    json={
        "title": "Pod Networking",
        "content": "Pods in Kubernetes get unique IP addresses...",
        "metadata": {"source": "k8s-docs", "topic": "networking"}
    }
)

result = response.json()
print(f"Added document with {result['chunk_count']} chunks")
```

### Method 2: Batch Upload (Parallel Processing)

**Use when:**
- Uploading multiple documents
- Need faster processing
- Have 10-1000+ documents

**Endpoint:** `POST /add-documents-batch`

**Example:**
```python
import requests

documents = [
    {
        "title": "Kubernetes Pods",
        "content": "Pod documentation content...",
        "metadata": {"source": "k8s-docs"}
    },
    {
        "title": "Kubernetes Services",
        "content": "Service documentation content...",
        "metadata": {"source": "k8s-docs"}
    }
    # ... more documents
]

response = requests.post(
    "http://localhost:8000/add-documents-batch",
    json={
        "documents": documents,
        "num_workers": 4,  # Parallel workers
        "max_concurrent_embeddings": 20  # Concurrent API calls
    }
)

result = response.json()
print(f"✅ {result['successful']}/{result['total_documents']} documents")
print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
```

**Performance:**
- 2 workers: ~2x speedup
- 4 workers: ~3-4x speedup
- 8 workers: ~5-7x speedup (diminishing returns)

### Method 3: File Upload

**Use when:**
- Uploading individual text files
- User-facing file upload feature

**Endpoint:** `POST /upload-document`

**Example (curl):**
```bash
curl -X POST http://localhost:8000/upload-document \
  -F "file=@kubernetes-guide.md" \
  -F "title=Kubernetes Guide"
```

**Example (Python):**
```python
import requests

with open("kubernetes-guide.md", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload-document",
        files={"file": f},
        data={"title": "Kubernetes Guide"}
    )

print(response.json())
```

---

## API Endpoints

### 1. Add Single Document

**Endpoint:** `POST /add-document`

**Request Body:**
```json
{
  "title": "string (required)",
  "content": "string (required)",
  "metadata": {
    "source": "string (optional)",
    "category": "string (optional)",
    "custom_field": "any (optional)"
  }
}
```

**Response:**
```json
{
  "doc_id": "string",
  "title": "string",
  "chunk_count": 5,
  "status": "added"
}
```

### 2. Batch Document Upload

**Endpoint:** `POST /add-documents-batch`

**Request Body:**
```json
{
  "documents": [
    {
      "title": "string",
      "content": "string",
      "metadata": {}
    }
  ],
  "num_workers": 4,
  "max_concurrent_embeddings": 20
}
```

**Response:**
```json
{
  "total_documents": 10,
  "successful": 10,
  "errors": 0,
  "total_chunks": 150,
  "processing_time": 5.23
}
```

**Parameters:**
- `num_workers`: Number of parallel workers (default: 4)
  - Recommended: 2-8 workers depending on CPU cores
- `max_concurrent_embeddings`: Concurrent embedding API calls (default: 20)
  - Recommended: 10-50 depending on network/Ollama capacity

### 3. Upload File

**Endpoint:** `POST /upload-document`

**Form Data:**
- `file`: File to upload (multipart/form-data)
- `title`: Document title (optional)

---

## Batch Processing

### Using Python Script

Create a batch ingestion script:

```python
#!/usr/bin/env python3
"""Batch ingest documentation files"""
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

def ingest_directory(docs_dir: str, source_name: str):
    """Ingest all markdown files from directory"""
    docs_path = Path(docs_dir)
    documents = []

    # Collect all markdown files
    for md_file in docs_path.rglob("*.md"):
        content = md_file.read_text(encoding='utf-8')
        documents.append({
            "title": md_file.stem.replace('-', ' ').title(),
            "content": content,
            "metadata": {
                "source": source_name,
                "file_path": str(md_file.relative_to(docs_path)),
                "category": md_file.parent.name
            }
        })

        print(f"📄 Loaded: {md_file.name}")

    # Batch upload
    print(f"\n📤 Uploading {len(documents)} documents...")
    response = requests.post(
        f"{API_URL}/add-documents-batch",
        json={
            "documents": documents,
            "num_workers": 4,
            "max_concurrent_embeddings": 20
        },
        timeout=300  # 5 minute timeout
    )

    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Success!")
        print(f"   Documents: {result['successful']}/{result['total_documents']}")
        print(f"   Chunks: {result['total_chunks']}")
        print(f"   Time: {result['processing_time']:.2f}s")
        print(f"   Speed: {result['total_documents']/result['processing_time']:.1f} docs/sec")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Ingest K8s documentation
    ingest_directory("data/docs/kubernetes", "k8s-docs")

    # Ingest Cilium documentation
    ingest_directory("data/docs/cilium", "cilium-docs")
```

Save as `scripts/batch_ingest.py` and run:

```bash
python scripts/batch_ingest.py
```

---

## Using Ingestion Scripts

### Kubernetes Documentation

The project includes pre-built ingestion scripts for K8s and Cilium docs.

**Script:** `scripts/ingest_k8s_docs.py`

**Usage:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Sequential ingestion (slower, safer)
python scripts/ingest_k8s_docs.py \
  --source data/docs/kubernetes \
  --version 1.28 \
  --component kubernetes \
  --batch-size 10

# Parallel ingestion (faster)
python scripts/ingest_k8s_docs.py \
  --source data/docs/kubernetes \
  --version 1.28 \
  --parallel \
  --workers 4 \
  --max-concurrent-embeddings 20
```

**Options:**
- `--source`: Path to K8s documentation directory
- `--version`: K8s version (e.g., 1.28, 1.29)
- `--component`: Component name for metadata
- `--batch-size`: Batch size for progress reporting
- `--parallel`: Enable parallel processing
- `--workers`: Number of parallel workers
- `--max-concurrent-embeddings`: Max concurrent embedding requests

### Cilium Documentation

**Script:** `scripts/ingest_cilium_docs.py`

**Usage:**
```bash
python scripts/ingest_cilium_docs.py \
  --source data/docs/cilium \
  --version latest \
  --parallel \
  --workers 4
```

### Quick Sample Ingestion

For testing or demonstration:

**Script:** `scripts/quick_ingest_sample.py`

```bash
python scripts/quick_ingest_sample.py
```

This ingests a small sample of K8s docs (3-5 documents) for quick testing.

---

## Performance Optimization

### Embedding Generation

**Bottleneck:** Embedding API (Ollama) is the slowest step

**Optimization:**
1. **Increase concurrent requests:**
   ```json
   {
     "max_concurrent_embeddings": 50  // Higher = faster
   }
   ```
   - Trade-off: Higher memory usage in Ollama
   - Recommended: 10-50 depending on Ollama capacity

2. **Use local Ollama with GPU:**
   - 10-100x faster than CPU
   - Ollama with NVIDIA GPU recommended for large-scale ingestion

### Parallel Workers

**Optimal Configuration:**
```json
{
  "num_workers": 4,  // CPU cores / 2
  "max_concurrent_embeddings": 20  // Network bandwidth dependent
}
```

**Performance Metrics:**

| Workers | Documents | Time   | Speedup |
|---------|-----------|--------|---------|
| 1       | 100 docs  | 120s   | 1x      |
| 2       | 100 docs  | 65s    | 1.8x    |
| 4       | 100 docs  | 35s    | 3.4x    |
| 8       | 100 docs  | 22s    | 5.5x    |

**Diminishing returns after 8 workers** due to:
- Embedding API rate limits
- Network bandwidth
- Memory constraints

### Chunking Strategy

**Configuration in `.env`:**
```bash
CHUNK_SIZE=800           # Larger = fewer chunks, less overhead
CHUNK_OVERLAP=200        # Balance between context and duplication
USE_CODE_AWARE_SPLITTING=true  # Preserves code blocks
```

**Guidelines:**
- **Technical docs:** 800-1000 chars (preserves code blocks)
- **General text:** 500-800 chars
- **Overlap:** 15-25% of chunk size

---

## Monitoring & Troubleshooting

### Check System Stats

```bash
curl http://localhost:8000/stats | jq .
```

**Response:**
```json
{
  "vector_backend": "In-Memory",
  "vector_store": {
    "vector_count": 1570,
    "dimension": 768,
    "total_points": 1570
  },
  "bm25": {
    "total_chunks": 1570,
    "total_documents": 0
  },
  "config": {
    "use_faiss": false,
    "use_code_aware_splitting": true,
    "fusion_method": "rrf",
    "use_heuristics": true
  }
}
```

### Monitor Ingestion Progress

**In Kubernetes:**
```bash
# Watch pod logs
kubectl logs -f deployment/rag-api -n rag-hybrid-search

# Check pod resource usage
kubectl top pods -n rag-hybrid-search
```

**Locally:**
```bash
# Check API logs
docker logs -f rag-api

# Monitor system resources
htop
```

### Common Issues

#### 1. Slow Ingestion

**Symptoms:** < 1 doc/sec processing speed

**Solutions:**
- Increase `max_concurrent_embeddings` (10 → 50)
- Check Ollama resource allocation
- Verify network connectivity to Ollama
- Use GPU-accelerated Ollama

#### 2. Out of Memory

**Symptoms:** Ollama pod crashes, embedding failures

**Solutions:**
- Reduce `max_concurrent_embeddings`
- Reduce `num_workers`
- Increase Ollama memory limits:
  ```yaml
  resources:
    limits:
      memory: "8Gi"  # Increase from default
  ```

#### 3. Timeout Errors

**Symptoms:** 504 Gateway Timeout

**Solutions:**
- Increase API timeout in client
- Reduce batch size
- Split large documents before ingestion

#### 4. Duplicate Documents

**Prevention:**
- Use unique `doc_id` if provided
- Track ingested files in database
- Use metadata filtering to avoid re-ingestion

### Health Checks

**Before ingestion:**
```bash
# Check API health
curl http://localhost:8000/health

# Check Ollama connectivity
curl http://localhost:11434/api/tags
```

**During ingestion:**
```bash
# Monitor stats in real-time
watch -n 5 'curl -s http://localhost:8000/stats | jq .'
```

---

## Best Practices

### 1. Document Metadata

Always include rich metadata for better filtering and boosting:

```json
{
  "title": "Kubernetes Pods",
  "content": "...",
  "metadata": {
    "source": "k8s-docs",
    "version": "1.28",
    "category": "workloads",
    "subcategory": "pods",
    "difficulty": "beginner",
    "last_updated": "2024-03-20"
  }
}
```

### 2. Content Preprocessing

**Clean content before ingestion:**
```python
def clean_content(text: str) -> str:
    """Clean markdown content"""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Normalize line endings
    text = text.replace('\r\n', '\n')

    return text.strip()
```

### 3. Error Handling

```python
import requests
import time

def ingest_with_retry(document, max_retries=3):
    """Ingest document with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/add-document",
                json=document,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                raise
```

### 4. Batch Size Guidelines

| Total Documents | Batch Size | Workers | Time Estimate |
|----------------|------------|---------|---------------|
| < 10           | 10         | 1       | < 1 min       |
| 10-100         | 10-20      | 2-4     | 2-5 min       |
| 100-1000       | 20-50      | 4-8     | 10-30 min     |
| 1000+          | 50-100     | 8       | 1-3 hours     |

---

## Examples

### Example 1: Ingest Local Files

```bash
#!/bin/bash
# ingest_local_docs.sh

API_URL="http://localhost:8000"
DOCS_DIR="./my-docs"

for file in "$DOCS_DIR"/*.md; do
    echo "Uploading: $(basename $file)"
    curl -X POST "$API_URL/upload-document" \
        -F "file=@$file" \
        -F "title=$(basename $file .md)"
    echo ""
done

echo "✅ Ingestion complete!"
curl -s "$API_URL/stats" | jq .
```

### Example 2: Ingest from URL

```python
import requests

def ingest_from_url(url: str, title: str):
    """Download and ingest content from URL"""
    # Download content
    response = requests.get(url)
    content = response.text

    # Ingest
    result = requests.post(
        "http://localhost:8000/add-document",
        json={
            "title": title,
            "content": content,
            "metadata": {"source_url": url}
        }
    )

    return result.json()

# Example
result = ingest_from_url(
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/overview/components.md",
    "Kubernetes Components"
)
print(f"Added: {result['chunk_count']} chunks")
```

### Example 3: Production Ingestion Pipeline

```python
#!/usr/bin/env python3
"""Production-grade ingestion pipeline"""
import requests
import logging
from pathlib import Path
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.stats = {"success": 0, "errors": 0, "total_chunks": 0}

    def ingest_directory(
        self,
        docs_dir: Path,
        source_name: str,
        file_pattern: str = "*.md",
        batch_size: int = 50
    ):
        """Ingest all files matching pattern"""
        files = list(docs_dir.rglob(file_pattern))
        logger.info(f"Found {len(files)} files to ingest")

        # Process in batches
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            self._process_batch(batch, source_name)

        self._print_summary()

    def _process_batch(self, files: List[Path], source_name: str):
        """Process a batch of files"""
        documents = []

        for file_path in files:
            try:
                content = file_path.read_text(encoding='utf-8')
                documents.append({
                    "title": file_path.stem.replace('-', ' ').title(),
                    "content": content,
                    "metadata": {
                        "source": source_name,
                        "file_path": str(file_path),
                        "file_size": file_path.stat().st_size
                    }
                })
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                self.stats["errors"] += 1

        # Upload batch
        try:
            response = requests.post(
                f"{self.api_url}/add-documents-batch",
                json={
                    "documents": documents,
                    "num_workers": 4,
                    "max_concurrent_embeddings": 20
                },
                timeout=300
            )

            if response.status_code == 200:
                result = response.json()
                self.stats["success"] += result["successful"]
                self.stats["errors"] += result["errors"]
                self.stats["total_chunks"] += result["total_chunks"]
                logger.info(
                    f"Batch complete: {result['successful']} docs, "
                    f"{result['total_chunks']} chunks, "
                    f"{result['processing_time']:.2f}s"
                )
            else:
                logger.error(f"API error: {response.status_code}")
                self.stats["errors"] += len(documents)
        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            self.stats["errors"] += len(documents)

    def _print_summary(self):
        """Print ingestion summary"""
        logger.info("\n" + "="*50)
        logger.info("INGESTION SUMMARY")
        logger.info("="*50)
        logger.info(f"✅ Success: {self.stats['success']} documents")
        logger.info(f"❌ Errors: {self.stats['errors']} documents")
        logger.info(f"📦 Total chunks: {self.stats['total_chunks']}")
        logger.info("="*50)

if __name__ == "__main__":
    pipeline = IngestionPipeline("http://localhost:8000")
    pipeline.ingest_directory(
        docs_dir=Path("data/docs/kubernetes"),
        source_name="k8s-docs",
        batch_size=50
    )
```

---

## Summary

**Quick Reference:**

| Use Case | Method | Endpoint | Speed |
|----------|--------|----------|-------|
| Single doc | API | `/add-document` | Slow |
| Multiple docs | API | `/add-documents-batch` | Fast |
| File upload | API | `/upload-document` | Medium |
| Bulk ingestion | Script | `ingest_k8s_docs.py` | Fastest |

**Performance Tips:**
- ✅ Use batch upload for 10+ documents
- ✅ Enable parallel processing
- ✅ Increase concurrent embeddings (10-50)
- ✅ Monitor system resources
- ✅ Include rich metadata

**For More Help:**
- API Documentation: `http://localhost:8000/docs`
- Troubleshooting: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Configuration: See [CONFIGURATION.md](CONFIGURATION.md)

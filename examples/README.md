# Examples

This directory contains example scripts demonstrating how to use the RAG Hybrid Search API.

## Available Examples

### 1. Client Example
**File:** [client_example.py](client_example.py)

Python client library for interacting with the RAG API.

```python
from client_example import RAGClient

client = RAGClient()
response = client.query("How do I create a Kubernetes Pod?")
print(response["answer"])
```

**Features:**
- Health check
- Add documents
- Upload files
- Search documents
- RAG queries (search + generate)

### 2. Full Test Suite
**File:** [example_test.py](example_test.py)

Comprehensive test script that validates the entire system.

```bash
python examples/example_test.py
```

**Tests:**
- Health check
- Document ingestion
- Hybrid search
- RAG query generation
- Performance timing

## Usage

### Using the Python Client

```python
from examples.client_example import RAGClient

# Initialize client
client = RAGClient(base_url="http://localhost:8000")

# Check system health
health = client.health_check()
print(health)

# Add a document
doc = client.add_document(
    title="Kubernetes Basics",
    content="Kubernetes is a container orchestration platform...",
    metadata={"version": "1.28"}
)

# Search
results = client.search("What is Kubernetes?", top_k=5)

# RAG query (search + generate)
response = client.query("Explain Kubernetes architecture", top_k=7)
print(response["answer"])
```

### Running the Test Suite

```bash
# Ensure services are running first
docker-compose up -d
python -m uvicorn app.main:app --reload

# In another terminal
python examples/example_test.py
```

## More Examples

For K8s/Cilium specific examples, see:
- [Quick Start Guide](../docs/quickstart.md)
- [K8s/Cilium Guide](../docs/kubernetes-cilium.md)
- [Test Ingestion Script](../scripts/test_ingestion.py)

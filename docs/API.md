# RAG API Reference

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication. For production, add API key validation.

## Response Format
All endpoints return JSON responses with standard HTTP status codes.

---

## Endpoints

### GET /health
Check system health and connectivity

**Response:**
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "llm_available": true,
  "version": "1.0.0"
}
```

---

### POST /add-document
Add a document to the RAG system

**Request Body:**
```json
{
  "title": "string (required)",
  "content": "string (required)",
  "metadata": {
    "key": "value"
  }
}
```

**Response:**
```json
{
  "doc_id": "doc_1234567890",
  "title": "string",
  "chunk_count": 5,
  "status": "added"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/add-document \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Python Basics",
    "content": "Python is a programming language...",
    "metadata": {"source": "tutorial"}
  }'
```

---

### POST /upload-document
Upload a text document file

**Request (multipart/form-data):**
- `file`: Text file to upload (required)
- `title`: Custom title (optional, defaults to filename)

**Response:**
```json
{
  "doc_id": "doc_1234567890",
  "title": "string",
  "chunk_count": 5,
  "filename": "document.txt",
  "status": "uploaded"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/upload-document \
  -F "file=@document.txt" \
  -F "title=My Document"
```

---

### POST /search
Perform hybrid search (BM25 + Vector)

**Request Body:**
```json
{
  "query": "string (required)",
  "top_k": 5,
  "use_hybrid": true
}
```

**Response:**
```json
{
  "query": "What is Python?",
  "results": [
    {
      "content": "Preview of content...",
      "score": 0.85,
      "document_id": "doc_123",
      "chunk_index": 0,
      "source": "hybrid"
    }
  ],
  "total_results": 1
}
```

**Parameters:**
- `query` (required): Search query string
- `top_k` (optional): Number of results to return (1-50, default: 5)
- `use_hybrid` (optional): Use hybrid search (default: true)

**Example:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "top_k": 3,
    "use_hybrid": true
  }'
```

---

### POST /query
Query with RAG (search + generate answer)

**Request Body:**
```json
{
  "query": "string (required)",
  "top_k": 5,
  "use_hybrid": true
}
```

**Response:**
```json
{
  "query": "How does Python compare to Java?",
  "answer": "Python and Java are both powerful languages. Python is known for its simplicity...",
  "sources": [
    {
      "content": "Python is...",
      "score": 0.85,
      "document_id": "doc_123",
      "chunk_index": 0,
      "source": "hybrid"
    }
  ],
  "generation_time": 2.34
}
```

**Parameters:**
- `query` (required): Question to answer
- `top_k` (optional): Number of context documents (default: 5)
- `use_hybrid` (optional): Use hybrid search (default: true)

**Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the benefits of machine learning?",
    "top_k": 5,
    "use_hybrid": true
  }' \
  --max-time 120
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Query must be at least 1 character long"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Rate Limiting
No rate limiting currently implemented. Add for production deployments.

## Timeouts
- `/search`: 5-10 seconds
- `/query`: 60-120 seconds (includes generation time)
- Large generations may take longer

## Performance Tips

1. **Batch Queries**: Send multiple queries sequentially rather than in parallel
2. **Optimize top_k**: Smaller values are faster (default 5 is good)
3. **Monitor Memory**: Check system memory during generation
4. **Use Keywords**: More specific queries return faster results

---

## Interactive API Documentation

FastAPI automatically generates interactive docs:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## SDK Examples

See `client_example.py` for Python client implementation.

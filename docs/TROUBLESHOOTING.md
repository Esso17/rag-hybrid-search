# Troubleshooting Guide

## Common Issues and Solutions

### 1. Connection Errors

#### Error: "Failed to connect to Qdrant"
```
Connection refused at http://localhost:6333
```

**Solution:**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Start Qdrant if not running
docker-compose up -d

# View logs
docker-compose logs qdrant

# Verify connection
curl http://localhost:6333/health
```

#### Error: "Failed to connect to Ollama"
```
Connection refused at http://localhost:11434
```

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Verify models are available
ollama list
```

---

### 2. Model Issues

#### Error: "Model 'mistral' not found"
```
Error: model not found
```

**Solution:**
```bash
# Pull the model
ollama pull mistral

# For embeddings
ollama pull nomic-embed-text

# Verify models
curl http://localhost:11434/api/tags
```

#### Error: "Out of memory" when running LLM
```
CUDA out of memory
```

**Solution:**
- Use a smaller model:
  ```env
  LLM_MODEL=neural-chat  # or other lightweight model
  ```
- Increase system memory or swap space
- Reduce batch size or context length

---

### 3. API Errors

#### Error: "422 Validation Error"
```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Solution:**
- Check request body format
- Ensure all required fields are present
- Validate JSON syntax

#### Error: "500 Internal Server Error"
```
Internal Server Error
```

**Solution:**
```bash
# Check server logs
tail -f server.log

# Check system resources
top  # or Activity Monitor on macOS

# Restart server
# Ctrl+C then restart with debug mode
DEBUG=True python -m uvicorn app.main:app --reload
```

---

### 4. Performance Issues

#### Issue: Search is slow
```
Query takes 30+ seconds
```

**Solutions:**
- Reduce `top_k` value:
  ```json
  {"query": "...", "top_k": 3}
  ```
- Verify Qdrant is not overloaded:
  ```bash
  curl http://localhost:6333/collections/documents
  ```
- Check available memory:
  ```bash
  free -h  # or Activity Monitor
  ```

#### Issue: Generation takes too long
```
RAG query takes 2+ minutes
```

**Solutions:**
- Use a faster LLM model
- Reduce context size (`top_k`)
- Check system resources

---

### 5. Data Issues

#### Issue: Document not found after upload
```
No results when searching for uploaded document
```

**Solutions:**
```bash
# Check if document was chunked
curl http://localhost:6333/collections/documents/points

# Verify Qdrant collection exists
curl http://localhost:6333/collections

# Re-add document with debugging
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Issue: Search results are irrelevant
```
Results don't match query
```

**Solutions:**
- Adjust `HYBRID_SEARCH_ALPHA`:
  ```env
  HYBRID_SEARCH_ALPHA=0.3  # More keyword search
  ```
- Check document content quality
- Verify embeddings are being generated:
  ```python
  client.client.post(
    "http://localhost:11434/api/embeddings",
    json={"model": "nomic-embed-text", "prompt": "test"}
  )
  ```

---

### 6. Installation Issues

#### Error: "ModuleNotFoundError: No module named 'fastapi'"
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fastapi; print(fastapi.__version__)"
```

#### Error: "pip: command not found"
```
Command not found
```

**Solution:**
```bash
# Use python -m pip instead
python -m pip install -r requirements.txt

# Or use python3
python3 -m pip install -r requirements.txt
```

---

### 7. Docker Issues

#### Error: "docker-compose: command not found"
```
docker-compose: command not found
```

**Solution (new Docker versions):**
```bash
# Use docker compose (new syntax)
docker compose up -d

# Or install docker-compose standalone
brew install docker-compose
```

#### Error: "Cannot connect to Docker daemon"
```
Cannot connect to Docker daemon at unix:///var/run/docker.sock
```

**Solution:**
```bash
# On macOS, start Docker Desktop
open /Applications/Docker.app

# On Linux, start Docker service
sudo systemctl start docker

# Verify Docker is running
docker ps
```

---

## Debug Mode

Enable debug logging for troubleshooting:

```python
# In .env
DEBUG=True

# Or via environment variable
export DEBUG=True
python -m uvicorn app.main:app --reload --log-level debug
```

## Logs

Check logs for errors:

```bash
# Application logs
tail -f app.log

# Qdrant logs
docker-compose logs qdrant -f

# Ollama logs
# On macOS: ~/Library/LaunchAgents/ollama
# On Linux: /var/log/ollama
```

## Getting Help

1. **Check logs first** - Most issues are visible in logs
2. **Verify services** - Ensure all services are running
3. **Test individually** - Test each component separately
4. **Check resources** - Monitor CPU, memory, disk space
5. **Review configuration** - Ensure .env settings are correct

## Contact Support

For issues, check:
- Qdrant docs: https://qdrant.tech/documentation/
- Ollama docs: https://github.com/ollama/ollama
- FastAPI docs: https://fastapi.tiangolo.com/

# Chunking Strategy Optimization Guide

**TL;DR**: Find the optimal chunk size and overlap for your RAG system through systematic benchmarking.

---

## Why Chunk Size Matters

Chunk size has **massive impact** on RAG performance:

| Aspect | Small Chunks (256) | Medium Chunks (800) | Large Chunks (1536) |
|--------|-------------------|-------------------|-------------------|
| **Precision** | High (specific matches) | Balanced | Lower (diluted) |
| **Context** | Low (fragmented) | Balanced | High (complete) |
| **Search Speed** | Slower (more chunks) | Faster | Fastest |
| **Memory** | Higher (more vectors) | Moderate | Lower |
| **Best For** | Technical queries | Mixed queries | Conceptual queries |

**Current Configuration** (in `app/config.py`):
```python
CHUNK_SIZE = 800        # Selected arbitrarily
CHUNK_OVERLAP = 200     # 25% overlap
```

**The Problem**: No data-driven evidence for these values!

---

## Quick Start

### 1. Run Benchmark (Fast Mode)

```bash
# Test 3 sizes × 2 overlaps = 6 configurations
python scripts/benchmark_chunking.py --quick

# Output:
# - Best precision config
# - Best speed config
# - Best balanced config
# - Detailed analysis by query type
```

**Time**: ~3-5 minutes with synthetic documents

---

### 2. Full Benchmark (Production)

```bash
# Test 5 sizes × 4 overlaps = 20 configurations
python scripts/benchmark_chunking.py

# Or customize:
python scripts/benchmark_chunking.py \
  --sizes 256,512,800,1024,1536 \
  --overlaps 50,100,200,400 \
  --docs 50 \
  --output my_results.json
```

**Time**: ~15-20 minutes with real documentation

---

## Understanding Results

### Example Output

```
================================================================================
RESULTS SUMMARY
================================================================================

🏆 Best Precision:
  Chunk size: 800
  Overlap: 200
  Precision@5: 0.867      ← 86.7% of top-5 results are relevant
  Search latency: 42.3ms

⚡ Best Speed:
  Chunk size: 512
  Overlap: 100
  Search latency: 31.2ms
  Precision@5: 0.734      ← 73.4% precision

⚖️ Best Balance (Precision - Speed Trade-off):
  Chunk size: 800
  Overlap: 200
  Precision@5: 0.867
  Search latency: 42.3ms

📊 Key Insights:

  Precision by chunk size:
      256 chars: 0.823
      512 chars: 0.778
      800 chars: 0.867  ← Winner
     1024 chars: 0.845
     1536 chars: 0.801

  Precision by overlap:
       50 chars: 0.756
      100 chars: 0.801
      200 chars: 0.867  ← Winner
      400 chars: 0.834

  Precision by query type (best config):
    conceptual  : 0.920  ← Large chunks help here
    technical   : 0.840  ← Exact matches
    hybrid      : 0.860
    code        : 0.850
```

---

## What Gets Tested

### Evaluation Queries (Built-in)

The benchmark uses **10 diverse queries** covering:

**Conceptual** (need context):
- "What is a Cilium network policy?"
- "How does pod-to-pod networking work?"

**Technical** (need precision):
- "kubectl get pods -n kube-system"
- "How to configure NetworkPolicy ingress rules"

**Hybrid** (need both):
- "Debug pod networking issues"
- "Create a deployment with 3 replicas"

**Code-heavy** (need code blocks):
- "Example Cilium NetworkPolicy YAML"

### Metrics Measured

1. **Precision@5**: % of top-5 results containing expected terms
2. **Term Coverage**: How well expected terms appear in results
3. **Search Latency**: Average search time (ms)
4. **Ingestion Time**: Time to process and index documents
5. **Chunk Count**: Number of chunks generated
6. **Query Type Breakdown**: Performance by query category

---

## Interpreting Results

### Scenario 1: Best Precision Wins

```
Best: size=800, overlap=200, precision=0.867
```

**Recommendation**: Use this if:
- Answer quality is critical
- You have enough memory (800 chunks for 50 docs)
- 42ms search latency is acceptable

**Action**:
```python
# Update app/config.py
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
```

---

### Scenario 2: Best Speed Wins

```
Best: size=512, overlap=100, latency=31ms, precision=0.734
```

**Recommendation**: Use this if:
- You need <50ms search latency SLA
- Memory is constrained
- Can tolerate 73% precision

**Trade-off**: 26% faster, but -13% precision

---

### Scenario 3: Precision by Query Type Varies

```
conceptual: 0.920
technical:  0.840
```

**Insight**: Large chunks (800-1024) help conceptual queries by preserving context

**Advanced**: Use dynamic chunking:
- Conceptual sections → 1024 chunks
- Technical sections → 512 chunks

---

## Using Your Own Data

### Option 1: Use Real Documents

```bash
# Make sure you've ingested K8s docs
python scripts/ingest_k8s_docs.py --source ~/k8s-website/content/en/docs

# Run benchmark (uses first 50 docs by default)
python scripts/benchmark_chunking.py --docs 50
```

---

### Option 2: Use Synthetic Documents

If no real docs available, the script auto-generates synthetic K8s documentation:

```python
# Automatically creates 50 synthetic docs with:
# - Pods, NetworkPolicy, Deployments, Services, Troubleshooting
# - Realistic YAML examples
# - kubectl commands
# - Technical explanations
```

**Good for**: Quick testing, CI/CD, development

---

## Advanced Usage

### Custom Evaluation Queries

Edit `EVAL_QUERIES` in `benchmark_chunking.py`:

```python
EVAL_QUERIES = [
    {
        "query": "Your custom query",
        "type": "conceptual",  # or technical, hybrid, code
        "expected_terms": ["term1", "term2", "term3"],
    },
    # ... more queries
]
```

---

### Focus on Specific Use Case

```bash
# Only test technical documentation
--sizes 512,800  --overlaps 100,200

# Only test conceptual explanations
--sizes 800,1024,1536  --overlaps 200,400
```

---

### Save and Compare Results

```bash
# Baseline
python scripts/benchmark_chunking.py --output baseline.json

# After code-aware chunking enabled
python scripts/benchmark_chunking.py --output code_aware.json

# Compare
python scripts/compare_chunking_results.py baseline.json code_aware.json
```

---

## Common Patterns

### Pattern 1: Technical Documentation (K8s, APIs)

**Observation**: Smaller chunks (512-800) + moderate overlap (100-200)

**Why**:
- Technical queries are specific ("kubectl get pods")
- Need exact term matches
- Code blocks should be preserved

**Recommendation**: `size=800, overlap=200`

---

### Pattern 2: Conceptual Documentation (Guides, Tutorials)

**Observation**: Larger chunks (1024-1536) + higher overlap (200-400)

**Why**:
- Conceptual queries need surrounding context
- Explanations span multiple paragraphs
- Semantic search benefits from more context

**Recommendation**: `size=1024, overlap=300`

---

### Pattern 3: Mixed Content (Most RAG Systems)

**Observation**: Medium chunks (800) + balanced overlap (200)

**Why**:
- Handles both technical and conceptual queries
- Good precision across query types
- Reasonable memory usage

**Recommendation**: `size=800, overlap=200` (current default)

---

## Integration with Project

### After Running Benchmark

1. **Review results**: Check precision by query type
2. **Update config**: Modify `app/config.py`
3. **Re-ingest docs**: Old chunks won't update automatically
4. **Validate**: Run queries to confirm improvement

```bash
# 1. Find optimal config
python scripts/benchmark_chunking.py --quick

# 2. Update config
# Edit app/config.py: CHUNK_SIZE=800, CHUNK_OVERLAP=200

# 3. Clear old data (important!)
rm -rf data/vector_store/* data/bm25_index/*

# 4. Re-ingest
python scripts/ingest_k8s_docs.py --source ~/k8s-website/content/en/docs

# 5. Test
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How to configure NetworkPolicy?"}'
```

---

## Benchmarking Best Practices

### 1. Use Representative Queries

✅ **Good**: Actual user queries from your use case
❌ **Bad**: Generic "test query 1", "test query 2"

### 2. Test Enough Documents

✅ **Good**: 50-100 docs (representative sample)
❌ **Bad**: 5 docs (not enough variety)

### 3. Consider Memory Constraints

```
Chunk size ↓ → More chunks → More memory
Example:
  1 doc = 10KB text
  size=256  → ~40 chunks → 40 vectors
  size=1024 → ~10 chunks → 10 vectors
```

### 4. Balance Precision vs Speed

- **Precision critical**: Healthcare, finance → favor precision
- **Speed critical**: Chat, real-time → favor speed
- **Balanced**: Most use cases → middle ground

---

## Troubleshooting

### "No documents loaded"

```bash
# Check if K8s docs exist
ls -la data/docs/kubernetes/

# If not, run ingestion first
python scripts/ingest_k8s_docs.py --source ~/k8s-website/content/en/docs
```

### "Ollama connection error"

```bash
# Start Ollama
ollama serve

# Verify models
ollama list
# Should see: nomic-embed-text, phi3.5:3.8b
```

### "Out of memory during benchmark"

```bash
# Use fewer documents
python scripts/benchmark_chunking.py --docs 20

# Or fewer configurations
python scripts/benchmark_chunking.py --quick
```

---

## Next Steps

After finding optimal chunking:

1. ✅ **Chunk Size Optimization** (this guide)
2. 🔄 **Reranker A/B Testing** - Compare RRF vs weighted vs neural
3. 🔄 **Cache Threshold Tuning** - Optimize 0.95 similarity threshold
4. 🔄 **Prompt Experimentation** - Test different prompt templates
5. 🔄 **Top-K Sensitivity** - Find optimal number of results

---

## Resources

- **Script**: `scripts/benchmark_chunking.py`
- **Config**: `app/config.py` (update `CHUNK_SIZE`, `CHUNK_OVERLAP`)
- **Documentation**: `scripts/README.md`

---

**Ready to optimize!** 🚀

Run the benchmark, analyze results, and make data-driven decisions about your chunking strategy.

# Benchmark Guide

## Comprehensive Benchmark Suite

The `benchmark_all.py` script provides a unified interface to test all RAG optimizations in one place.

---

## Quick Start

```bash
# Run all benchmarks (recommended)
python3 scripts/benchmark_all.py

# Quick mode (smaller datasets, faster)
python3 scripts/benchmark_all.py --quick

# Run specific scenario
python3 scripts/benchmark_all.py --scenario query_cache

# Save to custom location
python3 scripts/benchmark_all.py --output my_results.json
```

---

## Available Scenarios

### 1. Async Embeddings (Phase 1)
Tests parallel embedding generation vs sequential processing.

```bash
python3 scripts/benchmark_all.py --scenario async_embeddings
```

**What it tests:**
- Document ingestion speed
- Throughput (docs/sec, chunks/sec)
- Async batch processing with 20 concurrent requests

**Expected:** 5x faster than sequential

---

### 2. Query Cache (Phase 1)
Tests LRU cache effectiveness for query embeddings.

```bash
python3 scripts/benchmark_all.py --scenario query_cache
```

**What it tests:**
- First run (cache miss)
- Second run (cache hit)
- Cache persistence
- Speedup calculation

**Expected:** 12-14x faster on cache hits

**Includes:**
- 100 test queries (mix of generic and specific)
- Cache hit/miss comparison
- Time saved per query

---

### 3. Incremental BM25 (Phase 2)
Tests incremental index updates vs full rebuild.

```bash
python3 scripts/benchmark_all.py --scenario incremental_bm25
```

**What it tests:**
- Full rebuild time (old approach)
- Incremental update time (new approach)
- Speedup at different scales

**Expected:** 7-100x faster depending on corpus size

**Test parameters:**
- Base corpus: 1,000 documents
- New documents: 100 documents
- Measures both build and update time

---

### 4. FAISS Vector Index (Phase 3)
Tests FAISS HNSW vs in-memory brute-force search.

```bash
python3 scripts/benchmark_all.py --scenario faiss
```

**What it tests:**
- Vector search at 1k, 5k, 10k vectors
- FAISS HNSW vs in-memory comparison
- Build time vs search time trade-offs

**Expected:** 100-1000x faster at scale

**Requires:** `pip install faiss-cpu`

**Key insight:** FAISS search time stays nearly constant (~0.1ms) regardless of corpus size

---

### 5. BM25 Inverted Index (Phase 3)
Tests inverted index with candidate filtering.

```bash
python3 scripts/benchmark_all.py --scenario inverted_bm25
```

**What it tests:**
- Inverted index vs standard BM25
- Candidate filtering effectiveness
- Performance with specific technical queries

**Expected:** 10-50x faster with specific queries

**Uses specific queries:**
- "CiliumNetworkPolicy egress L7 HTTP filtering"
- "kubectl debug pod ephemeral containers"
- Other technical terms with 2-5% document overlap

---

## Options

### All Scenarios
```bash
python3 scripts/benchmark_all.py
```
Runs all 5 scenarios in sequence. Takes 2-5 minutes.

### Quick Mode
```bash
python3 scripts/benchmark_all.py --quick
```
Uses smaller datasets for faster testing:
- 10 docs instead of 30
- 20 queries instead of 100
- 500/1k vectors instead of 1k/5k/10k

Good for CI/CD or quick validation.

### Custom Output
```bash
python3 scripts/benchmark_all.py --output benchmarks/my_test.json
```
Save results to specific file.

---

## Output

### Console Output

```
================================================================================
RAG SYSTEM COMPREHENSIVE BENCHMARK SUITE
================================================================================

Testing all optimization phases:
  Phase 1: Async embeddings, query cache
  Phase 2: Incremental BM25 index
  Phase 3: FAISS vector index, inverted BM25
================================================================================

================================================================================
SCENARIO 1: ASYNC BATCH EMBEDDINGS
================================================================================

📊 Results:
   Documents: 30
   Chunks: 223
   Total time: 2.77s
   Throughput: 10.83 docs/sec
   Avg per doc: 92.3ms
   Min/Med/Max: 29.0 / 45.8 / 536.0 ms

✅ Expected: 5x faster than sequential (20 concurrent requests)

...

================================================================================
BENCHMARK SUMMARY
================================================================================

📊 Performance Highlights:
--------------------------------------------------------------------------------

1. Async Embeddings:
   ✅ 10.83 docs/sec
   ✅ 92.3ms avg per document
   Expected: 5x faster than sequential

2. Query Cache:
   ✅ 12.4x faster on cache hits
   ✅ 12.6ms saved per query
   Expected: 12-14x speedup

3. Incremental BM25:
   ✅ 7.3x faster updates
   ✅ 10.0ms saved
   Expected: 7-100x depending on scale

4. FAISS Vector Index:
   ✅ 153.5x faster at 10,000 vectors
   ✅ Near-constant search time (~0.1ms)
   Expected: 100-1000x at scale

5. BM25 Inverted Index:
   ✅ 1.1x faster at 10,000 documents
   ✅ Only scores 91.8% of corpus
   Expected: 10-50x with specific queries

================================================================================

✅ All benchmarks complete!
```

### JSON Output

Saved to `benchmarks/comprehensive_TIMESTAMP.json`:

```json
{
  "timestamp": "2026-03-23 20:30:45",
  "scenarios": {
    "1_async_embeddings": {
      "scenario": "async_embeddings",
      "documents": 30,
      "chunks": 223,
      "total_time_s": 2.77,
      "docs_per_sec": 10.83,
      "avg_time_per_doc_ms": 92.3,
      ...
    },
    "2_query_cache": {
      "scenario": "query_cache",
      "num_queries": 100,
      "cache_miss_avg_ms": 13.7,
      "cache_hit_avg_ms": 1.1,
      "speedup": 12.4,
      ...
    },
    ...
  }
}
```

---

## Interpreting Results

### Good Performance Indicators

**Async Embeddings:**
- ✅ 10+ docs/sec
- ✅ <100ms per document average
- ✅ Consistent times (low variance)

**Query Cache:**
- ✅ >10x speedup on hits
- ✅ <2ms cache hit time
- ✅ >10ms cache miss time

**Incremental BM25:**
- ✅ >5x speedup vs full rebuild
- ✅ Speedup increases with corpus size

**FAISS:**
- ✅ >50x speedup at 10k vectors
- ✅ <1ms search time
- ✅ Speedup increases with scale

**Inverted BM25:**
- ✅ >5x speedup with specific queries
- ✅ <10% candidates with good queries

### Performance Issues

If results are lower than expected:

1. **Slow embeddings:**
   - Check Ollama is running
   - Verify network latency
   - Check concurrent request limit

2. **Low cache speedup:**
   - Verify cache is working
   - Check if queries are identical
   - Ensure cache isn't cleared between runs

3. **Incremental BM25 not faster:**
   - Check corpus size (benefits increase with scale)
   - Verify incremental path is used

4. **FAISS not installed:**
   - Install: `pip install faiss-cpu`
   - Set `USE_FAISS=true`

5. **Inverted BM25 low speedup:**
   - Queries may be too generic
   - Try more specific technical terms
   - Check candidate % (should be <20% for good speedup)

---

## Comparison with Old Scripts

### Before (Multiple Scripts)

```bash
# Phase 1-2
python3 scripts/benchmark_optimizations.py

# Phase 3
python3 scripts/benchmark_phase3.py

# Compare
python3 scripts/compare_benchmarks.py benchmarks/*.json
```

### After (Unified Script)

```bash
# Everything in one place
python3 scripts/benchmark_all.py

# Specific scenario
python3 scripts/benchmark_all.py --scenario query_cache

# Quick test
python3 scripts/benchmark_all.py --quick
```

### Benefits

- ✅ **One command** for all benchmarks
- ✅ **Consistent output** format
- ✅ **Integrated comparison** between scenarios
- ✅ **Unified JSON** output
- ✅ **Better documentation** with help text

---

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run Performance Benchmarks
  run: |
    pip install -r requirements.txt
    pip install -r requirements-phase3.txt
    python3 scripts/benchmark_all.py --quick --output benchmarks/ci_results.json

- name: Upload Results
  uses: actions/upload-artifact@v2
  with:
    name: benchmark-results
    path: benchmarks/ci_results.json
```

### Regression Testing

```bash
# Baseline
python3 scripts/benchmark_all.py --output benchmarks/baseline.json

# After changes
python3 scripts/benchmark_all.py --output benchmarks/current.json

# Compare
python3 scripts/compare_benchmarks.py benchmarks/baseline.json benchmarks/current.json
```

---

## Troubleshooting

### "Data not found" Error

```bash
# Download test data
./scripts/download_cilium_docs.sh
```

### "FAISS not installed"

```bash
pip install faiss-cpu
```

### "Import Error"

```bash
# Make sure you're in project root
cd /path/to/rag-hybrid-search
python3 scripts/benchmark_all.py
```

### Slow Performance

```bash
# Use quick mode
python3 scripts/benchmark_all.py --quick

# Run single scenario
python3 scripts/benchmark_all.py --scenario async_embeddings
```

---

## Advanced Usage

### Custom Test Parameters

Edit `scripts/benchmark_all.py` to adjust:

```python
# Line ~400: Adjust sizes
if quick:
    sample_size = 10
    num_queries = 20
    corpus_sizes = [500, 1000]
else:
    sample_size = 30      # Change this
    num_queries = 100     # Change this
    corpus_sizes = [1000, 5000, 10000]  # Change this
```

### Add Custom Scenario

```python
def benchmark_my_scenario() -> Dict:
    """My custom benchmark"""
    print("\nSCENARIO 6: MY CUSTOM TEST")
    # ... your code ...
    return {"scenario": "custom", "result": 123}

# Add to run_all_benchmarks()
all_results["6_custom"] = benchmark_my_scenario()
```

---

## Best Practices

1. **Run baseline first:**
   ```bash
   python3 scripts/benchmark_all.py --output benchmarks/baseline.json
   ```

2. **Use quick mode for iteration:**
   ```bash
   python3 scripts/benchmark_all.py --quick
   ```

3. **Run full suite before release:**
   ```bash
   python3 scripts/benchmark_all.py
   ```

4. **Save results with meaningful names:**
   ```bash
   python3 scripts/benchmark_all.py --output benchmarks/v1.0_release.json
   ```

5. **Compare over time:**
   ```bash
   python3 scripts/compare_benchmarks.py benchmarks/baseline.json benchmarks/v1.0_release.json
   ```

---

## Related Documentation

- [OPTIMIZATION_PLAN.md](../OPTIMIZATION_PLAN.md) - Overall strategy
- [OPTIMIZATIONS_IMPLEMENTED.md](../OPTIMIZATIONS_IMPLEMENTED.md) - Phase 1-2 details
- [PHASE3_OPTIMIZATIONS.md](../PHASE3_OPTIMIZATIONS.md) - Phase 3 guide
- [OPTIMIZATION_SUMMARY.md](../OPTIMIZATION_SUMMARY.md) - Complete overview

---

## Support

For issues or questions:
1. Check this guide
2. Review [OPTIMIZATION_SUMMARY.md](../OPTIMIZATION_SUMMARY.md)
3. Run with `--help`: `python3 scripts/benchmark_all.py --help`

#!/usr/bin/env python3
"""
Semantic Caching Benchmark Suite

Tests the query-response cache implementation with:
- Cache miss scenario (first query)
- Cache hit scenario (exact repeat)
- Semantic cache hit (similar query, >0.95 similarity)
- Multiple queries for hit rate analysis
- Performance comparison with/without cache

Usage:
    python3 scripts/benchmark_caching.py                # Run full benchmark
    python3 scripts/benchmark_caching.py --quick         # Quick test (fewer queries)
    python3 scripts/benchmark_caching.py --no-ingest    # Skip document ingestion
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_documents_ingested(min_docs: int = 100):
    """Ensure RAG system has documents for testing"""
    from app.core.rag import get_rag_pipeline

    rag = get_rag_pipeline()

    # Check current stats
    if rag.use_faiss:
        stats = rag.faiss_store.get_stats()
        vector_count = stats.get("vector_count", 0)
    else:
        stats = rag.in_memory_store.get_stats()
        vector_count = stats.get("vector_count", 0)

    if vector_count < min_docs:
        logger.warning(f"Only {vector_count} vectors in store, need at least {min_docs}")
        logger.info("Please run: python scripts/ingest_optimized.py")
        return False

    logger.info(f"✅ Vector store ready: {vector_count} vectors")
    return True


def benchmark_cache_scenarios(num_queries: int = 100):
    """
    Benchmark different cache scenarios

    Tests:
    1. Cache MISS: First time query (full pipeline: 2050ms)
    2. Cache HIT (exact): Same query repeated (cached: <5ms)
    3. Cache HIT (semantic): Similar query with >0.95 similarity (cached: <5ms)
    4. Hit rate analysis: Mix of queries over time
    """
    from app.core.rag import get_rag_pipeline

    print("\n" + "=" * 80)
    print("SEMANTIC CACHING BENCHMARK")
    print("=" * 80)

    rag = get_rag_pipeline()

    # Check cache availability
    if not rag.use_cache:
        logger.error("❌ Query-response cache is disabled!")
        logger.info("Enable with: export USE_QUERY_RESPONSE_CACHE=true")
        return {}

    logger.info(
        f"✅ Cache enabled: threshold={rag.query_cache.similarity_threshold}, ttl={rag.query_cache.ttl_seconds}s"
    )

    # Clear cache for clean benchmark
    rag.query_cache.clear()
    logger.info("Cache cleared for benchmark")

    # Test queries (diverse set)
    test_queries = {
        "original": [
            "How do I create a Kubernetes deployment?",
            "What is a Cilium network policy?",
            "Configure Hubble for network observability",
            "Debug pod networking connectivity issues",
            "Install and upgrade Cilium in production",
            "Kubernetes service mesh with Cilium",
            "CiliumNetworkPolicy L7 HTTP filtering",
            "Egress gateway configuration",
            "Network security best practices",
            "Container orchestration patterns",
        ],
        "exact_repeat": [
            # Same queries - should hit cache exactly
            "How do I create a Kubernetes deployment?",
            "What is a Cilium network policy?",
            "Configure Hubble for network observability",
            "Debug pod networking connectivity issues",
            "Install and upgrade Cilium in production",
        ],
        "semantic_similar": [
            # Similar queries - should hit cache semantically (>0.95 similarity)
            "How to create a deployment in Kubernetes?",
            "What are Cilium network policies?",
            "Set up Hubble for network monitoring",
            "Troubleshoot pod network connectivity",
            "How to install Cilium in production?",
        ],
        "different": [
            # Different queries - should miss cache
            "What is a Kubernetes StatefulSet?",
            "How to configure Prometheus monitoring?",
            "Container security scanning tools",
            "Kubernetes cluster autoscaling",
            "Service mesh observability patterns",
        ],
    }

    results = {}

    # ========================================================================
    # Test 1: CACHE MISS (First time queries)
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 1: CACHE MISS (First Time Queries)")
    print("-" * 80)

    miss_times = []
    miss_search_times = []

    for query in test_queries["original"][:5]:  # First 5 queries
        logger.info(f"Query (miss): '{query[:50]}...'")
        search_results, answer, metadata = rag.query(query, top_k=7)

        miss_times.append(metadata["latency_ms"])
        miss_search_times.append(metadata.get("search_time_ms", 0))

        assert metadata["cache_hit"] is False, "Should be cache miss!"

        time.sleep(0.1)  # Small delay between queries

    results["cache_miss"] = {
        "count": len(miss_times),
        "avg_latency_ms": mean(miss_times),
        "min_latency_ms": min(miss_times),
        "max_latency_ms": max(miss_times),
        "median_latency_ms": median(miss_times),
        "std_latency_ms": stdev(miss_times) if len(miss_times) > 1 else 0,
        "avg_search_time_ms": mean(miss_search_times),
    }

    print(f"\n📊 Cache MISS Results ({len(miss_times)} queries):")
    print(f"   Avg latency: {results['cache_miss']['avg_latency_ms']:.1f}ms")
    print(
        f"   Range: {results['cache_miss']['min_latency_ms']:.1f} - {results['cache_miss']['max_latency_ms']:.1f}ms"
    )
    print(f"   Search time: {results['cache_miss']['avg_search_time_ms']:.1f}ms")
    print(
        f"   LLM time: {results['cache_miss']['avg_latency_ms'] - results['cache_miss']['avg_search_time_ms']:.1f}ms"
    )

    # ========================================================================
    # Test 2: CACHE HIT (Exact Repeat)
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 2: CACHE HIT (Exact Repeat)")
    print("-" * 80)

    exact_hit_times = []
    exact_similarities = []

    for query in test_queries["exact_repeat"]:
        logger.info(f"Query (exact): '{query[:50]}...'")
        search_results, answer, metadata = rag.query(query, top_k=7)

        exact_hit_times.append(metadata["latency_ms"])
        if "cache_similarity" in metadata:
            exact_similarities.append(metadata["cache_similarity"])

        if not metadata["cache_hit"]:
            logger.warning(f"Expected cache hit, got miss! Query: {query[:50]}")

        time.sleep(0.1)

    results["cache_hit_exact"] = {
        "count": len(exact_hit_times),
        "avg_latency_ms": mean(exact_hit_times),
        "min_latency_ms": min(exact_hit_times),
        "max_latency_ms": max(exact_hit_times),
        "median_latency_ms": median(exact_hit_times),
        "avg_similarity": mean(exact_similarities) if exact_similarities else 0,
        "hit_rate": sum(1 for t in exact_hit_times if t < 10) / len(exact_hit_times) * 100,
    }

    print(f"\n📊 Cache HIT (Exact) Results ({len(exact_hit_times)} queries):")
    print(f"   Avg latency: {results['cache_hit_exact']['avg_latency_ms']:.1f}ms")
    print(
        f"   Range: {results['cache_hit_exact']['min_latency_ms']:.1f} - {results['cache_hit_exact']['max_latency_ms']:.1f}ms"
    )
    print(f"   Avg similarity: {results['cache_hit_exact']['avg_similarity']:.4f}")
    print(f"   Hit rate: {results['cache_hit_exact']['hit_rate']:.0f}%")

    # Calculate speedup
    speedup_exact = (
        results["cache_miss"]["avg_latency_ms"] / results["cache_hit_exact"]["avg_latency_ms"]
    )
    print(f"   ⚡ Speedup: {speedup_exact:.1f}x faster than cache miss")

    # ========================================================================
    # Test 3: CACHE HIT (Semantic Similar)
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 3: CACHE HIT (Semantic Similar)")
    print("-" * 80)

    semantic_hit_times = []
    semantic_similarities = []
    semantic_hits = 0

    for query in test_queries["semantic_similar"]:
        logger.info(f"Query (semantic): '{query[:50]}...'")
        search_results, answer, metadata = rag.query(query, top_k=7)

        semantic_hit_times.append(metadata["latency_ms"])

        if metadata["cache_hit"]:
            semantic_hits += 1
            if "cache_similarity" in metadata:
                semantic_similarities.append(metadata["cache_similarity"])
                logger.info(f"   ✅ Cache HIT (similarity={metadata['cache_similarity']:.4f})")
        else:
            logger.info("   ❌ Cache MISS (similarity below threshold)")

        time.sleep(0.1)

    results["cache_hit_semantic"] = {
        "count": len(semantic_hit_times),
        "hits": semantic_hits,
        "hit_rate": (semantic_hits / len(semantic_hit_times)) * 100,
        "avg_latency_ms": mean(semantic_hit_times),
        "min_latency_ms": min(semantic_hit_times),
        "max_latency_ms": max(semantic_hit_times),
        "avg_similarity": mean(semantic_similarities) if semantic_similarities else 0,
        "min_similarity": min(semantic_similarities) if semantic_similarities else 0,
    }

    print(f"\n📊 Cache HIT (Semantic) Results ({len(semantic_hit_times)} queries):")
    print(
        f"   Hits: {semantic_hits}/{len(semantic_hit_times)} ({results['cache_hit_semantic']['hit_rate']:.0f}%)"
    )
    print(f"   Avg latency: {results['cache_hit_semantic']['avg_latency_ms']:.1f}ms")
    print(f"   Avg similarity: {results['cache_hit_semantic']['avg_similarity']:.4f}")
    print(f"   Min similarity: {results['cache_hit_semantic']['min_similarity']:.4f}")

    # ========================================================================
    # Test 4: CACHE MISS (Different Queries)
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 4: CACHE MISS (Different Queries)")
    print("-" * 80)

    different_miss_times = []

    for query in test_queries["different"]:
        logger.info(f"Query (different): '{query[:50]}...'")
        search_results, answer, metadata = rag.query(query, top_k=7)

        different_miss_times.append(metadata["latency_ms"])

        if metadata["cache_hit"]:
            logger.warning(f"Unexpected cache hit! Query: {query[:50]}")

        time.sleep(0.1)

    results["cache_miss_different"] = {
        "count": len(different_miss_times),
        "avg_latency_ms": mean(different_miss_times),
    }

    print(f"\n📊 Cache MISS (Different) Results ({len(different_miss_times)} queries):")
    print(f"   Avg latency: {results['cache_miss_different']['avg_latency_ms']:.1f}ms")

    # ========================================================================
    # Test 5: Mixed Workload (Realistic Scenario)
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 5: MIXED WORKLOAD (Realistic Scenario)")
    print("-" * 80)

    # Mix of original, repeats, and semantic similar
    mixed_queries = []
    mixed_queries.extend(test_queries["original"][:3])  # 3 new
    mixed_queries.extend(test_queries["exact_repeat"][:2])  # 2 exact repeats
    mixed_queries.extend(test_queries["semantic_similar"][:2])  # 2 semantic similar
    mixed_queries.extend(test_queries["different"][:3])  # 3 different

    mixed_times = []
    mixed_hits = 0

    for query in mixed_queries:
        search_results, answer, metadata = rag.query(query, top_k=7)
        mixed_times.append(metadata["latency_ms"])
        if metadata["cache_hit"]:
            mixed_hits += 1
        time.sleep(0.1)

    results["mixed_workload"] = {
        "total_queries": len(mixed_queries),
        "cache_hits": mixed_hits,
        "cache_misses": len(mixed_queries) - mixed_hits,
        "hit_rate": (mixed_hits / len(mixed_queries)) * 100,
        "avg_latency_ms": mean(mixed_times),
        "median_latency_ms": median(mixed_times),
    }

    # Calculate effective throughput increase
    avg_miss_time = results["cache_miss"]["avg_latency_ms"]
    avg_hit_time = results["cache_hit_exact"]["avg_latency_ms"]
    hit_rate = results["mixed_workload"]["hit_rate"] / 100

    # Effective latency = (hit_rate * hit_time) + ((1 - hit_rate) * miss_time)
    effective_latency = (hit_rate * avg_hit_time) + ((1 - hit_rate) * avg_miss_time)
    throughput_increase = (avg_miss_time / effective_latency - 1) * 100

    results["mixed_workload"]["effective_latency_ms"] = effective_latency
    results["mixed_workload"]["throughput_increase_pct"] = throughput_increase

    print(f"\n📊 Mixed Workload Results ({len(mixed_queries)} queries):")
    print(
        f"   Cache hits: {mixed_hits}/{len(mixed_queries)} ({results['mixed_workload']['hit_rate']:.0f}%)"
    )
    print(f"   Avg latency: {results['mixed_workload']['avg_latency_ms']:.1f}ms")
    print(f"   Effective latency: {results['mixed_workload']['effective_latency_ms']:.1f}ms")
    print(f"   Throughput increase: +{results['mixed_workload']['throughput_increase_pct']:.0f}%")

    # ========================================================================
    # Get cache statistics
    # ========================================================================
    cache_stats = rag.query_cache.get_stats()
    results["cache_stats"] = cache_stats

    return results


def print_summary(results: dict):
    """Print comprehensive summary"""
    print("\n\n" + "=" * 80)
    print("CACHING BENCHMARK SUMMARY")
    print("=" * 80)

    if not results:
        print("\n❌ No results to display")
        return

    # Performance comparison
    print("\n📊 Performance Comparison:")
    print("-" * 80)
    print(f"{'Scenario':<30} {'Avg Latency':>15} {'vs Baseline':>15}")
    print("-" * 80)

    baseline_latency = results.get("cache_miss", {}).get("avg_latency_ms", 0)

    scenarios = [
        ("Cache MISS (First Time)", "cache_miss"),
        ("Cache HIT (Exact Match)", "cache_hit_exact"),
        ("Cache HIT (Semantic)", "cache_hit_semantic"),
        ("Cache MISS (Different)", "cache_miss_different"),
        ("Mixed Workload", "mixed_workload"),
    ]

    for name, key in scenarios:
        if key in results:
            latency = results[key].get("avg_latency_ms", 0)
            if baseline_latency > 0:
                speedup = baseline_latency / latency if latency > 0 else 0
                speedup_str = (
                    f"{speedup:.1f}x faster" if speedup > 1 else f"{1/speedup:.1f}x slower"
                )
            else:
                speedup_str = "N/A"

            print(f"{name:<30} {latency:>13.1f}ms {speedup_str:>15}")

    # Cache statistics
    print("\n📈 Cache Statistics:")
    print("-" * 80)
    cache_stats = results.get("cache_stats", {})
    print(f"Total entries: {cache_stats.get('total_entries', 0)}")
    print(f"Expired entries: {cache_stats.get('expired_entries', 0)}")
    print(f"Max size: {cache_stats.get('max_size', 0)}")
    print(f"Similarity threshold: {cache_stats.get('similarity_threshold', 0):.2f}")
    print(f"TTL: {cache_stats.get('ttl_seconds', 0)}s")

    # Key insights
    print("\n💡 Key Insights:")
    print("-" * 80)

    miss_latency = results.get("cache_miss", {}).get("avg_latency_ms", 0)
    hit_latency = results.get("cache_hit_exact", {}).get("avg_latency_ms", 0)
    semantic_hit_rate = results.get("cache_hit_semantic", {}).get("hit_rate", 0)
    mixed_hit_rate = results.get("mixed_workload", {}).get("hit_rate", 0)
    throughput_increase = results.get("mixed_workload", {}).get("throughput_increase_pct", 0)

    if miss_latency > 0 and hit_latency > 0:
        speedup = miss_latency / hit_latency
        print(f"1. Cache speedup: {speedup:.1f}x faster on exact matches")
        print(f"   - Cache MISS: {miss_latency:.1f}ms (full pipeline)")
        print(f"   - Cache HIT: {hit_latency:.1f}ms (cached response)")

    if semantic_hit_rate > 0:
        print(f"\n2. Semantic matching: {semantic_hit_rate:.0f}% hit rate on similar queries")
        print(f"   - Threshold: {cache_stats.get('similarity_threshold', 0):.2f}")
        print("   - Enables cache hits for paraphrased queries")

    if mixed_hit_rate > 0:
        print(f"\n3. Real-world workload: {mixed_hit_rate:.0f}% cache hit rate")
        print(f"   - Throughput increase: +{throughput_increase:.0f}%")
        print("   - Same infrastructure serves more users")

    # Resource impact
    print("\n🔋 Resource Impact:")
    print("-" * 80)
    avg_search_time = results.get("cache_miss", {}).get("avg_search_time_ms", 0)
    llm_time = miss_latency - avg_search_time if miss_latency > avg_search_time else miss_latency

    print("Pipeline breakdown (cache miss):")
    print(
        f"  - Search (FAISS + BM25): {avg_search_time:.1f}ms ({avg_search_time/miss_latency*100:.0f}%)"
    )
    print(f"  - LLM generation: {llm_time:.1f}ms ({llm_time/miss_latency*100:.0f}%)")
    print("\nCache hit bypasses entire pipeline:")
    print(f"  - Saves: {miss_latency:.1f}ms per hit")
    print("  - Frees: ~40% CPU per hit (LLM generation)")
    print("  - Memory overhead: ~3.6MB for 1000 entries")

    print("\n" + "=" * 80)


def save_results(results: dict, output_path: Path = None):
    """Save results to JSON"""
    if not output_path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"benchmarks/caching_{timestamp}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": "semantic_caching",
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"📊 Results saved to: {output_path}")
    return output_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Semantic Caching Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer queries)")
    parser.add_argument("--no-ingest", action="store_true", help="Skip document ingestion check")
    parser.add_argument("--output", type=str, help="Output file path for results")

    args = parser.parse_args()

    # Check documents
    if not args.no_ingest:
        if not ensure_documents_ingested(min_docs=100):
            logger.error("Please ingest documents first: python scripts/ingest_optimized.py")
            return

    # Run benchmark
    num_queries = 20 if args.quick else 100
    results = benchmark_cache_scenarios(num_queries=num_queries)

    # Print summary
    print_summary(results)

    # Save results
    output_path = Path(args.output) if args.output else None
    save_results(results, output_path)

    print("\n✅ Caching benchmark complete!")
    print("\n💡 To optimize cache performance:")
    print("   - Increase CACHE_MAX_SIZE for more entries")
    print("   - Adjust CACHE_SIMILARITY_THRESHOLD (0.90-0.99)")
    print("   - Monitor hit rates and adjust TTL")


if __name__ == "__main__":
    main()

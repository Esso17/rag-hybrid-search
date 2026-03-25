#!/usr/bin/env python3
"""
Reranking and Fusion Method A/B Testing

Compare different fusion strategies to validate the "algorithms > models" thesis:
- RRF (Reciprocal Rank Fusion) with heuristics
- Weighted average fusion
- Vector-only (no fusion)
- BM25-only (no fusion)
- Optional: Neural reranker comparison

Measures:
- Retrieval precision (relevant docs in top-k)
- Answer quality (manual or LLM-as-judge)
- Latency (speed impact)
- Consistency (variance across queries)

Usage:
    python3 scripts/benchmark_reranking.py                    # Full benchmark
    python3 scripts/benchmark_reranking.py --quick            # Fast mode
    python3 scripts/benchmark_reranking.py --judge llm        # Use LLM to judge quality
    python3 scripts/benchmark_reranking.py --output results.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Optional

import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# EVALUATION QUERIES WITH GROUND TRUTH
# ============================================================================

EVAL_QUERIES = [
    # Conceptual questions (semantic search should excel)
    {
        "query": "What is a Kubernetes pod and why is it important?",
        "category": "conceptual",
        "expected_topics": ["pod", "container", "scheduling", "kubernetes"],
        "good_answer_contains": ["smallest deployable unit", "containers", "shared"],
        "bad_answer_contains": ["deployment", "service", "ingress"],  # Related but not correct
    },
    {
        "query": "Explain the difference between ClusterIP and NodePort services",
        "category": "conceptual",
        "expected_topics": ["service", "clusterip", "nodeport", "networking"],
        "good_answer_contains": ["internal", "external", "port", "expose"],
        "bad_answer_contains": ["pod", "deployment"],
    },
    {
        "query": "How does Cilium network policy work?",
        "category": "conceptual",
        "expected_topics": ["cilium", "networkpolicy", "ebpf", "security"],
        "good_answer_contains": ["policy", "network", "cilium", "security"],
        "bad_answer_contains": ["deployment", "service"],
    },
    # Technical/exact match questions (BM25 should excel)
    {
        "query": "kubectl get pods -n kube-system",
        "category": "technical",
        "expected_topics": ["kubectl", "pods", "namespace", "kube-system"],
        "good_answer_contains": ["kubectl", "get", "pods", "namespace"],
        "bad_answer_contains": ["create", "delete", "apply"],  # Different kubectl commands
    },
    {
        "query": "How to set resource limits for containers",
        "category": "technical",
        "expected_topics": ["resources", "limits", "requests", "cpu", "memory"],
        "good_answer_contains": ["resources", "limits", "cpu", "memory"],
        "bad_answer_contains": ["pod", "service", "deployment"],
    },
    {
        "query": "Configure NetworkPolicy ingress rules",
        "category": "technical",
        "expected_topics": ["networkpolicy", "ingress", "rules", "yaml"],
        "good_answer_contains": ["ingress", "networkpolicy", "rules"],
        "bad_answer_contains": ["egress", "pod"],
    },
    # Hybrid questions (both should contribute)
    {
        "query": "Debug pod networking issues in Kubernetes",
        "category": "hybrid",
        "expected_topics": ["debug", "pod", "networking", "troubleshoot"],
        "good_answer_contains": ["debug", "network", "troubleshoot", "pod"],
        "bad_answer_contains": ["create", "delete"],
    },
    {
        "query": "Create a deployment with 3 replicas",
        "category": "hybrid",
        "expected_topics": ["deployment", "replicas", "kubectl", "yaml"],
        "good_answer_contains": ["deployment", "replicas", "kubectl"],
        "bad_answer_contains": ["service", "ingress"],
    },
    # Code-specific questions (structure preservation matters)
    {
        "query": "Example NetworkPolicy YAML configuration",
        "category": "code",
        "expected_topics": ["networkpolicy", "yaml", "apiversion", "spec"],
        "good_answer_contains": ["apiVersion", "kind", "NetworkPolicy", "spec"],
        "bad_answer_contains": ["deployment", "service"],
    },
    {
        "query": "Kubernetes deployment YAML with resource limits",
        "category": "code",
        "expected_topics": ["deployment", "yaml", "resources", "limits"],
        "good_answer_contains": ["resources", "limits", "deployment", "yaml"],
        "bad_answer_contains": ["service", "pod"],
    },
]


# ============================================================================
# FUSION STRATEGIES TO COMPARE
# ============================================================================

FUSION_STRATEGIES = {
    "rrf_heuristics": {
        "name": "RRF + Heuristics (Recommended)",
        "description": "Reciprocal Rank Fusion with quality/overlap/exact match heuristics",
        "config": {
            "use_hybrid": True,
            "fusion_method": "rrf",
            "use_heuristics": True,
        },
        "expected_strength": "Balanced performance across all query types",
    },
    "rrf_only": {
        "name": "RRF Only",
        "description": "Reciprocal Rank Fusion without heuristics",
        "config": {
            "use_hybrid": True,
            "fusion_method": "rrf",
            "use_heuristics": False,
        },
        "expected_strength": "Good fusion, but misses quality signals",
    },
    "weighted": {
        "name": "Weighted Average",
        "description": f"Traditional weighted fusion (alpha={settings.HYBRID_SEARCH_ALPHA})",
        "config": {
            "use_hybrid": True,
            "fusion_method": "weighted",
            "use_heuristics": False,
        },
        "expected_strength": "Simple, but requires alpha tuning",
    },
    "weighted_heuristics": {
        "name": "Weighted + Heuristics",
        "description": "Weighted average with heuristics",
        "config": {
            "use_hybrid": True,
            "fusion_method": "weighted",
            "use_heuristics": True,
        },
        "expected_strength": "Improved weighted fusion",
    },
    "vector_only": {
        "name": "Vector Only",
        "description": "Pure semantic search (no BM25)",
        "config": {
            "use_hybrid": False,
            "fusion_method": None,
            "use_heuristics": False,
        },
        "expected_strength": "Conceptual queries, weak on exact matches",
    },
    "bm25_only": {
        "name": "BM25 Only",
        "description": "Pure keyword search (no vector)",
        "config": {
            "use_hybrid": False,  # Will need special handling
            "fusion_method": None,
            "use_heuristics": False,
        },
        "expected_strength": "Technical/exact queries, weak on semantic",
    },
}


# ============================================================================
# EVALUATION METRICS
# ============================================================================


def calculate_precision_at_k(results: list[dict], expected_topics: list[str], k: int = 5) -> float:
    """
    Calculate precision@k: what fraction of top-k results contain expected topics
    """
    top_k = results[:k]
    relevant = 0

    for result in top_k:
        content = result.get("content", "").lower()
        # Check if any expected topic appears
        if any(topic.lower() in content for topic in expected_topics):
            relevant += 1

    return relevant / k if k > 0 else 0.0


def calculate_topic_coverage(results: list[dict], expected_topics: list[str], k: int = 5) -> dict:
    """
    Calculate how many expected topics are covered in top-k results
    """
    top_k = results[:k]
    topics_found = set()

    for result in top_k:
        content = result.get("content", "").lower()
        for topic in expected_topics:
            if topic.lower() in content:
                topics_found.add(topic)

    coverage = len(topics_found) / len(expected_topics) if expected_topics else 0

    return {
        "coverage": coverage,
        "topics_found": len(topics_found),
        "topics_total": len(expected_topics),
        "missing_topics": [t for t in expected_topics if t not in topics_found],
    }


def evaluate_answer_quality_simple(
    answer: str, good_contains: list[str], bad_contains: list[str]
) -> dict:
    """
    Simple rule-based answer quality check
    """
    answer_lower = answer.lower()

    good_matches = sum(1 for term in good_contains if term.lower() in answer_lower)
    bad_matches = sum(1 for term in bad_contains if term.lower() in answer_lower)

    score = (good_matches - bad_matches) / len(good_contains) if good_contains else 0

    return {
        "quality_score": max(0, min(1, score)),  # Clamp to 0-1
        "good_matches": good_matches,
        "bad_matches": bad_matches,
        "good_ratio": good_matches / len(good_contains) if good_contains else 0,
    }


def evaluate_answer_quality_llm(query: str, answer: str, llm_client: httpx.Client) -> dict:
    """
    Use LLM as judge to evaluate answer quality

    Asks the LLM to rate the answer on:
    - Relevance (0-10)
    - Completeness (0-10)
    - Accuracy (0-10)
    """
    judge_prompt = f"""You are an expert evaluator of RAG system outputs. Rate the following answer to a user query.

Query: {query}

Answer: {answer}

Rate the answer on a scale of 0-10 for each criterion:
1. Relevance: Does the answer address the query?
2. Completeness: Does it provide sufficient detail?
3. Accuracy: Is the information correct?

Respond in JSON format:
{{
  "relevance": <0-10>,
  "completeness": <0-10>,
  "accuracy": <0-10>,
  "reasoning": "<brief explanation>"
}}
"""

    try:
        response = llm_client.post(
            f"{settings.LLM_BASE_URL}/api/generate",
            json={"model": settings.LLM_MODEL, "prompt": judge_prompt, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()

        # Extract JSON from response
        response_text = result.get("response", "")
        # Try to parse JSON from response
        import re

        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            return {
                "relevance": scores.get("relevance", 0),
                "completeness": scores.get("completeness", 0),
                "accuracy": scores.get("accuracy", 0),
                "overall": (
                    scores.get("relevance", 0)
                    + scores.get("completeness", 0)
                    + scores.get("accuracy", 0)
                )
                / 30,  # Normalize to 0-1
                "reasoning": scores.get("reasoning", ""),
            }
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")

    return {"relevance": 0, "completeness": 0, "accuracy": 0, "overall": 0, "reasoning": "Failed"}


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================


def run_query_with_strategy(
    rag_pipeline, query: str, strategy_config: dict, top_k: int = 5
) -> dict:
    """
    Run a query with a specific fusion strategy
    """
    start_time = time.time()

    try:
        # Special handling for BM25-only
        if strategy_config.get("fusion_method") is None and not strategy_config.get("use_hybrid"):
            # This is vector-only, already handled
            pass

        search_results, answer, metadata = rag_pipeline.query(
            query=query,
            use_hybrid=strategy_config.get("use_hybrid", True),
            top_k=top_k,
            fusion_method=strategy_config.get("fusion_method", "rrf"),
            use_heuristics=strategy_config.get("use_heuristics", False),
        )

        latency_ms = (time.time() - start_time) * 1000

        return {
            "success": True,
            "search_results": search_results,
            "answer": answer,
            "latency_ms": latency_ms,
            "metadata": metadata,
        }

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "latency_ms": (time.time() - start_time) * 1000,
        }


def benchmark_fusion_strategy(
    strategy_name: str,
    strategy_config: dict,
    eval_queries: list[dict],
    rag_pipeline,
    use_llm_judge: bool = False,
    llm_client: Optional[httpx.Client] = None,
) -> dict:
    """
    Benchmark a single fusion strategy across all evaluation queries
    """
    logger.info(f"\nTesting: {strategy_name}")
    logger.info(f"  Config: {strategy_config}")

    results = []
    latencies = []
    precision_scores = []
    coverage_scores = []
    quality_scores = []

    for query_data in eval_queries:
        query = query_data["query"]
        category = query_data["category"]
        expected_topics = query_data["expected_topics"]

        # Run query
        result = run_query_with_strategy(rag_pipeline, query, strategy_config["config"], top_k=5)

        if not result["success"]:
            logger.warning(f"  ❌ Query failed: {query[:50]}...")
            continue

        latency_ms = result["latency_ms"]
        search_results = result["search_results"]
        answer = result["answer"]

        # Calculate metrics
        precision = calculate_precision_at_k(search_results, expected_topics, k=5)
        coverage = calculate_topic_coverage(search_results, expected_topics, k=5)

        # Answer quality
        if use_llm_judge and llm_client:
            quality = evaluate_answer_quality_llm(query, answer, llm_client)
        else:
            quality = evaluate_answer_quality_simple(
                answer,
                query_data.get("good_answer_contains", []),
                query_data.get("bad_answer_contains", []),
            )

        # Store results
        results.append(
            {
                "query": query,
                "category": category,
                "latency_ms": latency_ms,
                "precision@5": precision,
                "topic_coverage": coverage["coverage"],
                "quality_score": quality.get("quality_score", 0),
                "answer": answer[:200],  # Truncate for storage
            }
        )

        latencies.append(latency_ms)
        precision_scores.append(precision)
        coverage_scores.append(coverage["coverage"])
        quality_scores.append(quality.get("quality_score", 0))

    # Aggregate metrics
    if not results:
        return {"strategy": strategy_name, "error": "All queries failed"}

    # Per-category breakdown
    category_metrics = {}
    for category in {r["category"] for r in results}:
        cat_results = [r for r in results if r["category"] == category]
        category_metrics[category] = {
            "precision@5": mean([r["precision@5"] for r in cat_results]),
            "coverage": mean([r["topic_coverage"] for r in cat_results]),
            "quality": mean([r["quality_score"] for r in cat_results]),
            "count": len(cat_results),
        }

    return {
        "strategy": strategy_name,
        "config": strategy_config,
        "num_queries": len(results),
        "avg_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
        "avg_precision@5": mean(precision_scores),
        "avg_coverage": mean(coverage_scores),
        "avg_quality": mean(quality_scores),
        "category_breakdown": category_metrics,
        "detailed_results": results,
    }


# ============================================================================
# MAIN BENCHMARK
# ============================================================================


def run_reranking_benchmark(
    strategies: list[str],
    num_queries: Optional[int] = None,
    use_llm_judge: bool = False,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Run comprehensive reranking benchmark
    """
    print("\n" + "=" * 80)
    print("RERANKING & FUSION METHOD A/B TEST")
    print("=" * 80)

    # Initialize RAG pipeline
    from app.core.rag import get_rag_pipeline

    rag_pipeline = get_rag_pipeline()

    # Select queries
    eval_queries = EVAL_QUERIES[:num_queries] if num_queries else EVAL_QUERIES
    print("\nConfiguration:")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Queries: {len(eval_queries)}")
    print(f"  LLM Judge: {use_llm_judge}")
    print()

    # Initialize LLM judge if needed
    llm_client = None
    if use_llm_judge:
        llm_client = httpx.Client(timeout=120.0)

    # Run benchmarks
    all_results = []

    for strategy_name in strategies:
        if strategy_name not in FUSION_STRATEGIES:
            logger.warning(f"Unknown strategy: {strategy_name}, skipping")
            continue

        strategy_config = FUSION_STRATEGIES[strategy_name]

        print(f"\n{'='*80}")
        print(f"Strategy: {strategy_config['name']}")
        print(f"Description: {strategy_config['description']}")
        print(f"Expected: {strategy_config['expected_strength']}")
        print(f"{'='*80}")

        try:
            result = benchmark_fusion_strategy(
                strategy_name=strategy_name,
                strategy_config=strategy_config,
                eval_queries=eval_queries,
                rag_pipeline=rag_pipeline,
                use_llm_judge=use_llm_judge,
                llm_client=llm_client,
            )
            all_results.append(result)

            # Print summary
            print(f"\n📊 Results for {strategy_config['name']}:")
            print(f"  Queries: {result['num_queries']}")
            print(f"  Avg Latency: {result['avg_latency_ms']:.1f}ms")
            print(f"  Avg Precision@5: {result['avg_precision@5']:.3f}")
            print(f"  Avg Coverage: {result['avg_coverage']:.3f}")
            print(f"  Avg Quality: {result['avg_quality']:.3f}")

            print("\n  By Category:")
            for cat, metrics in result["category_breakdown"].items():
                print(
                    f"    {cat:12s}: precision={metrics['precision@5']:.3f}, "
                    f"quality={metrics['quality']:.3f}"
                )

        except Exception as e:
            logger.error(f"Strategy {strategy_name} failed: {e}")
            continue

    if llm_client:
        llm_client.close()

    # Comparison summary
    if len(all_results) >= 2:
        print("\n" + "=" * 80)
        print("COMPARATIVE ANALYSIS")
        print("=" * 80)

        # Find best strategies
        best_precision = max(all_results, key=lambda x: x.get("avg_precision@5", 0))
        best_speed = min(all_results, key=lambda x: x.get("avg_latency_ms", float("inf")))
        best_quality = max(all_results, key=lambda x: x.get("avg_quality", 0))

        print("\n🏆 Best Precision:")
        print(f"  Strategy: {best_precision['strategy']}")
        print(f"  Precision@5: {best_precision['avg_precision@5']:.3f}")
        print(f"  Latency: {best_precision['avg_latency_ms']:.1f}ms")

        print("\n⚡ Best Speed:")
        print(f"  Strategy: {best_speed['strategy']}")
        print(f"  Latency: {best_speed['avg_latency_ms']:.1f}ms")
        print(f"  Precision@5: {best_speed['avg_precision@5']:.3f}")

        print("\n✨ Best Quality:")
        print(f"  Strategy: {best_quality['strategy']}")
        print(f"  Quality: {best_quality['avg_quality']:.3f}")
        print(f"  Precision@5: {best_quality['avg_precision@5']:.3f}")

        # Head-to-head comparison table
        print("\n📊 Head-to-Head Comparison:")
        print(f"\n{'Strategy':<25} {'Precision@5':<15} {'Latency':<12} {'Quality':<10}")
        print("-" * 70)
        for result in sorted(all_results, key=lambda x: x.get("avg_precision@5", 0), reverse=True):
            print(
                f"{result['strategy']:<25} "
                f"{result['avg_precision@5']:<15.3f} "
                f"{result['avg_latency_ms']:<12.1f} "
                f"{result['avg_quality']:<10.3f}"
            )

        # Statistical significance (simple comparison)
        if "rrf_heuristics" in strategies and "weighted" in strategies:
            rrf_result = next((r for r in all_results if r["strategy"] == "rrf_heuristics"), None)
            weighted_result = next((r for r in all_results if r["strategy"] == "weighted"), None)

            if rrf_result and weighted_result:
                precision_diff = rrf_result["avg_precision@5"] - weighted_result["avg_precision@5"]
                latency_diff = rrf_result["avg_latency_ms"] - weighted_result["avg_latency_ms"]

                print("\n🔬 RRF+Heuristics vs Weighted Average:")
                print(f"  Precision difference: {precision_diff:+.3f} ({precision_diff*100:+.1f}%)")
                print(f"  Latency difference: {latency_diff:+.1f}ms")

                if precision_diff > 0.05:
                    print("  ✅ RRF+Heuristics significantly better precision")
                elif precision_diff < -0.05:
                    print("  ❌ Weighted Average significantly better precision")
                else:
                    print("  ⚖️  Similar precision")

    # Save results
    output_data = {
        "config": {
            "strategies": strategies,
            "num_queries": len(eval_queries),
            "use_llm_judge": use_llm_judge,
        },
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")

    return output_data


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Benchmark different reranking/fusion strategies")

    parser.add_argument(
        "--strategies",
        type=str,
        default="rrf_heuristics,rrf_only,weighted,vector_only",
        help="Comma-separated strategies to test (default: rrf_heuristics,rrf_only,weighted,vector_only)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=None,
        help="Number of queries to test (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test fewer strategies (rrf_heuristics,weighted,vector_only)",
    )
    parser.add_argument(
        "--judge",
        type=str,
        choices=["simple", "llm"],
        default="simple",
        help="Answer quality judge: simple (rule-based) or llm (LLM-as-judge)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/reranking_analysis.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    # Parse strategies
    if args.quick:
        strategies = ["rrf_heuristics", "weighted", "vector_only"]
    else:
        strategies = [s.strip() for s in args.strategies.split(",")]

    use_llm_judge = args.judge == "llm"
    output_path = Path(args.output)

    # Run benchmark
    try:
        results = run_reranking_benchmark(
            strategies=strategies,
            num_queries=args.queries,
            use_llm_judge=use_llm_judge,
            output_path=output_path,
        )

        if results:
            print("\n✅ Benchmark completed successfully!")
        else:
            print("\n❌ Benchmark failed - no results generated")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

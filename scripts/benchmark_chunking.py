#!/usr/bin/env python3
"""
Chunk Size and Overlap Sensitivity Analysis

Tests different chunking strategies to find optimal configuration for:
- Retrieval precision (relevant chunks in top-k)
- Answer quality
- Ingestion time
- Search latency

Usage:
    python3 scripts/benchmark_chunking.py                          # Full benchmark
    python3 scripts/benchmark_chunking.py --quick                  # Fast mode (fewer combinations)
    python3 scripts/benchmark_chunking.py --sizes 512,800,1024     # Test specific sizes
    python3 scripts/benchmark_chunking.py --output results.json    # Save to custom path

Results:
    - Optimal chunk size for your corpus
    - Optimal overlap ratio
    - Trade-offs: precision vs speed vs memory
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# EVALUATION QUERIES - Technical Documentation Focus
# ============================================================================

EVAL_QUERIES = [
    # Conceptual questions (favor larger chunks with context)
    {
        "query": "What is a Cilium network policy?",
        "type": "conceptual",
        "expected_terms": ["cilium", "network", "policy", "kubernetes"],
    },
    {
        "query": "How does pod-to-pod networking work in Kubernetes?",
        "type": "conceptual",
        "expected_terms": ["pod", "networking", "kubernetes", "cni"],
    },
    {
        "query": "Explain the difference between ClusterIP and NodePort services",
        "type": "conceptual",
        "expected_terms": ["clusterip", "nodeport", "service"],
    },
    # Technical/specific questions (favor smaller chunks with precision)
    {
        "query": "kubectl get pods -n kube-system",
        "type": "technical",
        "expected_terms": ["kubectl", "pods", "kube-system", "namespace"],
    },
    {
        "query": "How to configure NetworkPolicy ingress rules",
        "type": "technical",
        "expected_terms": ["networkpolicy", "ingress", "rules", "yaml"],
    },
    {
        "query": "Set resource limits for containers",
        "type": "technical",
        "expected_terms": ["resources", "limits", "container", "cpu", "memory"],
    },
    # Hybrid questions (need both context and precision)
    {
        "query": "Debug pod networking issues",
        "type": "hybrid",
        "expected_terms": ["debug", "pod", "networking", "troubleshoot"],
    },
    {
        "query": "Create a deployment with 3 replicas",
        "type": "hybrid",
        "expected_terms": ["deployment", "replicas", "kubectl", "yaml"],
    },
    {
        "query": "Configure liveness and readiness probes",
        "type": "hybrid",
        "expected_terms": ["liveness", "readiness", "probe", "health"],
    },
    # Code-heavy questions (favor code-aware chunking)
    {
        "query": "Example Cilium NetworkPolicy YAML",
        "type": "code",
        "expected_terms": ["cilium", "networkpolicy", "yaml", "apiversion"],
    },
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_sample_documents(source_path: Path, max_docs: int = 50) -> list[dict]:
    """Load sample documents for testing"""
    from app.utils.loaders import load_k8s_docs

    if not source_path.exists():
        logger.warning(f"Source path not found: {source_path}")
        # Create synthetic documents for testing
        return create_synthetic_documents(max_docs)

    docs = load_k8s_docs(str(source_path))[:max_docs]
    logger.info(f"Loaded {len(docs)} documents from {source_path}")
    return docs


def create_synthetic_documents(count: int = 50) -> list[dict]:
    """Create synthetic Kubernetes documentation for testing"""
    synthetic_docs = []

    # Sample K8s topics with realistic content
    topics = [
        {
            "title": "Kubernetes Pods",
            "content": """
# Pods

A Pod is the smallest deployable unit in Kubernetes. Pods are the atomic unit
of scheduling and can contain one or more containers.

## Creating Pods

You can create a Pod using kubectl:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
spec:
  containers:
  - name: nginx
    image: nginx:1.14.2
    ports:
    - containerPort: 80
```

Apply with: `kubectl apply -f pod.yaml`

## Pod Lifecycle

Pods go through several phases:
- Pending: Pod accepted but not running
- Running: Pod bound to node, containers running
- Succeeded: All containers terminated successfully
- Failed: All containers terminated, at least one failed
- Unknown: Pod state cannot be obtained
            """,
        },
        {
            "title": "NetworkPolicy",
            "content": """
# Kubernetes NetworkPolicy

NetworkPolicies are Kubernetes resources that control traffic between pods.

## Basic NetworkPolicy

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: test-network-policy
  namespace: default
spec:
  podSelector:
    matchLabels:
      role: db
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: frontend
    ports:
    - protocol: TCP
      port: 6379
```

## Ingress Rules

Ingress rules allow incoming traffic from specified sources. You can filter by:
- Pod selectors
- Namespace selectors
- IP blocks

## Egress Rules

Egress rules control outgoing traffic to specific destinations.
            """,
        },
        {
            "title": "Deployments",
            "content": """
# Kubernetes Deployments

A Deployment provides declarative updates for Pods and ReplicaSets.

## Creating Deployments

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
        resources:
          limits:
            cpu: "500m"
            memory: "512Mi"
          requests:
            cpu: "250m"
            memory: "256Mi"
```

## Updating Deployments

Update the image: `kubectl set image deployment/nginx-deployment nginx=nginx:1.16.1`

## Scaling Deployments

Scale replicas: `kubectl scale deployment/nginx-deployment --replicas=5`
            """,
        },
        {
            "title": "Services",
            "content": """
# Kubernetes Services

Services expose applications running on pods.

## Service Types

### ClusterIP
Default type. Exposes service on cluster-internal IP.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  type: ClusterIP
  selector:
    app: MyApp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 9376
```

### NodePort
Exposes service on each Node's IP at a static port.

### LoadBalancer
Exposes service externally using cloud provider's load balancer.

### ExternalName
Maps service to external DNS name.
            """,
        },
        {
            "title": "Troubleshooting",
            "content": """
# Troubleshooting Kubernetes

## Common Issues

### Pod Not Starting

Check pod status: `kubectl get pods -n kube-system`
Describe pod: `kubectl describe pod <pod-name>`
Check logs: `kubectl logs <pod-name>`

### Networking Issues

Test connectivity:
```bash
kubectl exec -it <pod> -- ping <target-ip>
kubectl exec -it <pod> -- curl http://<service-name>
```

Check DNS: `kubectl exec -it <pod> -- nslookup kubernetes.default`

### Resource Issues

Check resource usage:
```bash
kubectl top nodes
kubectl top pods
```

View events: `kubectl get events --sort-by=.metadata.creationTimestamp`
            """,
        },
    ]

    # Replicate topics to reach desired count
    for i in range(count):
        topic = topics[i % len(topics)]
        synthetic_docs.append(
            {
                "doc_id": f"synthetic_{i}",
                "title": f"{topic['title']} - Part {i // len(topics) + 1}",
                "content": topic["content"],
                "metadata": {"source": "synthetic", "topic": topic["title"]},
            }
        )

    return synthetic_docs


def calculate_retrieval_precision(
    search_results: list[dict], expected_terms: list[str], top_k: int = 5
) -> dict:
    """
    Calculate precision metrics for retrieval results

    Returns:
        dict with precision@k, recall, term_coverage
    """
    top_results = search_results[:top_k]

    # Count how many results contain expected terms
    relevant_count = 0
    term_coverage = dict.fromkeys(expected_terms, 0)

    for result in top_results:
        content_lower = result.get("content", "").lower()

        # Check if result is relevant (contains any expected term)
        is_relevant = any(term.lower() in content_lower for term in expected_terms)
        if is_relevant:
            relevant_count += 1

        # Track term coverage
        for term in expected_terms:
            if term.lower() in content_lower:
                term_coverage[term] += 1

    precision_at_k = relevant_count / top_k if top_k > 0 else 0
    avg_term_coverage = mean(term_coverage.values()) / top_k if term_coverage else 0  # Normalized

    return {
        "precision@k": precision_at_k,
        "relevant_count": relevant_count,
        "term_coverage": avg_term_coverage,
        "terms_found": sum(1 for v in term_coverage.values() if v > 0),
        "total_terms": len(expected_terms),
    }


# ============================================================================
# CHUNKING BENCHMARK
# ============================================================================


def benchmark_chunking_config(
    documents: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    eval_queries: list[dict],
    top_k: int = 5,
) -> dict:
    """
    Benchmark a specific chunking configuration

    Args:
        documents: List of documents to index
        chunk_size: Chunk size in characters
        chunk_overlap: Overlap size in characters
        eval_queries: Evaluation queries with expected terms
        top_k: Number of results to retrieve

    Returns:
        dict with metrics (precision, latency, chunk_count, etc.)
    """
    from app.config import settings
    from app.core.embedding import embed_batch_async
    from app.core.retrieval import get_bm25_index
    from app.core.search.hybrid_search import hybrid_search
    from app.core.vector_stores import get_in_memory_vector_store

    logger.info(f"Testing chunk_size={chunk_size}, overlap={chunk_overlap}")

    # Create text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Initialize fresh stores
    vector_store = get_in_memory_vector_store(dimension=settings.EMBEDDING_DIMENSION)
    bm25_index = get_bm25_index()

    # Measure ingestion time
    ingestion_start = time.time()

    all_chunks = []
    all_embeddings = []
    all_payloads = []

    for doc in documents:
        chunks = splitter.split_text(doc["content"])

        # Generate embeddings (async batch)
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            embeddings = loop.run_until_complete(embed_batch_async(chunks, max_concurrent=10))
        finally:
            loop.close()

        # Prepare payloads
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_payloads.append(
                {
                    "document_id": doc["doc_id"],
                    "chunk_index": idx,
                    "content": chunk,
                    "title": doc["title"],
                    "metadata": doc.get("metadata", {}),
                }
            )

        all_embeddings.extend(embeddings)

    # Add to stores
    vector_store.add_points(all_embeddings, all_payloads)
    bm25_index.add_chunks(all_chunks)

    ingestion_time = time.time() - ingestion_start
    chunk_count = len(all_chunks)
    avg_chunk_length = mean(len(c) for c in all_chunks) if all_chunks else 0

    logger.info(
        f"  Ingested {chunk_count} chunks in {ingestion_time:.2f}s "
        f"(avg length: {avg_chunk_length:.0f} chars)"
    )

    # Measure search performance
    search_latencies = []
    precision_scores = []
    term_coverage_scores = []
    query_type_metrics = {"conceptual": [], "technical": [], "hybrid": [], "code": []}

    for query_data in eval_queries:
        query = query_data["query"]
        expected_terms = query_data["expected_terms"]
        query_type = query_data["type"]

        # Search
        search_start = time.time()
        results = hybrid_search(
            query=query,
            vector_store=vector_store,
            bm25_index=bm25_index,
            top_k=top_k,
            use_faiss=False,  # Use in-memory for consistency
            fusion_method="rrf",
            use_heuristics=True,
        )
        search_time = (time.time() - search_start) * 1000  # ms

        # Calculate precision
        metrics = calculate_retrieval_precision(results, expected_terms, top_k=top_k)

        search_latencies.append(search_time)
        precision_scores.append(metrics["precision@k"])
        term_coverage_scores.append(metrics["term_coverage"])
        query_type_metrics[query_type].append(metrics["precision@k"])

    # Aggregate metrics
    avg_precision = mean(precision_scores) if precision_scores else 0
    avg_search_latency = mean(search_latencies) if search_latencies else 0
    avg_term_coverage = mean(term_coverage_scores) if term_coverage_scores else 0

    # Per-query-type precision
    type_precision = {}
    for qtype, scores in query_type_metrics.items():
        if scores:
            type_precision[qtype] = mean(scores)

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "overlap_ratio": chunk_overlap / chunk_size if chunk_size > 0 else 0,
        "chunk_count": chunk_count,
        "avg_chunk_length": avg_chunk_length,
        "ingestion_time_sec": ingestion_time,
        "chunks_per_second": chunk_count / ingestion_time if ingestion_time > 0 else 0,
        "avg_search_latency_ms": avg_search_latency,
        "avg_precision@k": avg_precision,
        "avg_term_coverage": avg_term_coverage,
        "precision_by_type": type_precision,
        "num_queries": len(eval_queries),
    }


# ============================================================================
# MAIN BENCHMARK
# ============================================================================


def run_chunking_benchmark(
    chunk_sizes: list[int],
    overlaps: list[int],
    num_docs: int = 50,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Run comprehensive chunking benchmark

    Args:
        chunk_sizes: List of chunk sizes to test
        overlaps: List of overlap sizes to test
        num_docs: Number of documents to use
        output_path: Optional path to save results

    Returns:
        dict with all results
    """
    print("\n" + "=" * 80)
    print("CHUNK SIZE & OVERLAP SENSITIVITY ANALYSIS")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Chunk sizes: {chunk_sizes}")
    print(f"  Overlaps: {overlaps}")
    print(f"  Documents: {num_docs}")
    print(f"  Queries: {len(EVAL_QUERIES)}")
    print()

    # Load documents
    source_path = Path("data/docs/kubernetes")
    documents = load_sample_documents(source_path, max_docs=num_docs)

    if not documents:
        logger.error("No documents loaded, cannot run benchmark")
        return {}

    # Run benchmarks for all combinations
    results = []
    total_configs = len(chunk_sizes) * len(overlaps)
    current = 0

    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            current += 1

            # Skip invalid configs (overlap >= chunk_size)
            if overlap >= chunk_size:
                logger.warning(
                    f"Skipping invalid config: overlap {overlap} >= chunk_size {chunk_size}"
                )
                continue

            print(f"\n[{current}/{total_configs}] Testing: size={chunk_size}, overlap={overlap}")
            print("-" * 80)

            try:
                result = benchmark_chunking_config(
                    documents=documents,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    eval_queries=EVAL_QUERIES,
                    top_k=5,
                )
                results.append(result)

                # Print summary
                print(f"  Chunks: {result['chunk_count']}")
                print(f"  Avg chunk length: {result['avg_chunk_length']:.0f} chars")
                print(f"  Ingestion: {result['ingestion_time_sec']:.2f}s")
                print(f"  Search latency: {result['avg_search_latency_ms']:.1f}ms")
                print(f"  Precision@5: {result['avg_precision@k']:.3f}")
                print(f"  Term coverage: {result['avg_term_coverage']:.3f}")

            except Exception as e:
                logger.error(f"Error testing config: {e}")
                continue

    # Find best configurations
    if not results:
        logger.error("No successful benchmark runs")
        return {}

    best_precision = max(results, key=lambda x: x["avg_precision@k"])
    best_speed = min(results, key=lambda x: x["avg_search_latency_ms"])
    best_balance = max(
        results,
        key=lambda x: x["avg_precision@k"]
        - (x["avg_search_latency_ms"] / 1000),  # Precision - latency penalty
    )

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\n🏆 Best Precision:")
    print(f"  Chunk size: {best_precision['chunk_size']}")
    print(f"  Overlap: {best_precision['chunk_overlap']}")
    print(f"  Precision@5: {best_precision['avg_precision@k']:.3f}")
    print(f"  Search latency: {best_precision['avg_search_latency_ms']:.1f}ms")

    print("\n⚡ Best Speed:")
    print(f"  Chunk size: {best_speed['chunk_size']}")
    print(f"  Overlap: {best_speed['chunk_overlap']}")
    print(f"  Search latency: {best_speed['avg_search_latency_ms']:.1f}ms")
    print(f"  Precision@5: {best_speed['avg_precision@k']:.3f}")

    print("\n⚖️  Best Balance (Precision - Speed Trade-off):")
    print(f"  Chunk size: {best_balance['chunk_size']}")
    print(f"  Overlap: {best_balance['chunk_overlap']}")
    print(f"  Precision@5: {best_balance['avg_precision@k']:.3f}")
    print(f"  Search latency: {best_balance['avg_search_latency_ms']:.1f}ms")

    # Insights
    print("\n📊 Key Insights:")

    # Chunk size impact
    precision_by_size = {}
    for r in results:
        size = r["chunk_size"]
        if size not in precision_by_size:
            precision_by_size[size] = []
        precision_by_size[size].append(r["avg_precision@k"])

    print("\n  Precision by chunk size:")
    for size in sorted(precision_by_size.keys()):
        avg_prec = mean(precision_by_size[size])
        print(f"    {size:5d} chars: {avg_prec:.3f}")

    # Overlap impact
    precision_by_overlap = {}
    for r in results:
        overlap = r["chunk_overlap"]
        if overlap not in precision_by_overlap:
            precision_by_overlap[overlap] = []
        precision_by_overlap[overlap].append(r["avg_precision@k"])

    print("\n  Precision by overlap:")
    for overlap in sorted(precision_by_overlap.keys()):
        avg_prec = mean(precision_by_overlap[overlap])
        print(f"    {overlap:5d} chars: {avg_prec:.3f}")

    # Query type analysis (using best config)
    print("\n  Precision by query type (best config):")
    for qtype, prec in best_precision["precision_by_type"].items():
        print(f"    {qtype:12s}: {prec:.3f}")

    # Save results
    output_data = {
        "config": {
            "chunk_sizes": chunk_sizes,
            "overlaps": overlaps,
            "num_docs": num_docs,
            "num_queries": len(EVAL_QUERIES),
        },
        "all_results": results,
        "best_configurations": {
            "best_precision": best_precision,
            "best_speed": best_speed,
            "best_balance": best_balance,
        },
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
    parser = argparse.ArgumentParser(
        description="Benchmark different chunk sizes and overlaps for optimal RAG performance"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="256,512,800,1024,1536",
        help="Comma-separated chunk sizes to test (default: 256,512,800,1024,1536)",
    )
    parser.add_argument(
        "--overlaps",
        type=str,
        default="50,100,200,400",
        help="Comma-separated overlap sizes to test (default: 50,100,200,400)",
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=50,
        help="Number of documents to use (default: 50)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test fewer combinations (sizes: 512,800,1024; overlaps: 100,200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/chunking_analysis.json",
        help="Output path for results (default: benchmark_results/chunking_analysis.json)",
    )

    args = parser.parse_args()

    # Parse configurations
    if args.quick:
        chunk_sizes = [512, 800, 1024]
        overlaps = [100, 200]
    else:
        chunk_sizes = [int(x.strip()) for x in args.sizes.split(",")]
        overlaps = [int(x.strip()) for x in args.overlaps.split(",")]

    output_path = Path(args.output)

    # Run benchmark
    try:
        results = run_chunking_benchmark(
            chunk_sizes=chunk_sizes,
            overlaps=overlaps,
            num_docs=args.docs,
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

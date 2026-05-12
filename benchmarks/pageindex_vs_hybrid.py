"""
PageIndex vs Hybrid Search — standalone benchmark script.

Usage:
    python benchmarks/pageindex_vs_hybrid.py [--url http://localhost:8000]

Prerequisites:
    1. API server running  (uvicorn app.main:app)
    2. Documents already ingested into BOTH pipelines:
         POST /documents          → hybrid index
         POST /pageindex/documents → PageIndex index
    3. Ollama running with the configured model

What it measures:
    - Wall-clock latency per query (full round-trip)
    - Number of sources/sections retrieved
    - LLM-as-judge quality scores (faithfulness, answer_relevance,
      context_relevance, overall_score) via POST /evaluate
    - Winner per metric, aggregated across all queries

Output:
    - Console summary table
    - JSON file: benchmarks/pageindex_vs_hybrid_<timestamp>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

DEFAULT_URL = "http://localhost:8000"

TEST_QUERIES = [
    "How does Kubernetes handle pod networking between nodes?",
    "What is the difference between a Deployment and a StatefulSet?",
    "How do I set CPU and memory resource limits for a container?",
    "What are Kubernetes NetworkPolicies and how do I apply them?",
    "How does the Horizontal Pod Autoscaler decide when to scale?",
    "What is the role of etcd in a Kubernetes control plane?",
    "How do PersistentVolumeClaims bind to PersistentVolumes?",
    "What is the difference between ClusterIP, NodePort, and LoadBalancer?",
]


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def post(client: httpx.Client, base_url: str, path: str, body: dict) -> dict:
    try:
        resp = client.post(f"{base_url}{path}", json=body)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "detail": e.response.text[:200]}
    except Exception as e:
        return {"error": str(e)}


def run_hybrid(client: httpx.Client, base_url: str, query: str, top_k: int) -> tuple[dict, float]:
    t0 = time.perf_counter()
    data = post(
        client, base_url, "/query", {"query": query, "top_k": top_k, "retrieval_method": "hybrid"}
    )
    latency = time.perf_counter() - t0
    return data, latency


def run_pageindex(
    client: httpx.Client, base_url: str, query: str, top_k: int
) -> tuple[dict, float]:
    t0 = time.perf_counter()
    data = post(
        client,
        base_url,
        "/query",
        {"query": query, "top_k": top_k, "retrieval_method": "pageindex"},
    )
    latency = time.perf_counter() - t0
    return data, latency


def run_evaluate(
    client: httpx.Client, base_url: str, query: str, answer: str, context: list[str]
) -> dict:
    data = post(
        client,
        base_url,
        "/evaluate",
        {"query": query, "answer": answer, "context": context},
    )
    return data.get("metrics", {})


# ── Display helpers ───────────────────────────────────────────────────────────


def avg(lst: list, key: str) -> float:
    vals = [r[key] for r in lst if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def winner_arrow(h_val: float, pi_val: float, higher_better: bool) -> str:
    if higher_better:
        return "← Hybrid" if h_val > pi_val else "→ PageIndex"
    return "← Hybrid" if h_val < pi_val else "→ PageIndex"


def row(label: str, h: float, pi: float, higher_better: bool = True, fmt: str = ".3f") -> str:
    arrow = winner_arrow(h, pi, higher_better)
    return f"  {label:<32} {h:>10{fmt}}   {pi:>10{fmt}}   {arrow}"


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PageIndex vs Hybrid benchmark")
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip LLM-as-judge evaluation (faster)",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    top_k = args.top_k

    print("=" * 72)
    print("       PageIndex  vs  Hybrid Search  —  Benchmark")
    print(f"       API: {base_url}   top_k={top_k}")
    print("=" * 72)

    hybrid_rows: list[dict] = []
    pageindex_rows: list[dict] = []

    with httpx.Client(timeout=180.0) as client:
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n[{i}/{len(TEST_QUERIES)}] {query[:65]}...")

            h_data, h_latency = run_hybrid(client, base_url, query, top_k)
            if "error" in h_data:
                print(f"  Hybrid error: {h_data['error']}")
                continue

            pi_data, pi_latency = run_pageindex(client, base_url, query, top_k)
            if "error" in pi_data:
                print(f"  PageIndex error: {pi_data['error']}")
                continue

            h_ctx = [s["content"] for s in h_data.get("sources", [])]
            pi_ctx = [s["content"] for s in pi_data.get("sources", [])]

            h_eval: dict = {}
            pi_eval: dict = {}

            if not args.no_eval:
                h_eval = run_evaluate(client, base_url, query, h_data.get("answer", ""), h_ctx)
                pi_eval = run_evaluate(client, base_url, query, pi_data.get("answer", ""), pi_ctx)

            print(
                f"  Hybrid     latency={h_latency:.2f}s  "
                f"sources={len(h_ctx)}  "
                f"overall={h_eval.get('overall_score', 'n/a')}"
            )
            print(
                f"  PageIndex  latency={pi_latency:.2f}s  "
                f"sections={len(pi_ctx)}  "
                f"overall={pi_eval.get('overall_score', 'n/a')}"
            )

            hybrid_rows.append(
                {
                    "query": query,
                    "latency_s": h_latency,
                    "generation_time_s": h_data.get("generation_time", 0),
                    "num_sources": len(h_ctx),
                    "cache_hit": h_data.get("cache_hit", False),
                    **{f"eval_{k}": v for k, v in h_eval.items()},
                }
            )
            pageindex_rows.append(
                {
                    "query": query,
                    "latency_s": pi_latency,
                    "generation_time_s": pi_data.get("generation_time", 0),
                    "num_sources": len(pi_ctx),
                    **{f"eval_{k}": v for k, v in pi_eval.items()},
                }
            )

    if not hybrid_rows:
        print("\nNo results collected — is the server running and documents ingested?")
        sys.exit(1)

    # ── Summary table ─────────────────────────────────────────────────────────

    h_avg_l = avg(hybrid_rows, "latency_s")
    pi_avg_l = avg(pageindex_rows, "latency_s")

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print(f"  {'Metric':<32} {'Hybrid':>10}   {'PageIndex':>10}   Winner")
    print("-" * 72)
    print(row("Avg latency (s)", h_avg_l, pi_avg_l, higher_better=False))
    print(
        row(
            "Avg generation time (s)",
            avg(hybrid_rows, "generation_time_s"),
            avg(pageindex_rows, "generation_time_s"),
            higher_better=False,
        )
    )
    print(
        row(
            "Avg sources / sections",
            avg(hybrid_rows, "num_sources"),
            avg(pageindex_rows, "num_sources"),
        )
    )

    if not args.no_eval:
        print("-" * 72)
        for metric in ("faithfulness", "answer_relevance", "context_relevance", "overall_score"):
            print(
                row(
                    f"Avg {metric}",
                    avg(hybrid_rows, f"eval_{metric}"),
                    avg(pageindex_rows, f"eval_{metric}"),
                )
            )

    print("=" * 72)
    print("  ← Hybrid wins   │   → PageIndex wins")

    # ── Persist results ───────────────────────────────────────────────────────

    output = {
        "config": {"base_url": base_url, "top_k": top_k, "queries": len(hybrid_rows)},
        "hybrid": hybrid_rows,
        "pageindex": pageindex_rows,
        "summary": {
            "hybrid": {
                "avg_latency_s": h_avg_l,
                "avg_generation_time_s": avg(hybrid_rows, "generation_time_s"),
                "avg_sources": avg(hybrid_rows, "num_sources"),
                "avg_overall_score": avg(hybrid_rows, "eval_overall_score"),
                "avg_faithfulness": avg(hybrid_rows, "eval_faithfulness"),
                "avg_answer_relevance": avg(hybrid_rows, "eval_answer_relevance"),
                "avg_context_relevance": avg(hybrid_rows, "eval_context_relevance"),
            },
            "pageindex": {
                "avg_latency_s": pi_avg_l,
                "avg_generation_time_s": avg(pageindex_rows, "generation_time_s"),
                "avg_sources": avg(pageindex_rows, "num_sources"),
                "avg_overall_score": avg(pageindex_rows, "eval_overall_score"),
                "avg_faithfulness": avg(pageindex_rows, "eval_faithfulness"),
                "avg_answer_relevance": avg(pageindex_rows, "eval_answer_relevance"),
                "avg_context_relevance": avg(pageindex_rows, "eval_context_relevance"),
            },
        },
    }

    out_path = Path(__file__).parent / f"pageindex_vs_hybrid_{int(time.time())}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Full results → {out_path.name}\n")


if __name__ == "__main__":
    main()

"""Benchmarking utilities for ingestion performance measurement"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Performance metrics for a benchmark run"""

    total_documents: int = 0
    total_chunks: int = 0
    total_time_seconds: float = 0.0
    documents_per_second: float = 0.0
    chunks_per_second: float = 0.0
    avg_time_per_document: float = 0.0
    avg_chunks_per_document: float = 0.0

    # Detailed phase timings
    loading_time: float = 0.0
    embedding_time: float = 0.0
    vector_store_time: float = 0.0
    bm25_time: float = 0.0
    chunking_time: float = 0.0

    # Resource metrics
    peak_memory_mb: Optional[float] = None
    avg_cpu_percent: Optional[float] = None

    # Parallel processing stats
    num_workers: int = 1
    batch_size: int = 1
    parallel_mode: str = "sequential"

    # Error tracking
    errors: int = 0
    error_rate: float = 0.0


class PerformanceBenchmark:
    """Benchmark ingestion performance with detailed metrics"""

    def __init__(self, name: str = "ingestion"):
        self.name = name
        self.start_time = None
        self.end_time = None

        # Phase timers
        self.phase_timers = defaultdict(list)
        self.current_phase = None
        self.phase_start_time = None

        # Counters
        self.document_count = 0
        self.chunk_count = 0
        self.error_count = 0

        # Individual document timings
        self.document_times = []

        # Configuration
        self.config = {}

    def start(self, config: Optional[dict] = None):
        """Start the benchmark"""
        self.start_time = time.time()
        self.config = config or {}
        logger.info(f"🏁 Benchmark '{self.name}' started")

    def end(self):
        """End the benchmark"""
        self.end_time = time.time()
        logger.info(f"🏁 Benchmark '{self.name}' completed")

    def phase_start(self, phase_name: str):
        """Start timing a phase"""
        if self.current_phase:
            self.phase_end()
        self.current_phase = phase_name
        self.phase_start_time = time.time()

    def phase_end(self):
        """End timing current phase"""
        if self.current_phase and self.phase_start_time:
            duration = time.time() - self.phase_start_time
            self.phase_timers[self.current_phase].append(duration)
            self.current_phase = None
            self.phase_start_time = None

    def record_document(self, num_chunks: int, duration: float):
        """Record document processing metrics"""
        self.document_count += 1
        self.chunk_count += num_chunks
        self.document_times.append(duration)

    def record_error(self):
        """Record an error"""
        self.error_count += 1

    def get_metrics(self) -> BenchmarkMetrics:
        """Calculate and return benchmark metrics"""
        if not self.start_time or not self.end_time:
            logger.warning("Benchmark not completed, returning partial metrics")

        total_time = (self.end_time or time.time()) - (self.start_time or time.time())

        metrics = BenchmarkMetrics(
            total_documents=self.document_count,
            total_chunks=self.chunk_count,
            total_time_seconds=total_time,
            documents_per_second=(self.document_count / total_time if total_time > 0 else 0),
            chunks_per_second=self.chunk_count / total_time if total_time > 0 else 0,
            avg_time_per_document=(
                sum(self.document_times) / len(self.document_times) if self.document_times else 0
            ),
            avg_chunks_per_document=(
                self.chunk_count / self.document_count if self.document_count > 0 else 0
            ),
            # Phase timings
            loading_time=sum(self.phase_timers.get("loading", [])),
            embedding_time=sum(self.phase_timers.get("embedding", [])),
            vector_store_time=sum(self.phase_timers.get("vector_store", [])),
            bm25_time=sum(self.phase_timers.get("bm25", [])),
            chunking_time=sum(self.phase_timers.get("chunking", [])),
            # Configuration
            num_workers=self.config.get("num_workers", 1),
            batch_size=self.config.get("batch_size", 1),
            parallel_mode=self.config.get("parallel_mode", "sequential"),
            # Errors
            errors=self.error_count,
            error_rate=(self.error_count / self.document_count if self.document_count > 0 else 0),
        )

        return metrics

    def print_summary(self):
        """Print benchmark summary"""
        metrics = self.get_metrics()

        print("\n" + "=" * 80)
        print(f"PERFORMANCE BENCHMARK: {self.name}")
        print("=" * 80)
        print(f"Mode: {metrics.parallel_mode}")
        print(f"Workers: {metrics.num_workers}")
        print(f"Batch Size: {metrics.batch_size}")
        print()

        print("-" * 80)
        print("THROUGHPUT METRICS")
        print("-" * 80)
        print(f"Total Documents: {metrics.total_documents:,}")
        print(f"Total Chunks: {metrics.total_chunks:,}")
        print(f"Total Time: {timedelta(seconds=int(metrics.total_time_seconds))}")
        print()
        print(f"📊 Documents/sec: {metrics.documents_per_second:.2f}")
        print(f"📊 Chunks/sec: {metrics.chunks_per_second:.2f}")
        print(f"📊 Avg time/document: {metrics.avg_time_per_document:.3f}s")
        print(f"📊 Avg chunks/document: {metrics.avg_chunks_per_document:.1f}")
        print()

        if any([metrics.embedding_time, metrics.vector_store_time, metrics.chunking_time]):
            print("-" * 80)
            print("PHASE BREAKDOWN")
            print("-" * 80)
            if metrics.loading_time > 0:
                pct = (metrics.loading_time / metrics.total_time_seconds) * 100
                print(f"Loading:      {metrics.loading_time:7.2f}s ({pct:5.1f}%)")
            if metrics.chunking_time > 0:
                pct = (metrics.chunking_time / metrics.total_time_seconds) * 100
                print(f"Chunking:     {metrics.chunking_time:7.2f}s ({pct:5.1f}%)")
            if metrics.embedding_time > 0:
                pct = (metrics.embedding_time / metrics.total_time_seconds) * 100
                print(f"Embedding:    {metrics.embedding_time:7.2f}s ({pct:5.1f}%)")
            if metrics.vector_store_time > 0:
                pct = (metrics.vector_store_time / metrics.total_time_seconds) * 100
                print(f"Vector Store: {metrics.vector_store_time:7.2f}s ({pct:5.1f}%)")
            if metrics.bm25_time > 0:
                pct = (metrics.bm25_time / metrics.total_time_seconds) * 100
                print(f"BM25:         {metrics.bm25_time:7.2f}s ({pct:5.1f}%)")
            print()

        if metrics.errors > 0:
            print("-" * 80)
            print("ERROR METRICS")
            print("-" * 80)
            print(f"Errors: {metrics.errors}")
            print(f"Error Rate: {metrics.error_rate:.2%}")
            print()

        print("=" * 80)

    def save_report(self, output_path: Optional[Path] = None) -> Path:
        """Save detailed benchmark report to JSON"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"./benchmarks/{self.name}_{timestamp}.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = self.get_metrics()

        report = {
            "benchmark_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "start_time": (
                datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None
            ),
            "end_time": (
                datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None
            ),
            "configuration": self.config,
            "metrics": asdict(metrics),
            "document_times": {
                "min": min(self.document_times) if self.document_times else 0,
                "max": max(self.document_times) if self.document_times else 0,
                "median": (
                    sorted(self.document_times)[len(self.document_times) // 2]
                    if self.document_times
                    else 0
                ),
                "p95": (
                    sorted(self.document_times)[int(len(self.document_times) * 0.95)]
                    if self.document_times
                    else 0
                ),
                "p99": (
                    sorted(self.document_times)[int(len(self.document_times) * 0.99)]
                    if self.document_times
                    else 0
                ),
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 Benchmark report saved to: {output_path}")
        return output_path


class BenchmarkComparison:
    """Compare multiple benchmark runs"""

    @staticmethod
    def compare(reports: list[Path]) -> dict[str, Any]:
        """Compare multiple benchmark reports"""
        results = []

        for report_path in reports:
            with open(report_path) as f:
                results.append(json.load(f))

        if not results:
            return {}

        # Extract key metrics for comparison
        comparison = {
            "reports": len(results),
            "configurations": [r.get("configuration", {}) for r in results],
            "throughput": [
                {
                    "name": r.get("benchmark_name", "unknown"),
                    "mode": r.get("metrics", {}).get("parallel_mode", "unknown"),
                    "workers": r.get("metrics", {}).get("num_workers", 1),
                    "docs_per_sec": r.get("metrics", {}).get("documents_per_second", 0),
                    "chunks_per_sec": r.get("metrics", {}).get("chunks_per_second", 0),
                    "total_time": r.get("metrics", {}).get("total_time_seconds", 0),
                }
                for r in results
            ],
        }

        # Find fastest
        fastest = max(comparison["throughput"], key=lambda x: x["docs_per_sec"])
        comparison["fastest"] = fastest

        # Calculate speedup relative to first (assumed baseline)
        if len(results) > 1:
            baseline = comparison["throughput"][0]
            for i, perf in enumerate(comparison["throughput"][1:], 1):
                speedup = (
                    perf["docs_per_sec"] / baseline["docs_per_sec"]
                    if baseline["docs_per_sec"] > 0
                    else 0
                )
                time_reduction = (
                    ((baseline["total_time"] - perf["total_time"]) / baseline["total_time"]) * 100
                    if baseline["total_time"] > 0
                    else 0
                )
                comparison["throughput"][i]["speedup"] = speedup
                comparison["throughput"][i]["time_reduction_pct"] = time_reduction

        return comparison

    @staticmethod
    def print_comparison(reports: list[Path]):
        """Print comparison of benchmark reports"""
        comparison = BenchmarkComparison.compare(reports)

        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON")
        print("=" * 80)
        print(f"Comparing {comparison['reports']} benchmark runs")
        print()

        print("-" * 80)
        print("THROUGHPUT COMPARISON")
        print("-" * 80)
        print(
            f"{'Mode':<20} {'Workers':>8} {'Docs/sec':>12} {'Chunks/sec':>12} {'Time':>10} {'Speedup':>10}"
        )
        print("-" * 80)

        for i, perf in enumerate(comparison["throughput"]):
            mode = perf["mode"]
            workers = perf["workers"]
            docs_sec = perf["docs_per_sec"]
            chunks_sec = perf["chunks_per_sec"]
            total_time = timedelta(seconds=int(perf["total_time"]))
            speedup = perf.get("speedup", 1.0)

            speedup_str = f"{speedup:.2f}x" if i > 0 else "baseline"

            print(
                f"{mode:<20} {workers:>8} {docs_sec:>12.2f} {chunks_sec:>12.2f} {str(total_time):>10} {speedup_str:>10}"
            )

        print()
        print(
            f"🏆 Fastest: {comparison['fastest']['mode']} with {comparison['fastest']['workers']} workers"
        )
        print(f"   {comparison['fastest']['docs_per_sec']:.2f} docs/sec")

        if len(comparison["throughput"]) > 1:
            best_speedup = max([p.get("speedup", 1.0) for p in comparison["throughput"]])
            print(f"\n⚡ Best Speedup: {best_speedup:.2f}x faster than baseline")

        print("=" * 80)


def create_benchmark(name: str = "ingestion") -> PerformanceBenchmark:
    """Factory function to create a benchmark"""
    return PerformanceBenchmark(name=name)

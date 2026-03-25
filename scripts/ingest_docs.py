#!/usr/bin/env python3
"""
Unified document ingestion script

Supports both direct RAG pipeline and API-based ingestion with optimized settings
for preventing Ollama overload.

Usage:
    # Direct pipeline (fastest, recommended)
    python scripts/ingest_docs.py --source ~/k8s-website/content/en/docs

    # Via API (useful for remote server)
    python scripts/ingest_docs.py --source data/docs/kubernetes --api

    # Multiple sources
    python scripts/ingest_docs.py --source data/docs/kubernetes --source data/docs/cilium

    # Custom settings
    python scripts/ingest_docs.py --source ~/docs --batch-size 10 --workers 2
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.rag import get_rag_pipeline
from app.utils.benchmarking import create_benchmark
from app.utils.error_tracking import create_error_tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Optimized defaults (prevent Ollama overload)
DEFAULT_BATCH_SIZE = 5
DEFAULT_NUM_WORKERS = 1
DEFAULT_MAX_CONCURRENT_EMBEDDINGS = 2
API_URL = "http://localhost:8000"


def load_docs_from_path(docs_path: Path, source_name: str) -> list[dict]:
    """
    Load markdown documents from a path (file or directory)

    Args:
        docs_path: Path to file or directory
        source_name: Name to identify the source (e.g., "Kubernetes", "Cilium")

    Returns:
        List of document dictionaries
    """
    documents = []

    if docs_path.is_file():
        # Single file
        if docs_path.suffix == ".md":
            try:
                content = docs_path.read_text(encoding="utf-8")
                if len(content.strip()) >= 50:
                    title = docs_path.stem.replace("-", " ").replace("_", " ").title()
                    documents.append(
                        {
                            "title": f"{source_name}: {title}",
                            "content": content,
                            "metadata": {
                                "source": source_name.lower(),
                                "file": str(docs_path.name),
                                "category": docs_path.parent.name,
                            },
                        }
                    )
            except Exception as e:
                logger.warning(f"Error loading {docs_path}: {e}")

    elif docs_path.is_dir():
        # Directory of files
        md_files = list(docs_path.rglob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files in {docs_path}")

        for idx, file_path in enumerate(md_files, 1):
            # Skip index files (usually empty)
            if file_path.name.endswith("_index.md") or file_path.name == "_index.md":
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                if len(content.strip()) < 50:
                    continue

                title = file_path.stem.replace("-", " ").replace("_", " ").title()
                documents.append(
                    {
                        "title": f"{source_name}: {title}",
                        "content": content,
                        "metadata": {
                            "source": source_name.lower(),
                            "file": str(file_path.relative_to(docs_path)),
                            "category": file_path.parent.name,
                        },
                    }
                )

                if idx % 100 == 0:
                    logger.info(f"  Loaded {idx}/{len(md_files)} files...")

            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                continue

    logger.info(f"Loaded {len(documents)} valid documents from {source_name}")
    return documents


def ingest_via_api(
    documents: list[dict],
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    max_concurrent_embeddings: int = DEFAULT_MAX_CONCURRENT_EMBEDDINGS,
) -> tuple[int, int, int]:
    """
    Ingest documents via API endpoint

    Args:
        documents: List of document dictionaries
        batch_size: Documents per batch
        num_workers: Number of parallel workers
        max_concurrent_embeddings: Max concurrent embedding requests

    Returns:
        Tuple of (successful_count, error_count, total_chunks)
    """
    total = len(documents)
    successful = 0
    errors = 0
    total_chunks = 0

    logger.info(
        f"Ingesting {total} documents via API "
        f"(batch_size={batch_size}, workers={num_workers}, "
        f"concurrent_embeddings={max_concurrent_embeddings})"
    )

    start_time = time.time()

    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        logger.info(f"Batch {batch_num}/{total_batches} ({len(batch)} documents)...")

        try:
            response = requests.post(
                f"{API_URL}/add-documents-batch",
                json={
                    "documents": batch,
                    "num_workers": num_workers,
                    "max_concurrent_embeddings": max_concurrent_embeddings,
                },
                timeout=600,
            )

            if response.status_code == 200:
                result = response.json()
                successful += result["successful"]
                errors += result["errors"]
                total_chunks += result["total_chunks"]
                logger.info(
                    f"  ✓ {result['successful']} docs, "
                    f"{result['total_chunks']} chunks "
                    f"({result['processing_time']:.1f}s)"
                )
            else:
                logger.error(f"  ✗ API error {response.status_code}")
                errors += len(batch)

        except requests.exceptions.Timeout:
            logger.error("  ✗ Request timeout")
            errors += len(batch)
        except Exception as e:
            logger.error(f"  ✗ {str(e)[:100]}")
            errors += len(batch)

        # Delay between batches to prevent Ollama overload
        if i + batch_size < total:
            time.sleep(1.0)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("API Ingestion Complete:")
    logger.info(f"  Success: {successful}/{total} docs")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Chunks: {total_chunks}")
    logger.info(f"  Time: {elapsed:.1f}s ({elapsed / total:.2f}s/doc)")
    logger.info("=" * 70)

    return successful, errors, total_chunks


def ingest_via_pipeline(
    documents: list[dict],
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    max_concurrent_embeddings: int = DEFAULT_MAX_CONCURRENT_EMBEDDINGS,
    benchmark_name: Optional[str] = None,
) -> tuple[int, int, int]:
    """
    Ingest documents directly via RAG pipeline

    Args:
        documents: List of document dictionaries
        batch_size: Batch size for progress reporting
        num_workers: Number of parallel workers
        max_concurrent_embeddings: Max concurrent embedding requests
        benchmark_name: Optional benchmark name

    Returns:
        Tuple of (successful_count, error_count, total_chunks)
    """
    # Initialize error tracker and benchmark
    error_tracker = create_error_tracker(output_dir="./error_reports")
    benchmark = create_benchmark(name=benchmark_name or "ingestion")

    # Start benchmark
    benchmark.start(
        config={
            "mode": "parallel" if num_workers > 1 else "sequential",
            "num_workers": num_workers,
            "batch_size": batch_size,
            "max_concurrent_embeddings": max_concurrent_embeddings,
        }
    )

    # Prepare document data
    doc_data_list = []
    for idx, doc in enumerate(documents, 1):
        doc_id = f"doc_{int(time.time() * 1000)}_{idx}"
        doc_data_list.append(
            {
                "doc_id": doc_id,
                "title": doc["title"],
                "content": doc["content"],
                "metadata": doc["metadata"],
            }
        )

    # Initialize RAG pipeline
    try:
        rag = get_rag_pipeline()
        logger.info("RAG pipeline initialized")
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")
        return 0, len(documents), 0

    # Progress callback
    def progress_callback(batch_num, total_batches, chunks_so_far, errors_so_far):
        logger.info(
            f"Progress: Batch {batch_num}/{total_batches}, "
            f"{chunks_so_far} chunks, {errors_so_far} errors"
        )

    # Ingest with batching
    benchmark.phase_start("embedding")
    total_chunks, success_count, error_count = rag.add_documents_batched(
        doc_data_list,
        batch_size=batch_size,
        num_workers=num_workers,
        max_concurrent_embeddings=max_concurrent_embeddings,
        progress_callback=progress_callback,
        benchmark=benchmark,
    )
    benchmark.phase_end()

    # End benchmark
    benchmark.end()

    # Summary
    logger.info("=" * 70)
    logger.info("Pipeline Ingestion Complete:")
    logger.info(f"  Success: {success_count}/{len(documents)} docs")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Chunks: {total_chunks}")
    logger.info("=" * 70)

    # Print benchmark
    benchmark.print_summary()

    # Save benchmark report
    benchmark_path = benchmark.save_report()
    logger.info(f"📊 Benchmark report: {benchmark_path}")

    # Print error summary
    if error_tracker.errors:
        error_tracker.print_summary()
        report_path = error_tracker.save_detailed_report()
        logger.info(f"📄 Error report: {report_path}")

    return success_count, error_count, total_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest Kubernetes docs (direct pipeline)
  python scripts/ingest_docs.py --source ~/k8s-website/content/en/docs

  # Ingest via API (useful for remote server)
  python scripts/ingest_docs.py --source data/docs/kubernetes --api

  # Multiple sources
  python scripts/ingest_docs.py --source data/docs/kubernetes --source data/docs/cilium

  # Conservative settings (prevent Ollama overload)
  python scripts/ingest_docs.py --source ~/docs --batch-size 5 --workers 1 --concurrent 2
        """,
    )

    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Path to documentation (file or directory). Can be specified multiple times.",
    )
    parser.add_argument(
        "--name",
        action="append",
        help="Source name (e.g., 'Kubernetes', 'Cilium'). Defaults to directory name.",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Use API endpoint instead of direct pipeline (default: direct)",
    )
    parser.add_argument(
        "--api-url",
        default=API_URL,
        help=f"API URL (default: {API_URL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_EMBEDDINGS,
        help=f"Max concurrent embeddings (default: {DEFAULT_MAX_CONCURRENT_EMBEDDINGS})",
    )

    args = parser.parse_args()

    # Update API URL if specified
    global API_URL
    API_URL = args.api_url

    # Validate sources
    source_paths = []
    for source in args.source:
        path = Path(source)
        if not path.exists():
            logger.error(f"Source does not exist: {source}")
            sys.exit(1)
        source_paths.append(path)

    # Default names if not provided
    source_names = args.name or [path.name.replace("-", " ").title() for path in source_paths]

    # Ensure we have enough names
    if len(source_names) < len(source_paths):
        for i in range(len(source_names), len(source_paths)):
            source_names.append(source_paths[i].name.replace("-", " ").title())

    # Check API health if using API mode
    if args.api:
        try:
            logger.info(f"Checking API health at {API_URL}...")
            health = requests.get(f"{API_URL}/health", timeout=5).json()
            if health["status"] != "healthy":
                logger.warning(f"API status: {health['status']}")
            logger.info(
                f"API healthy | LLM: {health['llm_available']} | "
                f"Backend: {health.get('vector_store_type', 'Unknown')}"
            )
        except Exception as e:
            logger.error(f"Cannot connect to API at {API_URL}: {e}")
            sys.exit(1)

    # Process each source
    total_success = 0
    total_errors = 0
    total_chunks = 0

    for source_path, source_name in zip(source_paths, source_names):
        logger.info("\n" + "=" * 70)
        logger.info(f"📂 {source_name.upper()}")
        logger.info("=" * 70)

        # Load documents
        documents = load_docs_from_path(source_path, source_name)

        if not documents:
            logger.warning(f"No documents found in {source_path}")
            continue

        # Ingest
        if args.api:
            success, errors, chunks = ingest_via_api(
                documents,
                batch_size=args.batch_size,
                num_workers=args.workers,
                max_concurrent_embeddings=args.concurrent,
            )
        else:
            success, errors, chunks = ingest_via_pipeline(
                documents,
                batch_size=args.batch_size,
                num_workers=args.workers,
                max_concurrent_embeddings=args.concurrent,
                benchmark_name=f"ingestion_{source_name.lower().replace(' ', '_')}",
            )

        total_success += success
        total_errors += errors
        total_chunks += chunks

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("🎉 FINAL SUMMARY")
    logger.info("=" * 70)
    for source_name in source_names:
        logger.info(f"  {source_name}: ingested")
    logger.info(f"  TOTAL: {total_success} docs, {total_chunks} chunks ({total_errors} errors)")
    logger.info("=" * 70)

    # Verify final stats (API mode only)
    if args.api:
        try:
            stats = requests.get(f"{API_URL}/stats", timeout=5).json()
            logger.info("\n📊 System Stats:")
            logger.info(f"  Vector store: {stats['vector_store']['vector_count']} vectors")
            logger.info(f"  BM25 index: {stats['bm25']['total_chunks']} chunks")
            logger.info(f"  Backend: {stats['vector_backend']}")
        except Exception as e:
            logger.warning(f"Could not fetch stats: {e}")

    logger.info("\n✨ Ingestion complete!\n")


if __name__ == "__main__":
    main()

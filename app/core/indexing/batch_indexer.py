"""Batch and parallel document indexing"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from app.core.indexing.document_processor import process_document

logger = logging.getLogger(__name__)


def add_documents_parallel(
    documents: list[dict],
    text_splitter,
    vector_store,
    bm25_index,
    use_faiss: bool = False,
    num_workers: int = 4,
    max_concurrent_embeddings: int = 20,
    progress_callback: Optional[Callable[[int, int, int, int], None]] = None,
    benchmark=None,
) -> tuple[int, int, int]:
    """
    Add multiple documents in parallel

    Args:
        documents: List of document dicts with doc_id, title, content, metadata
        text_splitter: Text splitter instance
        vector_store: Vector store instance
        bm25_index: BM25 index instance
        use_faiss: Whether FAISS is being used
        num_workers: Number of parallel workers
        max_concurrent_embeddings: Max concurrent embedding requests per worker
        progress_callback: Optional callback(batch_num, total_batches, chunks, errors)
        benchmark: Optional benchmark object for timing

    Returns:
        Tuple of (total_chunks, successful_count, error_count)
    """
    total_chunks = 0
    success_count = 0
    error_count = 0

    # Add max_concurrent_embeddings to each document
    for doc in documents:
        doc["max_concurrent_embeddings"] = max_concurrent_embeddings

    # Process documents in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all documents
        future_to_doc = {
            executor.submit(
                process_document, doc, text_splitter, vector_store, bm25_index, use_faiss
            ): doc
            for doc in documents
        }

        # Collect results as they complete
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                chunks, error = future.result()

                if error:
                    error_count += 1
                    if benchmark:
                        benchmark.record_error()
                else:
                    total_chunks += chunks
                    success_count += 1

            except Exception as e:
                error_count += 1
                logger.error(
                    f"Unexpected error processing document {doc.get('doc_id', 'unknown')}: {e}"
                )
                if benchmark:
                    benchmark.record_error()

    return total_chunks, success_count, error_count


def add_documents_batched(
    documents: list[dict],
    text_splitter,
    vector_store,
    bm25_index,
    use_faiss: bool = False,
    batch_size: int = 10,
    num_workers: int = 4,
    max_concurrent_embeddings: int = 20,
    progress_callback: Optional[Callable[[int, int, int, int], None]] = None,
    benchmark=None,
) -> tuple[int, int, int]:
    """
    Add documents in batches with parallel processing for better progress tracking

    Args:
        documents: List of document dictionaries
        text_splitter: Text splitter instance
        vector_store: Vector store instance
        bm25_index: BM25 index instance
        use_faiss: Whether FAISS is being used
        batch_size: Documents per batch
        num_workers: Number of parallel workers
        max_concurrent_embeddings: Max concurrent embedding requests
        progress_callback: Optional callback(batch_num, total_batches, chunks_so_far, errors_so_far)
        benchmark: Optional benchmark object

    Returns:
        Tuple of (total_chunks, successful_count, error_count)
    """
    total_chunks = 0
    total_success = 0
    total_errors = 0

    # Split into batches
    num_batches = (len(documents) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(documents), batch_size):
        batch = documents[batch_idx : batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1

        # Process batch in parallel
        chunks, success, errors = add_documents_parallel(
            batch,
            text_splitter,
            vector_store,
            bm25_index,
            use_faiss=use_faiss,
            num_workers=num_workers,
            max_concurrent_embeddings=max_concurrent_embeddings,
            benchmark=benchmark,
        )

        total_chunks += chunks
        total_success += success
        total_errors += errors

        # Progress callback
        if progress_callback:
            progress_callback(batch_num, num_batches, total_chunks, total_errors)

    return total_chunks, total_success, total_errors

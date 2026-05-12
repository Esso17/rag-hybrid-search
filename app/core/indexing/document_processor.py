"""Document processing and chunking"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.core.embedding.async_embedder import embed_batch_async

logger = logging.getLogger(__name__)

# Thread pool for running async code from sync context
_executor = ThreadPoolExecutor(max_workers=1)


def process_document(
    doc_data: dict, text_splitter, vector_store, bm25_index, use_faiss: bool = False
) -> tuple[int, Optional[Exception]]:
    """
    Process a single document with async embeddings

    Args:
        doc_data: Dictionary with doc_id, title, content, metadata, max_concurrent_embeddings
        text_splitter: Text splitter instance for chunking
        vector_store: Vector store instance (FAISS or in-memory)
        bm25_index: BM25 index instance
        use_faiss: Whether FAISS is being used

    Returns:
        Tuple of (num_chunks, error_if_any)
    """
    try:
        doc_id = doc_data["doc_id"]
        title = doc_data["title"]
        content = doc_data["content"]
        metadata = doc_data.get("metadata", {})

        # Split document into chunks
        chunks = text_splitter.split_text(content)

        # Contextual prefixing: prepend [title] to each chunk before embedding.
        # The prefix anchors the embedding to the document topic, improving
        # retrieval accuracy for chunks with anaphoric references or sparse keywords.
        # The original (unprefixed) text is stored and displayed; only the
        # embedding input carries the prefix.
        from app.config import settings as _settings

        if _settings.USE_CONTEXTUAL_PREFIX:
            chunks_for_embedding = [f"[{title}] {chunk}" for chunk in chunks]
        else:
            chunks_for_embedding = chunks

        # Generate embeddings asynchronously
        max_concurrent = doc_data.get("max_concurrent_embeddings", 20)

        def run_async_embeddings():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    embed_batch_async(chunks_for_embedding, max_concurrent=max_concurrent)
                )
            finally:
                loop.close()

        try:
            # Check if we're already in an event loop (e.g., FastAPI)
            asyncio.get_running_loop()
            # We're in a running loop, execute in thread pool
            future = _executor.submit(run_async_embeddings)
            embeddings = future.result()
        except RuntimeError:
            # No running loop, execute directly
            embeddings = run_async_embeddings()

        # Filter out failed embeddings
        valid_chunks = []
        valid_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                valid_chunks.append(chunk)
                valid_embeddings.append(embedding)

        if not valid_embeddings:
            raise ValueError("All embeddings failed to generate")

        # Upload to vector store
        payloads = []
        for idx, chunk in enumerate(valid_chunks):
            payloads.append(
                {
                    "document_id": doc_id,
                    "chunk_index": idx,
                    "content": chunk,
                    "title": title,
                    "metadata": metadata,
                }
            )

        vector_store.add_points(valid_embeddings, payloads)

        # Add to BM25 index — include per-chunk metadata so search results carry it
        chunk_metadatas = [
            {**metadata, "title": title, "document_id": doc_id} for _ in valid_chunks
        ]
        bm25_index.add_chunks(valid_chunks, metadatas=chunk_metadatas)

        return len(valid_chunks), None

    except Exception as e:
        logger.error(f"Error processing document {doc_data.get('doc_id', 'unknown')}: {e}")
        return 0, e

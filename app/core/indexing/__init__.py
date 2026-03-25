"""Indexing module for document processing and batch operations"""

from app.core.indexing.batch_indexer import add_documents_batched, add_documents_parallel
from app.core.indexing.document_processor import process_document

__all__ = ["process_document", "add_documents_parallel", "add_documents_batched"]

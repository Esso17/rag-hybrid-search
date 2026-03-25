"""BM25 keyword search with technical tokenization for K8s/Cilium"""

import logging
import math
import re
from collections import defaultdict

from app.config import settings

logger = logging.getLogger(__name__)


class TechnicalTokenizer:
    """Tokenizer optimized for technical documentation (K8s, Cilium, DevOps)"""

    def __init__(self):
        # Common K8s/Cilium terms to preserve
        self.preserve_terms = {
            "kubectl",
            "k8s",
            "cilium",
            "apiversion",
            "namespace",
            "deployment",
            "pod",
            "service",
            "ingress",
            "networkpolicy",
            "configmap",
            "secret",
            "daemonset",
            "statefulset",
            "ciliumnetworkpolicy",
            "ciliumendpoint",
            "hubble",
        }

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text with awareness of technical terms.
        Handles: CamelCase, kebab-case, dots, underscores
        """
        text_lower = text.lower()
        tokens = []

        # First, extract preserved terms
        for term in self.preserve_terms:
            if term in text_lower:
                tokens.append(term)

        # Split on whitespace and special chars
        words = re.findall(r"\b[\w-]+\b", text_lower)

        for word in words:
            if word in self.preserve_terms:
                continue
            tokens.extend(self._tokenize_word(word))

        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token not in seen and len(token) > 1:
                seen.add(token)
                unique_tokens.append(token)

        return unique_tokens

    def _tokenize_word(self, word: str) -> list[str]:
        """Tokenize a single word with technical awareness"""
        tokens = [word]

        # Handle kebab-case: cilium-agent -> [cilium-agent, cilium, agent]
        if "-" in word:
            tokens.extend(word.split("-"))

        # Handle dots: k8s.io -> [k8s.io, k8s, io]
        if "." in word:
            tokens.extend(word.split("."))

        # Handle underscores
        if "_" in word:
            tokens.extend(word.split("_"))

        # Handle CamelCase: NetworkPolicy -> [networkpolicy, network, policy]
        if re.search(r"[a-z][A-Z]", word):
            camel_parts = re.sub("([A-Z])", r" \1", word).split()
            tokens.extend([p.lower() for p in camel_parts])

        # Handle version numbers
        if re.match(r"v?\d+(\.\d+)*", word):
            tokens.append(word)
            version_parts = re.findall(r"\d+", word)
            tokens.extend(version_parts)

        return tokens


class BM25:
    """BM25 ranking algorithm with optional technical tokenization"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, use_technical_tokenizer: bool = None):
        """
        Initialize BM25
        k1: controls term frequency saturation
        b: controls length normalization
        use_technical_tokenizer: Use technical tokenization (auto from config if None)
        """
        self.k1 = k1
        self.b = b

        # Use technical tokenizer if enabled in config
        if use_technical_tokenizer is None:
            use_technical_tokenizer = getattr(settings, "USE_ENHANCED_BM25", True)

        self.use_technical_tokenizer = use_technical_tokenizer
        self.tokenizer = TechnicalTokenizer() if use_technical_tokenizer else None

        self.doc_freqs = []
        self.idf = {}
        self.avg_doc_len = 0
        self.doc_lens = []
        self.tokenized_docs = []
        self.corpus_size = 0

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text using configured tokenizer"""
        if self.use_technical_tokenizer and self.tokenizer:
            return self.tokenizer.tokenize(text)
        else:
            # Simple tokenization (original)
            return text.lower().split()

    def fit(self, corpus: list[str]):
        """Build BM25 index from corpus"""
        self.corpus_size = len(corpus)

        # Tokenize documents
        self.tokenized_docs = [self._tokenize(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0

        # Calculate document frequencies and IDF
        doc_freq = defaultdict(int)
        for doc in self.tokenized_docs:
            unique_tokens = set(doc)
            for token in unique_tokens:
                doc_freq[token] += 1

        # Calculate IDF values
        for token, freq in doc_freq.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

        tokenizer_type = "technical" if self.use_technical_tokenizer else "simple"
        logger.info(
            f"BM25 index built: {self.corpus_size} docs, {len(self.idf)} unique terms ({tokenizer_type} tokenization)"
        )

    def add_documents_incremental(self, new_docs: list[str]):
        """
        Add new documents to existing index without full rebuild (10-100x faster)
        """
        if not new_docs:
            return

        # Track document frequencies for IDF calculation
        doc_freq = defaultdict(int)

        # First pass: count existing document frequencies
        if self.tokenized_docs:
            for doc in self.tokenized_docs:
                unique_tokens = set(doc)
                for token in unique_tokens:
                    doc_freq[token] += 1

        # Tokenize and add new documents
        new_tokenized = [self._tokenize(doc) for doc in new_docs]
        new_lens = [len(doc) for doc in new_tokenized]

        # Update document frequencies with new docs
        for doc in new_tokenized:
            unique_tokens = set(doc)
            for token in unique_tokens:
                doc_freq[token] += 1

        # Extend index
        self.tokenized_docs.extend(new_tokenized)
        self.doc_lens.extend(new_lens)
        self.corpus_size = len(self.tokenized_docs)

        # Recalculate average document length
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0

        # Recalculate IDF values (only this part is O(unique_tokens))
        self.idf.clear()
        for token, freq in doc_freq.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

        tokenizer_type = "technical" if self.use_technical_tokenizer else "simple"
        logger.info(
            f"BM25 index updated incrementally: {self.corpus_size} total docs (+{len(new_docs)} new), "
            f"{len(self.idf)} unique terms ({tokenizer_type} tokenization)"
        )

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Search for query and return top-k document indices with scores
        Returns: List of (doc_index, score) tuples
        """
        if not self.tokenized_docs:
            return []

        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for token in query_tokens:
            idf_score = self.idf.get(token, 0)

            for doc_idx, doc_tokens in enumerate(self.tokenized_docs):
                token_freq = doc_tokens.count(token)
                if token_freq > 0:
                    # BM25 formula
                    numerator = idf_score * token_freq * (self.k1 + 1)
                    denominator = token_freq + self.k1 * (
                        1 - self.b + self.b * (self.doc_lens[doc_idx] / self.avg_doc_len)
                    )
                    scores[doc_idx] += numerator / denominator

        # Return top-k with scores
        results = [(idx, score) for idx, score in enumerate(scores) if score > 0]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class BM25Index:
    """Wrapper for managing BM25 index of document chunks"""

    def __init__(self, use_technical_tokenizer: bool = None):
        """
        Initialize BM25 index
        use_technical_tokenizer: Use technical tokenization (auto from config if None)
        """
        self.chunks: list[str] = []
        self.bm25 = BM25(use_technical_tokenizer=use_technical_tokenizer)

    def add_chunks(self, chunks: list[str]):
        """Add chunks to index (uses incremental update for performance)"""
        self.chunks.extend(chunks)
        # Use incremental update if index already exists, otherwise do full fit
        if self.bm25.tokenized_docs:
            self.bm25.add_documents_incremental(chunks)
        else:
            self.bm25.fit(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search chunks by keyword"""
        results = self.bm25.search(query, top_k=top_k)
        return [
            {"chunk_index": idx, "content": self.chunks[idx], "score": score}
            for idx, score in results
        ]

    def reset(self):
        """Clear index"""
        self.chunks = []
        self.bm25 = BM25(use_technical_tokenizer=self.bm25.use_technical_tokenizer)


# Global BM25 index
_bm25_index = None


def get_bm25_index() -> BM25Index:
    """Get or create global BM25 index with config-based tokenization"""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index

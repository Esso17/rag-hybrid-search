"""BM25 with inverted index and technical tokenization for Kubernetes docs"""

import json
import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_PRESERVE = frozenset(
    {
        "kubectl",
        "k8s",
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
        "replicaset",
        "horizontalpodautoscaler",
        "persistentvolumeclaim",
    }
)


def _tokenize(text: str) -> list[str]:
    """Technical-aware tokenizer: preserves Kubernetes terms, splits CamelCase/kebab/dots."""
    text_lower = text.lower()
    tokens = [t for t in _PRESERVE if t in text_lower]
    seen = set(tokens)
    for word in re.findall(r"\b[\w-]+\b", text_lower):
        if word in seen:
            continue
        parts = [word]
        if "-" in word:
            parts += word.split("-")
        if "." in word:
            parts += word.split(".")
        if "_" in word:
            parts += word.split("_")
        if re.search(r"[a-z][A-Z]", word):
            parts += [p.lower() for p in re.sub("([A-Z])", r" \1", word).split()]
        for p in parts:
            if len(p) > 1 and p not in seen:
                seen.add(p)
                tokens.append(p)
    return tokens


class BM25:
    """BM25 with inverted index — O(candidates) search instead of O(corpus)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokenized_docs: list[list[str]] = []
        self.doc_lens: list[int] = []
        self.avg_doc_len: float = 0.0
        self.corpus_size: int = 0
        self.inverted: dict[str, dict[int, int]] = defaultdict(dict)
        self.idf: dict[str, float] = {}

    def _recalc_idf(self):
        n = self.corpus_size
        self.idf = {
            t: math.log((n - len(d) + 0.5) / (len(d) + 0.5) + 1) for t, d in self.inverted.items()
        }

    def add(self, docs: list[str]):
        start = len(self.tokenized_docs)
        tokenized = [_tokenize(d) for d in docs]
        self.tokenized_docs.extend(tokenized)
        self.doc_lens.extend(len(t) for t in tokenized)
        self.corpus_size = len(self.tokenized_docs)
        self.avg_doc_len = sum(self.doc_lens) / self.corpus_size if self.corpus_size else 0.0
        for i, tokens in enumerate(tokenized, start):
            tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for token, freq in tf.items():
                self.inverted[token][i] = freq
        self._recalc_idf()

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        if not self.tokenized_docs:
            return []
        q_tokens = _tokenize(query)
        candidates: set[int] = set()
        for t in q_tokens:
            candidates.update(self.inverted.get(t, {}).keys())
        if not candidates:
            return []
        k1, b, avdl = self.k1, self.b, self.avg_doc_len or 1.0
        scores: dict[int, float] = {}
        for doc_idx in candidates:
            dl = self.doc_lens[doc_idx]
            s = 0.0
            for token in q_tokens:
                tf = self.inverted.get(token, {}).get(doc_idx, 0)
                if tf:
                    s += (
                        self.idf.get(token, 0) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avdl))
                    )
            if s > 0:
                scores[doc_idx] = s
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class BM25Index:
    """Manages a BM25 index over document chunks with disk persistence."""

    def __init__(self):
        self.chunks: list[str] = []
        self.chunk_metadata: list[dict] = []
        self.bm25 = BM25()

    def add_chunks(self, chunks: list[str], metadatas: list[dict] | None = None):
        prev = len(self.chunks)
        self.chunks.extend(chunks)
        self.chunk_metadata.extend(
            metadatas if metadatas and len(metadatas) == len(chunks) else [{}] * len(chunks)
        )
        self.bm25.add(chunks)
        if len(self.chunks) // 100 > prev // 100:
            self.save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return [
            {
                "chunk_index": idx,
                "content": self.chunks[idx],
                "score": score,
                "metadata": self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {},
            }
            for idx, score in self.bm25.search(query, top_k)
        ]

    def reset(self):
        self.chunks, self.chunk_metadata, self.bm25 = [], [], BM25()

    def save(self, data_dir: str = "/app/data"):
        try:
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(data_dir, "bm25_index.json"), "w") as f:
                json.dump({"chunks": self.chunks, "chunk_metadata": self.chunk_metadata}, f)
            logger.info(f"BM25 saved: {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"BM25 save failed: {e}")

    def load(self, data_dir: str = "/app/data") -> bool:
        try:
            path = os.path.join(data_dir, "bm25_index.json")
            if not os.path.exists(path):
                return False
            with open(path) as f:
                data = json.load(f)
            self.chunks = data["chunks"]
            self.chunk_metadata = data.get("chunk_metadata", [{}] * len(self.chunks))
            if self.chunks:
                self.bm25.add(self.chunks)
            logger.info(f"BM25 loaded: {len(self.chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"BM25 load failed: {e}")
            return False


_bm25_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
        _bm25_index.load()
    return _bm25_index


# Aliases kept for backward compatibility
get_bm25_inverted_index = get_bm25_index

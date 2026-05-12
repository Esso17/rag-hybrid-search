"""
PageIndex retriever — LLM-based hierarchical tree navigation.

HOW IT WORKS (retrieval phase):
  Traditional RAG turns a query into an embedding and searches for the
  nearest vectors.  The assumption is that semantic similarity ≈ relevance.

  PageIndex disagrees: similarity is not the same as relevance.
  Instead, at query time it shows the LLM a compact serialisation of the
  document structure (just the section titles and their line ranges — no
  full text) and asks:

      "Given this query and this table of contents, which sections
       are most likely to contain the answer?"

  This is exactly how a human expert would navigate a technical manual:
  scan the headings, reason about which chapter to open, then read it.

  The LLM responds with a JSON list of node IDs:
      {"relevant_nodes": ["2.1", "3", "3.2"]}

  Those IDs are resolved back to line ranges, the actual text is sliced
  out of the stored document, and the resulting passages are returned as
  context for the answer-generation step.

  One LLM call per document is made during retrieval.  For large corpora
  a first-pass title filter (not yet implemented) would narrow the set
  before the per-document calls.
"""

import json
import logging
import re

import httpx

from app.core.indexing.pageindex_store import PageIndexStore

logger = logging.getLogger(__name__)

# Matches the first {...} block that contains "relevant_nodes"
_JSON_RE = re.compile(r'\{[^{}]*"relevant_nodes"[^{}]*\}', re.DOTALL)


class PageIndexRetriever:
    """
    Retrieves document sections by having the LLM reason over the tree structure.

    The key difference from vector retrieval:
      - Vector search:    embed(query)  →  nearest(embed(chunk))
      - PageIndex search: structure(doc) + query  →  LLM reasoning  →  node IDs
    """

    def __init__(self, store: PageIndexStore, llm_url: str, llm_model: str):
        self.store = store
        self.llm_url = llm_url.rstrip("/")
        self.llm_model = llm_model
        self._client = httpx.Client(timeout=60.0)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        For each indexed document, ask the LLM which sections answer the query.
        Aggregate, rank by LLM confidence order, return top_k.
        """
        all_docs = self.store.get_all_trees()
        if not all_docs:
            return []

        results: list[dict] = []
        for doc_id, doc in all_docs.items():
            sections = self._navigate_document(query, doc_id, doc)
            results.extend(sections)

        # Sort by score (rank-based: 1.0 for first pick, 0.9 for second, …)
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def close(self):
        self._client.close()

    # ── Core navigation ───────────────────────────────────────────────────────

    def _navigate_document(self, query: str, doc_id: str, doc: dict) -> list[dict]:
        """
        Single-document navigation:
          1. Serialise the tree to a compact "table of contents" string.
          2. Ask the LLM which node IDs are relevant to the query.
          3. Fetch the text for each identified node.
          4. Return as result dicts (score descends with LLM rank order).
        """
        structure_text = self._serialise_tree(doc["tree"])

        prompt = (
            f'Document: "{doc["title"]}"\n\n'
            f"Structure:\n{structure_text}\n\n"
            f'Query: "{query}"\n\n'
            "Identify the sections most likely to answer the query.\n"
            'Return ONLY valid JSON: {"relevant_nodes": ["1", "2.1"]}\n'
            "Include at most 4 node IDs, most relevant first. "
            "Return an empty list if nothing is relevant."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document navigator. Given a document's table of "
                    "contents and a user query, identify which sections contain "
                    "the answer. Respond only with a JSON object."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        raw_response = self._call_llm(messages)
        node_ids = self._parse_node_ids(raw_response)

        if not node_ids:
            logger.debug(f"PageIndex: no relevant sections found in '{doc['title']}'")
            return []

        # Build a flat id→node map for section title lookup
        node_map: dict[str, dict] = {}
        self._build_flat_map(doc["tree"], node_map)

        results = []
        for rank, node_id in enumerate(node_ids):
            content = self.store.get_section_content(doc_id, node_id)
            if not content:
                continue
            node = node_map.get(node_id, {})
            results.append(
                {
                    "content": content,
                    # Rank-based score: first pick = 1.0, second = 0.9, …
                    "score": max(0.1, 1.0 - rank * 0.1),
                    "document_id": doc_id,
                    "chunk_index": 0,
                    "source": "pageindex",
                    "title": doc["title"],
                    "section_title": node.get("title", ""),
                    "node_id": node_id,
                    "metadata": doc.get("metadata", {}),
                }
            )

        return results

    # ── Tree serialisation ────────────────────────────────────────────────────

    def _serialise_tree(self, nodes: list[dict], depth: int = 0) -> str:
        """
        Convert the tree to a compact indented list that the LLM can reason over:

            [1] Introduction (lines 1-10)
            [2] Architecture (lines 11-80)
              [2.1] Components (lines 12-40)
              [2.2] Networking (lines 41-80)
            [3] Configuration (lines 81-120)
        """
        lines = []
        for node in nodes:
            indent = "  " * depth
            lines.append(
                f"{indent}[{node['node_id']}] {node['title']} "
                f"(lines {node['line_start']}-{node['line_end']})"
            )
            if node["children"]:
                lines.append(self._serialise_tree(node["children"], depth + 1))
        return "\n".join(lines)

    # ── LLM interaction ───────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str:
        try:
            resp = self._client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"PageIndex: LLM navigation call failed: {e}")
            return ""

    def _parse_node_ids(self, llm_response: str) -> list[str]:
        """
        Extract the list of node IDs from the LLM's JSON response.
        Two strategies: regex search for the JSON block, then full-parse fallback.
        """
        # Strategy 1: find the JSON object with regex (handles leading prose)
        m = _JSON_RE.search(llm_response)
        if m:
            try:
                data = json.loads(m.group())
                return [str(n) for n in data.get("relevant_nodes", []) if n]
            except json.JSONDecodeError:
                pass

        # Strategy 2: try the entire response as JSON
        try:
            data = json.loads(llm_response.strip())
            return [str(n) for n in data.get("relevant_nodes", []) if n]
        except json.JSONDecodeError:
            pass

        logger.debug(f"PageIndex: unparseable response: {llm_response[:200]!r}")
        return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_flat_map(self, nodes: list[dict], result: dict) -> None:
        for node in nodes:
            result[node["node_id"]] = node
            self._build_flat_map(node["children"], result)

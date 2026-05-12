"""
PageIndex document store — builds a hierarchical tree from markdown headers.

HOW IT WORKS (indexing phase):
  Traditional RAG splits documents into fixed-size chunks and embeds them.
  PageIndex instead reads the document's own structure (headings: #, ##, ###…)
  to build a tree of named sections — exactly like a table of contents.

  Each tree node records:
    - node_id   : hierarchical label  ("1", "1.2", "1.2.3")
    - title     : the heading text
    - line_start: first line of the section (1-indexed)
    - line_end  : last line of the section (inclusive)
    - children  : list of sub-section nodes

  The full document text is stored as a list of lines so that any section's
  content can be fetched at retrieval time with a simple slice.

  No embeddings, no chunking, no vector store — just structured text.
"""

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")


class PageIndexStore:
    """Persists documents as hierarchical section trees derived from markdown headers."""

    def __init__(self, data_dir: str = "/app/data"):
        self.data_dir = data_dir
        # doc_id → {doc_id, title, lines, tree, metadata}
        self._store: dict[str, dict] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def add_document(self, doc_id: str, title: str, content: str, metadata: dict = None) -> dict:
        """Parse content into a section tree and persist it."""
        # Guarantee an H1 exists so the tree always has a root
        if not content.lstrip().startswith("#"):
            full_content = f"# {title}\n\n{content}"
        else:
            full_content = content

        lines = full_content.splitlines()
        tree = self._build_tree(lines)
        self._assign_ids(tree)
        self._assign_line_ends(tree, len(lines))

        self._store[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "lines": lines,
            "tree": tree,
            "metadata": metadata or {},
        }
        self.save()

        n = self._count_nodes(tree)
        logger.info(f"PageIndex: indexed '{title}' → {n} sections")
        return {"doc_id": doc_id, "sections": n}

    def get_document_tree(self, doc_id: str) -> Optional[dict]:
        return self._store.get(doc_id)

    def get_all_trees(self) -> dict:
        return self._store

    def get_section_content(self, doc_id: str, node_id: str) -> Optional[str]:
        """Return the raw text of a section identified by its node_id."""
        doc = self._store.get(doc_id)
        if not doc:
            return None
        node_map = self._build_node_map(doc["tree"])
        node = node_map.get(node_id)
        if not node:
            return None
        start = node["line_start"] - 1
        end = node["line_end"]
        return "\n".join(doc["lines"][start:end])

    def document_count(self) -> int:
        return len(self._store)

    def reset(self):
        self._store.clear()
        path = os.path.join(self.data_dir, "pageindex_store.json")
        if os.path.exists(path):
            os.remove(path)

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, "pageindex_store.json")
        with open(path, "w") as f:
            json.dump(self._store, f)

    def load(self) -> bool:
        path = os.path.join(self.data_dir, "pageindex_store.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                self._store = json.load(f)
            logger.info(f"PageIndex: loaded {len(self._store)} documents from disk")
            return True
        except Exception as e:
            logger.warning(f"PageIndex: failed to load store: {e}")
            return False

    # ── Tree construction ─────────────────────────────────────────────────────

    def _build_tree(self, lines: list[str]) -> list[dict]:
        """
        Walk through lines, detect markdown headers, and nest them into a tree.

        The algorithm keeps a stack of (level, node) pairs.  When a new header
        is found at level L:
          - pop everything from the stack with level >= L  (they are closed)
          - if the stack is non-empty, append the new node as a child of the top
          - otherwise append it as a root node
        """
        roots: list[dict] = []
        stack: list[tuple[int, dict]] = []

        for line_num, line in enumerate(lines, start=1):
            m = _HEADER_RE.match(line.rstrip())
            if not m:
                continue

            level = len(m.group(1))
            title = m.group(2).strip()

            node: dict = {
                "node_id": "",
                "title": title,
                "level": level,
                "line_start": line_num,
                "line_end": len(lines),
                "children": [],
            }

            while stack and stack[-1][0] >= level:
                stack.pop()

            if stack:
                stack[-1][1]["children"].append(node)
            else:
                roots.append(node)

            stack.append((level, node))

        if not roots:
            roots = [
                {
                    "node_id": "1",
                    "title": "Document",
                    "level": 1,
                    "line_start": 1,
                    "line_end": len(lines),
                    "children": [],
                }
            ]

        return roots

    def _assign_ids(self, nodes: list[dict], prefix: str = "") -> None:
        """Assign dot-separated hierarchical IDs: '1', '1.2', '1.2.3' …"""
        for i, node in enumerate(nodes, start=1):
            node["node_id"] = f"{prefix}{i}" if prefix else str(i)
            self._assign_ids(node["children"], f"{node['node_id']}.")

    def _assign_line_ends(self, nodes: list[dict], total_lines: int) -> None:
        """
        Set line_end for every node to (next node's line_start - 1).

        We flatten the tree in depth-first order to get the correct start-line
        sequence, then assign ends by looking at the successor.
        """
        flat: list[dict] = []
        self._flatten(nodes, flat)
        for i, node in enumerate(flat):
            if i + 1 < len(flat):
                node["line_end"] = flat[i + 1]["line_start"] - 1
            else:
                node["line_end"] = total_lines

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _flatten(self, nodes: list[dict], result: list) -> None:
        for node in nodes:
            result.append(node)
            self._flatten(node["children"], result)

    def _build_node_map(self, nodes: list[dict]) -> dict[str, dict]:
        """Build node_id → node dict for O(1) lookups."""
        result: dict[str, dict] = {}
        flat: list[dict] = []
        self._flatten(nodes, flat)
        for node in flat:
            result[node["node_id"]] = node
        return result

    def _count_nodes(self, nodes: list[dict]) -> int:
        total = len(nodes)
        for node in nodes:
            total += self._count_nodes(node["children"])
        return total


# ── Singleton ─────────────────────────────────────────────────────────────────

_store_instance: Optional[PageIndexStore] = None


def get_pageindex_store() -> PageIndexStore:
    global _store_instance
    if _store_instance is None:
        from app.config import settings

        data_dir = getattr(settings, "PAGEINDEX_DATA_DIR", "/app/data")
        _store_instance = PageIndexStore(data_dir=data_dir)
        _store_instance.load()
    return _store_instance

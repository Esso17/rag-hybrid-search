"""Base document loader with generic loading logic"""

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class BaseDocumentLoader:
    """Base class for loading and parsing documentation"""

    def __init__(self, source_path: str, version: str = "latest", component: str = "generic"):
        self.source_path = Path(source_path)
        self.version = version
        self.component = component
        self.supported_extensions = {".md", ".markdown", ".txt", ".yaml", ".yml"}

    def load_documents(self) -> list[dict]:
        """Load all documents from source"""
        documents = []

        if self.source_path.is_file():
            doc = self._load_single_file(self.source_path)
            if doc:
                documents.append(doc)
        elif self.source_path.is_dir():
            documents = self._load_directory(self.source_path)
        else:
            raise ValueError(f"Invalid source path: {self.source_path}")

        logger.info(f"Loaded {len(documents)} {self.component} documents from {self.source_path}")
        return documents

    def _load_directory(self, directory: Path) -> list[dict]:
        """Recursively load all documents from directory"""
        documents = []

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                doc = self._load_single_file(file_path)
                if doc:
                    documents.append(doc)

        return documents

    def _load_single_file(self, file_path: Path) -> Optional[dict]:
        """Load a single documentation file"""
        try:
            content = file_path.read_text(encoding="utf-8")

            # Extract metadata from file
            category = self._extract_category(file_path)
            doc_type = self._determine_doc_type(content, file_path)
            title = self._extract_title(content, file_path)
            tags = self._extract_tags(content)

            metadata = {
                "version": self.version,
                "category": category,
                "doc_type": doc_type,
                "component": self.component,
                "tags": tags,
                "source_file": str(file_path),
                "file_type": file_path.suffix,
            }

            return {"title": title, "content": content, "metadata": metadata}

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def _extract_title(self, content: str, file_path: Path) -> str:
        """Extract title from document"""
        # Try to find first H1 header
        h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()

        # Try YAML frontmatter
        yaml_match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if yaml_match:
            try:
                frontmatter = yaml.safe_load(yaml_match.group(1))
                if isinstance(frontmatter, dict) and "title" in frontmatter:
                    return frontmatter["title"]
            except Exception:  # noqa: S110  # nosec B110
                pass  # Ignore YAML parsing errors, fall back to filename

        # Fall back to filename
        return file_path.stem.replace("-", " ").replace("_", " ").title()

    def _extract_category(self, file_path: Path) -> str:
        """Extract category from file path - override in subclasses"""
        path_parts = file_path.parts

        # Common documentation categories
        categories = {
            "concept": "concepts",
            "task": "tasks",
            "tutorial": "tutorials",
            "reference": "reference",
            "setup": "setup",
            "config": "configuration",
            "admin": "administration",
        }

        for part in path_parts:
            part_lower = part.lower()
            for key, value in categories.items():
                if key in part_lower:
                    return value

        return "general"

    def _determine_doc_type(self, content: str, file_path: Path) -> str:
        """Determine document type"""
        content_lower = content.lower()

        # Check for tutorial indicators
        if any(
            word in content_lower
            for word in ["tutorial", "walkthrough", "getting started", "quickstart"]
        ):
            return "tutorial"

        # Check for reference indicators
        if file_path.suffix in [".yaml", ".yml"]:
            return "config"

        if any(word in content_lower for word in ["api reference", "specification", "schema"]):
            return "reference"

        # Check for task/guide indicators
        if any(word in content_lower for word in ["how to", "step-by-step", "instructions"]):
            return "guide"

        # Check for concept indicators
        if any(
            word in content_lower for word in ["overview", "introduction", "what is", "concepts"]
        ):
            return "concept"

        return "guide"

    def _extract_tags(self, content: str) -> list[str]:
        """Extract relevant tags from content - override in subclasses"""
        return []

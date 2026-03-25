"""Cilium documentation loader"""

from pathlib import Path

from .base_loader import BaseDocumentLoader


class CiliumDocumentLoader(BaseDocumentLoader):
    """Load and parse Cilium documentation"""

    def __init__(self, source_path: str, version: str = "latest"):
        super().__init__(source_path, version, component="cilium")

    def _extract_category(self, file_path: Path) -> str:
        """Extract category from file path with Cilium-specific categories"""
        path_parts = file_path.parts

        # Cilium-specific documentation categories
        cilium_categories = {
            "concept": "concepts",
            "task": "tasks",
            "tutorial": "tutorials",
            "reference": "reference",
            "setup": "setup",
            "network": "networking",
            "security": "security",
            "observability": "observability",
            "service-mesh": "service-mesh",
            "policy": "policy",
            "config": "configuration",
        }

        for part in path_parts:
            part_lower = part.lower()
            for key, value in cilium_categories.items():
                if key in part_lower:
                    return value

        return "general"

    def _extract_tags(self, content: str) -> list[str]:
        """Extract Cilium-specific tags from content"""
        tags = set()

        # Cilium technical terms
        cilium_terms = {
            "cilium",
            "hubble",
            "ciliumnetworkpolicy",
            "ciliumendpoint",
            "identity",
            "bpf",
            "ebpf",
            "kube-proxy",
            "cni",
            "service-mesh",
            "observability",
            "encryption",
            "wireguard",
            "ipsec",
            "clustermesh",
            "ingress-controller",
            "ciliumclusterwidenetworkpolicy",
            "ciliumnode",
            "envoy",
            "l7-policy",
            "network-policy",
        }

        content_lower = content.lower()
        for term in cilium_terms:
            if term in content_lower:
                tags.add(term)

        return list(tags)[:10]  # Limit to 10 tags


def load_cilium_docs(source_path: str, version: str = "latest") -> list[dict]:
    """Convenience function to load Cilium docs"""
    loader = CiliumDocumentLoader(source_path, version)
    return loader.load_documents()

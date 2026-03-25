"""Kubernetes documentation loader"""

from pathlib import Path

from .base_loader import BaseDocumentLoader


class K8sDocumentLoader(BaseDocumentLoader):
    """Load and parse Kubernetes documentation"""

    def __init__(self, source_path: str, version: str = "latest"):
        super().__init__(source_path, version, component="kubernetes")

    def _extract_category(self, file_path: Path) -> str:
        """Extract category from file path with K8s-specific categories"""
        path_parts = file_path.parts

        # Kubernetes-specific documentation categories
        k8s_categories = {
            "concept": "concepts",
            "task": "tasks",
            "tutorial": "tutorials",
            "reference": "reference",
            "setup": "setup",
            "network": "networking",
            "storage": "storage",
            "security": "security",
            "workload": "workloads",
            "config": "configuration",
            "admin": "administration",
        }

        for part in path_parts:
            part_lower = part.lower()
            for key, value in k8s_categories.items():
                if key in part_lower:
                    return value

        return "general"

    def _extract_tags(self, content: str) -> list[str]:
        """Extract Kubernetes-specific tags from content"""
        tags = set()

        # Kubernetes technical terms to look for
        k8s_terms = {
            "pod",
            "deployment",
            "service",
            "namespace",
            "configmap",
            "secret",
            "ingress",
            "networkpolicy",
            "daemonset",
            "statefulset",
            "replicaset",
            "job",
            "cronjob",
            "pvc",
            "pv",
            "storageclass",
            "rbac",
            "serviceaccount",
            "cni",
            "csi",
            "cri",
            "kubectl",
        }

        content_lower = content.lower()
        for term in k8s_terms:
            if term in content_lower:
                tags.add(term)

        return list(tags)[:10]  # Limit to 10 tags


def load_k8s_docs(source_path: str, version: str = "latest") -> list[dict]:
    """Convenience function to load Kubernetes docs"""
    loader = K8sDocumentLoader(source_path, version)
    return loader.load_documents()

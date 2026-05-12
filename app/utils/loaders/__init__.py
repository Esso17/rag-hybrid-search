"""Document loaders for Kubernetes documentation"""

from .base_loader import BaseDocumentLoader
from .k8s_loader import K8sDocumentLoader, load_k8s_docs

__all__ = [
    "BaseDocumentLoader",
    "K8sDocumentLoader",
    "load_k8s_docs",
]

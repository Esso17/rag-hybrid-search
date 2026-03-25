"""Document loaders for various documentation sources"""

from .base_loader import BaseDocumentLoader
from .cilium_loader import CiliumDocumentLoader, load_cilium_docs
from .k8s_loader import K8sDocumentLoader, load_k8s_docs

__all__ = [
    "BaseDocumentLoader",
    "K8sDocumentLoader",
    "CiliumDocumentLoader",
    "load_k8s_docs",
    "load_cilium_docs",
]

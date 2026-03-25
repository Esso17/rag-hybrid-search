"""Text processing utilities (chunking, splitting)"""

from app.core.text_processing.code_aware_splitter import (
    CodeAwareTextSplitter,
    get_code_aware_splitter,
)

__all__ = ["CodeAwareTextSplitter", "get_code_aware_splitter"]

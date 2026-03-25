"""Code-aware text splitter for technical documentation"""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


class CodeAwareTextSplitter:
    """
    Text splitter that preserves code blocks, YAML configs, and technical structures.
    Optimized for Kubernetes and Cilium documentation.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        preserve_code_blocks: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_code_blocks = preserve_code_blocks

        # Regex patterns for code blocks
        self.code_fence_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        self.yaml_block_pattern = re.compile(
            r"^(apiVersion|kind):.*?(?=\n\n|\n#|\Z)", re.MULTILINE | re.DOTALL
        )

    def split_text(self, text: str) -> list[str]:
        """Split text while preserving code blocks and technical structures"""

        if not self.preserve_code_blocks:
            # Fall back to standard splitting
            return self._standard_split(text)

        chunks = []
        current_pos = 0

        # Find all code blocks
        code_blocks = list(self.code_fence_pattern.finditer(text))

        for code_match in code_blocks:
            code_start, code_end = code_match.span()

            # Process text before code block
            if current_pos < code_start:
                before_text = text[current_pos:code_start].strip()
                if before_text:
                    chunks.extend(self._split_plain_text(before_text))

            # Add code block as single chunk if it fits, otherwise split carefully
            code_block = code_match.group()
            if len(code_block) <= self.chunk_size:
                # Find the nearest header or context before the code
                context_start = max(0, code_start - 300)
                context = text[context_start:code_start].strip()

                # Get the last sentence or header for context
                lines = context.split("\n")
                relevant_context = []
                for line in reversed(lines):
                    if line.startswith("#") or len(line) > 20:
                        relevant_context.insert(0, line)
                        break

                if relevant_context:
                    chunks.append("\n".join(relevant_context) + "\n\n" + code_block)
                else:
                    chunks.append(code_block)
            else:
                # Code block is too large, split it intelligently
                chunks.extend(self._split_large_code_block(code_block))

            current_pos = code_end

        # Process remaining text
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                chunks.extend(self._split_plain_text(remaining_text))

        # If no code blocks were found, use plain text splitting
        if not chunks:
            chunks = self._split_plain_text(text)

        return self._clean_chunks(chunks)

    def _split_plain_text(self, text: str) -> list[str]:
        """Split plain text on headers and paragraphs"""
        # Custom separators for technical documentation
        separators = [
            "\n## ",  # H2 headers
            "\n### ",  # H3 headers
            "\n#### ",  # H4 headers
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences
            " ",  # Words
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            keep_separator=True,
        )

        return splitter.split_text(text)

    def _split_large_code_block(self, code_block: str) -> list[str]:
        """Split large code blocks on logical boundaries"""
        # Remove fence markers
        code_content = re.sub(r"^```\w*\n|```$", "", code_block, flags=re.MULTILINE).strip()

        # Split on YAML document separators or significant blank lines
        parts = re.split(r"\n---\n|\n\n\n", code_content)

        chunks = []
        current_chunk = []
        current_size = 0

        for part in parts:
            part_size = len(part)

            if current_size + part_size <= self.chunk_size:
                current_chunk.append(part)
                current_size += part_size
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append("```\n" + "\n\n".join(current_chunk) + "\n```")

                # Start new chunk
                if part_size <= self.chunk_size:
                    current_chunk = [part]
                    current_size = part_size
                else:
                    # Part is too large, split by lines
                    lines = part.split("\n")
                    temp_chunk = []
                    temp_size = 0

                    for line in lines:
                        if temp_size + len(line) <= self.chunk_size:
                            temp_chunk.append(line)
                            temp_size += len(line)
                        else:
                            if temp_chunk:
                                chunks.append("```\n" + "\n".join(temp_chunk) + "\n```")
                            temp_chunk = [line]
                            temp_size = len(line)

                    if temp_chunk:
                        current_chunk = temp_chunk
                        current_size = temp_size

        # Add final chunk
        if current_chunk:
            chunks.append("```\n" + "\n\n".join(current_chunk) + "\n```")

        return chunks

    def _standard_split(self, text: str) -> list[str]:
        """Standard splitting without code awareness"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(text)

    def _clean_chunks(self, chunks: list[str]) -> list[str]:
        """Clean and validate chunks"""
        cleaned = []
        for chunk in chunks:
            # Remove excessive whitespace
            chunk = re.sub(r"\n{4,}", "\n\n", chunk)
            chunk = chunk.strip()

            # Skip very small chunks (less than 50 chars)
            if len(chunk) >= 50:
                cleaned.append(chunk)

        return cleaned


def get_code_aware_splitter(
    chunk_size: int = 800, chunk_overlap: int = 200, preserve_code_blocks: bool = True
) -> CodeAwareTextSplitter:
    """Factory function to create code-aware splitter"""
    return CodeAwareTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_code_blocks=preserve_code_blocks,
    )

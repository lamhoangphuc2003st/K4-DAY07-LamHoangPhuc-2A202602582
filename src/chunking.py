from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    # Split on whitespace that FOLLOWS a sentence-ending mark. The lookbehind
    # keeps ".", "!" and "?" attached to the sentence they close, which a plain
    # split on r"[.!?]\s+" would swallow.
    _BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = [part.strip() for part in self._BOUNDARY.split(text)]
        sentences = [sentence for sentence in sentences if sentence]
        if not sentences:
            return []

        step = self.max_sentences_per_chunk
        return [" ".join(sentences[start : start + step]) for start in range(0, len(sentences), step)]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case 1 — nothing left to split.
        if not current_text:
            return []
        # Base case 2 — the piece already fits, keep it whole.
        if len(current_text) <= self.chunk_size:
            return [current_text]
        # Base case 3 — out of separators (or the empty one): hard cut.
        if not remaining_separators or remaining_separators[0] == "":
            return self._hard_cut(current_text)

        separator, rest = remaining_separators[0], remaining_separators[1:]
        if separator not in current_text:
            # This boundary does not exist in the text, try the next one down.
            return self._split(current_text, rest)

        # Re-attach the separator to the piece it followed. split() would eat it,
        # and a chunk boundary landing on ". " would silently drop the period.
        pieces = current_text.split(separator)
        pieces = [piece + separator for piece in pieces[:-1]] + pieces[-1:]

        chunks: list[str] = []
        buffer = ""
        for piece in pieces:
            candidate = buffer + piece
            if len(candidate) <= self.chunk_size:
                # Merge up: keep packing neighbouring pieces until chunk_size.
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer)
                buffer = ""
            if len(piece) <= self.chunk_size:
                buffer = piece
            else:
                # Recurse down: this piece alone is still too long.
                sub_chunks = self._split(piece, rest)
                chunks.extend(sub_chunks[:-1])
                buffer = sub_chunks[-1] if sub_chunks else ""

        if buffer:
            chunks.append(buffer)
        # rstrip only: the separator now trails each piece, so trimming the tail
        # drops the blank space a boundary leaves behind but keeps the period.
        return [stripped for chunk in chunks if (stripped := chunk.rstrip())]

    def _hard_cut(self, current_text: str) -> list[str]:
        size = max(1, self.chunk_size)
        return [current_text[start : start + size] for start in range(0, len(current_text), size)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0

    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        overlap = chunk_size // 10  # 10% overlap keeps step > 0 for any chunk_size
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=overlap),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(chunk) for chunk in chunks) / count if count else 0.0
            comparison[name] = {
                "count": count,
                "avg_length": round(avg_length, 2),
                "chunks": chunks,
            }
        return comparison

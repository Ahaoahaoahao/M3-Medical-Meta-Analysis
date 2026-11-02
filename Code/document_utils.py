"""Bounded document segments and complete JSON record batches."""

from dataclasses import dataclass
import json
from typing import Any, List


@dataclass(frozen=True)
class TextChunk:
    index: int
    start: int
    end: int
    text: str


def split_text(text: str, max_chars: int = 18000, overlap: int = 500) -> List[TextChunk]:
    """Cover the full text with overlapping segments and original offsets."""
    if not isinstance(text, str):
        raise TypeError("Document text must be a string.")
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Require max_chars > overlap >= 0.")
    chunks: List[TextChunk] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + max_chars // 2, end)
            if boundary >= 0:
                end = boundary + 1
        chunks.append(TextChunk(len(chunks) + 1, start, end, text[start:end]))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def format_chunk(chunk: TextChunk) -> str:
    return (
        f"[Document segment {chunk.index}; original character offsets "
        f"{chunk.start}:{chunk.end}, zero-based, end-exclusive]\n{chunk.text}"
    )


def batch_records(records: List[Any], max_chars: int) -> List[List[Any]]:
    """Pack complete records within the serialized JSON character budget."""
    if max_chars < 2:
        raise ValueError("JSON batch budget must be at least 2 characters.")
    batches: List[List[Any]] = []
    batch: List[Any] = []
    size = 2
    for index, record in enumerate(records):
        record_size = len(json.dumps(record, ensure_ascii=False))
        if record_size + 2 > max_chars:
            raise ValueError(f"Record {index} exceeds the JSON batch budget ({max_chars} characters).")
        extra = record_size + (2 if batch else 0)
        if size + extra > max_chars:
            batches.append(batch)
            batch = []
            size = 2
            extra = record_size
        batch.append(record)
        size += extra
    if batch:
        batches.append(batch)
    return batches

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class RetrievedChunk:
    text: str
    score: float
    document_name: Optional[str] = None
    source: Optional[str] = None
    page: Optional[str] = None
    chunk_id: Optional[str] = None


@dataclass
class RAGResult:
    answer: str
    chunks: List[RetrievedChunk]
    provider: str
    model: str
    latency_ms: int

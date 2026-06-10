from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    embedding: Optional[list] = None

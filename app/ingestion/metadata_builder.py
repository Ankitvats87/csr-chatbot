import hashlib
from typing import Dict


def build_chunk_metadata(
    *,
    file_id: str,
    document_name: str,
    page: int,
    chunk_index: int,
    text: str,
    upload_date: str,
) -> Dict[str, str]:
    chunk_id = f"{file_id}::p{page}::c{chunk_index}"
    return {
        "file_id": file_id,
        "document_name": document_name,
        "source": "google_drive",
        "chunk_id": chunk_id,
        "upload_date": upload_date,
        "page": str(page),
        "text": text,
        # short fingerprint helps debugging without storing the full text twice
        "text_hash": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
    }


def vector_id(file_id: str, page: int, chunk_index: int) -> str:
    return f"{file_id}-p{page}-c{chunk_index}"

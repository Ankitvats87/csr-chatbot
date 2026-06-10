import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout to use UTF-8 to prevent cp1252 crashes on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.utils.env import get_settings
from app.db.sqlite_client import SQLiteClient
from app.db.pinecone_client import PineconeClient
from app.services.embedding_service import EmbeddingService

def print_matches():
    settings = get_settings()
    
    # Query Pinecone
    pinecone = PineconeClient(settings)
    pinecone.connect()
    embedder = EmbeddingService(settings)
    
    # 1. Search Pinecone for "Manoj Jhawar" chunks
    query_text = "Manoj Jhawar"
    print(f"Embedding search query: '{query_text}'")
    embedding = embedder.embed(query_text)
    
    raw_resp = pinecone.index.query(
        vector=embedding,
        top_k=10,
        namespace=settings.pinecone_namespace,
        include_metadata=True,
    )
    
    matches = raw_resp.get("matches") if isinstance(raw_resp, dict) else getattr(raw_resp, "matches", [])
    print(f"\n--- Pinecone Matches for '{query_text}' ---")
    for idx, m in enumerate(matches):
        score = m["score"] if isinstance(m, dict) else m.score
        md = m["metadata"] if isinstance(m, dict) else (m.metadata or {})
        doc_name = md.get("document_name", "Unknown")
        text = md.get("text", "")
        print(f"\nMatch {idx+1} (Score: {score:.4f}, Document: {doc_name}):\n{text}\n" + "-"*40)

if __name__ == "__main__":
    load_dotenv()
    print_matches()

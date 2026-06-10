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
from app.services.vector_service import VectorService
from app.repositories.ingested_files_repo import IngestedFilesRepo

def test_query():
    settings = get_settings()
    
    # 1. Check SQLite
    sqlite = SQLiteClient(settings.sqlite_path)
    sqlite.connect()
    repo = IngestedFilesRepo(sqlite)
    all_files = repo.all_indexed()
    
    print("--- Ingested Files in SQLite ---")
    for file_id, file_info in all_files.items():
        print(f"ID: {file_id} | Name: {file_info.name} | Status: {file_info.status} | Chunks: {file_info.n_chunks}")
    
    # 2. Query Pinecone
    pinecone = PineconeClient(settings)
    pinecone.connect()
    embedder = EmbeddingService(settings)
    vectors = VectorService(pinecone, settings)
    
    query_text = "Which CSR Committee meetings did Dr. Manoj Kumar Jhawar attend?"
    print(f"\nEmbedding query: '{query_text}'")
    embedding = embedder.embed(query_text)
    
    print("\nQuerying Pinecone...")
    raw_resp = pinecone.index.query(
        vector=embedding,
        top_k=10,
        namespace=settings.pinecone_namespace,
        include_metadata=True,
    )
    
    matches = raw_resp.get("matches") if isinstance(raw_resp, dict) else getattr(raw_resp, "matches", [])
    print(f"\n--- Pinecone Matches (Raw count: {len(matches)}) ---")
    for idx, m in enumerate(matches):
        score = m["score"] if isinstance(m, dict) else m.score
        md = m["metadata"] if isinstance(m, dict) else (m.metadata or {})
        doc_name = md.get("document_name", "Unknown")
        text = md.get("text", "")
        # Preview first 150 chars
        text_preview = text.replace('\n', ' ')[:150]
        print(f"{idx+1}. Score: {score:.4f} | Document: {doc_name} | Text: {text_preview}...")
        
    sqlite.close()

if __name__ == "__main__":
    load_dotenv()
    test_query()

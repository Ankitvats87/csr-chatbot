import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime

# Reconfigure stdout to use UTF-8 to prevent cp1252 crashes on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.utils.env import get_settings
from app.db.sqlite_client import SQLiteClient
from app.db.pinecone_client import PineconeClient
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.repositories.ingested_files_repo import IngestedFilesRepo
from app.ingestion.chunker import Chunker
from app.ingestion.embed_and_upsert import embed_and_upsert

def ingest_local_file(filepath: str):
    settings = get_settings()
    path = Path(filepath)
    if not path.exists():
        print(f"Error: file {filepath} not found.")
        sys.exit(1)
        
    print(f"Starting local ingestion for: {path.name}")
    
    # Generate a unique pseudo file ID using MD5 of filename
    file_id = "local_" + hashlib.md5(path.name.encode('utf-8')).hexdigest()[:16]
    
    # 1. Initialize Clients
    sqlite = SQLiteClient(settings.sqlite_path)
    sqlite.connect()
    
    pinecone = PineconeClient(settings)
    pinecone.connect()
    
    repo = IngestedFilesRepo(sqlite)
    embedder = EmbeddingService(settings)
    vectors = VectorService(pinecone, settings)
    chunker = Chunker(settings)
    
    # 2. Read file content and check for delimiter (only for TXT files)
    ext = path.suffix.lower().lstrip(".")
    delim = "=================================================="
    is_compiled_txt = False
    content = ""
    
    if ext == "txt":
        content = path.read_text(encoding="utf-8", errors="ignore")
        if delim in content:
            is_compiled_txt = True
            
    # 3. Check for multi-document compiler separator
    if is_compiled_txt:
        print(f"Detected multi-document compilation separator in {path.name}")
        blocks = content.split(delim)
        all_chunks = []
        for index, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue
            
            # Find a suitable title/context header for this block
            lines = [line.strip() for line in block.split("\n") if line.strip()]
            header = ""
            for line in lines[:15]:
                # Skip separator lines and agenda item codes
                if "CONTENT OF DOCUMENT" in line.upper():
                    continue
                if line.lower().startswith("agenda-") or line.lower().startswith("agenda item"):
                    continue
                if len(line) < 5:
                    continue
                
                # Look for meeting descriptors
                if any(x in line.upper() for x in ["MINUTES OF", "AGENDA", "SUBJECT:", "CONFIRMATION OF", "APPROVAL FOR"]):
                    header = line
                    break
            
            # Fallback if no matching keywords found
            if not header:
                for line in lines[:10]:
                    if "CONTENT OF DOCUMENT" not in line.upper() and not line.lower().startswith("agenda-") and len(line) >= 5:
                        header = line
                        break
                
            header = header.replace("#", "").replace("---", "").strip()
            print(f"Block {index+1} context header identified: '{header}'")
            
            # Chunk the block using the RecursiveCharacterTextSplitter directly
            raw_chunks = chunker.splitter.split_text(block)
            
            # Prepend context header to every chunk
            for chunk_index, chunk_text in enumerate(raw_chunks):
                chunk_text = chunk_text.strip()
                if chunk_text:
                    context_chunk = f"[Context: {header}]\n\n{chunk_text}"
                    all_chunks.append((context_chunk, 1)) # page is 1 for txt files
        
        print(f"Total contextual chunks generated: {len(all_chunks)}")
        
        # 4. Embed and Upsert
        n = embed_and_upsert(
            embedder=embedder,
            vectors=vectors,
            file_id=file_id,
            document_name=path.name,
            upload_date=datetime.utcnow().isoformat() + "Z",
            chunks=all_chunks,
        )
        repo.upsert_success(file_id, path.name, datetime.utcnow().isoformat() + "Z", n)
        print(f"Ingestion successful! Upserted {n} chunks to Pinecone.")
        
    else:
        # Standard ingestion (no separator)
        print("Processing as single standard document.")
        chunks = chunker.chunk_file(str(path))
        n = embed_and_upsert(
            embedder=embedder,
            vectors=vectors,
            file_id=file_id,
            document_name=path.name,
            upload_date=datetime.utcnow().isoformat() + "Z",
            chunks=chunks,
        )
        repo.upsert_success(file_id, path.name, datetime.utcnow().isoformat() + "Z", n)
        print(f"Ingestion successful! Upserted {n} chunks to Pinecone.")
        
    sqlite.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_local_file.py <path_to_file>")
        sys.exit(1)
    ingest_local_file(sys.argv[1])

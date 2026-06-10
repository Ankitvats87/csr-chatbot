import sys
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.utils.env import get_settings
from app.db.pinecone_client import PineconeClient
from app.ingestion.chunker import Chunker

def debug_chunks():
    settings = get_settings()
    path = Path("data/raw_docs/Test CSR 21_26 .txt")
    if not path.exists():
        print("File not found.")
        return
        
    file_id = "local_" + hashlib.md5(path.name.encode('utf-8')).hexdigest()[:16]
    content = path.read_text(encoding="utf-8", errors="ignore")
    delim = "=================================================="
    blocks = content.split(delim)
    
    chunker = Chunker(settings)
    all_chunks = []
    
    for index, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        
        # Identify context header
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        header = ""
        for line in lines[:15]:
            if "CONTENT OF DOCUMENT" in line.upper():
                continue
            if line.lower().startswith("agenda-") or line.lower().startswith("agenda item"):
                continue
            if len(line) < 5:
                continue
            if any(x in line.upper() for x in ["MINUTES OF", "AGENDA", "SUBJECT:", "CONFIRMATION OF", "APPROVAL FOR"]):
                header = line
                break
        
        if not header:
            for line in lines[:10]:
                if "CONTENT OF DOCUMENT" not in line.upper() and not line.lower().startswith("agenda-") and len(line) >= 5:
                    header = line
                    break
            
        header = header.replace("#", "").replace("---", "").strip()
        raw_chunks = chunker.splitter.split_text(block)
        
        for chunk_index, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if chunk_text:
                context_chunk = f"[Context: {header}]\n\n{chunk_text}"
                all_chunks.append((context_chunk, chunk_index, header))

    print(f"Total chunks recreated: {len(all_chunks)}")
    
    # Let's search for "Jhawar" in the recreated chunks
    print("\n--- Chunks containing 'Jhawar' ---")
    found_any = False
    
    # Connect to Pinecone to check if these vectors exist
    pinecone = PineconeClient(settings)
    pinecone.connect()
    
    for global_idx, (text, chunk_index, header) in enumerate(all_chunks):
        if "jhawar" in text.lower():
            found_any = True
            vec_id = f"{file_id}-p1-c{global_idx}" # wait, let's see how vector_id is constructed in ingest_local_file.py
            # In ingest_local_file.py:
            # For multi-document: all_chunks.append((context_chunk, 1))
            # And then: vector_id(file_id, page, chunk_index) = f"{file_id}-p{page}-c{chunk_index}"
            # Let's print details
            print(f"\nGlobal Index: {global_idx} | Chunk Index: {chunk_index} | Header: {header}")
            print(f"Vector ID: {vec_id}")
            print(f"Text Preview:\n{text[:300]}...")
            
            # Fetch from Pinecone
            try:
                fetch_resp = pinecone.index.fetch(ids=[vec_id], namespace=settings.pinecone_namespace)
                vectors = fetch_resp.get("vectors", {})
                if vec_id in vectors:
                    print(f"--> Vector exists in Pinecone! Metadata:")
                    md = vectors[vec_id].get("metadata", {})
                    # Print keys and document_name
                    print(f"    Document: {md.get('document_name')}")
                    print(f"    Text length in Pinecone: {len(md.get('text', ''))}")
                else:
                    print(f"--> Vector DOES NOT exist in Pinecone.")
            except Exception as e:
                print(f"--> Error fetching vector: {e}")
                
    if not found_any:
        print("No chunks containing 'Jhawar' were found in the chunked text!")

if __name__ == "__main__":
    load_dotenv()
    debug_chunks()

import sys
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.utils.env import get_settings
from app.db.pinecone_client import PineconeClient

def count_vectors():
    settings = get_settings()
    pinecone = PineconeClient(settings)
    pinecone.connect()
    
    file_id = "local_48fd14345211663a"
    print(f"Index: {settings.pinecone_index_name}")
    print(f"Namespace: {settings.pinecone_namespace}")
    sys.stdout.flush()
    
    total_exists = 0
    missing_indices = []
    
    batch_size = 100
    for start in range(0, 2583, batch_size):
        end = min(start + batch_size, 2583)
        ids = [f"{file_id}-p1-c{i}" for i in range(start, end)]
        print(f"Fetching batch {start} to {end}...")
        sys.stdout.flush()
        try:
            resp = pinecone.index.fetch(ids=ids, namespace=settings.pinecone_namespace)
            vectors = resp.get("vectors", {})
            total_exists += len(vectors)
            print(f"Batch {start} to {end} fetched. Found {len(vectors)} existing vectors.")
            sys.stdout.flush()
            for idx in range(start, end):
                vid = f"{file_id}-p1-c{idx}"
                if vid not in vectors:
                    missing_indices.append(idx)
        except Exception as e:
            print(f"Error fetching batch {start}-{end}: {e}")
            sys.stdout.flush()
            
    print(f"Total checked: 2583")
    print(f"Total existing in Pinecone: {total_exists}")
    print(f"Total missing: {len(missing_indices)}")
    sys.stdout.flush()
    if missing_indices:
        print(f"First 10 missing indices: {missing_indices[:10]}")
        sys.stdout.flush()

if __name__ == "__main__":
    load_dotenv()
    count_vectors()

import sys
from pathlib import Path

# Reconfigure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.utils.env import get_settings
from app.db.sqlite_client import SQLiteClient

def read_logs():
    settings = get_settings()
    sqlite = SQLiteClient(settings.sqlite_path)
    sqlite.connect()
    
    rows = sqlite.fetchall(
        "SELECT id, chat_id, question, answer, n_retrieved, latency_ms, status, error, created_at "
        "FROM request_logs ORDER BY id DESC LIMIT 10"
    )
    
    print(f"--- Last 10 Request Logs from SQLite ---")
    for r in rows:
        print(f"\nID: {r['id']} | Timestamp: {r['created_at']} | Status: {r['status']}")
        print(f"Question: {r['question']}")
        print(f"Answer: {r['answer']}")
        print(f"Retrieved Chunks: {r['n_retrieved']} | Latency: {r['latency_ms']}ms")
        if r['error']:
            print(f"Error: {r['error']}")
        print("-" * 50)
        
    sqlite.close()

if __name__ == "__main__":
    read_logs()

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.utils.env import get_settings
from openai import OpenAI

def test_embeddings():
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    
    texts = ["Hello world test embedding"] * 64
    print("Sending batch of 64 embeddings to OpenAI...")
    sys.stdout.flush()
    try:
        resp = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        )
        print("Success! Received response from OpenAI.")
        print(f"Data length: {len(resp.data)}")
        print(f"Dimension: {len(resp.data[0].embedding)}")
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
    sys.stdout.flush()

if __name__ == "__main__":
    load_dotenv()
    test_embeddings()

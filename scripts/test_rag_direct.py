import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.utils.env import get_settings
from app.db.sqlite_client import SQLiteClient
from app.db.pinecone_client import PineconeClient
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.repositories.memory_repo import MemoryRepo
from app.services.memory_service import MemoryService
from app.services.prompt_service import PromptService
from app.services.response_service import ResponseService
from app.repositories.request_log_repo import RequestLogRepo
from app.services.rag_service import RAGService
from app.services.document_directory_service import DocumentDirectoryService

async def test_rag():
    settings = get_settings()
    
    # Initialize Clients
    sqlite = SQLiteClient(settings.sqlite_path)
    sqlite.connect()
    pinecone = PineconeClient(settings)
    pinecone.connect()
    
    # Repos and Services
    embedder = EmbeddingService(settings)
    vectors = VectorService(pinecone, settings)
    memory_repo = MemoryRepo(sqlite)
    memory = MemoryService(memory_repo, settings)
    prompts = PromptService()
    responder = ResponseService(settings)
    request_log = RequestLogRepo(sqlite)
    directory = DocumentDirectoryService(sqlite)
    
    from app.services.intelligence_layer import IntelligenceLayerService
    intelligence_layer = IntelligenceLayerService(
        embedder=embedder,
        vectors=vectors,
        responder=responder,
        directory=directory,
        settings=settings,
    )
    
    rag = RAGService(
        embedder=embedder,
        vectors=vectors,
        memory=memory,
        prompts=prompts,
        responder=responder,
        request_log=request_log,
        directory=directory,
        settings=settings,
        intelligence_layer=intelligence_layer,
    )
    
    queries = [
        "What is the budget for FY 2025-2026?",
        "What is the budget of CT Scan machine?",
        "List down the date of the 25th CSR meeting"
    ]
    
    chat_id = 999999
    # Clear conversation history for testing
    memory.clear(chat_id)
    
    print("=" * 70)
    print("              RAG END-TO-END PIPELINE DIAGNOSTIC")
    print("=" * 70)
    
    for q in queries:
        print(f"\nQUERY: '{q}'")
        try:
            result = await rag.answer(chat_id, q)
            print(f"LATENCY: {result.latency_ms}ms")
            print(f"RETRIEVED CHUNKS: {len(result.chunks)}")
            for idx, c in enumerate(result.chunks):
                print(f"  [{idx+1}] Score: {c.score:.4f} | Source: {c.document_name} (p.{c.page})")
            print(f"\nANSWER:\n{result.answer}")
        except Exception as e:
            print(f"ERROR: {e}")
        print("-" * 70)
        
    sqlite.close()

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(test_rag())

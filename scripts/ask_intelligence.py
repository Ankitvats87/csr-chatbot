"""One-shot end-to-end pipeline test (planner → hybrid → rerank → dual-pass)."""
import asyncio
import sys

sys.path.insert(0, "/app")

from app.db.pinecone_client import PineconeClient
from app.db.sqlite_client import SQLiteClient
from app.services.document_directory_service import DocumentDirectoryService
from app.services.embedding_service import EmbeddingService
from app.services.hybrid_search import HybridSearchService
from app.services.intelligence_layer import IntelligenceLayerService
from app.services.reranker import LLMReranker
from app.services.response_service import ResponseService
from app.services.vector_service_v2 import VectorServiceV2
from app.utils.env import get_settings
from app.utils.logger import setup_logging

settings = get_settings()
setup_logging("WARNING")

sqlite = SQLiteClient(settings.sqlite_path); sqlite.connect()
pinecone = PineconeClient(settings); pinecone.connect()
embedder = EmbeddingService(settings)
responder = ResponseService(settings)
directory = DocumentDirectoryService(sqlite)
hybrid = HybridSearchService(settings)
reranker = LLMReranker(responder)
vectors = VectorServiceV2(pinecone, settings, hybrid=hybrid)
layer = IntelligenceLayerService(
    embedder=embedder, vectors=vectors, responder=responder,
    directory=directory, settings=settings, hybrid=hybrid, reranker=reranker,
)

q = sys.argv[1] if len(sys.argv) > 1 else "What projects were approved in the 26th CSR committee meeting?"
print(f"QUESTION: {q}\n{'='*70}")
result = asyncio.run(layer.answer(chat_id=0, question=q, history=[]))
print(result.answer)
print(f"\n{'='*70}\nchunks used: {len(result.chunks)} | hybrid: {hybrid.available}")

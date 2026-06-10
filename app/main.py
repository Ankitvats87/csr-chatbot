from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from app.db.pinecone_client import PineconeClient
from app.db.sqlite_client import SQLiteClient
from app.repositories.access_repo import AccessRepo
from app.repositories.memory_repo import MemoryRepo
from app.repositories.request_log_repo import RequestLogRepo
from app.routes.health import router as health_router
from app.routes.telegram_webhook import router as webhook_router
from app.services.access_service import AccessService
from app.services.document_directory_service import DocumentDirectoryService
from app.services.embedding_service import EmbeddingService
from app.services.memory_service import MemoryService
from app.services.prompt_service import PromptService
from app.services.rag_service import RAGService
from app.services.response_service import ResponseService
from app.services.telegram_service import TelegramService
from app.services.vector_service import VectorService
from app.utils.env import Settings, get_settings
from app.utils.logger import get_logger, setup_logging


from app.services.ingestion_service import IngestionService
from app.ingestion.loader import DriveLoader
from app.ingestion.chunker import Chunker
from app.repositories.ingested_files_repo import IngestedFilesRepo


@dataclass
class Deps:
    settings: Settings
    sqlite: SQLiteClient
    pinecone: PineconeClient
    telegram: TelegramService
    memory: MemoryService
    rag: RAGService
    access: AccessService
    ingestion: IngestionService


def _build_deps(settings: Settings) -> Deps:
    sqlite = SQLiteClient(settings.sqlite_path)
    sqlite.connect()

    pinecone = PineconeClient(settings)
    pinecone.connect()

    memory_repo = MemoryRepo(sqlite)
    request_log_repo = RequestLogRepo(sqlite)
    access_repo = AccessRepo(sqlite)
    ingested_files_repo = IngestedFilesRepo(sqlite)

    embedder = EmbeddingService(settings)

    # Hybrid lexical layer (BM25 over local chunk store) — shared by V2
    # retrieval and the intelligence layer. Degrades to None when disabled
    # or when the chunk store hasn't been built yet.
    hybrid = None
    if settings.enable_hybrid_retrieval:
        from app.services.hybrid_search import HybridSearchService
        hybrid = HybridSearchService(settings)
        get_logger("app.boot").info(
            "hybrid retrieval: %s", "ACTIVE" if hybrid.available else "enabled but chunk store missing"
        )

    if settings.rag_version == "v2":
        from app.services.vector_service_v2 import VectorServiceV2
        vectors = VectorServiceV2(pinecone, settings, hybrid=hybrid)
        get_logger("app.boot").info("retrieval mode: V2 (csr_v2_enriched + csr_project_master)")
    else:
        vectors = VectorService(pinecone, settings)
        get_logger("app.boot").info("retrieval mode: V1 (knowledgebase)")
    memory = MemoryService(memory_repo, settings)
    prompts = PromptService()
    responder = ResponseService(settings)
    telegram = TelegramService(settings)
    directory = DocumentDirectoryService(sqlite)

    reranker = None
    if settings.enable_reranker:
        from app.services.reranker import LLMReranker
        reranker = LLMReranker(responder)

    from app.services.intelligence_layer import IntelligenceLayerService
    intelligence_layer = IntelligenceLayerService(
        embedder=embedder,
        vectors=vectors,
        responder=responder,
        directory=directory,
        settings=settings,
        hybrid=hybrid,
        reranker=reranker,
    )
    rag = RAGService(
        embedder=embedder,
        vectors=vectors,
        memory=memory,
        prompts=prompts,
        responder=responder,
        request_log=request_log_repo,
        directory=directory,
        settings=settings,
        intelligence_layer=intelligence_layer,
    )
    access = AccessService(access_repo, telegram, settings)

    loader = DriveLoader(settings)
    if settings.drive_configured():
        try:
            loader.connect()
        except Exception:
            get_logger("app.boot").exception("failed to connect DriveLoader during startup")
    chunker = Chunker(settings)
    ingestion = IngestionService(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vectors=vectors,
        repo=ingested_files_repo,
        settings=settings,
    )

    return Deps(
        settings=settings,
        sqlite=sqlite,
        pinecone=pinecone,
        telegram=telegram,
        memory=memory,
        rag=rag,
        access=access,
        ingestion=ingestion,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger("app.boot")
    logger.info(
        "starting telegram-rag-bot",
        extra={
            "llm_provider": settings.llm_provider,
            "pinecone_index": settings.pinecone_index_name,
            "webhook_url": settings.webhook_url(),
            "drive_configured": settings.drive_configured(),
            "admins_configured": bool(settings.admin_chat_ids),
        },
    )
    deps = _build_deps(settings)
    app.state.deps = deps
    try:
        yield
    finally:
        await deps.telegram.aclose()
        deps.sqlite.close()
        logger.info("shutdown complete")


from app.routes.admin import router as admin_router

app = FastAPI(title="Spark63 CSR Bot", lifespan=lifespan)
app.include_router(health_router)
app.include_router(webhook_router)
app.include_router(admin_router)

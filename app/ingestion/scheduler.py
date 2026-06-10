import argparse
import sys
import time

from apscheduler.schedulers.blocking import BlockingScheduler

from app.db.pinecone_client import PineconeClient
from app.db.sqlite_client import SQLiteClient
from app.ingestion.chunker import Chunker
from app.ingestion.loader import DriveLoader
from app.repositories.ingested_files_repo import IngestedFilesRepo
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.vector_service import VectorService
from app.utils.env import get_settings
from app.utils.logger import get_logger, setup_logging


def _build_service() -> IngestionService:
    settings = get_settings()
    sqlite = SQLiteClient(settings.sqlite_path)
    sqlite.connect()
    pinecone = PineconeClient(settings)
    pinecone.connect()

    loader = DriveLoader(settings)
    if settings.drive_configured():
        loader.connect()

    return IngestionService(
        loader=loader,
        chunker=Chunker(settings),
        embedder=EmbeddingService(settings),
        vectors=VectorService(pinecone, settings),
        repo=IngestedFilesRepo(sqlite),
        settings=settings,
    )


def run_once() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger("ingestion.once")
    service = _build_service()
    counts = service.sync_once()
    logger.info("one-shot sync complete", extra=counts)
    return 0


def run_loop() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger("ingestion.loop")
    if not settings.drive_configured():
        logger.warning(
            "google drive not configured — scheduler will idle. "
            "Fill GOOGLE_DRIVE_* values in .env and restart this container."
        )
        # Idle so the container stays up (compose will keep restarting otherwise).
        while True:
            time.sleep(3600)

    service = _build_service()

    # Run once on startup, then on a schedule.
    try:
        service.sync_once()
    except Exception as e:
        logger.exception("initial sync failed", extra={"err": str(e)})

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        service.sync_once,
        "interval",
        seconds=settings.ingestion_poll_interval_seconds,
        id="drive_sync",
        max_instances=1,
        coalesce=True,
    )
    logger.info(
        "ingestion scheduler started",
        extra={"interval_seconds": settings.ingestion_poll_interval_seconds},
    )
    scheduler.start()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single sync pass and exit.")
    args = parser.parse_args()
    return run_once() if args.once else run_loop()


if __name__ == "__main__":
    sys.exit(main())

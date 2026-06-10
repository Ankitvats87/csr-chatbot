from pathlib import Path
from typing import Dict

from app.ingestion.chunker import Chunker
from app.ingestion.embed_and_upsert import embed_and_upsert
from app.ingestion.loader import DriveFile, DriveLoader
from app.repositories.ingested_files_repo import IngestedFile, IngestedFilesRepo
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class IngestionService:
    """Orchestrates a single Drive→Pinecone sync pass.

    Add detection:   new file_id OR modifiedTime increased.
    Delete detection: file_id present in DB but absent from Drive listing.
    """

    DOWNLOAD_DIR = str(PROJECT_ROOT / "data" / "raw_docs")

    def __init__(
        self,
        loader: DriveLoader,
        chunker: Chunker,
        embedder: EmbeddingService,
        vectors: VectorService,
        repo: IngestedFilesRepo,
        settings: Settings,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vectors = vectors
        self.repo = repo
        self.settings = settings

    def sync_once(self) -> Dict[str, int]:
        """Runs one full sync. Returns a counts dict."""
        if not self.settings.drive_configured():
            logger.info("drive not configured; skipping sync")
            return {"skipped": 1}

        if self.settings.rag_version == "v2":
            return self.sync_once_v2()

        drive_files = {f.id: f for f in self.loader.list_files()}
        indexed = self.repo.all_indexed()

        added = updated = deleted = failed = 0

        # Add / update
        for file_id, drive_file in drive_files.items():
            existing = indexed.get(file_id)
            if existing and existing.modified_time == drive_file.modified_time and existing.status == "indexed" and existing.n_chunks > 0:
                continue
            try:
                self._index_one(drive_file)
                if existing is None:
                    added += 1
                else:
                    updated += 1
            except Exception as e:
                logger.exception(
                    "ingestion failed",
                    extra={"file_id": file_id, "file_name": drive_file.name, "err": str(e)},
                )
                self.repo.mark_failed(file_id, drive_file.name, drive_file.modified_time)
                failed += 1

        # Delete: gone from Drive → drop vectors + repo row
        for file_id, ifile in indexed.items():
            if file_id in drive_files:
                continue
            try:
                self.vectors.delete_by_file_id(file_id)
                self.repo.delete(file_id)
                deleted += 1
                logger.info("deleted vectors for removed file", extra={"file_id": file_id, "file_name": ifile.name})
            except Exception as e:
                logger.exception(
                    "delete failed", extra={"file_id": file_id, "file_name": ifile.name, "err": str(e)}
                )

        counts = {"added": added, "updated": updated, "deleted": deleted, "failed": failed}
        logger.info("ingestion sync complete", extra=counts)
        return counts

    def sync_once_v2(self) -> Dict[str, int]:
        """Runs one full sync in V2 mode. Reuses the raw_docs folder but routes processing
        through the V2Pipeline.
        """
        try:
            drive_files = {f.id: f for f in self.loader.list_files()}
        except Exception as e:
            logger.exception("failed to list drive files for v2", extra={"err": str(e)})
            return {"skipped": 1}

        indexed = self.repo.all_indexed()
        added = updated = deleted = failed = 0
        changed_file_ids = []

        # 1. Add / Update detection (download phase only)
        for file_id, drive_file in drive_files.items():
            existing = indexed.get(file_id)
            if existing and existing.modified_time == drive_file.modified_time and existing.status == "indexed" and existing.n_chunks > 0:
                continue
            try:
                local_path = self.loader.download(drive_file, self.DOWNLOAD_DIR)
                if local_path:
                    changed_file_ids.append(file_id)
                    if existing is None:
                        added += 1
                    else:
                        updated += 1
            except Exception as e:
                logger.exception(
                    "v2 download failed during sync",
                    extra={"file_id": file_id, "file_name": drive_file.name, "err": str(e)},
                )
                self.repo.mark_failed(file_id, drive_file.name, drive_file.modified_time)
                failed += 1

        # 2. Delete detection (Pinecone prefix wipe + SQLite remove + local files delete)
        for file_id, ifile in indexed.items():
            if file_id in drive_files:
                continue
            try:
                self.vectors.delete_by_file_id(file_id)
                self.repo.delete(file_id)
                deleted += 1

                # Clean up local raw doc
                for ext in ["pdf", "docx", "txt", "csv"]:
                    local_p = Path(self.DOWNLOAD_DIR) / f"{file_id}.{ext}"
                    if local_p.is_file():
                        local_p.unlink()

                # Clean up V2 cached processed files
                from app.ingestion_v2.pipeline import PROCESSED_DIR
                for p_ext in ["md", "json"]:
                    proc_p = PROCESSED_DIR / f"{file_id}.{p_ext}"
                    if proc_p.is_file():
                        proc_p.unlink()

                logger.info("v2 deleted vectors and cache for removed file", extra={"file_id": file_id, "file_name": ifile.name})
            except Exception as e:
                logger.exception(
                    "v2 delete failed during sync", extra={"file_id": file_id, "file_name": ifile.name, "err": str(e)}
                )
                failed += 1

        # 3. If any changes occurred, run V2 pipeline on all raw docs
        if added > 0 or updated > 0 or deleted > 0:
            try:
                from app.ingestion_v2.pipeline import V2Pipeline
                # self.vectors.pinecone is PineconeClient, self.repo.db is SQLiteClient
                pipeline = V2Pipeline(
                    settings=self.settings,
                    pinecone=self.vectors.pinecone,
                    sqlite=self.repo.db,
                )
                counts = pipeline.run_on_all_raw_docs()
                logger.info("v2 pipeline execution complete during sync", extra={
                    "files_seen": counts.files_seen,
                    "files_indexed": counts.files_indexed,
                    "files_failed": counts.files_failed,
                    "enriched_chunks_upserted": counts.enriched_chunks_upserted,
                    "project_masters_upserted": counts.project_masters_upserted,
                })
                failed += counts.files_failed
            except Exception as e:
                logger.exception("v2 pipeline run failed during sync", extra={"err": str(e)})
                failed += len(changed_file_ids)

        sync_counts = {"added": added, "updated": updated, "deleted": deleted, "failed": failed}
        logger.info("v2 ingestion sync complete", extra=sync_counts)
        return sync_counts

    def _index_one(self, drive_file: DriveFile) -> None:
        local_path = self.loader.download(drive_file, self.DOWNLOAD_DIR)
        if not local_path:
            return
        chunks = self.chunker.chunk_file(local_path)
        if not chunks:
            logger.warning(
                "no chunks produced",
                extra={"file_id": drive_file.id, "file_name": drive_file.name},
            )
            self.repo.upsert_success(drive_file.id, drive_file.name, drive_file.modified_time, 0)
            return
        n = embed_and_upsert(
            embedder=self.embedder,
            vectors=self.vectors,
            file_id=drive_file.id,
            document_name=drive_file.name,
            upload_date=drive_file.modified_time,
            chunks=chunks,
        )
        self.repo.upsert_success(drive_file.id, drive_file.name, drive_file.modified_time, n)

"""V2 end-to-end ingestion orchestrator.

Reads files from data/raw_docs/ (already downloaded by the V1 Drive
loader) so V2 doesn't re-hit the Drive API or LlamaParse credits
unnecessarily. Each file goes through:

   parse → classify → extract → chunk → tag lifecycle → embed → upsert
                                                          ↓
                              (project entries collected by ProjectMasterBuilder)
                                                          ↓
                                              build & embed masters → upsert

The pipeline saves the LlamaParse markdown to data/processed_v2/<file_id>.md
(closing the gap we identified earlier — V2 produces inspectable markdown).
The structured JSON per document is saved to data/processed_v2/<file_id>.json.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.db.pinecone_client import PineconeClient
from app.db.sqlite_client import SQLiteClient
from app.ingestion_v2.document_classifier import classify
from app.ingestion_v2.entity_extractor import EntityExtractor
from app.ingestion_v2.lifecycle_tagger import tag_chunk
from app.ingestion_v2.logical_chunker import LogicalChunker, LogicalChunk
from app.ingestion_v2.parser import joined_markdown, parse_to_markdown
from app.ingestion_v2.pinecone_v2 import V2Pinecone
from app.ingestion_v2.project_master_builder import ProjectMasterBuilder
from app.ingestion_v2.schemas import (
    DocumentType,
    ExtractedDocument,
    LifecycleStage,
    ProjectMaster,
)
from app.repositories.ingested_files_repo import IngestedFilesRepo
from app.services.embedding_service import EmbeddingService
from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_v2"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for metadata-safe Pinecone payloads.
# Pinecone metadata values must be strings, numbers, booleans, or lists of strings.
# ─────────────────────────────────────────────────────────────────────────────

def _safe_meta(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return str(value)


@dataclass
class PipelineCounts:
    files_seen: int = 0
    files_indexed: int = 0
    files_failed: int = 0
    enriched_chunks_upserted: int = 0
    project_masters_upserted: int = 0
    project_master_count: int = 0
    errors: List[Tuple[str, str]] = field(default_factory=list)  # (filename, error)


class V2Pipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        pinecone: PineconeClient,
        sqlite: SQLiteClient,
    ):
        self.settings = settings
        self.embedder = EmbeddingService(settings)
        self.chunker = LogicalChunker()
        self.extractor = EntityExtractor(settings)
        self.v2 = V2Pinecone(pinecone)
        self.ingested_repo = IngestedFilesRepo(sqlite)
        self.master_builder = ProjectMasterBuilder()
        # In-memory chunk staging so we can batch embeds across the whole run.
        self._stage_chunks: List[dict] = []  # Pinecone-ready dicts
        # File ids re-processed this run — used to merge the local chunk store
        # (BM25 corpus for hybrid retrieval) without touching cached files.
        self._processed_file_ids: set = set()

    # ──────────────────────────────────────────────────────────────────
    # Run on local file lists.
    # ──────────────────────────────────────────────────────────────────
    def run_on_all_raw_docs(self, only_filenames: Optional[List[str]] = None) -> PipelineCounts:
        counts = PipelineCounts()
        ingested = self.ingested_repo.all_indexed()

        # Map raw_docs/<file_id>.<ext> → human document_name via SQLite.
        file_id_to_name = {fid: f.name for fid, f in ingested.items()}

        for fname in sorted(os.listdir(RAW_DOCS_DIR)):
            if fname.startswith("."):
                continue
            local_path = RAW_DOCS_DIR / fname
            if not local_path.is_file():
                continue
            counts.files_seen += 1
            file_id = local_path.stem  # filename without extension
            document_name = file_id_to_name.get(file_id, fname)

            if only_filenames and document_name not in only_filenames and file_id not in only_filenames:
                continue

            try:
                # ── Caching logic ─────────────────────────────────────────────
                json_path = PROCESSED_DIR / f"{file_id}.json"
                md_path = PROCESSED_DIR / f"{file_id}.md"
                is_cached = False
                
                if json_path.is_file() and md_path.is_file():
                    raw_mtime = os.path.getmtime(local_path)
                    json_mtime = os.path.getmtime(json_path)
                    db_record = ingested.get(file_id)
                    if json_mtime > raw_mtime and db_record and db_record.status == "indexed" and db_record.n_chunks > 0:
                        is_cached = True

                if is_cached:
                    logger.info(
                        "using cached V2 extraction (skipping parse & extract)",
                        extra={"file_id": file_id, "document_name": document_name},
                    )
                    with open(json_path, "r", encoding="utf-8") as f:
                        extracted = ExtractedDocument.model_validate_json(f.read())
                    self.master_builder.add_document(extracted, document_name)
                    counts.files_indexed += 1
                else:
                    self._process_one(str(local_path), file_id, document_name)
                    counts.files_indexed += 1
            except Exception as e:
                logger.exception(
                    "v2 ingestion failed",
                    extra={"file_id": file_id, "document_name": document_name, "err": str(e)},
                )
                counts.files_failed += 1
                counts.errors.append((document_name, str(e)))
                try:
                    mtime = os.path.getmtime(local_path)
                    modified_time = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                    self.ingested_repo.mark_failed(file_id, document_name, modified_time)
                except Exception:
                    pass

        # Flush enriched chunks
        counts.enriched_chunks_upserted = self.v2.upsert_enriched_chunks(self._stage_chunks)
        self._sync_chunk_store()
        self._stage_chunks.clear()
        self._processed_file_ids.clear()

        # Build + upsert masters
        masters = self.master_builder.build()
        counts.project_master_count = len(masters)
        if masters:
            # Purge the project master namespace first to prevent orphaned records!
            try:
                self.v2.pc.index.delete(delete_all=True, namespace="csr_project_master")
                logger.info("purged csr_project_master namespace before rebuilt project masters upsert")
            except Exception as e:
                logger.warning("failed to purge csr_project_master namespace", extra={"err": str(e)})

            master_vectors = self._embed_masters(masters)
            counts.project_masters_upserted = self.v2.upsert_project_masters(master_vectors)

        return counts

    # ──────────────────────────────────────────────────────────────────
    # Per-document
    # ──────────────────────────────────────────────────────────────────
    def _process_one(self, local_path: str, file_id: str, document_name: str) -> None:
        self._processed_file_ids.add(file_id)
        # 0. Wipe prior vectors to ensure idempotency.
        try:
            self.v2.delete_chunks_by_file_id(file_id)
            logger.info("wiped prior V2 enriched vectors", extra={"file_id": file_id})
        except Exception as e:
            logger.debug("delete prior V2 chunks failed", extra={"file_id": file_id, "err": str(e)})

        # 1. Parse to markdown (sanitized inside parse_to_markdown).
        segments = parse_to_markdown(local_path, self.settings)
        if not segments:
            logger.warning("no segments produced", extra={"file_id": file_id})
            return
        full_md = joined_markdown(segments)

        # 1b. Validation gate — fail CLOSED if any fabrication marker survived.
        from app.ingestion_v2.sanitizer import validate_markdown
        issues = validate_markdown(full_md)
        if issues:
            raise RuntimeError(
                f"validation gate: {len(issues)} fabrication marker(s) survived sanitization "
                f"in {document_name}; refusing to index. First: {issues[0]}"
            )

        # 1a. Save the markdown for inspection (closes the V1 gap).
        md_path = PROCESSED_DIR / f"{file_id}.md"
        md_path.write_text(full_md, encoding="utf-8")

        # 2. Classify document type using filename + first segment.
        heuristic = classify(document_name, segments[0].text if segments else None)

        # 3. Extract structured entities (one LLM call).
        extracted: ExtractedDocument = self.extractor.extract(
            filename=document_name,
            markdown=full_md,
            heuristic_type=heuristic,
        )
        # Save extraction JSON for inspection.
        json_path = PROCESSED_DIR / f"{file_id}.json"
        json_path.write_text(
            extracted.model_dump_json(indent=2),
            encoding="utf-8",
        )

        # 4. Aggregate into master builder.
        self.master_builder.add_document(extracted, document_name)

        # 5. Logical chunk + tag + embed.
        all_chunks: List[LogicalChunk] = []
        for seg in segments:
            for ch in self.chunker.chunk_page(seg.text, seg.page):
                all_chunks.append(ch)

        if not all_chunks:
            # Even if no chunks were produced, record a SQLite success with 0 chunks
            try:
                mtime = os.path.getmtime(local_path)
                modified_time = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                self.ingested_repo.upsert_success(file_id, document_name, modified_time, 0)
            except Exception:
                pass
            return

        # Tag lifecycle, build embed input
        texts_to_embed: List[str] = []
        local_meta: List[dict] = []
        upload_date = datetime.utcnow().isoformat() + "Z"
        for idx, ch in enumerate(all_chunks):
            stage = tag_chunk(ch.text)
            # Prepend a small header context so the embedding captures the section
            # identity, not just raw prose. This is a known accuracy win.
            embed_text = (
                f"[Document: {document_name}] "
                f"[Section: {ch.section_path}] "
                f"[Page {ch.page}]\n\n{ch.text}"
            )
            texts_to_embed.append(embed_text)
            local_meta.append(
                {
                    "id": f"{file_id}-p{ch.page}-c{idx}",
                    "metadata": self._chunk_metadata(
                        file_id=file_id,
                        document_name=document_name,
                        chunk=ch,
                        idx=idx,
                        upload_date=upload_date,
                        lifecycle_stage=stage,
                        extracted=extracted,
                        embed_text=embed_text,
                    ),
                }
            )

        # 6. Embed in batches.
        BATCH = 64
        for start in range(0, len(texts_to_embed), BATCH):
            batch_texts = texts_to_embed[start : start + BATCH]
            batch_embs = self.embedder.embed_batch(batch_texts)
            for offset, emb in enumerate(batch_embs):
                pos = start + offset
                self._stage_chunks.append(
                    {
                        "id": local_meta[pos]["id"],
                        "values": emb,
                        "metadata": local_meta[pos]["metadata"],
                    }
                )

        # Update SQLite table to indexed status
        try:
            mtime = os.path.getmtime(local_path)
            modified_time = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            self.ingested_repo.upsert_success(file_id, document_name, modified_time, len(all_chunks))
            logger.info("recorded V2 ingestion success to SQLite", extra={"file_id": file_id, "n_chunks": len(all_chunks)})
        except Exception as e:
            logger.warning("failed to update SQLite with ingestion success", extra={"file_id": file_id, "err": str(e)})

        logger.info(
            "v2 file processed",
            extra={
                "file_id": file_id,
                "document_name": document_name,
                "n_chunks": len(all_chunks),
                "doc_type": extracted.document_type.value,
                "n_projects": len(extracted.projects),
            },
        )

    # ──────────────────────────────────────────────────────────────────
    # Local chunk store (BM25 corpus for hybrid retrieval)
    # ──────────────────────────────────────────────────────────────────
    def _sync_chunk_store(self) -> None:
        """Mirror this run's staged chunks into the local chunk store JSON so
        the hybrid BM25 index stays in lock-step with Pinecone. Files that were
        served from cache this run keep their existing store entries."""
        if not self._stage_chunks:
            return
        p = Path(self.settings.chunk_store_path)
        store_path = p if p.is_absolute() else PROJECT_ROOT / p
        store_path.parent.mkdir(parents=True, exist_ok=True)

        existing: List[dict] = []
        if store_path.is_file():
            try:
                existing = json.loads(store_path.read_text(encoding="utf-8")).get("chunks", [])
            except Exception as e:
                logger.warning("chunk store unreadable — rebuilding from this run only", extra={"err": str(e)})

        kept = [
            c for c in existing
            if c.get("metadata", {}).get("file_id") not in self._processed_file_ids
        ]
        for v in self._stage_chunks:
            md = dict(v["metadata"])
            text = md.pop("text", "")
            md.pop("embedded_text_preview", None)
            kept.append({"id": v["id"], "text": text, "metadata": md})

        store_path.write_text(
            json.dumps(
                {"built_at": datetime.utcnow().isoformat() + "Z", "chunks": kept},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(
            "chunk store synced",
            extra={"path": str(store_path), "n_chunks": len(kept), "n_new": len(self._stage_chunks)},
        )

    # ──────────────────────────────────────────────────────────────────
    # Metadata builders
    # ──────────────────────────────────────────────────────────────────
    def _chunk_metadata(
        self,
        *,
        file_id: str,
        document_name: str,
        chunk: LogicalChunk,
        idx: int,
        upload_date: str,
        lifecycle_stage: LifecycleStage,
        extracted: ExtractedDocument,
        embed_text: str,
    ) -> dict:
        # Project linkage — list of all projects mentioned in this document
        # (chunk-level project linkage would require per-chunk LLM extraction,
        # which is out of scope per the user's "no eval this round" constraint).
        project_names = [p.project_name for p in extracted.projects if p.project_name]
        ngo_names = list({p.ngo.ngo_name for p in extracted.projects if p.ngo and p.ngo.ngo_name})
        sectors = list({p.classification.csr_sector for p in extracted.projects if p.classification and p.classification.csr_sector})
        schedules = list({p.classification.csr_schedule for p in extracted.projects if p.classification and p.classification.csr_schedule})
        states = list({p.geography.state for p in extracted.projects if p.geography and p.geography.state})

        md = {
            "text": _safe_meta(chunk.text),
            "embedded_text_preview": _safe_meta(embed_text[:400]),
            "file_id": _safe_meta(file_id),
            "document_name": _safe_meta(document_name),
            "document_type": _safe_meta(extracted.document_type.value),
            "page": _safe_meta(chunk.page),
            "chunk_index": _safe_meta(idx),
            "section_path": _safe_meta(chunk.section_path),
            "lifecycle_stage": _safe_meta(lifecycle_stage.value),
            "upload_date": _safe_meta(upload_date),
            "meeting_number": _safe_meta(extracted.meeting.meeting_number),
            "meeting_date": _safe_meta(extracted.meeting.meeting_date),
            "financial_year": _safe_meta(extracted.meeting.financial_year),
            "committee_name": _safe_meta(extracted.governance.committee_name),
            "project_names": _safe_meta(project_names),
            "ngo_names": _safe_meta(ngo_names),
            "csr_sectors": _safe_meta(sectors),
            "csr_schedules": _safe_meta(schedules),
            "states": _safe_meta(states),
            "source_namespace": "csr_v2_enriched",
        }
        return md

    # ──────────────────────────────────────────────────────────────────
    # Master embedding
    # ──────────────────────────────────────────────────────────────────
    def _embed_masters(self, masters: List[ProjectMaster]) -> List[dict]:
        texts = [m.summary_text for m in masters]
        # Batched embeds
        out: List[dict] = []
        BATCH = 64
        for start in range(0, len(texts), BATCH):
            batch_texts = texts[start : start + BATCH]
            embeddings = self.embedder.embed_batch(batch_texts)
            for offset, emb in enumerate(embeddings):
                m = masters[start + offset]
                out.append(
                    {
                        "id": m.project_id,
                        "values": emb,
                        "metadata": {
                            "project_id": _safe_meta(m.project_id),
                            "project_name": _safe_meta(m.project_name),
                            "aliases": _safe_meta(m.aliases),
                            "ngo_name": _safe_meta(m.ngo_name),
                            "sector": _safe_meta(m.sector),
                            "schedule_vii_clause": _safe_meta(m.schedule_vii_clause),
                            "current_status": _safe_meta(m.current_status.value),
                            "state": _safe_meta(m.geography.state if m.geography else None),
                            "district": _safe_meta(m.geography.district if m.geography else None),
                            "city": _safe_meta(m.geography.city if m.geography else None),
                            "approved_cost": _safe_meta(m.approved_cost),
                            "disbursed_amount": _safe_meta(m.disbursed_amount),
                            "balance_amount": _safe_meta(m.balance_amount),
                            "beneficiary_count": _safe_meta(m.beneficiary_count),
                            "beneficiary_type": _safe_meta(m.beneficiary_type),
                            "lifecycle_stages_crossed": _safe_meta([s.value for s in m.lifecycle_stages_crossed]),
                            "meeting_numbers_referenced": _safe_meta([str(n) for n in m.meeting_numbers_referenced]),
                            "source_documents": _safe_meta(m.source_documents),
                            "summary_text": _safe_meta(m.summary_text[:2000]),
                            "source_namespace": "csr_project_master",
                        },
                    }
                )
        return out

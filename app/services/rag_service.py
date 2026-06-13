import asyncio
import re
from typing import List, Set

from app.models.message_model import RAGResult, RetrievedChunk
from app.repositories.request_log_repo import RequestLogRepo
from app.services.document_directory_service import DocumentDirectoryService
from app.services.embedding_service import EmbeddingService
from app.services.memory_service import MemoryService
from app.services.prompt_service import PromptService
from app.services.response_service import ResponseService
from app.services.vector_service import VectorService
from app.utils.helpers import timed
from app.utils.logger import get_logger
from app.services.intelligence_layer import IntelligenceLayerService
from app.utils.env import Settings
from typing import List, Set, Optional

logger = get_logger(__name__)

# Matches "26th", "27", "29th meeting", "meeting 26" — any 1-3 digit ordinal mention.
_ORDINAL_IN_QUERY = re.compile(
    r"\b(\d{1,3})(?:st|nd|rd|th)?\b", re.IGNORECASE
)


class RAGService:
    def __init__(
        self,
        embedder: EmbeddingService,
        vectors: VectorService,
        memory: MemoryService,
        prompts: PromptService,
        responder: ResponseService,
        request_log: RequestLogRepo,
        directory: DocumentDirectoryService,
        settings: Settings,
        intelligence_layer: Optional[IntelligenceLayerService] = None,
    ):
        self.embedder = embedder
        self.vectors = vectors
        self.memory = memory
        self.prompts = prompts
        self.responder = responder
        self.request_log = request_log
        self.directory = directory
        self.settings = settings
        self.intelligence_layer = intelligence_layer

    async def answer(self, chat_id: int, question: str) -> RAGResult:
        with timed() as total:
            try:
                history = await asyncio.to_thread(self.memory.recent, chat_id)

                if self.settings.enable_intelligence_layer and self.intelligence_layer:
                    result = await self.intelligence_layer.answer(chat_id, question, history)
                    await asyncio.to_thread(self.memory.record_user, chat_id, question)
                    clean_ans = result.answer.split("\n\nSources:\n")[0]
                    await asyncio.to_thread(self.memory.record_assistant, chat_id, clean_ans)
                else:
                    embedding = await asyncio.to_thread(self.embedder.embed, question)

                    # Primary similarity retrieval (passes question so V2 can apply Phase 1 metadata filter).
                    primary_chunks = await asyncio.to_thread(self.vectors.query, embedding, question)

                    # Targeted per-meeting retrieval whenever the question mentions an
                    # ordinal that matches an indexed meeting (e.g. "26th"). This
                    # guarantees the right document's chunks land in context even when
                    # similarity search ranks neighbouring meetings higher.
                    targeted_chunks = await asyncio.to_thread(
                        self._targeted_retrieval, question, embedding
                    )

                    chunks = self._merge_chunks(primary_chunks, targeted_chunks)

                    directory_text = self.directory.format_for_prompt()
                    messages = self.prompts.build_messages(
                        question=question,
                        chunks=chunks,
                        memory=history,
                        document_directory=directory_text,
                    )
                    llm = await asyncio.to_thread(self.responder.generate, messages)
                    answer_text = self._with_sources(llm.text, chunks)

                    await asyncio.to_thread(self.memory.record_user, chat_id, question)
                    await asyncio.to_thread(self.memory.record_assistant, chat_id, llm.text)

                    result = RAGResult(
                        answer=answer_text,
                        chunks=chunks,
                        provider=llm.provider,
                        model=llm.model,
                        latency_ms=0,
                    )
            except Exception as e:
                logger.exception("rag pipeline failed", extra={"chat_id": chat_id, "err": str(e)})
                await asyncio.to_thread(
                    self.request_log.log,
                    chat_id=chat_id,
                    question=question,
                    answer=None,
                    n_retrieved=0,
                    latency_ms=total.get("ms", 0),
                    model="",
                    provider="",
                    status="error",
                    error=str(e),
                )
                raise

        result.latency_ms = total["ms"]
        logger.info(
            "rag answered",
            extra={
                "chat_id": chat_id,
                "latency_ms": result.latency_ms,
                "n_chunks": len(result.chunks),
                "n_memory_turns": len(history),
                "provider": result.provider,
                "model": result.model,
            },
        )
        await asyncio.to_thread(
            self.request_log.log,
            chat_id=chat_id,
            question=question,
            answer=result.answer,
            n_retrieved=len(result.chunks),
            latency_ms=result.latency_ms,
            model=result.model,
            provider=result.provider,
            status="ok",
        )
        return result

    def _targeted_retrieval(self, question: str, embedding: List[float]) -> List[RetrievedChunk]:
        """If the question mentions any indexed meeting ordinal, run a per-meeting
        metadata-filtered Pinecone query so those chunks are guaranteed in context.
        """
        mentioned: Set[int] = set()
        for m in _ORDINAL_IN_QUERY.finditer(question):
            try:
                n = int(m.group(1))
                if 1 <= n <= 999:
                    mentioned.add(n)
            except ValueError:
                continue
        if not mentioned:
            return []

        # Restrict to ordinals that actually correspond to an indexed document
        # (avoids spurious "Top 5 / last 2 / next 3" matches).
        indexed_ordinals = {e.meeting_number for e in self.directory.all() if e.meeting_number is not None}
        relevant = mentioned & indexed_ordinals
        if not relevant:
            return []

        all_chunks: List[RetrievedChunk] = []
        for n in sorted(relevant):
            doc_names = self.directory.document_names_for_meeting(n)
            for doc_name in doc_names:
                try:
                    sub = self.vectors.query(
                        embedding,
                        metadata_filter={"document_name": {"$eq": doc_name}},
                        top_k=8,
                    )
                except Exception as e:
                    logger.warning(
                        "targeted retrieval failed",
                        extra={"meeting": n, "doc_name": doc_name, "err": str(e)},
                    )
                    continue
                all_chunks.extend(sub)
        logger.info(
            "targeted retrieval",
            extra={
                "mentioned_ordinals": sorted(list(mentioned)),
                "matched_to_index": sorted(list(relevant)),
                "n_targeted_chunks": len(all_chunks),
            },
        )
        return all_chunks

    @staticmethod
    def _merge_chunks(primary: List[RetrievedChunk], targeted: List[RetrievedChunk]) -> List[RetrievedChunk]:
        # Dedupe by chunk_id (falls back to text hash if missing) preserving order:
        # targeted first (so the LLM is more likely to see them in case of truncation),
        # then primary fills out breadth.
        seen: Set[str] = set()
        out: List[RetrievedChunk] = []
        for c in list(targeted) + list(primary):
            key = c.chunk_id or (c.document_name or "") + "::" + (c.text[:40] if c.text else "")
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _with_sources(self, answer: str, chunks) -> str:
        if not chunks:
            return answer
        seen = []
        for c in chunks:
            if not c.document_name:
                continue
            label = self.directory.humanize_source(c.document_name)
            pg = self.directory.clean_page(c.page)
            if pg:
                label += f" (p.{pg})"
            if label not in seen:
                seen.append(label)
        if not seen:
            return answer
        # Cap source list at 8 entries for readability.
        return f"{answer}\n\nSources:\n" + "\n".join(f"• {s}" for s in seen[:8])

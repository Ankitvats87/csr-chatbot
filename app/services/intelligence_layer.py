import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import difflib

from app.models.message_model import RetrievedChunk, Turn, RAGResult
from app.services.embedding_service import EmbeddingService
from app.services.response_service import ResponseService
from app.services.document_directory_service import DocumentDirectoryService
from app.repositories.request_log_repo import RequestLogRepo
from app.ingestion_v2.project_master_builder import ProjectMasterBuilder
from app.ingestion_v2.schemas import ExtractedDocument, ProjectMaster
from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Constants for project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_v2"


class IntelligenceLayerService:
    def __init__(
        self,
        embedder: EmbeddingService,
        vectors,  # Can be VectorService or VectorServiceV2
        responder: ResponseService,
        directory: DocumentDirectoryService,
        settings: Settings,
        hybrid=None,  # Optional[HybridSearchService]
        reranker=None,  # Optional[LLMReranker]
        kb=None,  # Optional[KBQueryService] — exact structured records
    ):
        self.embedder = embedder
        self.vectors = vectors
        self.responder = responder
        self.directory = directory
        self.settings = settings
        self.hybrid = hybrid
        self.reranker = reranker
        self.kb = kb
        
        # In-memory corpus index
        self.project_masters: List[ProjectMaster] = []
        self.projects_by_name: Dict[str, ProjectMaster] = {}
        self.ngos_by_name: Dict[str, List[ProjectMaster]] = {}
        self.meetings: Dict[int, List[ProjectMaster]] = {}
        
        self._load_corpus()

    def _load_corpus(self) -> None:
        """Loads processed V2 JSON files and canonicalizes them in-memory."""
        start_time = time.time()
        if not PROCESSED_DIR.is_dir():
            logger.warning("Processed V2 directory does not exist. In-memory indexing skipped.", extra={"dir": str(PROCESSED_DIR)})
            return

        try:
            builder = ProjectMasterBuilder()
            loaded_files = 0
            for f in PROCESSED_DIR.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        extracted = ExtractedDocument.model_validate_json(file.read())
                        # Reconstruct document_name from directory or default
                        doc_name = f.stem
                        # Find corresponding doc name in directory service if possible
                        for entry in self.directory.all():
                            if entry.document_name.startswith(f.stem) or f.stem in entry.document_name:
                                doc_name = entry.document_name
                                break
                        builder.add_document(extracted, doc_name)
                        loaded_files += 1
                except Exception as ex:
                    logger.warning("failed to load json file in corpus loader", extra={"file": str(f), "err": str(ex)})

            self.project_masters = builder.build()
            
            # Map canonical names and aliases
            for pm in self.project_masters:
                self.projects_by_name[pm.project_name.lower().strip()] = pm
                for alias in pm.aliases:
                    self.projects_by_name[alias.lower().strip()] = pm
                if pm.ngo_name:
                    self.ngos_by_name.setdefault(pm.ngo_name.lower().strip(), []).append(pm)
                for mnum in pm.meeting_numbers_referenced:
                    self.meetings.setdefault(mnum, []).append(pm)
                    
            logger.info(
                "corpus loaded successfully",
                extra={
                    "files_loaded": loaded_files,
                    "projects_mapped": len(self.projects_by_name),
                    "ngos_mapped": len(self.ngos_by_name),
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
            )
        except Exception as e:
            logger.exception("failed to load corpus database", extra={"err": str(e)})

    # ───── Fuzzy Matching Entity Resolution ─────
    def _clean_name(self, name: str) -> str:
        s = name.lower().strip()
        s = s.replace("project", "").replace("ngo", "").replace("foundation", "").replace("trust", "").strip()
        return re.sub(r"[^\w\s]", "", s)

    def resolve_project(self, raw_name: str) -> Optional[str]:
        if not raw_name:
            return None
        qn = raw_name.lower().strip()
        
        # 1. Exact match
        if qn in self.projects_by_name:
            return self.projects_by_name[qn].project_name

        # 2. Cleaned exact match / Substring match
        q_clean = self._clean_name(qn)
        if not q_clean:
            return None
            
        for name in self.projects_by_name.keys():
            name_clean = self._clean_name(name)
            if not name_clean:
                continue
            if q_clean == name_clean or q_clean in name_clean or name_clean in q_clean:
                return self.projects_by_name[name].project_name

        # 3. Token set overlap match
        q_tokens = set(q_clean.split())
        best_overlap = 0
        best_match = None
        for name, pm in self.projects_by_name.items():
            name_clean = self._clean_name(name)
            name_tokens = set(name_clean.split())
            overlap = len(q_tokens & name_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = pm.project_name
                
        # Require significant token overlap relative to query size
        if best_overlap > 0 and best_overlap >= len(q_tokens) * 0.5:
            return best_match

        # 4. Cleaned difflib backup (run on cleaned candidate names to avoid suffix matching)
        candidates_clean = {}
        for name, pm in self.projects_by_name.items():
            c_clean = self._clean_name(name)
            if c_clean:
                candidates_clean[c_clean] = pm.project_name
                
        matches = difflib.get_close_matches(q_clean, list(candidates_clean.keys()), n=1, cutoff=0.6)
        if matches:
            return candidates_clean[matches[0]]

        return None

    def resolve_ngo(self, raw_name: str) -> Optional[str]:
        if not raw_name:
            return None
        qn = raw_name.lower().strip()
        
        # 1. Exact match
        if qn in self.ngos_by_name:
            # Return canonical capitalization
            return self.ngos_by_name[qn][0].ngo_name

        # 2. Substring match on NGO names
        q_clean = self._clean_name(qn)
        if not q_clean:
            return None
            
        for name in self.ngos_by_name.keys():
            name_clean = self._clean_name(name)
            if q_clean == name_clean or q_clean in name_clean or name_clean in q_clean:
                return self.ngos_by_name[name][0].ngo_name

        # 3. Cleaned difflib fallback
        candidates_clean = {}
        for name, pms in self.ngos_by_name.items():
            c_clean = self._clean_name(name)
            if c_clean:
                candidates_clean[c_clean] = pms[0].ngo_name
        matches = difflib.get_close_matches(q_clean, list(candidates_clean.keys()), n=1, cutoff=0.6)
        if matches:
            return candidates_clean[matches[0]]

        return None

    # ───── Unified Planning & Extraction LLM Call ─────
    def _parse_plan(self, question: str, history: List[Turn]) -> dict:
        history_text = "\n".join(f"{t.role.upper()}: {t.content}" for t in history) if history else "(no history)"
        
        prompt = f"""You are a CSR Query Planner and Analyzer.
Your task is to analyze the user's latest query, taking into account the recent conversation history, and produce a structured JSON query plan.

CONVERSATION HISTORY:
{history_text}

LATEST USER QUERY:
{question}

INSTRUCTIONS:
1. Rewrite the query if it contains conversational pronouns or refers to earlier topics. The rewritten query must be a standalone, self-contained search query. If the query is already standalone, keep it as is.
2. Classify the query intent into one of the following exact categories:
   - meeting_summary
   - meeting_decisions
   - meeting_projects
   - meeting_budget
   - project_status
   - project_budget
   - project_history
   - project_approval
   - project_timeline
   - ngo_projects
   - ngo_budget
   - ngo_history
   - resolution_lookup
   - board_approval_lookup
   - amendment_lookup
   - budget_lookup
   - expenditure_lookup
   - allocation_lookup
   - timeline_lookup
   - person_attendance
   - general_faq
3. Extract any specific entities mentioned in the query:
   - meeting_number: integer (e.g. 26)
   - board_meeting_number: string or null
   - project_name: string or null
   - ngo_name: string or null
   - person_name: string or null (full name of a committee member, attendee, or officer — e.g. "Dr. Manoj Kumar Jhawar")
   - resolution_number: string or null
   - agenda_item: string or null
   - financial_year: string or null (e.g. "2024-25")
   - state: string or null
   - district: string or null

   NUMBERING CONVENTION: agenda items are referenced as <meeting>.<item>,
   e.g. "23.1" means CSR Meeting 23, Agenda Item 1. If the query contains
   such a pattern (e.g. "agenda 26.3", "item 24.2"), set meeting_number to
   the integer before the dot and agenda_item to the number after the dot.
   Do NOT treat "Project A"/"Project B" style letters as project names —
   those are per-meeting table labels, not real project identities.
4. If the query is completely vague, incomplete, or invalid (e.g., just "Status" or "details" without naming a project, meeting, or NGO), set "is_invalid" to true and explain.

You must output a valid JSON object ONLY. Do not output any markdown formatting (like ```json ... ```), code blocks, or extra text.

JSON Schema:
{{
  "rewritten_query": "standalone query text",
  "intent": "detected_intent",
  "entities": {{
    "meeting_number": null or integer,
    "board_meeting_number": null or string,
    "project_name": null or string,
    "ngo_name": null or string,
    "person_name": null or string,
    "resolution_number": null or string,
    "agenda_item": null or string,
    "financial_year": null or string,
    "state": null or string,
    "district": null or string
  }},
  "confidence": 0.0 to 1.0,
  "is_invalid": true or false,
  "invalid_explanation": null or string
}}
"""
        messages = [{"role": "system", "content": prompt}]
        resp = self.responder.generate(messages)
        
        # Clean response string
        text = resp.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            plan = json.loads(text)
            logger.info("query planner analyzed request", extra={"plan": plan})
            return plan
        except Exception as e:
            logger.warning("failed to parse JSON from query planner", extra={"text": text, "err": str(e)})
            # Safe fallback plan
            return {
                "rewritten_query": question,
                "intent": "general_faq",
                "entities": {
                    "meeting_number": None, "board_meeting_number": None, "project_name": None,
                    "ngo_name": None, "person_name": None, "resolution_number": None,
                    "agenda_item": None, "financial_year": None, "state": None, "district": None
                },
                "confidence": 0.5,
                "is_invalid": False,
                "invalid_explanation": None
            }

    # ───── Multi-Hop Retrieval Router ─────
    def _query_pinecone(self, embedding: List[float], namespace: str, top_k: int, metadata_filter: Optional[dict] = None) -> List[RetrievedChunk]:
        kwargs = {
            "vector": embedding,
            "top_k": top_k,
            "namespace": namespace,
            "include_metadata": True
        }
        if metadata_filter:
            kwargs["filter"] = metadata_filter

        try:
            resp = self.vectors.pinecone.index.query(**kwargs)
            matches = resp.get("matches") if isinstance(resp, dict) else getattr(resp, "matches", [])
            chunks: List[RetrievedChunk] = []
            
            # Use specific similarity threshold per namespace
            # Pinecone score thresholds: V2 enriched is diluted, so we use lower thresholds
            min_score = 0.40 if namespace == "csr_v2_enriched" else 0.35
            
            from app.ingestion_v2.project_master_builder import is_generic_label

            for m in matches or []:
                score = m["score"] if isinstance(m, dict) else m.score
                if score is None or score < min_score:
                    continue
                md = (m["metadata"] if isinstance(m, dict) else (m.metadata or {})) or {}
                # Skip stale letter-label project masters ("Project C") that
                # may remain in Pinecone until the next full ingestion.
                if namespace == "csr_project_master" and is_generic_label(md.get("project_name")):
                    continue
                chunk_id = m["id"] if isinstance(m, dict) else m.id
                page = md.get("page")
                
                text_content = md.get("text") or md.get("summary_text") or ""
                chunks.append(
                    RetrievedChunk(
                        text=text_content,
                        score=float(score),
                        document_name=md.get("document_name") or md.get("project_name") or "Unknown",
                        source=namespace,
                        page=str(page) if page is not None else None,
                        chunk_id=chunk_id,
                    )
                )
            return chunks
        except Exception as e:
            logger.exception("pinecone query failed in intelligence layer", extra={"namespace": namespace, "err": str(e)})
            return []

    def _retrieve_attendance(self, person_name: str) -> List[RetrievedChunk]:
        """Deterministic cross-meeting attendance retrieval.

        Attendance counting fails with plain semantic search because a person
        listed "By Invitation" (CMD/CFO) scores low against a name query, so
        their meeting drops out of the top-k. The corpus is small (~11 meetings),
        so instead we walk EVERY indexed meeting and pull its top attendee-bearing
        chunks via a per-meeting metadata filter. This guarantees every meeting's
        attendee list reaches the LLM, which then scans for the person by name
        across all roles. We return these directly (bypassing the rerank/context
        cap) so no meeting is silently dropped before generation."""
        meeting_nums = sorted(
            {e.meeting_number for e in self.directory.all() if e.meeting_number is not None}
        )
        if not meeting_nums:
            return []
        # Name + role focused embedding ranks the attendee block highest within
        # each meeting (covers "Member", "Chairperson", "By Invitation", CMD/CFO).
        name_emb = self.embedder.embed(
            f"{person_name} attendance members present in attendance "
            f"by invitation chairperson CMD CFO CSR committee meeting minutes"
        )
        out: List[RetrievedChunk] = []
        seen: Set[str] = set()
        covered = 0
        for n in meeting_nums:
            hits = self._query_pinecone(
                name_emb, "csr_v2_enriched", top_k=3,
                metadata_filter={"meeting_number": {"$eq": n}},
            )
            if hits:
                covered += 1
            for c in hits:
                key = c.chunk_id or f"{c.document_name}::{c.text[:50]}"
                if key not in seen:
                    seen.add(key)
                    out.append(c)
        logger.info(
            "attendance retrieval (per-meeting)",
            extra={"person": person_name, "meetings_total": len(meeting_nums),
                   "meetings_covered": covered, "n_chunks": len(out)},
        )
        return out

    def _execute_retrieval(self, plan: dict, embedding: List[float]) -> List[RetrievedChunk]:
        intent = plan.get("intent", "general_faq")
        entities = plan.get("entities", {})
        rewritten_query = plan.get("rewritten_query", "")

        chunks: List[RetrievedChunk] = []

        # Person-attendance: handled by a dedicated exhaustive walk over every
        # meeting (see _retrieve_attendance). Returned directly so the rerank +
        # context cap below can't drop a meeting before the count is computed.
        # Gate on the attendance intent OR an attendance-phrased query so person
        # questions like "what did X propose" are NOT hijacked into this path.
        person_name = (entities.get("person_name") or "").strip()
        rq = (rewritten_query or "").lower()
        _attendance_words = ("attend", "present in", "present at", "meetings did",
                             "how many meeting", "which meeting", "was in the", "part of the meeting")
        if intent == "person_attendance" or (person_name and any(w in rq for w in _attendance_words)):
            att = self._retrieve_attendance(person_name or rewritten_query)
            if att:
                return att
            # Fall through to standard retrieval if the per-meeting walk found nothing.

        # 1. Resolve Project Entity (Fuzzy Match)
        resolved_project = None
        raw_project = entities.get("project_name")
        if raw_project:
            resolved_project = self.resolve_project(raw_project)
            logger.info("project entity resolved", extra={"raw": raw_project, "resolved": resolved_project})

        # 2. Resolve NGO Entity (Fuzzy Match)
        resolved_ngo = None
        raw_ngo = entities.get("ngo_name")
        if raw_ngo:
            resolved_ngo = self.resolve_ngo(raw_ngo)
            logger.info("ngo entity resolved", extra={"raw": raw_ngo, "resolved": resolved_ngo})
            
        # 3. Route based on resolved entities & intent
        if resolved_project:
            pm = self.projects_by_name.get(resolved_project.lower())
            # Hop 1: Add canonical project master summary from in-memory (guaranteed lookup)
            if pm:
                chunks.append(RetrievedChunk(
                    text=f"[Project Master — {pm.project_name}]\n{pm.summary_text}",
                    score=1.0,
                    document_name=f"Project Master — {pm.project_name}",
                    source="csr_project_master",
                    page=None,
                    chunk_id=pm.project_id
                ))
            
            # Hop 2: Query csr_v2_enriched for this specific project
            proj_filter = {"project_names": {"$in": [resolved_project]}}
            chunks.extend(self._query_pinecone(embedding, "csr_v2_enriched", top_k=10, metadata_filter=proj_filter))
            
            # Hop 2b: If it refers to specific meetings, pull those meetings' chunks
            if pm and pm.meeting_numbers_referenced:
                meeting_filter = {"meeting_number": {"$in": pm.meeting_numbers_referenced}}
                chunks.extend(self._query_pinecone(embedding, "csr_v2_enriched", top_k=8, metadata_filter=meeting_filter))
                
            # Hop 3: Standard semantic fallback to fill context gaps
            chunks.extend(self._query_pinecone(embedding, "csr_v2_enriched", top_k=5))
            
        elif resolved_ngo:
            # Hop 1: Load all project master entries for this NGO
            ngo_pms = self.ngos_by_name.get(resolved_ngo.lower(), [])
            for pm in ngo_pms:
                chunks.append(RetrievedChunk(
                    text=f"[Project Master — {pm.project_name}]\n{pm.summary_text}",
                    score=1.0,
                    document_name=f"Project Master — {pm.project_name}",
                    source="csr_project_master",
                    page=None,
                    chunk_id=pm.project_id
                ))
            # Hop 2: Query enriched chunks with NGO filter
            ngo_filter = {"ngo_names": {"$in": [resolved_ngo]}}
            chunks.extend(self._query_pinecone(embedding, "csr_v2_enriched", top_k=10, metadata_filter=ngo_filter))
            chunks.extend(self._query_pinecone(embedding, "csr_v2_enriched", top_k=5))

        elif entities.get("meeting_number") is not None:
            # Hop 1: pull the requested meeting's chunks via metadata filter.
            mnum = int(entities.get("meeting_number"))
            meeting_filter = {"meeting_number": {"$eq": mnum}}
            chunks.extend(self._query_pinecone(embedding, "csr_v2_enriched", top_k=15, metadata_filter=meeting_filter))
            # Hop 2: ALWAYS run a broad semantic fallback. meeting_number metadata
            # is unreliable (some docs were extracted with null/wrong numbers — e.g.
            # a "Minutes of the 28th meeting" tagged null), so a filter-only path
            # silently loses the very document that holds the answer. We retrieve
            # broadly and PREFER meeting-N via boost_by_entities below — never hard
            # exclude. The LLM then corroborates the fact across the retrieved set.
            chunks.extend(self._query_pinecone(embedding, "csr_v2_enriched", top_k=10))

        else:
            # General standard multi-namespace retrieval
            chunks.extend(self._query_pinecone(embedding, "csr_project_master", top_k=4))
            chunks.extend(self._query_pinecone(embedding, "csr_v2_enriched", top_k=15))

        # Deduplicate chunks
        seen = set()
        deduped: List[RetrievedChunk] = []
        for c in chunks:
            key = c.chunk_id or f"{c.document_name}::{c.text[:50]}"
            if key not in seen:
                seen.add(key)
                deduped.append(c)
                
        # Sort chunks: Project Masters first, then enriched sorted by similarity score descending
        masters = [c for c in deduped if c.source == "csr_project_master"]
        enriched = sorted([c for c in deduped if c.source == "csr_v2_enriched"], key=lambda x: x.score, reverse=True)

        # ── Hybrid lexical fusion + entity boost + rerank ──────────────
        # BM25 catches exact tokens (amounts, resolution numbers, names) that
        # dense embeddings rank poorly; RRF rewards chunks both systems agree on.
        if self.settings.enable_hybrid_retrieval and self.hybrid and self.hybrid.available:
            lexical = self.hybrid.lexical_search(rewritten_query or "")
            enriched = self.hybrid.fuse(enriched, lexical)

            meeting_set = None
            if entities.get("meeting_number") is not None:
                try:
                    meeting_set = {int(entities["meeting_number"])}
                except (TypeError, ValueError):
                    meeting_set = None
            enriched = self.hybrid.boost_by_entities(
                enriched,
                meeting_numbers=meeting_set,
                project_names={resolved_project} if resolved_project else None,
                ngo_names={resolved_ngo} if resolved_ngo else None,
            )

            if self.settings.enable_reranker and self.reranker:
                candidates = enriched[: self.settings.rerank_candidates]
                enriched = self.reranker.rerank(
                    rewritten_query or "", candidates, top_n=self.settings.context_max_chunks
                )

        # NOTE: We deliberately do NOT hard-filter chunks to the requested
        # meeting here. meeting_number metadata is unreliable, so excluding
        # "out-of-scope" chunks was silently dropping the correct document
        # (e.g. a mistagged minutes file) and producing "insufficient evidence".
        # Instead we PREFER meeting-N via boost_by_entities above and let the
        # LLM corroborate the fact across the retrieved set.

        # Cap context size so the prompt stays within budget.
        enriched = enriched[: self.settings.context_max_chunks]

        # If intent is a timeline query, sort the enriched chunks chronologically by meeting number
        if "timeline" in intent:
            def get_meeting_num(c: RetrievedChunk) -> float:
                # Extract meeting number from document name if not present in metadata
                m = re.search(r"\b(\d+)(st|nd|rd|th)?\b", c.document_name or "")
                return float(m.group(1)) if m else 999.0
            enriched.sort(key=get_meeting_num)
            
        final_chunks = masters + enriched
        logger.info("retrieval executed", extra={"n_retrieved": len(final_chunks), "has_project": bool(resolved_project)})
        return final_chunks

    # ───── Context Aggregation & Formatting ─────
    def _build_context(self, chunks: List[RetrievedChunk], kb_blocks: str = "") -> str:
        # The document directory makes corpus-wide aggregate questions
        # ("list all meetings", "how many documents") answerable without
        # depending on similarity search.
        directory_block = (
            "=== DOCUMENT DIRECTORY (every indexed CSR document, with meeting numbers and dates) ===\n"
            + self.directory.format_for_prompt()
            + "\n"
        )

        kb_block = ""
        if kb_blocks:
            kb_block = (
                "=== STRUCTURED CROSS-CHECK REFERENCE (an independent index of meeting facts — "
                "use it to VERIFY the chunks below. If the reference and the chunks AGREE, answer "
                "confidently. If they CONFLICT, report the conflict and cite both; do NOT silently "
                "pick one) ===\n"
                + kb_blocks
                + "\n"
            )

        if not chunks:
            return kb_block + directory_block + "\n(no relevant document chunks retrieved)"

        masters = [c for c in chunks if c.source == "csr_project_master"]
        enriched = [c for c in chunks if c.source == "csr_v2_enriched"]

        lines: List[str] = [b for b in (kb_block, directory_block) if b]

        if masters:
            lines.append("=== CANONICAL PROJECT SUMMARIES ===")
            for i, c in enumerate(masters, 1):
                c.chunk_id = f"Ref-{i}"  # Set identifier tag
                lines.append(f"[Ref {i}] {c.document_name}:\n{c.text.strip()}\n")
                
        if enriched:
            lines.append("=== DETAILED CSR DOCUMENT CHUNKS ===")
            for i, c in enumerate(enriched, len(masters) + 1):
                c.chunk_id = f"Ref-{i}"
                header = f"[Ref {i}] Source: {c.document_name or 'Unknown Document'}"
                if c.page:
                    header += f", Page {c.page}"
                lines.append(f"{header} (Score: {c.score:.2f}):\n{c.text.strip()}\n")
                
        return "\n".join(lines)

    # ───── Dual-Pass Generation & Critique Loop ─────
    def _generate_answer(self, question: str, plan: dict, context_block: str, history: List[Turn]) -> str:
        intent = plan.get("intent", "general_faq")
        history_text = "\n".join(f"{t.role.upper()}: {t.content}" for t in history) if history else "(no history)"
        
        # Pass 1: drafting the reply using templates
        draft_prompt = f"""You are a CSR Governance Intelligence Assistant.
Your task is to write a draft answer to the user's question using ONLY the provided verified context.

CONVERSATION HISTORY:
{history_text}

VERIFIED CONTEXT:
{context_block}

USER QUERY:
{question}

INSTRUCTIONS:
1. ANSWER THE QUESTION THAT WAS ASKED. The user's specific question always
   takes priority over any template. If they ask about the annual action plan,
   give the action plan details (with its budget table if present in context);
   do not substitute a generic meeting summary.
2. Write a direct, factual, and concise answer. No fillers, pleasantries, or preamble.
3. Cite your sources using inline tags like [Ref 1], [Ref 2] matching the references
   in the verified context. Every major fact or number must have a citation tag.
4. NAMING RULES (critical):
   - NEVER refer to a project as "Project A", "Project B", etc. Those letters are
     per-meeting table labels in the source documents, not project identities.
     Always use the descriptive project title and implementing NGO from the context,
     e.g. "CT-Scanner for Safdarjung Hospital (Doctors for You)".
   - Agenda items are numbered <meeting>.<item>: agenda item 26.3 means CSR Meeting 26,
     Agenda Item 3. Use this numbering when referring to agenda items.
   - If the context only shows a letter label with no descriptive name, write
     "an unnamed proposal (listed as item {{letter}} in the source table)" instead
     of presenting the letter as a project name.
5. If the context contains a relevant table (budgets, allocations, project lists),
   reproduce the relevant rows as a markdown table with real figures.
6. ATTENDANCE & COUNT QUERIES — when the user asks "how many meetings did X attend",
   "which meetings did X attend", or "did X attend meeting Y":
   - Scan EVERY chunk in the verified context for the person's name (first name, last name,
     any variant — e.g. "Jhawar" matches "Dr. Manoj Kumar Jhawar").
   - Include every meeting where the person appears in ANY role: Member, Chairperson,
     CMD, CFO, "By Invitation", Invitee, or any other designation. All roles count.
   - List each meeting found with its number and date.
   - After listing, state the total count. If the DOCUMENT DIRECTORY shows more meetings
     in the knowledge base than your chunks cover, add exactly one sentence:
     "Note: Count is based on retrieved document chunks; additional meetings may exist
     in the knowledge base."
7. TEMPLATES — when the question is a broad meeting/project/NGO/timeline lookup,
   structure the answer with these layouts (filling every placeholder with REAL
   values from the context; omit a section if the context has nothing for it):

=== Meeting Summary layout (broad "what happened in meeting N" questions) ===
## CSR Meeting {{N}} — {{meeting date}}
### Key Decisions
- {{real decision}}
### Projects Discussed / Approved
- {{agenda item number}} — {{real project title}} ({{NGO}}) — ₹{{amount}}
### Budget Approved
₹{{real amount}}

=== Project Status layout ===
## Project Status
**Project:** {{real project title}}
**Implementing NGO:** {{NGO}}
**Current Status:** {{status}}
**Approved Budget:** ₹{{amount}}
**Latest Update:** {{update with meeting reference}}

=== NGO layout ===
## NGO Information
**NGO:** {{NGO name}}
### Related Projects
- {{real project title}} — ₹{{amount}}
### Total Allocation
₹{{sum}}

=== Timeline layout ===
## Project Timeline — {{real project title}}
1. {{date / meeting}} — {{lifecycle event}}
2. ...

For any other question, write a direct factual response with inline [Ref X] citations.
"""
        logger.info("executing Pass 1: drafting answer", extra={"intent": intent})
        messages = [{"role": "system", "content": draft_prompt}]
        draft_resp = self.responder.generate(messages)
        draft = draft_resp.text.strip()
        
        # Pass 2: auditing and verification
        verify_prompt = f"""You are an expert CSR Auditor and Fact-Checker.
Your task is to review the DRAFT ANSWER and verify its accuracy, citations, and compliance against the raw VERIFIED CONTEXT.

USER QUERY:
{question}

VERIFIED CONTEXT:
{context_block}

DRAFT ANSWER:
{draft}

CRITICAL AUDITING RULES:
1. **Citation Verification**: Every claim in the DRAFT ANSWER must be supported by the VERIFIED CONTEXT. If the draft makes a claim that is NOT supported, delete it or modify it to be completely factual.
2. **Missing Information**: If the draft tries to guess or answer something not in the context (e.g. implementation status when only proposal is present), replace it with:
   "{{Project Name}} was approved, however no implementation status was found in the available CSR records." (or similar specific missing info sentence).
3. **Contradictions**: If the context contains contradictory numbers or statuses, do NOT pick one. Output:
   "Conflicting information was found across available records." followed by both sources.
4. **Insufficient Evidence**: Use the exact refusal sentence
   "The available records do not contain sufficient evidence to answer this question confidently."
   ONLY when the context contains NOTHING relevant to the question. If the
   context contains the answer — even in a single chunk, even if other chunks
   are silent on it — give that answer with its [Ref] citation. A fact present
   in one relevant chunk is sufficient; do NOT refuse just because most chunks
   don't mention it, or because a date/number appears in only one document.
5. **No Fabrication**: Do not invent facts, names, or numbers.
6. **No Fillers**: Maintain the template structure if the draft uses one.
7. **No Letter Labels**: If the draft lists projects as "Project A", "Project B" etc.,
   replace each with the real project title and NGO found in the VERIFIED CONTEXT
   (these letters are per-meeting table shorthand, not project names). If the real
   title is genuinely absent from the context, summarize what IS known about the
   item (sector, NGO, amount) instead of using the letter.
8. **Answer Relevance**: If the draft answers a more generic question than the one
   asked (e.g. gives a meeting summary when the user asked for the annual action
   plan), rewrite it to address the user's actual question using the context.
9. **Attendance Completeness**: If the question asks how many meetings a person
   attended, scan the VERIFIED CONTEXT one more time. Check for the person's name
   (and surname alone) in EVERY chunk — including chunks where they appear
   "By Invitation", as CMD, CFO, or any other role. Each distinct meeting where
   the name appears must be listed and counted. If the draft's count is lower than
   the number of distinct meetings you can find, correct it.

Output the final, verified, and polished answer. Do not include any meta-commentary, introductory remarks, or explanations. Output the final markdown response directly.
"""
        logger.info("executing Pass 2: fact-check critique")
        verify_messages = [{"role": "system", "content": verify_prompt}]
        final_resp = self.responder.generate(verify_messages)
        
        return final_resp.text.strip()

    def _is_vague_query(self, question: str, plan: dict) -> Tuple[bool, Optional[str]]:
        # Heuristic short-circuits for exact vague queries
        q_clean = question.lower().strip().rstrip('?').strip()
        if q_clean == "status":
            return True, "Please specify a project name, NGO name, or meeting number."
        if q_clean in ("give me details", "details", "give details"):
            return True, "Please specify what information you need:\n- Project\n- NGO\n- Meeting\n- Resolution\n- Budget"
            
        if plan.get("is_invalid"):
            return True, plan.get("invalid_explanation") or "Please specify a project name, NGO name, or meeting number."

        # NOTE: we deliberately do NOT require a specific meeting/project/NGO
        # entity here. Corpus-wide aggregate questions ("list all meetings
        # date wise", "which projects are delayed?") are legitimate and are
        # answered from the document directory + broad retrieval. Unresolvable
        # entities are handled downstream by evidence validation ("No records
        # for X were found..."), which the spec prefers over asking again.
        return False, None

    # ───── Public API ─────
    async def answer(self, chat_id: int, question: str, history: List[Turn]) -> RAGResult:
        # 1. Planner & Extraction
        plan = self._parse_plan(question, history)
        
        # Immediate short-circuit for completely invalid/vague queries
        is_vague, explanation = self._is_vague_query(question, plan)
        if is_vague:
            return RAGResult(
                answer=explanation or "Please specify a project name, NGO name, or meeting number.",
                chunks=[],
                provider=self.settings.llm_provider,
                model=self.settings.openai_chat_model,
                latency_ms=0,
            )
            
        # 2. Embedding creation
        embedding = self.embedder.embed(plan.get("rewritten_query", question))
        
        # 3. Retrieval Strategy Selection
        chunks = self._execute_retrieval(plan, embedding)

        # 3b. Exact structured records (only when the KB is wired — currently
        # kb is None, so this is dormant). We intentionally do NOT inject the
        # extraction "meeting_reference" date as a cross-check: that date is
        # often the agenda-issue date, not the held date, so it manufactured a
        # false conflict and pushed the verifier to "insufficient evidence".
        # The document chunks themselves are the source of truth for dates.
        kb_blocks = ""
        if self.kb is not None:
            try:
                kb_blocks = self.kb.blocks_for_plan(plan, question)
            except Exception:
                logger.exception("kb block fetch failed; continuing vector-only")

        # 4. Context Builder
        context_block = self._build_context(chunks, kb_blocks)
        
        # 5. Dual-pass LLM Generation
        answer_text = self._generate_answer(question, plan, context_block, history)
        
        # Append source references cleanly at the bottom if sources were cited.
        # Raw Pinecone labels are Drive file-ids; humanize_source() maps them to
        # reader-friendly citations like "23rd CSR Committee Minutes (held 22.07.2024)".
        def _label_for(c) -> str:
            if c.source == "csr_project_master":
                return f"Project Master — {self.directory.humanize_source(c.document_name)}"
            label = self.directory.humanize_source(c.document_name or "Unknown Document")
            pg = self.directory.clean_page(c.page)
            if pg:
                label += f" (p.{pg})"
            return label

        sources_list = []
        for c in chunks:
            label = _label_for(c)
            if label not in sources_list:
                sources_list.append(label)

        if sources_list and "sufficient evidence" not in answer_text.lower():
            # Extract citation references used in text
            refs_used = re.findall(r"\[Ref (\d+)\]", answer_text)
            if refs_used:
                # Only show sources that were actually cited in the response text
                indices_used = sorted(list(set(int(idx) for idx in refs_used)))
                citation_lines = []
                for idx in indices_used:
                    if idx <= len(chunks):
                        c = chunks[idx - 1]
                        citation_lines.append(f"• [Ref {idx}] {_label_for(c)}")
                if citation_lines:
                    answer_text += "\n\nSources:\n" + "\n".join(citation_lines)
            else:
                # Fallback to listing top 5 retrieved source documents
                answer_text += "\n\nSources:\n" + "\n".join(f"• {s}" for s in sources_list[:5])

        # Logging results as structured extra payload
        logger.info(
            "middleware_intelligence_run",
            extra={
                "user_query": question,
                "rewritten_query": plan.get("rewritten_query"),
                "detected_intent": plan.get("intent"),
                "extracted_entities": plan.get("entities"),
                "n_chunks_used": len(chunks),
                "sources_retrieved": sources_list[:8],
                "final_response": answer_text[:200] + "..."
            }
        )

        return RAGResult(
            answer=answer_text,
            chunks=chunks,
            provider=self.settings.llm_provider,
            model=self.settings.openai_chat_model if self.settings.llm_provider == "openai" else self.settings.openrouter_model,
            latency_ms=0,
        )

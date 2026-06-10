"""LLM listwise reranker (Enhance Retrieval spec, stage 4).

A cross-encoder (sentence-transformers) would add a ~2 GB torch dependency to
the Docker image, so we rerank with one cheap LLM call instead: the model sees
the query plus numbered chunk previews and returns the indices of the most
relevant chunks in order. On any failure the original order is preserved —
reranking can only be a no-op, never a regression.
"""
from __future__ import annotations

import json
import re
from typing import List

from app.models.message_model import RetrievedChunk
from app.services.response_service import ResponseService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_PREVIEW_CHARS = 400


class LLMReranker:
    def __init__(self, responder: ResponseService):
        self.responder = responder

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
        if len(chunks) <= top_n:
            return chunks

        previews = []
        for i, c in enumerate(chunks, 1):
            text = (c.text or "").strip().replace("\n", " ")[:_PREVIEW_CHARS]
            previews.append(f"[{i}] ({c.document_name or 'unknown'}) {text}")

        prompt = (
            "You are a retrieval reranker for a CSR governance document assistant.\n"
            f"QUERY: {query}\n\n"
            "PASSAGES:\n" + "\n".join(previews) + "\n\n"
            f"Select the {top_n} passages most useful for answering the query, "
            "ordered from most to least relevant. Prefer passages containing exact "
            "entities (meeting numbers, project names, NGO names, amounts) mentioned "
            "in the query.\n"
            'Respond with JSON only, no other text: {"ranking": [passage numbers]}'
        )
        try:
            resp = self.responder.generate([{"role": "system", "content": prompt}])
            text = resp.text.strip()
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise ValueError(f"no JSON object in reranker output: {text[:120]}")
            ranking = json.loads(m.group(0)).get("ranking", [])
            seen = set()
            ordered: List[RetrievedChunk] = []
            for idx in ranking:
                i = int(idx) - 1
                if 0 <= i < len(chunks) and i not in seen:
                    seen.add(i)
                    ordered.append(chunks[i])
                if len(ordered) >= top_n:
                    break
            # Backfill with original order if the model returned too few.
            for i, c in enumerate(chunks):
                if len(ordered) >= top_n:
                    break
                if i not in seen:
                    ordered.append(c)
            logger.info(
                "llm rerank applied",
                extra={"in": len(chunks), "out": len(ordered), "model_ranked": len(seen)},
            )
            return ordered
        except Exception as e:
            logger.warning("rerank failed — keeping original order", extra={"err": str(e)})
            return chunks[:top_n]

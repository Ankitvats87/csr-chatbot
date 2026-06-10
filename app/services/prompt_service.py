from pathlib import Path
from typing import List

from app.models.message_model import RetrievedChunk, Turn

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptService:
    def __init__(self) -> None:
        self.system_prompt = (PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8")
        self.answer_template = (PROMPTS_DIR / "answer_template.txt").read_text(encoding="utf-8")

    def build_messages(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        memory: List[Turn],
        document_directory: str = "",
    ) -> List[dict]:
        # Priority: Document Directory + Retrieved Documents > User Question > Memory.
        # The directory is authoritative for "what documents exist" questions
        # (e.g. "list all meetings"); the retrieved context provides the actual
        # content needed to answer detail questions.
        context_block = self._format_chunks(chunks)
        memory_block = self._format_memory(memory)
        directory_block = document_directory.strip() or "(directory unavailable)"

        system = (
            f"{self.system_prompt}\n\n"
            f"=== DOCUMENT DIRECTORY (authoritative list of every indexed document) ===\n"
            f"{directory_block}\n\n"
            f"=== RETRIEVED CONTEXT (chunks most similar to the user's question) ===\n"
            f"{context_block}\n\n"
            f"=== RECENT CONVERSATION (oldest first) ===\n{memory_block}\n\n"
            f"=== INSTRUCTIONS ===\n{self.answer_template}"
        )

        messages: List[dict] = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _format_chunks(chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "(no relevant context retrieved)"
        out = []
        for i, c in enumerate(chunks, 1):
            header = f"[{i}] {c.document_name or 'unknown'}"
            if c.page:
                header += f" (page {c.page})"
            header += f"  score={c.score:.2f}"
            out.append(f"{header}\n{c.text.strip()}")
        return "\n\n".join(out)

    @staticmethod
    def _format_memory(memory: List[Turn]) -> str:
        if not memory:
            return "(no prior turns)"
        return "\n".join(f"{t.role.upper()}: {t.content}" for t in memory)

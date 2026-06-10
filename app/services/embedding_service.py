from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Wraps OpenAI embeddings. Used by both retrieval and (Phase 2) ingestion."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for embeddings. Set it in .env."
            )
        self._client = OpenAI(api_key=settings.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def embed(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=text,
        )
        return resp.data[0].embedding

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=texts,
        )
        return [d.embedding for d in resp.data]

import time
from typing import Optional

from pinecone import Pinecone, ServerlessSpec

from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# text-embedding-3-small has 1536 dimensions.
EMBEDDING_DIMENSION = 1536


class PineconeClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pc: Optional[Pinecone] = None
        self._index = None

    def connect(self) -> None:
        """Connect with bounded retries. Transient SSL EOFs and 5xx during
        startup should not leave us in a permanently broken state — the index
        is known to exist, so we always optimistically bind to it and let
        per-query retries handle the occasional network blip."""
        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                self._pc = Pinecone(api_key=self.settings.pinecone_api_key)
                try:
                    self._ensure_index()
                except Exception as e:
                    logger.warning(
                        "pinecone _ensure_index failed — binding to index directly",
                        extra={"err": str(e)[:200], "attempt": attempt},
                    )
                self._index = self._pc.Index(self.settings.pinecone_index_name)
                logger.info(
                    "pinecone connected",
                    extra={
                        "index": self.settings.pinecone_index_name,
                        "namespace": self.settings.pinecone_namespace,
                        "attempt": attempt,
                    },
                )
                return
            except Exception as e:
                last_err = e
                logger.warning(
                    "pinecone connect attempt failed",
                    extra={"attempt": attempt, "err": str(e)[:200]},
                )
                time.sleep(2 ** attempt)

        logger.error(f"Failed to connect to Pinecone after 3 attempts: {last_err}")
        self._index = None

    def _ensure_index(self) -> None:
        names = {idx.name for idx in self._pc.list_indexes()}
        if self.settings.pinecone_index_name in names:
            return
        logger.info("creating pinecone index", extra={"index": self.settings.pinecone_index_name})
        self._pc.create_index(
            name=self.settings.pinecone_index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=self.settings.pinecone_cloud,
                region=self.settings.pinecone_region,
            ),
        )

    @property
    def index(self):
        if self._index is None:
            raise RuntimeError("PineconeClient not connected. Call connect() first.")
        return self._index

    def health_ok(self) -> bool:
        try:
            self.index.describe_index_stats()
            return True
        except Exception:
            return False

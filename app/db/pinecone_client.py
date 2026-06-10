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
        try:
            self._pc = Pinecone(api_key=self.settings.pinecone_api_key)
            self._ensure_index()
            self._index = self._pc.Index(self.settings.pinecone_index_name)
            logger.info(
                "pinecone connected",
                extra={
                    "index": self.settings.pinecone_index_name,
                    "namespace": self.settings.pinecone_namespace,
                },
            )
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone on startup: {e}")
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

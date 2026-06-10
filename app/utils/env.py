from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")

    # OpenRouter
    llm_provider: str = Field(default="openrouter", alias="LLM_PROVIDER")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct:free", alias="OPENROUTER_MODEL"
    )
    openrouter_fallback_to_openai: bool = Field(default=True, alias="OPENROUTER_FALLBACK_TO_OPENAI")
    openrouter_referer: str = Field(default="https://srv988340.hstgr.cloud", alias="OPENROUTER_REFERER")
    openrouter_app_title: str = Field(default="Spark63 CSR Bot", alias="OPENROUTER_APP_TITLE")

    # Pinecone
    pinecone_api_key: str = Field(alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(alias="PINECONE_INDEX_NAME")
    pinecone_namespace: str = Field(default="knowledgebase", alias="PINECONE_NAMESPACE")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")

    # Telegram
    telegram_bot_token: str = Field(alias="TELEGRAM_BOT_TOKEN")
    telegram_webhook_secret: str = Field(default="", alias="TELEGRAM_WEBHOOK_SECRET")

    # Access control
    admin_chat_ids: List[int] = Field(default_factory=list, alias="ADMIN_CHAT_IDS")
    allowed_chat_ids: List[int] = Field(default_factory=list, alias="ALLOWED_CHAT_IDS")

    # Google Drive ingestion
    google_drive_client_id: Optional[str] = Field(default=None, alias="GOOGLE_DRIVE_CLIENT_ID")
    google_drive_client_secret: Optional[str] = Field(default=None, alias="GOOGLE_DRIVE_CLIENT_SECRET")
    google_drive_refresh_token: Optional[str] = Field(default=None, alias="GOOGLE_DRIVE_REFRESH_TOKEN")
    google_drive_folder_id: Optional[str] = Field(default=None, alias="GOOGLE_DRIVE_FOLDER_ID")
    ingestion_poll_interval_seconds: int = Field(default=120, alias="INGESTION_POLL_INTERVAL_SECONDS")
    ingestion_max_file_mb: int = Field(default=50, alias="INGESTION_MAX_FILE_MB")
    llama_cloud_api_key: Optional[str] = Field(default=None, alias="LLAMA_CLOUD_API_KEY")

    # Memory + retrieval
    memory_window: int = Field(default=10, alias="MEMORY_WINDOW")
    top_k: int = Field(default=5, alias="TOP_K")
    similarity_threshold: float = Field(default=0.75, alias="SIMILARITY_THRESHOLD")
    chunk_size: int = Field(default=600, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")

    # Retrieval version: "v1" uses knowledgebase namespace; "v2" uses csr_v2_enriched + csr_project_master
    rag_version: str = Field(default="v1", alias="RAG_VERSION")

    # Middleware Intelligence Layer
    enable_intelligence_layer: bool = Field(default=False, alias="ENABLE_INTELLIGENCE_LAYER")

    # Hybrid retrieval (BM25 + vector + Reciprocal Rank Fusion + reranking)
    enable_hybrid_retrieval: bool = Field(default=False, alias="ENABLE_HYBRID_RETRIEVAL")
    enable_reranker: bool = Field(default=False, alias="ENABLE_RERANKER")
    hybrid_bm25_top_k: int = Field(default=25, alias="HYBRID_BM25_TOP_K")
    rerank_candidates: int = Field(default=30, alias="RERANK_CANDIDATES")
    context_max_chunks: int = Field(default=24, alias="CONTEXT_MAX_CHUNKS")
    chunk_store_path: str = Field(default="data/v2_chunk_store.json", alias="CHUNK_STORE_PATH")
    llm_max_tokens: int = Field(default=1600, alias="LLM_MAX_TOKENS")
    # Model used ONLY for final answer synthesis (planner/extraction stay on the
    # default chat model). Empty = use the default chat model.
    answer_model: str = Field(default="", alias="ANSWER_MODEL")

    # App
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    app_port: int = Field(default=8000, alias="APP_PORT")
    sqlite_path: str = Field(default="/app/data/sqlite/app.db", alias="SQLITE_PATH")

    # Admin Web Dashboard
    admin_username: str = Field(default="admin", alias="ADMIN_USERNAME")
    admin_password: str = Field(default="admin", alias="ADMIN_PASSWORD")

    # Deployment
    public_domain: str = Field(default="srv988340.hstgr.cloud", alias="PUBLIC_DOMAIN")
    webhook_path: str = Field(default="/webhook/telegram", alias="WEBHOOK_PATH")
    acme_email: Optional[str] = Field(default=None, alias="ACME_EMAIL")

    @field_validator("admin_chat_ids", "allowed_chat_ids", mode="before")
    @classmethod
    def _split_csv_ids(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, list):
            return v
        return [int(x.strip()) for x in str(v).split(",") if x.strip() and x.strip() != "REPLACE_ME"]

    @field_validator("openrouter_fallback_to_openai", mode="before")
    @classmethod
    def _parse_bool(cls, v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    @field_validator("telegram_webhook_secret", mode="after")
    @classmethod
    def _require_strong_secret(cls, v: str) -> str:
        # Empty secret would make /webhook/telegram accept anonymous traffic.
        # We allow short secrets for local dev but refuse blank.
        if not v or not v.strip():
            raise ValueError(
                "TELEGRAM_WEBHOOK_SECRET must be set in .env. "
                "Run scripts/setup_webhook.sh to auto-generate one."
            )
        if len(v) < 6:
            raise ValueError("TELEGRAM_WEBHOOK_SECRET must be at least 6 characters.")
        return v

    def webhook_url(self) -> str:
        return f"https://{self.public_domain}{self.webhook_path}"

    def drive_configured(self) -> bool:
        return all(
            [
                self.google_drive_client_id,
                self.google_drive_client_secret,
                self.google_drive_refresh_token,
                self.google_drive_folder_id,
            ]
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()

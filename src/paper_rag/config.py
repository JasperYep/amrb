from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    workspace_root: Path = PROJECT_ROOT
    embedding_provider: Literal["openai"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_API_KEY", "OPENAI_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    hindsight_api_llm_api_key: SecretStr | None = None
    embedding_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_BASE_URL", "OPENAI_BASE_URL"),
    )
    qdrant_path: Path = Path(".state/qdrant")
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "agent_memory_papers"
    qdrant_dashboard_url: str | None = None
    rag_top_k: int = Field(default=8, ge=1, le=100)
    rag_chunk_size: int = Field(default=700, ge=1)
    rag_chunk_overlap: int = Field(default=100, ge=0)
    hindsight_api_llm_provider: str = "openai"
    hindsight_api_llm_model: str = "gpt-4o-mini"
    hindsight_api_llm_base_url: str | None = None
    hindsight_ui_url: str | None = None

    @field_validator(
        "qdrant_url",
        "qdrant_api_key",
        "qdrant_dashboard_url",
        "embedding_api_key",
        "openai_api_key",
        "hindsight_api_llm_api_key",
        "hindsight_api_llm_base_url",
        "embedding_base_url",
        "hindsight_ui_url",
        mode="before",
    )
    @classmethod
    def empty_optional_value_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def papers_dir(self) -> Path:
        return self.workspace_root / "papers"

    @property
    def state_dir(self) -> Path:
        return self.workspace_root / ".state"

    @property
    def manifest_path(self) -> Path:
        return self.state_dir / "index.json"

    @property
    def resolved_qdrant_path(self) -> Path:
        if self.qdrant_path.is_absolute():
            return self.qdrant_path
        return self.workspace_root / self.qdrant_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def create_embedding_model(settings: Settings | None = None) -> BaseEmbedding:
    from llama_index.embeddings.openai import OpenAIEmbedding

    current = settings or get_settings()
    if current.embedding_provider != "openai":
        raise ValueError(f"Unsupported embedding provider: {current.embedding_provider}")
    if not current.embedding_api_key or not current.embedding_api_key.get_secret_value():
        raise ValueError(
            "缺少 EMBEDDING_API_KEY（也可使用旧变量 OPENAI_API_KEY）。\n\n"
            "请复制 .env.example 为 .env，并配置 embedding provider。"
        )
    return OpenAIEmbedding(
        model_name=current.embedding_model,
        embed_batch_size=20,
        api_key=current.embedding_api_key.get_secret_value(),
        api_base=current.embedding_base_url,
        max_retries=3,
    )


def create_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    current = settings or get_settings()
    if current.qdrant_url:
        api_key = (
            current.qdrant_api_key.get_secret_value()
            if current.qdrant_api_key
            else None
        )
        return QdrantClient(url=current.qdrant_url, api_key=api_key)
    current.resolved_qdrant_path.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(current.resolved_qdrant_path))


def ensure_paper_id_payload_index(
    client: QdrantClient,
    settings: Settings | None = None,
) -> None:
    current = settings or get_settings()
    collection = current.qdrant_collection
    if not client.collection_exists(collection):
        return

    payload_schema = client.get_collection(collection).payload_schema or {}
    existing = payload_schema.get("paper_id")
    if existing is not None:
        existing_type = getattr(existing, "data_type", existing)
        if existing_type != PayloadSchemaType.KEYWORD:
            raise ValueError(
                "Qdrant payload index for 'paper_id' must use the keyword type; "
                f"found {existing_type!s}."
            )
        return

    client.create_payload_index(
        collection_name=collection,
        field_name="paper_id",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )


def create_vector_store(
    client: QdrantClient,
    settings: Settings | None = None,
) -> QdrantVectorStore:
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    current = settings or get_settings()
    return QdrantVectorStore(
        client=client,
        collection_name=current.qdrant_collection,
    )

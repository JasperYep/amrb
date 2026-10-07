from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient

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
    )

    workspace_root: Path = PROJECT_ROOT
    embedding_provider: Literal["openai"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: SecretStr | None = None
    qdrant_path: Path = Path(".state/qdrant")
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "agent_memory_papers"
    rag_top_k: int = Field(default=8, ge=1, le=100)
    rag_chunk_size: int = Field(default=700, ge=1)
    rag_chunk_overlap: int = Field(default=100, ge=0)

    @field_validator("qdrant_url", "qdrant_api_key", "openai_api_key", mode="before")
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
    if not current.openai_api_key or not current.openai_api_key.get_secret_value():
        raise ValueError(
            "缺少 OPENAI_API_KEY。\n\n"
            "请复制 .env.example 为 .env，并配置 embedding provider。"
        )
    return OpenAIEmbedding(
        model=current.embedding_model,
        api_key=current.openai_api_key.get_secret_value(),
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

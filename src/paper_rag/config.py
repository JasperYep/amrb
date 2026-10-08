from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import AliasChoices, Field, SecretStr, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Deployment values use process environment > workspace TOML/init > .env.
        # load_settings then locks persisted workspace identity and index-profile values.
        return (env_settings, init_settings, dotenv_settings, file_secret_settings)

    workspace_root: Path = PROJECT_ROOT
    workspace_bound_path: Path | None = None
    workspace_id: str = "uninitialized-workspace"
    workspace_name: str = "Research workspace"
    discipline: str = ""
    research_focus: str = ""
    papers_directory: Path = Path("papers")
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
    qdrant_collection: str = "amrb_papers"
    qdrant_resource_accepted: bool = False
    hindsight_bank_id: str = "amrb-research"
    hindsight_resource_accepted: bool = False
    legacy_data_compatibility: bool = False
    embedding_dimension: int | None = None
    legacy_index_accepted: bool = False
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
        if self.papers_directory.is_absolute():
            return self.papers_directory
        return self.workspace_root / self.papers_directory

    @property
    def state_dir(self) -> Path:
        return self.workspace_root / ".state"

    @property
    def embedding_batches_dir(self) -> Path:
        return self.state_dir / "embedding_batches"

    @property
    def manifest_path(self) -> Path:
        return self.state_dir / "index.json"

    @property
    def resolved_qdrant_path(self) -> Path:
        if self.qdrant_path.is_absolute():
            return self.qdrant_path
        return self.workspace_root / self.qdrant_path


def find_workspace_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "amrb.toml").is_file():
            return candidate
    raise FileNotFoundError(
        f"No amrb.toml found from {current}. Initialize this project with `amrb setup --workspace {current}`."
    )


def _settings_from_toml(path: Path) -> dict[str, object]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Could not read workspace configuration: {path}") from exc

    version = data.get("config_version", 1)
    if version != 1:
        raise ValueError(f"Unsupported amrb.toml config_version: {version!r}")
    workspace = data.get("workspace", {})
    papers = data.get("papers", {})
    index = data.get("index", {})
    memory = data.get("memory", {})
    if not all(isinstance(section, dict) for section in (workspace, papers, index, memory)):
        raise ValueError("Workspace configuration sections must be TOML tables.")
    workspace_id = str(workspace.get("id", "")).strip()
    if not workspace_id:
        raise ValueError("amrb.toml must contain a stable [workspace].id.")

    values: dict[str, object] = {}
    for section, key_map in (
        (workspace, {
            "id": "workspace_id",
            "bound_path": "workspace_bound_path",
            "name": "workspace_name",
            "discipline": "discipline",
            "research_focus": "research_focus",
            "legacy_data_compatibility": "legacy_data_compatibility",
        }),
        (papers, {"directory": "papers_directory"}),
        (index, {
            "embedding_provider": "embedding_provider",
            "embedding_model": "embedding_model",
            "qdrant_path": "qdrant_path",
            "collection": "qdrant_collection",
            "chunk_size": "rag_chunk_size",
            "chunk_overlap": "rag_chunk_overlap",
            "embedding_dimension": "embedding_dimension",
            "legacy_index_accepted": "legacy_index_accepted",
            "resource_accepted": "qdrant_resource_accepted",
        }),
        (memory, {
            "bank_id": "hindsight_bank_id",
            "resource_accepted": "hindsight_resource_accepted",
        }),
    ):
        for source_key, target_key in key_map.items():
            if source_key in section:
                values[target_key] = section[source_key]
    collection, bank_id = derived_resource_ids(workspace_id)
    values.setdefault("workspace_id", workspace_id)
    values.setdefault("qdrant_collection", collection)
    values.setdefault("hindsight_bank_id", bank_id)
    return values


def copied_workspace_origin(workspace_root: Path) -> Path | None:
    root = workspace_root.expanduser().resolve()
    config_path = root / "amrb.toml"
    if not config_path.is_file():
        return None
    values = _settings_from_toml(config_path)
    bound_path = values.get("workspace_bound_path")
    if not bound_path:
        return None
    origin = Path(str(bound_path)).expanduser().resolve()
    return None if origin == root else origin


def load_settings(workspace_root: Path, *, allow_copied_workspace: bool = False) -> Settings:
    root = workspace_root.expanduser().resolve()
    config_path = root / "amrb.toml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Workspace is not initialized: {config_path}. Run `amrb setup --workspace {root}`."
        )
    values = _settings_from_toml(config_path)
    bound_path = values.get("workspace_bound_path")
    if bound_path:
        origin = Path(str(bound_path)).expanduser().resolve()
        if origin != root and not allow_copied_workspace:
            raise ValueError(
                f"This workspace config is bound to {origin} but is located at {root}. "
                "It may have been copied. Run `amrb setup` and explicitly choose "
                "--copied-identity reuse or --copied-identity new before connecting it."
            )
    values["workspace_root"] = root
    settings = Settings(
        _env_file=root / ".env",
        _env_file_encoding="utf-8",
        **values,
    )
    locked_fields = (
        "workspace_root",
        "workspace_bound_path",
        "workspace_id",
        "workspace_name",
        "discipline",
        "research_focus",
        "papers_directory",
        "qdrant_path",
        "qdrant_collection",
        "qdrant_resource_accepted",
        "hindsight_bank_id",
        "hindsight_resource_accepted",
        "legacy_data_compatibility",
    )
    locked_values = {
        field: TypeAdapter(Settings.model_fields[field].annotation).validate_python(
            root if field == "workspace_root" else values.get(field, Settings.model_fields[field].default)
        )
        for field in locked_fields
    }
    if not locked_values["qdrant_path"].is_absolute():
        locked_values["qdrant_path"] = (root / locked_values["qdrant_path"]).resolve()
    for field in (
        "embedding_provider",
        "embedding_model",
        "rag_chunk_size",
        "rag_chunk_overlap",
        "embedding_dimension",
        "legacy_index_accepted",
    ):
        if field in values:
            locked_values[field] = TypeAdapter(Settings.model_fields[field].annotation).validate_python(
                values[field]
            )
    return settings.model_copy(update=locked_values)


def initialize_workspace_config(
    workspace_root: Path,
    *,
    name: str,
    discipline: str = "",
    research_focus: str = "",
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    adopt_legacy: bool = False,
) -> tuple[Settings, bool]:
    root = workspace_root.expanduser().resolve()
    config_path = root / "amrb.toml"
    if config_path.exists():
        return load_settings(root), False

    root.mkdir(parents=True, exist_ok=True)
    has_legacy_data = (
        (root / ".env").is_file()
        or (root / ".state" / "index.json").is_file()
        or any((root / "papers").glob("*.pdf"))
    )
    if has_legacy_data and not adopt_legacy:
        raise ValueError(
            "This directory already contains AMRB data. Explicitly choose whether to adopt its existing "
            "collection, bank, papers, and local paths before initializing it."
        )

    legacy = Settings(
        _env_file=root / ".env",
        _env_file_encoding="utf-8",
        workspace_root=root,
    ) if adopt_legacy else Settings(_env_file=None, workspace_root=root)
    workspace_id = str(uuid4())
    collection, bank_id = derived_resource_ids(workspace_id)
    if adopt_legacy:
        legacy_env_text = (root / ".env").read_text(encoding="utf-8") if (root / ".env").exists() else ""
        has_collection_override = "QDRANT_COLLECTION" in os.environ or any(
            line.lstrip().startswith("QDRANT_COLLECTION=") for line in legacy_env_text.splitlines()
        )
        has_bank_override = "HINDSIGHT_BANK_ID" in os.environ or any(
            line.lstrip().startswith("HINDSIGHT_BANK_ID=") for line in legacy_env_text.splitlines()
        )
        collection = legacy.qdrant_collection if has_collection_override else "agent_memory_papers"
        bank_id = legacy.hindsight_bank_id if has_bank_override else "agent-memory-research"

    def quote(value: str) -> str:
        import json

        return json.dumps(value, ensure_ascii=False)

    lines = [
        "config_version = 1",
        "",
        "[workspace]",
        f"id = {quote(workspace_id)}",
        f"bound_path = {quote(str(root))}",
        f"name = {quote(name.strip())}",
        f"discipline = {quote(discipline.strip())}",
        f"research_focus = {quote(research_focus.strip())}",
    ]
    if adopt_legacy:
        lines.append("legacy_data_compatibility = true")
    lines.extend([
        "",
        "[papers]",
        'directory = "papers"',
        "",
        "[index]",
        f"embedding_provider = {quote(embedding_provider or legacy.embedding_provider)}",
        f"embedding_model = {quote(embedding_model or legacy.embedding_model)}",
        f"qdrant_path = {quote(str(legacy.qdrant_path))}",
        f"collection = {quote(collection)}",
        f"chunk_size = {chunk_size if chunk_size is not None else legacy.rag_chunk_size}",
        f"chunk_overlap = {chunk_overlap if chunk_overlap is not None else legacy.rag_chunk_overlap}",
        "legacy_index_accepted = false",
        "",
        "[memory]",
        f"bank_id = {quote(bank_id)}",
        "",
    ])
    content = "\n".join(lines)
    root.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root,
            prefix=".amrb.toml.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_path, config_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return load_settings(root), True


def update_workspace_config(
    workspace_root: Path,
    section: str,
    values: dict[str, str | int | float | bool],
) -> None:
    path = workspace_root.expanduser().resolve() / "amrb.toml"
    original = path.read_text(encoding="utf-8")
    tomllib.loads(original)
    lines = original.splitlines()
    header = f"[{section}]"
    start = next((i for i, line in enumerate(lines) if line.strip() == header), None)
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(header)
        start = len(lines) - 1
        end = len(lines)
    else:
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
            len(lines),
        )
    import json
    import re

    for key, value in values.items():
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, str) else str(value).lower() if isinstance(value, bool) else str(value)
        assignment = re.compile(rf"^\s*{re.escape(key)}\s*=")
        existing = next((i for i in range(start + 1, end) if assignment.match(lines[i])), None)
        if existing is not None:
            lines[existing] = f"{key} = {rendered}"
        else:
            lines.insert(end, f"{key} = {rendered}")
            end += 1
    content = "\n".join(lines).rstrip() + "\n"
    tomllib.loads(content)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def get_settings(workspace_root: Path | None = None) -> Settings:
    """Load settings for the selected workspace without consulting the install directory."""
    if workspace_root is not None:
        root = workspace_root
    elif configured_root := os.environ.get("AMRB_WORKSPACE"):
        root = Path(configured_root)
    else:
        root = find_workspace_root()
    return load_settings(root)


def derived_resource_ids(workspace_id: str) -> tuple[str, str]:
    """Return stable default collection and bank names for a persisted workspace ID."""
    import hashlib

    suffix = hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:12]
    return f"amrb_papers_{suffix}", f"amrb-{suffix}"


def stable_paper_id(workspace_id: str, relative_path: str) -> str:
    normalized = Path(relative_path).as_posix()
    return str(uuid5(NAMESPACE_URL, f"amrb:{workspace_id}:{normalized}"))


def _safe_endpoint(value: str | None) -> str:
    if not value:
        return "not configured"
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return "configured (value redacted)"
    authority = parsed.hostname
    if parsed.port:
        authority += f":{parsed.port}"
    return f"{parsed.scheme}://{authority}"


def resolved_configuration(settings: Settings) -> dict[str, str]:
    config_path = settings.workspace_root / "amrb.toml"
    has_config = config_path.is_file()
    values = _settings_from_toml(config_path) if has_config else {}
    env_path = settings.workspace_root / ".env"
    dotenv_keys = set()
    if env_path.is_file():
        import re

        dotenv_keys = {
            match.group(1)
            for line in env_path.read_text(encoding="utf-8").splitlines()
            if (match := re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line))
        }

    def source(env_names: tuple[str, ...], toml_field: str | None = None) -> str:
        if any(name in os.environ for name in env_names):
            return "process environment"
        if toml_field and toml_field in values:
            return "amrb.toml"
        if any(name in dotenv_keys for name in env_names):
            return "workspace .env"
        return "default"

    def profile_source(field: str, env_names: tuple[str, ...]) -> str:
        if field in values:
            return "amrb.toml"
        return source(env_names)

    remote_qdrant = bool(settings.qdrant_url)
    from paper_rag.ui import probe_web_ui

    hindsight_ui_status = (
        probe_web_ui(settings.hindsight_ui_url)
        if settings.hindsight_ui_url
        else "not configured (unavailable)"
    )
    qdrant_ui_status = (
        probe_web_ui(settings.qdrant_dashboard_url)
        if settings.qdrant_dashboard_url
        else "not configured (unavailable)"
    ) if remote_qdrant else "not applicable to local embedded Qdrant"
    return {
        "workspace_root": str(settings.workspace_root),
        "workspace_id": settings.workspace_id,
        "workspace_name": settings.workspace_name,
        "workspace_source": "amrb.toml" if "workspace_name" in values else "default",
        "papers_directory": str(settings.papers_dir.resolve()),
        "papers_directory_source": "amrb.toml" if "papers_directory" in values else "default",
        "qdrant_backend": "remote" if remote_qdrant else "local embedded",
        "qdrant_target": _safe_endpoint(settings.qdrant_url) if remote_qdrant else str(settings.resolved_qdrant_path),
        "qdrant_target_source": source(("QDRANT_URL",), "qdrant_path"),
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_collection_source": (
            "amrb.toml" if "qdrant_collection" in values else "derived from workspace ID"
        ),
        "embedding_provider": settings.embedding_provider,
        "embedding_provider_source": profile_source("embedding_provider", ("EMBEDDING_PROVIDER",)),
        "embedding_model": settings.embedding_model,
        "embedding_model_source": profile_source("embedding_model", ("EMBEDDING_MODEL",)),
        "embedding_endpoint": _safe_endpoint(settings.embedding_base_url),
        "embedding_endpoint_source": source(("EMBEDDING_BASE_URL", "OPENAI_BASE_URL")),
        "embedding_api_key_source": source(("EMBEDDING_API_KEY", "OPENAI_API_KEY")),
        "embedding_dimension": str(settings.embedding_dimension or "not set in workspace config"),
        "chunk_size": str(settings.rag_chunk_size),
        "chunk_overlap": str(settings.rag_chunk_overlap),
        "chunk_profile_source": (
            "amrb.toml"
            if "rag_chunk_size" in values or "rag_chunk_overlap" in values
            else source(("RAG_CHUNK_SIZE", "RAG_CHUNK_OVERLAP"))
        ),
        "hindsight_bank": settings.hindsight_bank_id,
        "hindsight_bank_source": (
            "amrb.toml" if "hindsight_bank_id" in values else "derived from workspace ID"
        ),
        "hindsight_service": "shared local service at 127.0.0.1:8888",
        "hindsight_ui": _safe_endpoint(settings.hindsight_ui_url),
        "hindsight_ui_status": hindsight_ui_status,
        "qdrant_ui": _safe_endpoint(settings.qdrant_dashboard_url) if remote_qdrant else "not applicable to local embedded Qdrant",
        "qdrant_ui_status": qdrant_ui_status,
    }


def redact_sensitive_text(text: str, settings: Settings) -> str:
    sensitive_values = [
        settings.embedding_api_key.get_secret_value() if settings.embedding_api_key else "",
        settings.openai_api_key.get_secret_value() if settings.openai_api_key else "",
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else "",
        settings.hindsight_api_llm_api_key.get_secret_value() if settings.hindsight_api_llm_api_key else "",
        settings.embedding_base_url or "",
        settings.qdrant_url or "",
        settings.hindsight_api_llm_base_url or "",
        settings.hindsight_ui_url or "",
        settings.qdrant_dashboard_url or "",
    ]
    for value in sorted((value for value in sensitive_values if value), key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return text


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


def inspect_qdrant_collection_identity(
    client: QdrantClient,
    settings: Settings,
) -> tuple[bool, int, str | None]:
    collection_name = settings.qdrant_collection
    if not client.collection_exists(collection_name):
        return False, 0, None
    collection = client.get_collection(collection_name)
    count = int(client.count(collection_name, exact=True).count)
    metadata = getattr(collection.config, "metadata", None) or {}
    owner = metadata.get("amrb_workspace_id") if isinstance(metadata, dict) else None
    return True, count, str(owner) if owner else None


def verify_qdrant_collection_identity(
    client: QdrantClient,
    settings: Settings,
) -> tuple[bool, int, str | None]:
    exists, count, owner = inspect_qdrant_collection_identity(client, settings)
    if owner and owner != settings.workspace_id:
        raise ValueError(
            f"Qdrant collection '{settings.qdrant_collection}' belongs to workspace {owner}, "
            f"not {settings.workspace_id}."
        )
    if exists and count and not owner and not (
        settings.qdrant_resource_accepted
        or settings.legacy_index_accepted
        or settings.legacy_data_compatibility
    ):
        raise ValueError(
            f"Qdrant collection '{settings.qdrant_collection}' has populated vectors but unknown ownership. "
            "Run `amrb setup --adopt-existing-resources` after reviewing its target."
        )
    return exists, count, owner


def claim_qdrant_collection(
    client: QdrantClient,
    settings: Settings,
) -> tuple[bool, int, str | None]:
    """Verify or record collection ownership, blocking unknown populated collections."""
    exists, count, owner = verify_qdrant_collection_identity(client, settings)
    if not exists:
        return exists, count, owner
    if owner and owner != settings.workspace_id:
        raise ValueError(
            f"Qdrant collection '{settings.qdrant_collection}' belongs to workspace {owner}, "
            f"not {settings.workspace_id}. Choose a different collection or resolve the copied workspace identity."
        )
    metadata = getattr(client.get_collection(settings.qdrant_collection).config, "metadata", None) or {}
    if not owner:
        metadata = dict(metadata)
        metadata["amrb_workspace_id"] = settings.workspace_id
        client.update_collection(settings.qdrant_collection, metadata=metadata)
    return exists, count, owner


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

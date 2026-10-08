from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp import Client

from paper_rag.config import (
    Settings,
    create_embedding_model,
    create_qdrant_client,
    get_settings,
    redact_sensitive_text,
    resolved_configuration,
    verify_qdrant_collection_identity,
)
from paper_rag.ingest import load_manifest
from paper_rag.models import ComponentHealth, DoctorReport, SystemStatus
from paper_rag.service import hindsight_base_url, hindsight_healthcheck
from paper_rag.hindsight_setup import inspect_hindsight_bank

def _health(status: str, message: str | None = None) -> ComponentHealth:
    return ComponentHealth(status=status, message=message)


def _redact_secrets(message: str, settings: Settings) -> str:
    return redact_sensitive_text(message, settings)


def _has_embedding_config(settings: Settings) -> bool:
    return bool(
        settings.embedding_api_key
        and settings.embedding_api_key.get_secret_value()
        and settings.embedding_base_url
        and settings.embedding_model.strip()
    )


def _manifest_counts(settings: Settings) -> tuple[dict[str, dict[str, Any]], int, bool]:
    manifest = load_manifest(settings.manifest_path)
    chunks = sum(int(entry.get("chunk_count", 0)) for entry in manifest.values())
    papers_root = settings.papers_dir.resolve()
    missing = any(
        not (settings.workspace_root / relative_path).resolve().is_relative_to(papers_root)
        or not (settings.workspace_root / relative_path).is_file()
        for relative_path in manifest
    )
    return manifest, chunks, missing


def _qdrant_client_if_present(settings: Settings):
    if not settings.qdrant_url and not settings.resolved_qdrant_path.exists():
        return None
    return create_qdrant_client(settings)


def get_system_status(settings: Settings | None = None) -> SystemStatus:
    current = settings or get_settings()
    manifest, _, missing_files = _manifest_counts(current)
    unverified_index = any(
        entry.get("paper_id") and not isinstance(entry.get("index_config"), dict)
        for entry in manifest.values()
    ) and not current.legacy_index_accepted
    qdrant = "ready"
    chunks = 0
    try:
        client = _qdrant_client_if_present(current)
        if client is None:
            if manifest:
                qdrant = "unavailable"
        else:
            try:
                if client.collection_exists(current.qdrant_collection):
                    verify_qdrant_collection_identity(client, current)
                    chunks = client.count(current.qdrant_collection, exact=True).count
            finally:
                client.close()
    except Exception:
        qdrant = "unavailable"

    hindsight = "ready" if hindsight_healthcheck(current) else "unavailable"
    paper_rag = "ready"
    if qdrant != "ready" or not _has_embedding_config(current) or missing_files or unverified_index:
        paper_rag = "degraded"
    if manifest and chunks == 0:
        paper_rag = "degraded"

    return SystemStatus(
        paper_rag=paper_rag,
        hindsight=hindsight,
        qdrant=qdrant,
        papers=len(manifest),
        chunks=chunks,
        workspace_id=current.workspace_id,
        workspace_name=current.workspace_name,
        memory_bank=current.hindsight_bank_id,
        collection=current.qdrant_collection,
        configuration=resolved_configuration(current),
    )


def _check_environment(settings: Settings) -> ComponentHealth:
    if not _has_embedding_config(settings):
        return _health(
            "unavailable",
            "请配置 EMBEDDING_API_KEY、EMBEDDING_BASE_URL 和 EMBEDDING_MODEL（可来自进程环境或 workspace .env）。",
        )
    return _health("ready")


def _check_embedding(settings: Settings, verbose: bool) -> tuple[ComponentHealth, str | None]:
    if not _has_embedding_config(settings):
        return _health("unavailable", "Embedding 配置不完整。"), None
    try:
        model = create_embedding_model(settings)
        vector = model.get_text_embedding("AMRB embedding connectivity check")
        if not vector:
            return _health("unavailable", "Embedding 服务返回空向量。"), None
        return _health("ready"), None
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code in {401, 403}:
            message = "Embedding 服务拒绝了当前 Key。"
        elif status_code == 404:
            message = "Embedding endpoint 或 model 不存在。"
        else:
            message = "无法连接 Embedding 服务。"
        details = _redact_secrets(str(exc), settings) if verbose else None
        return _health("unavailable", message), details


def _check_qdrant(settings: Settings) -> tuple[ComponentHealth, int, str | None]:
    manifest, expected_chunks, _ = _manifest_counts(settings)
    try:
        client = _qdrant_client_if_present(settings)
        if client is None:
            if manifest:
                return (
                    _health("degraded", "索引清单有论文，但本地 Qdrant 数据库不存在。"),
                    0,
                    None,
                )
            return _health("ready"), 0, None
        try:
            if not client.collection_exists(settings.qdrant_collection):
                if manifest:
                    return (
                        _health("degraded", "索引清单有论文，但 Qdrant collection 不存在。"),
                        0,
                        None,
                    )
                return _health("ready"), 0, None
            collection = client.get_collection(settings.qdrant_collection)
            count = client.count(settings.qdrant_collection, exact=True).count
            try:
                verify_qdrant_collection_identity(client, settings)
            except ValueError as exc:
                return _health("degraded", str(exc)), count, None
            schema = collection.payload_schema or {}
            if "paper_id" in schema:
                schema_type = getattr(schema["paper_id"], "data_type", schema["paper_id"])
                if str(getattr(schema_type, "value", schema_type)).casefold() != "keyword":
                    return _health("degraded", "paper_id payload index 类型不是 keyword。"), count, None
            elif count and settings.qdrant_url:
                return _health("degraded", "缺少 paper_id keyword payload index。"), count, None
            if manifest and count == 0:
                return _health("degraded", "索引清单有论文，但 collection 中没有 chunks。"), count, None
            if expected_chunks and count != expected_chunks:
                return _health("degraded", "manifest 与 Qdrant chunk 数量不一致。"), count, None
            return _health("ready"), count, None
        finally:
            client.close()
    except Exception as exc:
        return _health("unavailable", "无法连接 Qdrant。"), 0, _redact_secrets(str(exc), settings)


def _check_paper_index(settings: Settings, chunks: int, qdrant_status: str) -> ComponentHealth:
    manifest, expected_chunks, missing_files = _manifest_counts(settings)
    if missing_files:
        return _health("degraded", "索引清单引用的 PDF 文件缺失或路径无效。")
    if any(
        entry.get("paper_id") and not isinstance(entry.get("index_config"), dict)
        for entry in manifest.values()
    ) and not settings.legacy_index_accepted:
        return _health(
            "degraded",
            "旧索引未记录 embedding 模型、维度和切分配置；运行 `amrb setup` 并显式确认后再按已验证索引使用。",
        )
    if manifest and (qdrant_status != "ready" or chunks != expected_chunks):
        return _health("degraded", "论文索引与 Qdrant 状态不一致。")
    if not manifest:
        return _health("ready", "论文库为空。")
    return _health("ready")


async def _bank_exists(settings: Settings) -> bool:
    url = f"{hindsight_base_url(settings)}/mcp/"
    async with Client(url, read_timeout_seconds=10) as client:
        result = await client.call_tool("get_bank", {"bank_id": settings.hindsight_bank_id})
        if result.is_error:
            return False
        for item in result.content:
            if item.type != "text":
                continue
            try:
                response = json.loads(item.text)
            except json.JSONDecodeError:
                continue
            return response.get("bank_id") == settings.hindsight_bank_id
    return False


def _check_bank(settings: Settings) -> ComponentHealth:
    if not hindsight_healthcheck(settings):
        return _health("unavailable", "Hindsight API 未就绪。")
    try:
        if asyncio.run(_bank_exists(settings)):
            if not settings.hindsight_resource_accepted and not settings.legacy_data_compatibility:
                return _health(
                    "degraded",
                    "Hindsight bank exists but workspace ownership is unconfirmed. "
                    "Review it with `amrb setup --adopt-existing-resources`.",
                )
            return _health("ready")
        return _health("unavailable", f"找不到 Hindsight bank：{settings.hindsight_bank_id}。")
    except Exception as exc:
        return _health("unavailable", "无法读取 Hindsight memory bank。")


def run_doctor(
    settings: Settings | None = None,
    verbose: bool = False,
    verify_embedding: bool = True,
) -> tuple[DoctorReport, dict[str, str]]:
    current = settings or get_settings()
    environment = _check_environment(current)
    if verify_embedding:
        embedding, embedding_details = _check_embedding(current, verbose)
    elif _has_embedding_config(current):
        embedding, embedding_details = (
            _health("unverified", "Network check skipped; pass --verify-embedding to send one minimal request."),
            None,
        )
    else:
        embedding, embedding_details = _health("unavailable", "Embedding configuration is incomplete."), None
    qdrant, chunks, qdrant_details = _check_qdrant(current)
    index = _check_paper_index(current, chunks, qdrant.status)
    hindsight = (
        _health("ready")
        if hindsight_healthcheck(current)
        else _health("unavailable", "Hindsight API 未就绪。")
    )
    bank = _check_bank(current)
    papers, _, _ = _manifest_counts(current)
    report = DoctorReport(
        environment=environment,
        embedding=embedding,
        qdrant=qdrant,
        paper_index=index,
        hindsight=hindsight,
        memory_bank=bank,
        papers=len(papers),
        chunks=chunks,
        workspace_id=current.workspace_id,
        workspace_name=current.workspace_name,
        memory_bank_id=current.hindsight_bank_id,
        collection=current.qdrant_collection,
        configuration=resolved_configuration(current),
    )
    details = {
        key: value
        for key, value in {
            "embedding": embedding_details,
            "qdrant": qdrant_details,
        }.items()
        if value
    }
    return report, details

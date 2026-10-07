from __future__ import annotations

import asyncio
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from paper_rag.config import Settings, create_qdrant_client, ensure_paper_id_payload_index, get_settings
from paper_rag.hindsight_setup import configure_hindsight_bank
from paper_rag.health import (
    _check_bank,
    _check_qdrant,
    _has_embedding_config,
)
from paper_rag.service import (
    HINDSIGHT_API_PORT,
    hindsight_healthcheck,
    install_user_service,
    manual_hindsight_command,
    systemd_user_available,
)


@dataclass(frozen=True)
class SetupResult:
    python: str
    uv: str
    environment: str
    qdrant: str
    hindsight: str
    memory_bank: str
    notes: tuple[str, ...] = ()


def _wait_for_hindsight(timeout: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if hindsight_healthcheck():
            return True
        time.sleep(0.25)
    return False


def setup_workspace(settings: Settings | None = None) -> SetupResult:
    current = settings or get_settings()
    python_ready = sys.version_info >= (3, 12)
    if not python_ready:
        raise RuntimeError("AMRB requires Python 3.12 or newer.")
    uv_ready = shutil.which("uv") is not None
    if not uv_ready:
        raise RuntimeError("uv was not found in PATH. Install uv and rerun `uv run amrb setup`.")

    env_path = current.workspace_root / ".env"
    if not env_path.is_file():
        raise RuntimeError("缺少 .env。请先从 .env.example 复制并填写 Embedding 配置。")
    if not _has_embedding_config(current):
        raise RuntimeError(
            "Embedding 配置不完整。请检查 EMBEDDING_API_KEY、EMBEDDING_BASE_URL 和 EMBEDDING_MODEL。"
        )
    current.papers_dir.mkdir(parents=True, exist_ok=True)

    qdrant_health, _, qdrant_details = _check_qdrant(current)
    if qdrant_health.status == "unavailable":
        raise RuntimeError(qdrant_health.message or qdrant_details or "Qdrant is unavailable.")
    qdrant_client = create_qdrant_client(current)
    try:
        ensure_paper_id_payload_index(qdrant_client, current)
    finally:
        qdrant_client.close()

    notes: list[str] = []
    if systemd_user_available():
        install_user_service(current)
        if not _wait_for_hindsight():
            raise RuntimeError("Hindsight service started, but its API did not become ready.")
    else:
        notes.append(
            "Hindsight automatic service setup unavailable. Start it manually with: "
            + manual_hindsight_command(current)
        )

    if hindsight_healthcheck(current):
        asyncio.run(configure_hindsight_bank())
        bank_health = _check_bank(current)
        bank_status = "ready" if bank_health.status == "ready" else "unavailable"
    else:
        bank_status = "unavailable"
        notes.append(f"Hindsight is not available at 127.0.0.1:{HINDSIGHT_API_PORT}.")

    hindsight_status = "ready" if hindsight_healthcheck(current) else "unavailable"
    return SetupResult(
        python="ready" if python_ready else "unavailable",
        uv="ready" if uv_ready else "unavailable",
        environment="ready",
        qdrant=qdrant_health.status,
        hindsight=hindsight_status,
        memory_bank=bank_status,
        notes=tuple(notes),
    )

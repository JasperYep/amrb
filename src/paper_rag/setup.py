from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import sys
import subprocess
import tempfile
import time
import tomllib
import uuid
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit

from paper_rag.config import (
    Settings,
    copied_workspace_origin,
    claim_qdrant_collection,
    create_qdrant_client,
    derived_resource_ids,
    ensure_paper_id_payload_index,
    get_settings,
    inspect_qdrant_collection_identity,
    load_settings,
    update_workspace_config,
)
from paper_rag.ingest import sha256_file
from paper_rag.hindsight_setup import configure_hindsight_bank
from paper_rag.health import (
    _manifest_counts,
    _check_bank,
    _check_qdrant,
    _has_embedding_config,
)
from paper_rag.service import (
    HINDSIGHT_API_PORT,
    hindsight_healthcheck,
    install_user_service,
    manual_hindsight_command,
    service_paths,
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
    embedding: str = "unverified"


@dataclass(frozen=True)
class LegacyIndexInspection:
    papers: int
    manifest_chunks: int
    qdrant_chunks: int
    vector_dimension: int
    paper_ids: tuple[str, ...]


@dataclass(frozen=True)
class WorkspaceFileChange:
    path: Path
    content: str
    summary: str
    existed: bool


def resolve_copied_workspace_identity(
    root: Path,
    choice: str,
    *,
    new_workspace_id: str | None = None,
) -> Settings:
    """Bind a moved workspace to this path or create an isolated identity for a copy."""
    workspace_root = root.expanduser().resolve()
    origin = copied_workspace_origin(workspace_root)
    if origin is None:
        raise ValueError("This workspace does not have a copied or moved identity to resolve.")
    if choice not in {"reuse", "new"}:
        raise ValueError("Copied workspace identity must be either 'reuse' or 'new'.")

    current = load_settings(workspace_root, allow_copied_workspace=True)
    config_path = workspace_root / "amrb.toml"
    original_config = config_path.read_text(encoding="utf-8")
    document = tomlkit.parse(original_config)
    workspace_table = document["workspace"]
    workspace_table["bound_path"] = str(workspace_root)

    archive_files: list[tuple[Path, Path]] = []
    cleared_files: list[tuple[Path, str]] = []
    if choice == "new":
        old_workspace_id = current.workspace_id
        new_workspace_id = str(uuid.UUID(new_workspace_id)) if new_workspace_id else str(uuid.uuid4())
        collection, bank_id = derived_resource_ids(new_workspace_id)
        workspace_table["id"] = new_workspace_id
        workspace_table["copied_from_id"] = old_workspace_id
        workspace_table.pop("legacy_data_compatibility", None)
        if "index" not in document:
            document["index"] = tomlkit.table()
        if "memory" not in document:
            document["memory"] = tomlkit.table()
        index_table = document["index"]
        index_table["collection"] = collection
        index_table["legacy_index_accepted"] = False
        index_table["resource_accepted"] = False
        index_table.pop("embedding_dimension", None)
        if not current.qdrant_url:
            suffix = collection.removeprefix("amrb_papers_")
            index_table["qdrant_path"] = f".state/qdrant-{suffix}"
        document["memory"]["bank_id"] = bank_id
        document["memory"]["resource_accepted"] = False

        archive_root = current.state_dir / "copied-workspace" / old_workspace_id
        for name in ("index.json", "index-failures.json"):
            source = current.state_dir / name
            if not source.is_file():
                continue
            destination = archive_root / name
            if destination.exists():
                raise ValueError(f"Copied workspace archive already exists: {destination}")
            archive_files.append((source, destination))
            cleared_files.append((source, "{}\n"))

    rendered = tomlkit.dumps(document)
    tomllib.loads(rendered)
    config_temp: Path | None = None
    created_archives: list[tuple[Path, Path]] = []
    previous_contents = {path: path.read_text(encoding="utf-8") for path, _ in cleared_files}
    previous_modes = {path: path.stat().st_mode & 0o777 for path, _ in cleared_files}
    original_config_mode = config_path.stat().st_mode & 0o777
    config_written = False
    try:
        for source, destination in archive_files:
            destination.parent.mkdir(parents=True, exist_ok=True)
            created_archives.append((source, destination))
            shutil.copy2(source, destination)
        for path, content in cleared_files:
            _atomic_write_text(path, content, previous_modes[path])
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=workspace_root,
            prefix=".amrb.toml.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            config_temp = Path(temporary.name)
            os.fchmod(temporary.fileno(), original_config_mode)
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(config_temp, config_path)
        config_written = True
        return load_settings(workspace_root)
    except Exception:
        if config_written:
            _atomic_write_text(config_path, original_config, original_config_mode)
        for path, content in previous_contents.items():
            _atomic_write_text(path, content, previous_modes[path])
        for _, destination in created_archives:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if config_temp and config_temp.exists():
            config_temp.unlink()


def _atomic_write_text(path: Path, content: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            os.fchmod(temporary.fileno(), mode)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def inspect_legacy_index(settings: Settings) -> LegacyIndexInspection:
    manifest, expected_chunks, missing_files = _manifest_counts(settings)
    if not manifest or not any(entry.get("paper_id") for entry in manifest.values()):
        raise ValueError("No legacy paper index was found to confirm.")
    if missing_files:
        raise ValueError("The legacy manifest references missing paper files or invalid paths.")
    for relative_path, entry in manifest.items():
        if entry.get("sha256"):
            source = settings.workspace_root / relative_path
            if sha256_file(source) != entry["sha256"]:
                raise ValueError("A legacy paper source has changed since it was indexed.")

    client = create_qdrant_client(settings)
    try:
        if not client.collection_exists(settings.qdrant_collection):
            raise ValueError("The configured Qdrant collection does not exist.")
        collection = client.get_collection(settings.qdrant_collection)
        qdrant_chunks = int(client.count(settings.qdrant_collection, exact=True).count)
        if qdrant_chunks != expected_chunks:
            raise ValueError(
                f"Legacy chunk counts disagree: manifest={expected_chunks}, Qdrant={qdrant_chunks}."
            )
        vectors = collection.config.params.vectors
        if isinstance(vectors, dict):
            if "" in vectors:
                vector = vectors[""]
            elif len(vectors) == 1:
                vector = next(iter(vectors.values()))
            else:
                raise ValueError("Qdrant has multiple named vectors; explicit selection is required.")
        else:
            vector = vectors
        dimension = int(getattr(vector, "size", 0) or 0)
        if dimension < 1:
            raise ValueError("Could not verify the legacy Qdrant vector dimension.")
    finally:
        client.close()
    paper_ids = tuple(sorted(str(entry["paper_id"]) for entry in manifest.values() if entry.get("paper_id")))
    return LegacyIndexInspection(
        papers=len(paper_ids),
        manifest_chunks=expected_chunks,
        qdrant_chunks=qdrant_chunks,
        vector_dimension=dimension,
        paper_ids=paper_ids,
    )


def accept_legacy_index(settings: Settings, inspection: LegacyIndexInspection) -> Settings:
    update_workspace_config(
        settings.workspace_root,
        "index",
        {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.rag_chunk_size,
            "chunk_overlap": settings.rag_chunk_overlap,
            "embedding_dimension": inspection.vector_dimension,
            "legacy_index_accepted": True,
        },
    )
    return get_settings(settings.workspace_root)


def inspect_qdrant_resource(settings: Settings) -> tuple[bool, int, str | None]:
    client = create_qdrant_client(settings)
    try:
        return inspect_qdrant_collection_identity(client, settings)
    finally:
        client.close()


def accept_existing_qdrant_resource(settings: Settings) -> Settings:
    update_workspace_config(
        settings.workspace_root,
        "index",
        {"resource_accepted": True},
    )
    return load_settings(settings.workspace_root)


def accept_existing_hindsight_resource(settings: Settings) -> Settings:
    update_workspace_config(
        settings.workspace_root,
        "memory",
        {"resource_accepted": True},
    )
    return load_settings(settings.workspace_root)


def write_environment_values(path: Path, values: dict[str, str]) -> tuple[str, ...]:
    """Atomically add missing workspace settings without replacing nonempty values."""
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = original.splitlines()
    changed: list[str] = []

    def quoted(value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("Environment values must be single-line.")
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    for key, value in values.items():
        match_index = next(
            (index for index, line in enumerate(lines) if re.match(rf"^\s*{re.escape(key)}\s*=", line)),
            None,
        )
        if match_index is None:
            lines.append(f"{key}={quoted(value)}")
            changed.append(key)
            continue
        line = lines[match_index]
        rhs = line.split("=", 1)[1].strip()
        if rhs and rhs not in {'""', "''"}:
            continue
        lines[match_index] = f"{key}={quoted(value)}"
        changed.append(key)
    if not changed:
        if path.exists():
            path.chmod(0o600)
        return ()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines).rstrip() + "\n"
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
            os.fchmod(temporary.fileno(), 0o600)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o600)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return tuple(changed)


def _research_rules(settings: Settings) -> str:
    name = settings.workspace_name.strip() or "this research workspace"
    focus = settings.research_focus.strip()
    focus_line = f"Current focus: {focus}\n\n" if focus else ""
    return f'''# Research workflow

Use this workspace for {name}.
{focus_line}AMRB keeps paper evidence separate from the researcher's personal long-term research cognition. The MCP client remains responsible for conversation, reasoning, and tool choice.

## Evidence and memory

- Use Paper RAG for claims about what papers define, propose, evaluate, or report. Read continuous PDF pages when search snippets are insufficient, and cite the paper and PDF page number.
- Use Hindsight recall for a specific prior judgment or decision, and reflect when synthesizing multiple long-term judgments or their evolution.
- Keep external paper evidence, remembered researcher judgments, and current analysis explicitly distinct.
- A paper fact or retrieved passage does not become personal memory just because it was found.
- A question, tentative discussion, or paper summary does not establish the researcher's view.
- Retain only an explicitly stated or confirmed durable judgment, hypothesis, decision, concept, or stable preference. Do not retain ordinary questions, temporary tasks, or paper text.
- When a current judgment revises an earlier one, mark the current view and preserve only useful history.

## Workspace operations

Use the AMRB MCP for research evidence and memory. Use the `amrb` CLI or the `amrb-ops` skill for workspace maintenance, paper inventory, system status, and official infrastructure UI links. Do not treat Qdrant vector counts as a paper inventory or Hindsight synthesis as a complete memory listing.
'''


def _mcp_launch_command(workspace_root: Path) -> tuple[str, list[str]]:
    workspace = workspace_root.expanduser().resolve()
    executable = shutil.which("amrb-mcp")
    if executable:
        return str(Path(executable).resolve()), ["--workspace", str(workspace)]
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Could not locate amrb-mcp or uv to generate a stable Codex MCP command.")
    install_root = Path(__file__).resolve().parents[2]
    return str(Path(uv).resolve()), [
        "--directory",
        str(install_root),
        "run",
        "amrb-mcp",
        "--workspace",
        str(workspace),
    ]


def plan_workspace_client_files(root: Path, settings: Settings) -> tuple[WorkspaceFileChange, ...]:
    workspace_root = root.expanduser().resolve()
    changes: list[WorkspaceFileChange] = []
    rules_path = workspace_root / "AGENTS.md"
    if not rules_path.exists():
        changes.append(
            WorkspaceFileChange(
                path=rules_path,
                content=_research_rules(settings),
                summary="Create generic research collaboration rules; preserve existing rules when present.",
                existed=False,
            )
        )

    config_path = workspace_root / ".codex" / "config.toml"
    original = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    document = tomlkit.loads(original) if original.strip() else tomlkit.document()
    servers = document.get("mcp_servers")
    if servers is None:
        servers = tomlkit.table()
        document["mcp_servers"] = servers
    if not isinstance(servers, MutableMapping):
        raise ValueError("Codex config mcp_servers is not a table; setup left it unchanged.")
    entry = servers.get("amrb")
    if entry is None:
        entry = tomlkit.table()
        servers["amrb"] = entry
    if not isinstance(entry, MutableMapping):
        raise ValueError("Codex mcp_servers.amrb is not a table; setup left it unchanged.")

    command, arguments = _mcp_launch_command(workspace_root)
    managed_values = {
        "command": command,
        "args": arguments,
        "required": True,
        "enabled_tools": [
            "search_papers",
            "list_papers",
            "get_paper_pages",
            "retain",
            "recall",
            "reflect",
        ],
        "startup_timeout_sec": 30,
    }
    changed_keys = [key for key, value in managed_values.items() if entry.get(key) != value]
    if changed_keys:
        for key in changed_keys:
            entry[key] = managed_values[key]
        changes.append(
            WorkspaceFileChange(
                path=config_path,
                content=tomlkit.dumps(document),
                summary="Set AMRB MCP launch path and workspace binding; update only "
                + ", ".join(changed_keys)
                + "; preserve other MCP servers and unknown AMRB fields.",
                existed=config_path.exists(),
            )
        )
    return tuple(changes)


def write_workspace_client_files(
    changes: tuple[WorkspaceFileChange, ...], root: Path
) -> tuple[Path, ...]:
    workspace_root = root.expanduser().resolve()
    written: list[Path] = []
    try:
        for change in changes:
            parent = change.path.parent
            if not parent.resolve().is_relative_to(workspace_root):
                raise ValueError(f"Refusing to write outside the workspace: {change.path}")
            parent.mkdir(parents=True, exist_ok=True)
            previous_mode = change.path.stat().st_mode & 0o777 if change.path.exists() else 0o644
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=parent,
                    prefix=f".{change.path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    os.fchmod(temporary.fileno(), previous_mode)
                    temporary.write(change.content)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_path, change.path)
                written.append(change.path)
            finally:
                if temporary_path and temporary_path.exists():
                    temporary_path.unlink()
    except Exception as exc:
        completed = ", ".join(str(path) for path in written) or "none"
        raise RuntimeError(
            f"Client setup stopped after writing {completed}. Each completed file is valid; rerun setup "
            "to apply any remaining client changes."
        ) from exc
    return tuple(written)


def verify_mcp_startup(workspace_root: Path, settings: Settings, timeout: float = 20.0) -> dict[str, Any]:
    command, arguments = _mcp_launch_command(workspace_root)
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "amrb-setup", "version": "0.1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    request = "".join(json.dumps(message, separators=(",", ":")) + "\n" for message in messages)
    try:
        result = subprocess.run(
            [command, *arguments],
            input=request,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=Path(tempfile.gettempdir()),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("MCP initialization did not complete from an unrelated working directory.") from exc
    if result.returncode != 0:
        raise RuntimeError("MCP server exited during initialization; inspect the workspace config and rerun setup.")
    responses: dict[int, dict[str, Any]] = {}
    try:
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            response = json.loads(line)
            if isinstance(response.get("id"), int):
                responses[response["id"]] = response
    except json.JSONDecodeError as exc:
        raise RuntimeError("MCP server returned a non-JSON response during initialization.") from exc
    initialized = responses.get(1, {}).get("result", {})
    tool_result = responses.get(2, {}).get("result", {})
    instructions = str(initialized.get("instructions", ""))
    tools = tool_result.get("tools", [])
    tool_names = sorted(str(tool.get("name")) for tool in tools if isinstance(tool, dict))
    expected_names = sorted((
        "search_papers",
        "list_papers",
        "get_paper_pages",
        "retain",
        "recall",
        "reflect",
    ))
    if expected_names != tool_names:
        raise RuntimeError("MCP tool enumeration did not match the configured research tool set.")
    expected_binding = (settings.workspace_id, settings.qdrant_collection, settings.hindsight_bank_id)
    if not all(value in instructions for value in expected_binding):
        raise RuntimeError("MCP server initialized with an unexpected workspace binding.")
    return {
        "workspace_id": settings.workspace_id,
        "tools": tool_names,
        "cwd_independent": True,
    }


def _wait_for_hindsight(settings: Settings, timeout: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if hindsight_healthcheck(settings):
            return True
        time.sleep(0.25)
    return False


def setup_workspace(
    settings: Settings | None = None,
    *,
    update_hindsight_missions: bool = False,
) -> SetupResult:
    current = settings or get_settings()
    python_ready = sys.version_info >= (3, 12)
    if not python_ready:
        raise RuntimeError("AMRB requires Python 3.12 or newer.")
    uv_ready = shutil.which("uv") is not None
    if not uv_ready:
        raise RuntimeError("uv was not found in PATH. Install uv and rerun `uv run amrb setup`.")

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
        claim_qdrant_collection(qdrant_client, current)
        ensure_paper_id_payload_index(qdrant_client, current)
    finally:
        qdrant_client.close()

    notes: list[str] = []
    service = service_paths()
    service_already_configured = service.env_file.exists() or service.unit_file.exists()
    api_ready = hindsight_healthcheck(current)
    if not api_ready and service_already_configured:
        notes.append(
            "An existing shared Hindsight service configuration was found, but its API is unavailable. "
            "Setup left that service configuration unchanged; inspect it with `amrb service status`."
        )
    elif not api_ready and systemd_user_available():
        install_user_service(current)
        if not _wait_for_hindsight(current):
            raise RuntimeError("Hindsight service started, but its API did not become ready.")
        api_ready = True
    elif not api_ready:
        notes.append(
            "Hindsight automatic service setup unavailable. Start it manually with: "
            + manual_hindsight_command(current)
        )

    if api_ready:
        bank_action = asyncio.run(
            configure_hindsight_bank(
                current,
                update_existing=update_hindsight_missions,
                adopt_existing=(
                    current.hindsight_resource_accepted or current.legacy_data_compatibility
                ),
            )
        )
        if bank_action == "created" and not current.hindsight_resource_accepted:
            update_workspace_config(
                current.workspace_root,
                "memory",
                {"resource_accepted": True},
            )
            current = load_settings(current.workspace_root)
        elif bank_action == "needs-confirmation":
            notes.append(
                "The existing Hindsight missions were left unchanged. Review them and rerun setup "
                "with --update-missions to apply the displayed changes."
            )
        elif bank_action == "updated":
            notes.append("Updated the confirmed Hindsight bank missions.")
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

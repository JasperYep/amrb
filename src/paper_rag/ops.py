from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import pymupdf

from paper_rag.config import (
    Settings,
    create_qdrant_client,
    create_vector_store,
    get_settings,
    redact_sensitive_text,
)
from paper_rag.ingest import (
    _delete_paper,
    _paper_nodes,
    ingest_papers,
    load_manifest,
    sha256_file,
    validate_index_compatibility,
    write_manifest,
)
from paper_rag.models import (
    PaperAddResult,
    PaperFailure,
    PaperRecord,
    PaperRemoveResult,
    PaperSyncResult,
)
from paper_rag.retrieval import get_status


def _paper_path(relative_path: str, settings: Settings) -> Path | None:
    papers_root = settings.papers_dir.resolve()
    path = (settings.workspace_root / relative_path).resolve()
    if path.suffix.lower() != ".pdf" or not path.is_relative_to(papers_root):
        return None
    return path


def _index_failures_path(settings: Settings) -> Path:
    return settings.state_dir / "index-failures.json"


def _clear_index_failure(settings: Settings, relative_path: str) -> None:
    path = _index_failures_path(settings)
    failures = load_manifest(path)
    if relative_path in failures:
        del failures[relative_path]
        write_manifest(path, failures)


def _mark_index_failure(
    settings: Settings,
    relative_path: str,
    source_path: Path,
    message: object,
) -> None:
    failures = load_manifest(_index_failures_path(settings))
    failure = _failure(relative_path, message, False, settings)
    failures[relative_path] = {
        "source_sha256": sha256_file(source_path),
        "message": failure.message,
    }
    write_manifest(_index_failures_path(settings), failures)


def _record(
    relative_path: str,
    entry: dict[str, Any] | None,
    settings: Settings,
    failure: dict[str, Any] | None = None,
) -> PaperRecord:
    path = _paper_path(relative_path, settings)
    pages: int | None = None
    if path is not None and path.is_file():
        try:
            with pymupdf.open(path) as pdf:
                pages = pdf.page_count
        except Exception:
            pages = None
    title = str(entry.get("title") or (path.stem if path else Path(relative_path).stem)) if entry else (
        path.stem if path else Path(relative_path).stem
    )
    if path is not None and path.is_file() and not (entry or {}).get("title"):
        try:
            with pymupdf.open(path) as pdf:
                title = str((pdf.metadata or {}).get("title") or "").strip() or title
        except Exception:
            pass
    source_hash = sha256_file(path) if path is not None and path.is_file() else None
    failure_matches_source = bool(
        failure and source_hash and failure.get("source_sha256") == source_hash
    )
    if failure_matches_source:
        status = "failed"
    elif entry is None:
        status = "not_indexed"
    elif path is None or not path.is_file():
        status = "source_missing"
    elif entry.get("sha256") and source_hash != entry["sha256"]:
        status = "source_changed"
    elif not entry.get("paper_id"):
        status = "failed"
    elif not isinstance(entry.get("index_config"), dict) and not settings.legacy_index_accepted:
        status = "unverified"
    else:
        status = "indexed"
    return PaperRecord(
        paper_id=str(entry["paper_id"]) if entry and entry.get("paper_id") else None,
        title=title,
        file_name=Path(relative_path).name,
        pages=pages,
        chunks=int((entry or {}).get("chunk_count", 0)),
        index_status=status,
        index_message=str(failure.get("message")) if failure_matches_source and failure else None,
    )


def list_paper_records(settings: Settings | None = None) -> list[PaperRecord]:
    current = settings or get_settings()
    manifest = load_manifest(current.manifest_path)
    failures = load_manifest(_index_failures_path(current))
    records = [
        _record(path, entry, current, failures.get(path))
        for path, entry in sorted(manifest.items())
    ]
    tracked = set(manifest)
    for path in sorted(current.papers_dir.glob("*.pdf")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(current.workspace_root).as_posix()
        if relative_path not in tracked:
            records.append(_record(relative_path, None, current, failures.get(relative_path)))
    return records


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _failure(
    source: str,
    message: object,
    copied: bool,
    settings: Settings,
) -> PaperFailure:
    clean_message = " ".join(redact_sensitive_text(str(message), settings).split())
    if len(clean_message) > 500:
        clean_message = clean_message[:497] + "..."
    return PaperFailure(source=source, message=clean_message, copied=copied)


def add_papers(
    paths: list[Path],
    settings: Settings | None = None,
    embedding_model: Any | None = None,
    qdrant_client: Any | None = None,
) -> PaperAddResult:
    current = settings or get_settings()
    validate_index_compatibility(load_manifest(current.manifest_path), current)
    if not paths:
        raise ValueError("至少指定一个 PDF 文件路径。")
    current.papers_dir.mkdir(parents=True, exist_ok=True)
    result = PaperAddResult()
    planned_destinations: dict[str, str] = {}

    for supplied_path in paths:
        source_label = str(supplied_path)
        copied = False
        destination: Path | None = None
        relative_path: str | None = None
        try:
            source = Path(supplied_path).expanduser().resolve(strict=True)
            if not source.is_file():
                raise ValueError("路径不是文件。")
            if source.suffix.lower() != ".pdf":
                raise ValueError("只支持 PDF 文件。")
            with pymupdf.open(source) as pdf:
                pages = pdf.page_count
                if pages < 1:
                    raise ValueError("PDF 没有页面。")

            digest = sha256_file(source)
            manifest = load_manifest(current.manifest_path)
            duplicate = next(
                (
                    (relative_path, entry)
                    for relative_path, entry in manifest.items()
                    if entry.get("sha256") == digest and entry.get("paper_id")
                ),
                None,
            )
            if duplicate is not None:
                duplicate_path = _paper_path(duplicate[0], current)
                if duplicate_path is not None and not duplicate_path.is_file():
                    _copy_atomic(source, duplicate_path)
                _clear_index_failure(current, duplicate[0])
                result.skipped.append(_record(duplicate[0], duplicate[1], current))
                continue

            destination = current.papers_dir / source.name
            relative_path = destination.relative_to(current.workspace_root).as_posix()
            previous = manifest.get(relative_path)
            earlier_hash = planned_destinations.get(relative_path)
            if earlier_hash is not None and earlier_hash != digest:
                raise ValueError(
                    f"本次添加中有多个同名但内容不同的文件：{destination.name}。"
                )
            planned_destinations[relative_path] = digest

            if destination.exists():
                destination_hash = sha256_file(destination)
                if destination_hash != digest and previous is None:
                    raise ValueError(
                        f"papers/ 中已存在未纳入索引的同名文件：{destination.name}；拒绝覆盖。"
                    )
                if destination_hash != digest:
                    _copy_atomic(source, destination)
                    copied = source != destination
            elif source != destination:
                _copy_atomic(source, destination)
                copied = True

            old_hash = previous.get("sha256") if previous else None
            ingest_papers(
                current,
                embedding_model,
                qdrant_client,
                paths=[destination],
            )
            updated_manifest = load_manifest(current.manifest_path)
            entry = updated_manifest.get(relative_path)
            if not entry or entry.get("sha256") != digest:
                raise RuntimeError("文件已复制，但索引清单未确认该论文。")
            _clear_index_failure(current, relative_path)
            record = _record(relative_path, entry, current)
            if old_hash and old_hash != digest:
                result.updated.append(record)
            else:
                result.added.append(record)
        except Exception as exc:
            failure = _failure(source_label, exc, copied, current)
            result.failed.append(failure)
            if destination is not None and relative_path and destination.is_file():
                try:
                    _mark_index_failure(current, relative_path, destination, exc)
                except Exception:
                    failure.message += " Index failure status could not be persisted; run `amrb paper sync`."

    return result



def sync_papers(
    settings: Settings | None = None,
    embedding_model: Any | None = None,
    qdrant_client: Any | None = None,
) -> PaperSyncResult:
    current = settings or get_settings()
    before = load_manifest(current.manifest_path)
    error: str | None = None
    ingest_error: str | None = None
    try:
        ingest_papers(current, embedding_model, qdrant_client)
    except Exception as exc:
        ingest_error = _failure("paper sync", exc, False, current).message
        error = ingest_error

    after = load_manifest(current.manifest_path)
    current_file_hashes = {
        path.relative_to(current.workspace_root).as_posix(): sha256_file(path)
        for path in current.papers_dir.glob("*.pdf")
        if path.is_file()
    }
    if ingest_error:
        for relative_path, digest in current_file_hashes.items():
            if before.get(relative_path, {}).get("sha256") == digest:
                continue
            try:
                _mark_index_failure(
                    current,
                    relative_path,
                    current.workspace_root / relative_path,
                    ingest_error,
                )
            except Exception:
                pass
    else:
        for relative_path in after:
            _clear_index_failure(current, relative_path)
    added = sum(path not in before for path in after)
    updated = sum(
        path in before and before[path].get("sha256") != entry.get("sha256")
        for path, entry in after.items()
    )
    removed = sum(path not in after for path in before)
    unchanged = sum(
        path in after
        and after[path].get("sha256") == entry.get("sha256")
        and current_file_hashes.get(path) == entry.get("sha256")
        for path, entry in before.items()
    )
    try:
        status = get_status(current, qdrant_client)
        chunks = status.chunks
    except Exception as exc:
        chunks = 0
        if error is None:
            error = " ".join(str(exc).split())[:500]
    return PaperSyncResult(
        added=added,
        updated=updated,
        removed=removed,
        unchanged=unchanged,
        papers=len(after),
        chunks=chunks,
        error=error,
    )


def _match_manifest_path(identifier: str, manifest: dict[str, dict[str, Any]]) -> str | None:
    query = identifier.casefold().strip()
    exact: list[str] = []
    partial: list[str] = []
    for relative_path, entry in manifest.items():
        path = Path(relative_path)
        aliases = {
            relative_path.casefold(),
            path.name.casefold(),
            path.stem.casefold(),
            str(entry.get("paper_id", "")).casefold(),
            str(entry.get("title", "")).casefold(),
        }
        if query in aliases:
            exact.append(relative_path)
        elif any(query in alias for alias in aliases):
            partial.append(relative_path)
    matches = exact or partial
    if len(matches) > 1:
        names = ", ".join(sorted(Path(path).name for path in matches))
        raise ValueError(f"匹配到多篇论文，请使用文件名或 paper_id：{names}")
    return matches[0] if matches else None


def remove_paper(
    identifier: str,
    settings: Settings | None = None,
    qdrant_client: Any | None = None,
) -> PaperRemoveResult | None:
    current = settings or get_settings()
    manifest = load_manifest(current.manifest_path)
    relative_path = _match_manifest_path(identifier, manifest)
    if relative_path is None:
        return None
    entry = manifest[relative_path]
    record = _record(relative_path, entry, current)
    path = _paper_path(relative_path, current)
    if path is None:
        raise ValueError("索引清单中的论文路径无效，拒绝删除。")

    paper_id = str(entry["paper_id"])
    remaining_ids = {
        str(other["paper_id"])
        for other_path, other in manifest.items()
        if other_path != relative_path and other.get("paper_id")
    }
    previous_manifest = dict(manifest)
    del manifest[relative_path]

    staged: Path | None = None
    client = qdrant_client
    owns_client = False
    try:
        if path.is_file():
            trash = current.state_dir / "trash"
            trash.mkdir(parents=True, exist_ok=True)
            staged = trash / f"{uuid4().hex}-{path.name}"
            os.replace(path, staged)
        write_manifest(current.manifest_path, manifest)
        if paper_id not in remaining_ids:
            if client is None:
                client = create_qdrant_client(current)
                owns_client = True
            if client.collection_exists(current.qdrant_collection):
                store = create_vector_store(client, current)
                store.delete(ref_doc_id=paper_id)
    except Exception:
        write_manifest(current.manifest_path, previous_manifest)
        if staged is not None and staged.exists():
            os.replace(staged, path)
        raise
    finally:
        if owns_client and client is not None:
            client.close()

    if staged is not None:
        staged.unlink(missing_ok=True)
    _clear_index_failure(current, relative_path)
    return PaperRemoveResult(removed=record)

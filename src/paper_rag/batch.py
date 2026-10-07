from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pymupdf
from llama_index.core.schema import MetadataMode

from paper_rag.config import Settings, create_qdrant_client, create_vector_store, get_settings
from paper_rag.ingest import _delete_paper, _paper_nodes, load_manifest, sha256_file, write_manifest
from paper_rag.models import PaperAddResult, PaperRecord
from paper_rag.ops import _copy_atomic, _failure, _paper_path, _record

logger = logging.getLogger(__name__)
BATCH_MAX_REQUESTS = 50_000
BATCH_MAX_FILE_BYTES = 500 * 1024 * 1024
BATCH_MAX_LINE_BYTES = 6 * 1024 * 1024
BATCH_POLL_SECONDS = 30
BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _batch_state_path(job_id: str, settings: Settings) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Batch job ID 格式无效。")
    return settings.embedding_batches_dir / f"{job_id}.json"


def _write_batch_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(state, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def _load_batch_state(job_id: str, settings: Settings) -> tuple[Path, dict[str, Any]]:
    path = _batch_state_path(job_id, settings)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到本地 Batch job：{job_id}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 Batch job 状态：{job_id}") from exc
    if not isinstance(state, dict) or state.get("job_id") != job_id:
        raise ValueError(f"Batch job 状态格式无效：{job_id}")
    return path, state


def _create_batch_client(settings: Settings) -> Any:
    from openai import OpenAI

    key = settings.embedding_api_key.get_secret_value() if settings.embedding_api_key else ""
    if not key:
        raise ValueError("缺少 EMBEDDING_API_KEY，无法提交 Batch embedding 任务。")
    return OpenAI(
        api_key=key,
        base_url=settings.embedding_base_url,
        timeout=300,
        max_retries=3,
    )


def _batch_row(custom_id: str, model: str, text: str) -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": model,
                "input": text,
                "encoding_format": "float",
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _write_batch_jsonl(
    path: Path,
    candidates: list[dict[str, Any]],
    rows: list[dict[str, str]],
    model: str,
    settings: Settings,
) -> None:
    rows_by_paper: dict[str, dict[str, str]] = {}
    for row in rows:
        rows_by_paper.setdefault(row["relative_path"], {})[row["node_id"]] = row[
            "custom_id"
        ]
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for candidate in candidates:
            nodes, _ = _paper_nodes(
                candidate["destination"],
                candidate["paper_id"],
                settings,
            )
            if len(nodes) != int(candidate["chunk_count"]):
                raise ValueError(
                    f"重新切分后的 chunk 数与预处理不一致：{candidate['source']}。"
                )
            custom_ids = rows_by_paper.get(candidate["relative_path"], {})
            for node in nodes:
                custom_id = custom_ids.get(str(node.id_))
                if custom_id is None:
                    raise ValueError(f"Batch 请求缺少 chunk：{node.id_}")
                text = node.get_content(metadata_mode=MetadataMode.EMBED)
                output.write(_batch_row(custom_id, model, text))
                output.write("\n")


def _consume_batch_result_file(
    client: Any,
    file_id: str | None,
    request_index: dict[str, dict[str, str]],
    database: sqlite3.Connection,
    errors: dict[str, str],
) -> str | None:
    if not file_id:
        return None
    try:
        with client.files.with_streaming_response.content(file_id) as response:
            pending: list[tuple[str, str]] = []
            for line_number, raw_line in enumerate(response.iter_lines(), start=1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError:
                    return f"Batch 结果文件第 {line_number} 行不是有效 JSON。"
                if not isinstance(row, dict):
                    return f"Batch 结果文件第 {line_number} 行格式无效。"
                custom_id = str(row.get("custom_id") or "")
                if not custom_id or custom_id not in request_index:
                    continue
                error = row.get("error")
                response_data = row.get("response") or {}
                status_code = response_data.get("status_code")
                body = response_data.get("body") or {}
                data = body.get("data") or []
                if error or (
                    status_code is not None
                    and not 200 <= int(status_code) < 300
                ):
                    detail = error or body.get("error") or f"HTTP {status_code}"
                    if isinstance(detail, dict):
                        detail = detail.get("message") or detail.get("code") or "请求失败"
                    errors[custom_id] = " ".join(str(detail).split())[:300]
                    continue
                if len(data) != 1 or not isinstance(data[0], dict):
                    errors[custom_id] = "Embedding 返回的向量数量不是 1。"
                    continue
                vector = data[0].get("embedding")
                if (
                    not isinstance(vector, list)
                    or not vector
                    or any(
                        not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in vector
                    )
                ):
                    errors[custom_id] = "Embedding 返回的向量格式无效。"
                    continue
                pending.append(
                    (
                        custom_id,
                        json.dumps(vector, separators=(",", ":")),
                    )
                )
                if len(pending) >= 256:
                    database.executemany(
                        "INSERT OR REPLACE INTO vectors (custom_id, embedding) VALUES (?, ?)",
                        pending,
                    )
                    database.commit()
                    pending.clear()
            if pending:
                database.executemany(
                    "INSERT OR REPLACE INTO vectors (custom_id, embedding) VALUES (?, ?)",
                    pending,
                )
                database.commit()
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        return f"无法读取 Batch 结果文件 (HTTP {status or 'unknown'})。"
    return None


def _delete_batch_files(client: Any, state: dict[str, Any], batch: Any) -> None:
    file_ids = {
        state.get("input_file_id"),
        getattr(batch, "output_file_id", None),
        getattr(batch, "error_file_id", None),
    }
    for file_id in file_ids - {None}:
        try:
            client.files.delete(str(file_id))
        except Exception as exc:
            logger.info("Could not delete consumed Batch file (%s).", type(exc).__name__)


def _batch_counts(batch: Any) -> dict[str, int] | None:
    counts = getattr(batch, "request_counts", None)
    if counts is None:
        return None
    if hasattr(counts, "model_dump"):
        counts = counts.model_dump()
    return {
        key: int(value)
        for key, value in counts.items()
        if key in {"total", "completed", "failed"} and value is not None
    }


def _collect_paper_batch(
    client: Any,
    batch: Any,
    state: dict[str, Any],
    settings: Settings,
    qdrant_client: Any | None,
) -> PaperAddResult:
    rows_by_paper: dict[str, list[dict[str, str]]] = {}
    request_index = {row["custom_id"]: row for row in state["requests"]}
    for row in state["requests"]:
        rows_by_paper.setdefault(row["relative_path"], []).append(row)

    settings.embedding_batches_dir.mkdir(parents=True, exist_ok=True)
    result_path = settings.embedding_batches_dir / f"{state['job_id']}.results.sqlite3"
    result_path.unlink(missing_ok=True)
    database = sqlite3.connect(result_path)
    database.execute(
        "CREATE TABLE vectors (custom_id TEXT PRIMARY KEY, embedding TEXT NOT NULL)"
    )
    output_errors: dict[str, str] = {}
    global_error = _consume_batch_result_file(
        client,
        getattr(batch, "output_file_id", None),
        request_index,
        database,
        output_errors,
    )
    error_file_error = _consume_batch_result_file(
        client,
        getattr(batch, "error_file_id", None),
        request_index,
        database,
        output_errors,
    )
    global_error = global_error or error_file_error
    if global_error:
        database.close()
        result_path.unlink(missing_ok=True)
        raise RuntimeError(global_error)
    result = PaperAddResult()
    manifest = load_manifest(settings.manifest_path)
    client_qdrant = qdrant_client
    owns_qdrant = False
    vector_store: Any | None = None
    try:
        for paper in state["papers"]:
            relative_path = paper["relative_path"]
            copied = bool(paper.get("copied"))
            try:
                if global_error:
                    raise RuntimeError(global_error)
                path = _paper_path(relative_path, settings)
                if path is None or not path.is_file():
                    raise RuntimeError("本地论文文件不存在，无法应用 Batch 结果。")
                if sha256_file(path) != paper["sha256"]:
                    raise RuntimeError("论文在 Batch 任务执行期间发生变化；请重新索引。")
                previous = manifest.get(relative_path)
                if previous and previous.get("sha256") == paper["sha256"]:
                    result.skipped.append(_record(relative_path, previous, settings))
                    continue

                nodes, title = _paper_nodes(path, paper["paper_id"], settings)
                if len(nodes) != int(paper["chunk_count"]):
                    raise RuntimeError("重新切分后的 chunk 数与 Batch 提交时不一致。")
                request_rows = rows_by_paper.get(relative_path, [])
                row_by_node = {row["node_id"]: row for row in request_rows}
                if len(row_by_node) != len(nodes):
                    raise RuntimeError("Batch job 中的 chunk 映射不完整。")
                custom_ids = [row["custom_id"] for row in request_rows]
                vectors: dict[str, list[float]] = {}
                for offset in range(0, len(custom_ids), 500):
                    ids = custom_ids[offset : offset + 500]
                    placeholders = ",".join("?" for _ in ids)
                    found = database.execute(
                        f"SELECT custom_id, embedding FROM vectors WHERE custom_id IN ({placeholders})",
                        ids,
                    )
                    vectors.update(
                        {
                            custom_id: json.loads(embedding)
                            for custom_id, embedding in found
                        }
                    )
                failures = [
                    (custom_id, output_errors.get(custom_id))
                    for custom_id in custom_ids
                    if custom_id not in vectors or custom_id in output_errors
                ]
                if failures:
                    first_detail = next((detail for _, detail in failures if detail), None)
                    detail = f"{len(failures)} / {len(request_rows)} 个 chunk embedding 失败"
                    if first_detail:
                        detail += f"：{first_detail}"
                    raise RuntimeError(detail)
                for node in nodes:
                    row = row_by_node.get(str(node.id_))
                    if row is None:
                        raise RuntimeError(f"Batch job 缺少 chunk：{node.id_}")
                    if row["custom_id"] not in vectors:
                        raise RuntimeError(
                            f"Batch 结果缺少 chunk embedding：{row['custom_id']}"
                        )
                    node.embedding = vectors[row["custom_id"]]

                if vector_store is None:
                    if client_qdrant is None:
                        client_qdrant = create_qdrant_client(settings)
                        owns_qdrant = True
                    vector_store = create_vector_store(client_qdrant, settings)
                vector_store.add(nodes)

                old_paper_id = previous.get("paper_id") if previous else None
                other_paper_ids = {
                    str(entry.get("paper_id"))
                    for other_path, entry in manifest.items()
                    if other_path != relative_path and entry.get("paper_id")
                }
                if (
                    old_paper_id
                    and old_paper_id != paper["paper_id"]
                    and old_paper_id not in other_paper_ids
                ):
                    _delete_paper(
                        vector_store,
                        str(old_paper_id),
                        client_qdrant,
                        settings.qdrant_collection,
                    )
                manifest[relative_path] = {
                    "sha256": paper["sha256"],
                    "paper_id": paper["paper_id"],
                    "indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "title": title,
                    "chunk_count": len(nodes),
                }
                write_manifest(settings.manifest_path, manifest)
                record = _record(relative_path, manifest[relative_path], settings)
                if previous and previous.get("sha256") != paper["sha256"]:
                    result.updated.append(record)
                else:
                    result.added.append(record)
            except Exception as exc:
                result.failed.append(
                    _failure(paper["source"], exc, copied, settings)
                )
    finally:
        if owns_qdrant and client_qdrant is not None:
            client_qdrant.close()
        database.close()
        result_path.unlink(missing_ok=True)
    return result


def _save_batch_result(
    path: Path,
    state: dict[str, Any],
    result: PaperAddResult,
    status: str,
) -> PaperAddResult:
    state["status"] = status
    state["result"] = result.model_dump(mode="json")
    _write_batch_state(path, state)
    return result


def _stored_batch_result(state: dict[str, Any]) -> PaperAddResult | None:
    saved = state.get("result")
    return PaperAddResult.model_validate(saved) if isinstance(saved, dict) else None


def _merge_paper_add_results(
    first: PaperAddResult,
    second: PaperAddResult,
) -> PaperAddResult:
    return PaperAddResult(
        added=[*first.added, *second.added],
        updated=[*first.updated, *second.updated],
        skipped=[*first.skipped, *second.skipped],
        failed=[*first.failed, *second.failed],
    )


def _batch_progress_info(batch: Any) -> dict[str, Any]:
    return {
        "status": str(getattr(batch, "status", "unknown")),
        "request_counts": _batch_counts(batch),
    }


def paper_batch_status(
    job_id: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    current = settings or get_settings()
    _, state = _load_batch_state(job_id, current)
    info: dict[str, Any] = {
        "job_id": job_id,
        "status": state.get("status", "unknown"),
        "batch_id": state.get("batch_id"),
    }
    if state.get("result"):
        return info
    if not state.get("batch_id"):
        return info
    client = _create_batch_client(current)
    try:
        batch = client.batches.retrieve(state["batch_id"])
        info.update(_batch_progress_info(batch))
        info["status"] = state.get("status") if state.get("result") else batch.status
        return info
    finally:
        client.close()


def wait_paper_batch(
    job_id: str,
    settings: Settings | None = None,
    qdrant_client: Any | None = None,
    on_status: Callable[[str, str, dict[str, int] | None], None] | None = None,
    poll_seconds: int = BATCH_POLL_SECONDS,
) -> PaperAddResult:
    current = settings or get_settings()
    state_path, state = _load_batch_state(job_id, current)
    stored = _stored_batch_result(state)
    if stored is not None:
        return stored
    batch_id = state.get("batch_id")
    if not batch_id:
        raise ValueError(f"Batch job {job_id} 尚未成功提交，不能等待结果。")

    client = _create_batch_client(current)
    try:
        batch = client.batches.retrieve(batch_id)
        status = str(batch.status)
        while status not in BATCH_TERMINAL_STATUSES:
            counts = _batch_counts(batch)
            if on_status:
                on_status(job_id, status, counts)
            time.sleep(poll_seconds)
            batch = client.batches.retrieve(batch_id)
            status = str(batch.status)
        if on_status:
            on_status(job_id, status, _batch_counts(batch))

        if status != "completed":
            result = PaperAddResult(
                failed=[
                    _failure(
                        paper["source"],
                        f"Batch embedding job 结束状态为 {status}。",
                        bool(paper.get("copied")),
                        current,
                    )
                    for paper in state["papers"]
                ]
            )
            initial_result = PaperAddResult.model_validate(
                state.get("initial_result", {})
            )
            result = _merge_paper_add_results(initial_result, result)
            saved = _save_batch_result(state_path, state, result, status)
            _delete_batch_files(client, state, batch)
            return saved

        result = _collect_paper_batch(
            client,
            batch,
            state,
            current,
            qdrant_client,
        )
        initial_result = PaperAddResult.model_validate(
            state.get("initial_result", {})
        )
        result = _merge_paper_add_results(initial_result, result)
        saved = _save_batch_result(state_path, state, result, "processed")
        _delete_batch_files(client, state, batch)
        return saved
    finally:
        client.close()


def add_papers_batch(
    paths: list[Path],
    settings: Settings | None = None,
    qdrant_client: Any | None = None,
    on_submitted: Callable[[str, str], None] | None = None,
    on_status: Callable[[str, str, dict[str, int] | None], None] | None = None,
) -> PaperAddResult:
    current = settings or get_settings()
    if not paths:
        raise ValueError("至少指定一个 PDF 文件路径。")
    current.papers_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(current.manifest_path)
    result = PaperAddResult()
    candidates: list[dict[str, Any]] = []
    planned_destinations: dict[str, str] = {}
    staged_hashes: dict[str, dict[str, Any]] = {}
    request_rows: list[dict[str, str]] = []
    input_bytes = 0

    for supplied_path in paths:
        source_label = str(supplied_path)
        copied = False
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
                result.skipped.append(_record(duplicate[0], duplicate[1], current))
                continue

            if digest in staged_hashes:
                existing = staged_hashes[digest]
                result.skipped.append(
                    PaperRecord(
                        paper_id=existing["paper_id"],
                        title=existing["title"],
                        file_name=source.name,
                        pages=pages,
                        chunks=int(existing["chunk_count"]),
                    )
                )
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

            paper_id = digest[:16]
            nodes, title = _paper_nodes(source, paper_id, current)
            if not nodes:
                raise ValueError(f"PDF 中没有可检索的正文文本：{source.name}")
            candidate_rows: list[dict[str, str]] = []
            candidate_bytes = 0
            for node in nodes:
                custom_id = f"amrb-{len(request_rows) + len(candidate_rows):06d}"
                text = node.get_content(metadata_mode=MetadataMode.EMBED)
                line_bytes = len(
                    (_batch_row(custom_id, current.embedding_model, text) + "\n").encode(
                        "utf-8"
                    )
                )
                if line_bytes > BATCH_MAX_LINE_BYTES:
                    raise ValueError(
                        f"Batch 请求行超过 6 MB：{source.name}，chunk {node.id_}。"
                    )
                candidate_bytes += line_bytes
                candidate_rows.append(
                    {
                        "custom_id": custom_id,
                        "relative_path": relative_path,
                        "node_id": str(node.id_),
                    }
                )
                if len(request_rows) + len(candidate_rows) > BATCH_MAX_REQUESTS:
                    raise ValueError(
                        "单个 Batch 输入最多 50,000 个请求；请拆成多个批次。"
                    )
            if input_bytes + candidate_bytes > BATCH_MAX_FILE_BYTES:
                raise ValueError("Batch 输入文件超过 500 MB；请拆成多个批次。")
            candidate = {
                "source": source_label,
                "source_path": source,
                "destination": destination,
                "relative_path": relative_path,
                "sha256": digest,
                "paper_id": paper_id,
                "title": title,
                "chunk_count": len(nodes),
                "copied": False,
            }
            candidates.append(candidate)
            staged_hashes[digest] = candidate
            request_rows.extend(candidate_rows)
            input_bytes += candidate_bytes
        except Exception as exc:
            result.failed.append(_failure(source_label, exc, copied, current))

    if not candidates:
        return result

    copied_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        source = candidate["source_path"]
        destination = candidate["destination"]
        try:
            if destination.exists():
                if sha256_file(destination) != candidate["sha256"]:
                    _copy_atomic(source, destination)
                    candidate["copied"] = source != destination
            elif source != destination:
                _copy_atomic(source, destination)
                candidate["copied"] = True
            if sha256_file(destination) != candidate["sha256"]:
                raise RuntimeError("PDF 在准备 Batch 任务时发生变化；请重新添加。")
            copied_candidates.append(candidate)
        except Exception as exc:
            result.failed.append(
                _failure(candidate["source"], exc, candidate["copied"], current)
            )

    if not copied_candidates:
        return result
    copied_paths = {candidate["relative_path"] for candidate in copied_candidates}
    rows = [row for row in request_rows if row["relative_path"] in copied_paths]
    job_id = uuid4().hex
    state_path = _batch_state_path(job_id, current)
    state: dict[str, Any] = {
        "schema_version": 1,
        "job_id": job_id,
        "status": "prepared",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": current.embedding_model,
        "initial_result": result.model_dump(mode="json"),
        "papers": [
            {
                "source": candidate["source"],
                "relative_path": candidate["relative_path"],
                "sha256": candidate["sha256"],
                "paper_id": candidate["paper_id"],
                "title": candidate["title"],
                "chunk_count": candidate["chunk_count"],
                "copied": candidate["copied"],
            }
            for candidate in copied_candidates
        ],
        "requests": rows,
    }
    _write_batch_state(state_path, state)

    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f"{job_id}.", suffix=".jsonl", dir=current.embedding_batches_dir
    )
    os.close(temp_fd)
    input_path = Path(temp_name)
    client = None
    uploaded_file_id: str | None = None
    try:
        _write_batch_jsonl(
            input_path,
            copied_candidates,
            rows,
            current.embedding_model,
            current,
        )
        client = _create_batch_client(current)
        state["status"] = "uploading"
        _write_batch_state(state_path, state)
        with input_path.open("rb") as input_file:
            uploaded = client.files.create(file=input_file, purpose="batch")
        uploaded_file_id = uploaded.id
        state["input_file_id"] = uploaded_file_id
        state["status"] = "uploaded"
        _write_batch_state(state_path, state)
        batch = client.batches.create(
            input_file_id=uploaded_file_id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"application": "amrb", "local_job_id": job_id},
        )
        state["batch_id"] = batch.id
        state["status"] = "submitted"
        _write_batch_state(state_path, state)
    except Exception as exc:
        if client is not None and uploaded_file_id:
            try:
                client.files.delete(uploaded_file_id)
            except Exception:
                pass
        failures = [
            _failure(candidate["source"], exc, candidate["copied"], current)
            for candidate in copied_candidates
        ]
        result.failed.extend(failures)
        result = _merge_paper_add_results(result, PaperAddResult(failed=failures))
        _save_batch_result(state_path, state, result, "submission_failed")
        return result
    finally:
        input_path.unlink(missing_ok=True)
        if client is not None:
            client.close()

    if on_submitted:
        on_submitted(job_id, str(batch.id))
    return wait_paper_batch(
        job_id,
        settings=current,
        qdrant_client=qdrant_client,
        on_status=on_status,
    )

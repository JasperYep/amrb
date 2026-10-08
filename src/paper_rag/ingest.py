import hashlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import (
    BaseNode,
    Document,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
)
from llama_index.readers.file import PyMuPDFReader
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    VectorParams,
)

from paper_rag.config import (
    Settings,
    claim_qdrant_collection,
    create_embedding_model,
    create_qdrant_client,
    create_vector_store,
    get_settings,
    stable_paper_id,
)
from paper_rag.models import IngestResult

logger = logging.getLogger(__name__)
Manifest = dict[str, dict[str, Any]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> Manifest:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取索引 manifest：{path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"索引 manifest 格式无效：{path}")
    return data


def write_manifest(path: Path, manifest: Manifest) -> None:
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
            json.dump(manifest, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def _pdf_title(path: Path) -> str:
    import pymupdf

    with pymupdf.open(path) as pdf:
        title = str((pdf.metadata or {}).get("title") or "").strip()
    return title if title and title.lower() not in {"untitled", "unknown"} else path.stem


def _page_number(metadata: dict[str, Any], fallback: int) -> int:
    for key in ("page_number", "page", "page_label", "source"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page > 0:
            return page
        if key == "page_number" and page == 0:
            return 1
    return fallback


def _paper_nodes(path: Path, paper_id: str, settings: Settings) -> tuple[list[BaseNode], str]:
    reader = PyMuPDFReader()
    source_pages = reader.load_data(file_path=path)
    title = _pdf_title(path)
    pages: list[Document] = []
    for index, source_page in enumerate(source_pages, start=1):
        metadata = {
            "paper_id": paper_id,
            "workspace_id": settings.workspace_id,
            "file_name": path.name,
            "title": title,
            "page_number": _page_number(source_page.metadata, index),
        }
        pages.append(
            Document(
                text=source_page.text,
                metadata=metadata,
                excluded_embed_metadata_keys=list(metadata),
                excluded_llm_metadata_keys=list(metadata),
            )
        )

    splitter = SentenceSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(pages)
    for index, node in enumerate(nodes):
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        if not text.strip():
            continue
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        page = int(node.metadata["page_number"])
        node.metadata["content_hash"] = content_hash
        node.id_ = str(uuid5(NAMESPACE_URL, f"amrb:{paper_id}:{page}:{index}"))
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=paper_id)
    return [node for node in nodes if node.get_content(metadata_mode=MetadataMode.NONE).strip()], title


def _index_pdf(
    path: Path,
    paper_id: str,
    settings: Settings,
    client: Any,
    vector_store: Any,
    embedding_model: Any,
    expected_dimension: int | None = None,
) -> tuple[str, int, int]:
    nodes, title = _paper_nodes(path, paper_id, settings)
    if not nodes:
        raise ValueError(f"PDF 中没有可检索的正文文本：{path.name}")
    texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]
    try:
        embeddings = embedding_model.get_text_embedding_batch(texts)
    except Exception as exc:
        if getattr(exc, "status_code", None) in {401, 403}:
            raise RuntimeError(
                "Embedding API authentication or access failed "
                f"(HTTP {exc.status_code}) while indexing {path.name}. "
                "Check that EMBEDDING_API_KEY is valid for EMBEDDING_BASE_URL."
            ) from exc
        if getattr(exc, "status_code", None) == 404:
            raise RuntimeError(
                f"Embedding API returned HTTP 404 while indexing {path.name}. "
                "Check that EMBEDDING_BASE_URL supports OpenAI-compatible embeddings "
                "and EMBEDDING_MODEL names an available embedding model."
            ) from exc
        raise RuntimeError(f"Embedding failed for {path.name}: {exc}") from exc
    if len(embeddings) != len(nodes):
        raise ValueError(f"Embedding 返回数量不匹配：{path.name}")
    dimensions = {len(vector) for vector in embeddings}
    if len(dimensions) != 1:
        raise ValueError(f"Embedding 返回向量维度不一致：{path.name}")
    dimension = dimensions.pop()
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(
            f"Embedding returned dimension {dimension}, but this workspace index requires {expected_dimension}. "
            "No vectors were written. Explicitly rebuild the workspace index to change dimensions."
        )
    for node, embedding in zip(nodes, embeddings, strict=True):
        node.embedding = embedding
    exists, _, _ = claim_qdrant_collection(client, settings)
    if not exists:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            metadata={"amrb_workspace_id": settings.workspace_id},
        )
        if hasattr(vector_store, "_collection_initialized"):
            vector_store._collection_initialized = True
    vector_store.add(nodes)
    active_ids = {node.id_ for node in nodes}
    offset = None
    stale_ids: list[str | int] = []
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
            limit=1000,
            with_payload=False,
            with_vectors=False,
            offset=offset,
        )
        stale_ids.extend(point.id for point in points if point.id not in active_ids)
        if offset is None:
            break
    if stale_ids:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=PointIdsList(points=stale_ids),
            wait=True,
        )
    return title, len(nodes), dimension


def index_profile(settings: Settings, dimension: int) -> dict[str, Any]:
    return {
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "dimension": dimension,
        "chunk_size": settings.rag_chunk_size,
        "chunk_overlap": settings.rag_chunk_overlap,
    }


def validate_index_compatibility(manifest: Manifest, settings: Settings) -> None:
    entries = [entry for entry in manifest.values() if entry.get("paper_id")]
    if not entries:
        return
    unverified = [entry for entry in entries if not isinstance(entry.get("index_config"), dict)]
    if unverified and not settings.legacy_index_accepted:
        raise ValueError(
            "This index predates embedding and chunk metadata and is unverified. "
            "Run `amrb setup` and explicitly confirm its existing configuration before searching or indexing."
        )
    expected = {
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "chunk_size": settings.rag_chunk_size,
        "chunk_overlap": settings.rag_chunk_overlap,
    }
    recorded_dimensions = {
        int(profile["dimension"])
        for entry in entries
        if isinstance((profile := entry.get("index_config")), dict) and profile.get("dimension")
    }
    if len(recorded_dimensions) > 1:
        raise ValueError(
            "The workspace contains papers indexed with different vector dimensions. "
            "Queries and mixed writes are blocked until the index is explicitly rebuilt."
        )
    if settings.embedding_dimension and recorded_dimensions and recorded_dimensions != {settings.embedding_dimension}:
        raise ValueError(
            "The workspace vector dimension does not match its recorded paper index. "
            "Queries and mixed writes are blocked until the index is explicitly rebuilt."
        )
    for entry in entries:
        profile = entry.get("index_config")
        if not isinstance(profile, dict):
            continue
        mismatch = {
            key: (profile.get(key), value)
            for key, value in expected.items()
            if profile.get(key) != value
        }
        if mismatch:
            fields = ", ".join(sorted(mismatch))
            raise ValueError(
                f"Index configuration mismatch for paper {entry.get('paper_id')}: {fields}. "
                "Queries and mixed writes are blocked; explicitly rebuild this workspace index."
            )
        if settings.embedding_dimension and profile.get("dimension") != settings.embedding_dimension:
            raise ValueError(
                "Embedding vector dimension does not match the workspace index profile. "
                "Queries and mixed writes are blocked; explicitly rebuild this workspace index."
            )


def _delete_paper(vector_store: Any, paper_id: str, client: Any, collection: str) -> None:
    if client.collection_exists(collection):
        vector_store.delete(ref_doc_id=paper_id)


def _index_changed_files(
    pdfs: list[Path],
    current_hashes: dict[str, str],
    desired_ids: set[str],
    manifest: Manifest,
    settings: Settings,
    embedding_model: Any | None,
    qdrant_client: Any | None,
) -> tuple[int, int, int]:
    changes = [
        (path, path.relative_to(settings.workspace_root).as_posix(), current_hashes[
            path.relative_to(settings.workspace_root).as_posix()
        ])
        for path in pdfs
        if manifest.get(path.relative_to(settings.workspace_root).as_posix(), {}).get("sha256")
        != current_hashes[path.relative_to(settings.workspace_root).as_posix()]
    ]
    skipped = len(pdfs) - len(changes)
    if not changes:
        return 0, skipped, 0

    client = qdrant_client or create_qdrant_client(settings)
    owns_client = qdrant_client is None
    try:
        vector_store = create_vector_store(client, settings)
        active_embedding = embedding_model or create_embedding_model(settings)
        chunks_indexed = 0
        for path, relative_path, file_hash in changes:
            previous = manifest.get(relative_path)
            known_dimensions = {
                int(profile["dimension"])
                for item in manifest.values()
                if isinstance((profile := item.get("index_config")), dict) and profile.get("dimension")
            }
            expected_dimension = settings.embedding_dimension
            if expected_dimension is None and len(known_dimensions) == 1:
                expected_dimension = next(iter(known_dimensions))
            paper_id = str(previous.get("paper_id")) if previous and previous.get("paper_id") else stable_paper_id(
                settings.workspace_id, relative_path
            )
            title, chunk_count, dimension = _index_pdf(
                path,
                paper_id,
                settings,
                client,
                vector_store,
                active_embedding,
                expected_dimension,
            )
            old_paper_id = previous.get("paper_id") if previous else None
            if old_paper_id and old_paper_id != paper_id and old_paper_id not in desired_ids:
                _delete_paper(vector_store, old_paper_id, client, settings.qdrant_collection)
            manifest[relative_path] = {
                "sha256": file_hash,
                "paper_id": paper_id,
                "indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "title": title,
                "chunk_count": chunk_count,
                "index_config": index_profile(settings, dimension),
            }
            write_manifest(settings.manifest_path, manifest)
            chunks_indexed += chunk_count
            logger.info("Indexed %s (%d chunks)", path.name, chunk_count)
        return len(changes), skipped, chunks_indexed
    finally:
        if owns_client:
            client.close()


def _remove_missing_files(
    current_hashes: dict[str, str],
    desired_ids: set[str],
    manifest: Manifest,
    settings: Settings,
    qdrant_client: Any | None,
) -> int:
    missing_paths = sorted(set(manifest) - set(current_hashes))
    if not missing_paths:
        return 0
    orphaned_ids = {
        str(manifest[path]["paper_id"])
        for path in missing_paths
        if manifest[path].get("paper_id") not in desired_ids
    }
    client = qdrant_client
    owns_client = False
    try:
        vector_store = None
        if orphaned_ids:
            if client is None:
                client = create_qdrant_client(settings)
                owns_client = True
            vector_store = create_vector_store(client, settings)
        for relative_path in missing_paths:
            entry = manifest[relative_path]
            paper_id = entry.get("paper_id")
            if paper_id in orphaned_ids and vector_store is not None:
                _delete_paper(vector_store, str(paper_id), client, settings.qdrant_collection)
                orphaned_ids.discard(str(paper_id))
            del manifest[relative_path]
            write_manifest(settings.manifest_path, manifest)
        return len(missing_paths)
    finally:
        if owns_client and client is not None:
            client.close()


def ingest_papers(
    settings: Settings | None = None,
    embedding_model: Any | None = None,
    qdrant_client: Any | None = None,
    paths: list[Path] | None = None,
) -> IngestResult:
    current = settings or get_settings()
    current.papers_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(current.manifest_path)
    validate_index_compatibility(manifest, current)
    full_sync = paths is None
    if paths is None:
        pdfs = sorted(current.papers_dir.glob("*.pdf"))
    else:
        papers_root = current.papers_dir.resolve()
        pdfs = sorted({Path(path).expanduser().resolve() for path in paths})
        for path in pdfs:
            if path.parent != papers_root or path.suffix.lower() != ".pdf":
                raise ValueError("定向 ingest 只接受 papers/ 第一层中的 PDF 文件。")
            if not path.is_file():
                raise ValueError(f"PDF 文件不存在：{path}")
    current_hashes = {
        path.relative_to(current.workspace_root).as_posix(): sha256_file(path)
        for path in pdfs
    }
    if full_sync:
        desired_ids = {
            str(manifest.get(path, {}).get("paper_id") or stable_paper_id(current.workspace_id, path))
            for path in current_hashes
        }
    else:
        targeted_paths = set(current_hashes)
        desired_ids = {
            str(entry["paper_id"])
            for relative_path, entry in manifest.items()
            if relative_path not in targeted_paths and entry.get("paper_id")
        }
        desired_ids.update(
            str(manifest.get(path, {}).get("paper_id") or stable_paper_id(current.workspace_id, path))
            for path in current_hashes
        )
    indexed, skipped, chunks = _index_changed_files(
        pdfs,
        current_hashes,
        desired_ids,
        manifest,
        current,
        embedding_model,
        qdrant_client,
    )
    removed = (
        _remove_missing_files(
            current_hashes,
            desired_ids,
            manifest,
            current,
            qdrant_client,
        )
        if full_sync
        else 0
    )
    return IngestResult(
        discovered=len(pdfs),
        indexed=indexed,
        skipped=skipped,
        removed=removed,
        chunks=chunks,
    )

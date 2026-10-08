from atexit import register
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from paper_rag.config import (
    Settings,
    create_embedding_model,
    create_qdrant_client,
    create_vector_store,
    ensure_paper_id_payload_index,
    get_settings,
)
from paper_rag.ingest import load_manifest
from paper_rag.models import PaperBrief, PaperHit, PaperPage, PaperRAGStatus


@dataclass
class _SearchResources:
    client: Any
    store: Any
    embedding_model: Any
    index: Any


_RESOURCE_LOCK = RLock()
_DEFAULT_EMBEDDING_MODEL: Any | None = None
_DEFAULT_REMOTE_SEARCH_RESOURCES: _SearchResources | None = None


def _default_embedding_model() -> Any:
    global _DEFAULT_EMBEDDING_MODEL
    with _RESOURCE_LOCK:
        if _DEFAULT_EMBEDDING_MODEL is None:
            _DEFAULT_EMBEDDING_MODEL = create_embedding_model(get_settings())
        return _DEFAULT_EMBEDDING_MODEL


def _default_remote_search_resources(settings: Settings) -> _SearchResources:
    global _DEFAULT_REMOTE_SEARCH_RESOURCES
    with _RESOURCE_LOCK:
        if _DEFAULT_REMOTE_SEARCH_RESOURCES is None:
            client = create_qdrant_client(settings)
            try:
                if not client.collection_exists(settings.qdrant_collection):
                    raise ValueError(
                        "Paper RAG 尚未建立索引。\n\n"
                        "请先运行：\n\nuv run paper-rag ingest"
                    )
                store = create_vector_store(client, settings)
                embedding_model = _default_embedding_model()
                index = VectorStoreIndex.from_vector_store(
                    vector_store=store,
                    embed_model=embedding_model,
                )
            except Exception:
                client.close()
                raise
            _DEFAULT_REMOTE_SEARCH_RESOURCES = _SearchResources(
                client=client,
                store=store,
                embedding_model=embedding_model,
                index=index,
            )
        return _DEFAULT_REMOTE_SEARCH_RESOURCES


def _close_default_remote_search_client() -> None:
    with _RESOURCE_LOCK:
        if _DEFAULT_REMOTE_SEARCH_RESOURCES is not None:
            _DEFAULT_REMOTE_SEARCH_RESOURCES.client.close()


register(_close_default_remote_search_client)


def list_papers(settings: Settings | None = None) -> list[PaperBrief]:
    current = settings or get_settings()
    manifest = load_manifest(current.manifest_path)
    papers = [
        PaperBrief(
            paper_id=str(entry["paper_id"]),
            title=str(entry.get("title") or Path(path).stem),
            file_name=Path(path).name,
        )
        for path, entry in sorted(manifest.items())
        if entry.get("paper_id")
    ]
    return papers


def _match_paper_id(paper: str, settings: Settings) -> str | None:
    query = paper.casefold().strip()
    candidates: dict[str, str] = {}
    for brief in list_papers(settings):
        aliases = {
            brief.paper_id.casefold(),
            brief.title.casefold(),
            brief.file_name.casefold(),
            Path(brief.file_name).stem.casefold(),
        }
        if query in aliases:
            return brief.paper_id
        if any(query in alias for alias in aliases):
            candidates[brief.paper_id] = brief.file_name
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        names = ", ".join(sorted(candidates.values()))
        raise ValueError(f"paper 参数匹配到多篇论文，请使用文件名或 paper_id：{names}")
    return None


def search_papers(
    query: str,
    top_k: int | None = None,
    paper: str | None = None,
    settings: Settings | None = None,
    qdrant_client: Any | None = None,
    embedding_model: Any | None = None,
) -> list[PaperHit]:
    current = settings or get_settings()
    if not query.strip():
        raise ValueError("query 不能为空。")
    count = top_k if top_k is not None else current.rag_top_k
    if count < 1 or count > 100:
        raise ValueError("top_k 必须在 1 到 100 之间。")
    if paper:
        paper_id = _match_paper_id(paper, current)
        if paper_id is None:
            return []
    else:
        paper_id = None

    cached_resources = None
    # Keep remote connections open across MCP calls. Embedded Qdrant holds a
    # storage lock, so its client must stay request-scoped while CLI commands
    # may access the same local database.
    if (
        settings is None
        and qdrant_client is None
        and embedding_model is None
        and current.qdrant_url
    ):
        cached_resources = _default_remote_search_resources(current)

    if cached_resources is not None:
        client = cached_resources.client
    elif qdrant_client is not None:
        client = qdrant_client
    else:
        client = create_qdrant_client(current)
    close_client = cached_resources is None and qdrant_client is None
    try:
        if cached_resources is None:
            if not client.collection_exists(current.qdrant_collection):
                raise ValueError(
                    "Paper RAG 尚未建立索引。\n\n"
                    "请先运行：\n\nuv run paper-rag ingest"
                )
            store = create_vector_store(client, current)
        else:
            store = cached_resources.store
        if paper_id is not None:
            ensure_paper_id_payload_index(client, current)
        if cached_resources is None:
            active_embedding = embedding_model or (
                _default_embedding_model()
                if settings is None
                else create_embedding_model(current)
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store=store,
                embed_model=active_embedding,
            )
        else:
            index = cached_resources.index
        filters = (
            MetadataFilters(filters=[MetadataFilter(key="paper_id", value=paper_id)])
            if paper_id
            else None
        )
        retriever = index.as_retriever(similarity_top_k=count, filters=filters)
        hits = []
        for result in retriever.retrieve(query):
            metadata = result.node.metadata
            page = metadata.get("page_number")
            if page is None:
                raise ValueError("索引节点缺少页码；请重新运行 paper-rag ingest。")
            hits.append(
                PaperHit(
                    paper_id=str(metadata["paper_id"]),
                    title=str(metadata.get("title") or metadata["file_name"]),
                    file_name=str(metadata["file_name"]),
                    page=int(page),
                    score=float(result.score or 0.0),
                    text=result.node.get_content(metadata_mode=MetadataMode.NONE),
                )
            )
        return hits
    finally:
        if close_client:
            client.close()


def get_paper_pages(
    paper_id: str,
    start_page: int,
    end_page: int | None = None,
    settings: Settings | None = None,
) -> list[PaperPage]:
    import pymupdf

    current = settings or get_settings()
    last_page = end_page if end_page is not None else start_page
    if start_page < 1 or last_page < start_page:
        raise ValueError("页码范围无效；页码从 1 开始。")
    if last_page - start_page + 1 > 10:
        raise ValueError("单次最多读取 10 页。")

    matched = next(
        (
            (path, entry)
            for path, entry in load_manifest(current.manifest_path).items()
            if entry.get("paper_id") == paper_id
        ),
        None,
    )
    if matched is None:
        raise ValueError(f"未找到 paper_id：{paper_id}")
    relative_path, entry = matched
    papers_root = current.papers_dir.resolve()
    source_path = (current.workspace_root / relative_path).resolve()
    if not source_path.is_relative_to(papers_root) or source_path.suffix.lower() != ".pdf":
        raise ValueError("索引 manifest 中的论文路径无效。")
    if not source_path.is_file():
        raise ValueError(f"论文文件不存在：{source_path.name}；请先运行 paper-rag ingest。")

    title = str(entry.get("title") or source_path.stem)
    pages: list[PaperPage] = []
    with pymupdf.open(source_path) as pdf:
        if start_page > pdf.page_count:
            raise ValueError(f"论文只有 {pdf.page_count} 页，起始页超出范围。")
        for number in range(start_page, min(last_page, pdf.page_count) + 1):
            citation = f"{title}, p. {number}"
            pages.append(
                PaperPage(
                    paper_id=paper_id,
                    title=title,
                    file_name=source_path.name,
                    page=number,
                    citation=citation,
                    text=pdf[number - 1].get_text("text"),
                )
            )
    return pages


def get_status(
    settings: Settings | None = None,
    qdrant_client: Any | None = None,
) -> PaperRAGStatus:
    current = settings or get_settings()
    manifest = load_manifest(current.manifest_path)
    latest = max(
        (str(entry["indexed_at"]) for entry in manifest.values() if entry.get("indexed_at")),
        default=None,
    )
    client = qdrant_client if qdrant_client is not None else create_qdrant_client(current)
    close_client = qdrant_client is None
    try:
        chunks = (
            client.count(collection_name=current.qdrant_collection, exact=True).count
            if client.collection_exists(current.qdrant_collection)
            else 0
        )
    finally:
        if close_client:
            client.close()
    return PaperRAGStatus(
        papers=len(manifest),
        chunks=chunks,
        vector_backend="server" if current.qdrant_url else "local",
        collection=current.qdrant_collection,
        last_ingest=latest,
    )

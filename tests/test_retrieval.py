import json

import pytest
from qdrant_client.models import Distance, VectorParams

from paper_rag.config import create_qdrant_client
from paper_rag.ingest import ingest_papers
from paper_rag.retrieval import get_paper_pages, list_papers, search_papers


def test_retrieval_returns_expected_paper_page_and_top_k(workspace, make_pdf):
    settings, embedding = workspace
    pdf = settings.papers_dir / "fixture.pdf"
    make_pdf(pdf)
    client = create_qdrant_client(settings)
    try:
        ingest_papers(settings, embedding, client)
        hits = search_papers(
            "How is stored information modified?",
            top_k=8,
            settings=settings,
            qdrant_client=client,
            embedding_model=embedding,
        )
        one_hit = search_papers(
            "How is stored information modified?",
            top_k=1,
            settings=settings,
            qdrant_client=client,
            embedding_model=embedding,
        )
        paper_hits = search_papers(
            "How is stored information modified?",
            top_k=8,
            paper="fixture.pdf",
            settings=settings,
            qdrant_client=client,
            embedding_model=embedding,
        )
        papers = list_papers(settings)
    finally:
        client.close()

    assert hits[0].file_name == "fixture.pdf"
    assert hits[0].page == 3
    assert len(hits) == 3
    assert len(one_hit) == 1
    assert [hit.paper_id for hit in paper_hits] == [hit.paper_id for hit in hits]
    assert [(item.title, item.file_name) for item in papers] == [
        ("Fixture Research", "fixture.pdf")
    ]

    page = get_paper_pages(hits[0].paper_id, 2, 3, settings)
    assert [item.page for item in page] == [2, 3]
    assert page[-1].text.strip() == "Memory update modifies previously retained information."


def test_get_paper_pages_enforces_ten_page_limit(workspace):
    settings, _ = workspace
    try:
        get_paper_pages("unknown", 1, 11, settings)
    except ValueError as exc:
        assert "最多读取 10 页" in str(exc)
    else:
        raise AssertionError("expected page limit validation")


def test_paper_inventory_marks_unindexed_and_changed_sources(workspace, make_pdf):
    settings, embedding = workspace
    unindexed = settings.papers_dir / "unindexed.pdf"
    make_pdf(unindexed, "Pending Paper")
    inventory = list_papers(settings)
    assert inventory[0].title == "Pending Paper"
    assert inventory[0].index_status == "not_indexed"
    assert inventory[0].paper_id is None

    client = create_qdrant_client(settings)
    try:
        ingest_papers(settings, embedding, client)
    finally:
        client.close()
    unindexed.write_bytes(unindexed.read_bytes() + b"changed")

    assert list_papers(settings)[0].index_status == "source_changed"


def test_legacy_index_is_reported_and_search_requires_confirmation(workspace, make_pdf):
    settings, embedding = workspace
    make_pdf(settings.papers_dir / "legacy.pdf")
    client = create_qdrant_client(settings)
    try:
        ingest_papers(settings, embedding, client)
    finally:
        client.close()
    manifest = json.loads(settings.manifest_path.read_text())
    manifest["papers/legacy.pdf"].pop("index_config")
    settings.manifest_path.write_text(json.dumps(manifest))

    from paper_rag.retrieval import list_papers

    assert list_papers(settings)[0].index_status == "unverified"
    try:
        search_papers("legacy", settings=settings, embedding_model=embedding)
    except ValueError as exc:
        assert "unverified" in str(exc)
        assert "amrb setup" in str(exc)
    else:
        raise AssertionError("unverified legacy index must require explicit confirmation")


def test_search_rejects_a_collection_owned_by_another_workspace(workspace):
    settings, embedding = workspace
    client = create_qdrant_client(settings)
    try:
        client.create_collection(
            settings.qdrant_collection,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
            metadata={"amrb_workspace_id": "other-workspace"},
        )
        with pytest.raises(ValueError, match="belongs to workspace other-workspace"):
            search_papers(
                "memory",
                settings=settings,
                qdrant_client=client,
                embedding_model=embedding,
            )
    finally:
        client.close()

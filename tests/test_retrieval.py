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

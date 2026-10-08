from paper_rag.config import create_qdrant_client
from paper_rag.ingest import load_manifest
from paper_rag.ops import add_papers, list_paper_records, remove_paper, sync_papers


def test_add_paper_and_duplicate_do_not_reembed(workspace, make_pdf, tmp_path):
    settings, embedding = workspace
    source = tmp_path / "downloads" / "paper.pdf"
    source.parent.mkdir()
    make_pdf(source, "Fixture Research")
    client = create_qdrant_client(settings)
    try:
        result = add_papers([source], settings, embedding, client)
        calls_after_add = embedding._batch_calls
        duplicate_source = tmp_path / "another-copy.pdf"
        duplicate_source.write_bytes(source.read_bytes())
        duplicate = add_papers([duplicate_source], settings, embedding, client)
        count = client.count(settings.qdrant_collection, exact=True).count
    finally:
        client.close()

    assert len(result.added) == 1
    assert result.added[0].file_name == "paper.pdf"
    assert result.added[0].pages == 3
    assert not result.failed
    assert len(duplicate.skipped) == 1
    assert embedding._batch_calls == calls_after_add
    assert count == 3
    assert (settings.papers_dir / "paper.pdf").is_file()


def test_sync_unchanged_and_changed_pdf(workspace, make_pdf):
    settings, embedding = workspace
    paper = settings.papers_dir / "paper.pdf"
    make_pdf(paper, "Original")
    client = create_qdrant_client(settings)
    try:
        first = sync_papers(settings, embedding, client)
        first_record = list_paper_records(settings)[0]
        unchanged = sync_papers(settings, embedding, client)
        make_pdf(paper, "Updated title")
        updated = sync_papers(settings, embedding, client)
        updated_record = list_paper_records(settings)[0]
        count = client.count(settings.qdrant_collection, exact=True).count
    finally:
        client.close()

    assert first.added == 1
    assert first.chunks == 3
    assert unchanged.unchanged == 1
    assert unchanged.added == unchanged.updated == unchanged.removed == 0
    assert updated.updated == 1
    assert updated.papers == 1
    assert count == 3
    assert updated_record.paper_id == first_record.paper_id


def test_add_failure_leaves_file_retriable_by_sync(workspace, make_pdf, tmp_path):
    settings, embedding = workspace
    source = tmp_path / "download.pdf"
    make_pdf(source)
    client = create_qdrant_client(settings)

    class FailingEmbedding:
        def get_text_embedding_batch(self, texts):
            raise RuntimeError("provider unavailable")

    try:
        failed = add_papers([source], settings, FailingEmbedding(), client)
        manifest_after_failure = load_manifest(settings.manifest_path)
        failed_inventory = list_paper_records(settings)
        retried = sync_papers(settings, embedding, client)
        recovered_inventory = list_paper_records(settings)
        count = client.count(settings.qdrant_collection, exact=True).count
    finally:
        client.close()

    assert not failed.added
    assert len(failed.failed) == 1
    assert failed.failed[0].copied is True
    assert "provider unavailable" in failed.failed[0].message
    assert failed_inventory[0].index_status == "failed"
    assert recovered_inventory[0].index_status == "indexed"
    assert not manifest_after_failure
    assert retried.error is None
    assert retried.added == 1
    assert count == 3


def test_remove_deletes_pdf_manifest_and_vectors(workspace, make_pdf):
    settings, embedding = workspace
    paper = settings.papers_dir / "paper.pdf"
    make_pdf(paper)
    client = create_qdrant_client(settings)
    try:
        sync_papers(settings, embedding, client)
        removed = remove_paper("paper.pdf", settings, client)
        count = client.count(settings.qdrant_collection, exact=True).count
    finally:
        client.close()

    assert removed is not None
    assert removed.removed.file_name == "paper.pdf"
    assert not paper.exists()
    assert not load_manifest(settings.manifest_path)
    assert count == 0


def test_remove_ambiguous_title_fails_without_deleting(workspace, make_pdf):
    settings, embedding = workspace
    make_pdf(settings.papers_dir / "study-a.pdf", "Shared Study")
    make_pdf(settings.papers_dir / "study-b.pdf", "Shared Study")
    client = create_qdrant_client(settings)
    try:
        sync_papers(settings, embedding, client)
        try:
            remove_paper("Shared Study", settings, client)
        except ValueError as exc:
            assert "多篇论文" in str(exc)
        else:
            raise AssertionError("expected ambiguous paper selection to fail")
        remaining = sorted(path.name for path in settings.papers_dir.glob("*.pdf"))
    finally:
        client.close()
    assert remaining == ["study-a.pdf", "study-b.pdf"]


def test_list_paper_records_reports_pages(workspace, make_pdf):
    settings, embedding = workspace
    make_pdf(settings.papers_dir / "paper.pdf")
    client = create_qdrant_client(settings)
    try:
        sync_papers(settings, embedding, client)
    finally:
        client.close()
    records = list_paper_records(settings)
    assert len(records) == 1
    assert records[0].pages == 3

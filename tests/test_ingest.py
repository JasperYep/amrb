import json

from paper_rag.config import create_qdrant_client
from paper_rag.ingest import ingest_papers, sha256_file


def test_ingest_indexes_fixture_and_skips_unchanged(workspace, make_pdf):
    settings, embedding = workspace
    pdf = settings.papers_dir / "fixture.pdf"
    make_pdf(pdf)
    client = create_qdrant_client(settings)
    try:
        first = ingest_papers(settings, embedding, client)
        manifest_path = settings.manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["papers/fixture.pdf"]
        count_after_first = client.count(settings.qdrant_collection, exact=True).count
        embedding_calls = embedding._batch_calls

        second = ingest_papers(settings, embedding, client)
        count_after_second = client.count(settings.qdrant_collection, exact=True).count
    finally:
        client.close()

    assert first.indexed == 1
    assert first.chunks == 3
    assert entry["sha256"] == sha256_file(pdf)
    assert entry["paper_id"] == entry["sha256"][:16]
    assert entry["title"] == "Fixture Research"
    assert second.indexed == 0
    assert second.skipped == 1
    assert count_after_first == count_after_second == 3
    assert embedding._batch_calls == embedding_calls


def test_failed_reindex_preserves_manifest_and_old_vectors(workspace, make_pdf):
    settings, embedding = workspace
    pdf = settings.papers_dir / "fixture.pdf"
    make_pdf(pdf)
    client = create_qdrant_client(settings)
    try:
        ingest_papers(settings, embedding, client)
        old_manifest = settings.manifest_path.read_text(encoding="utf-8")
        old_count = client.count(settings.qdrant_collection, exact=True).count
        pdf.write_bytes(pdf.read_bytes() + b" ")

        class FailingEmbedding:
            def get_text_embedding_batch(self, texts):
                raise RuntimeError("fixture embedding failure")

        try:
            ingest_papers(settings, FailingEmbedding(), client)
        except RuntimeError as exc:
            assert "fixture embedding failure" in str(exc)
        else:
            raise AssertionError("expected embedding failure")

        new_manifest = settings.manifest_path.read_text(encoding="utf-8")
        new_count = client.count(settings.qdrant_collection, exact=True).count
    finally:
        client.close()

    assert new_manifest == old_manifest
    assert new_count == old_count == 3


def test_ingest_removes_deleted_paper_and_vectors(workspace, make_pdf):
    settings, embedding = workspace
    pdf = settings.papers_dir / "fixture.pdf"
    make_pdf(pdf)
    client = create_qdrant_client(settings)
    try:
        ingest_papers(settings, embedding, client)
        pdf.unlink()
        result = ingest_papers(settings, embedding, client)
        count = client.count(settings.qdrant_collection, exact=True).count
    finally:
        client.close()

    assert result.removed == 1
    assert settings.manifest_path.read_text(encoding="utf-8").strip() == "{}"
    assert count == 0

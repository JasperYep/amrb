import json

import pytest

from paper_rag.config import create_qdrant_client, stable_paper_id
from paper_rag.ingest import ingest_papers, sha256_file, validate_index_compatibility


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
        collection_metadata = client.get_collection(settings.qdrant_collection).config.metadata
        embedding_calls = embedding._batch_calls

        second = ingest_papers(settings, embedding, client)
        count_after_second = client.count(settings.qdrant_collection, exact=True).count
    finally:
        client.close()

    assert first.indexed == 1
    assert first.chunks == 3
    assert entry["sha256"] == sha256_file(pdf)
    assert entry["paper_id"] == stable_paper_id(settings.workspace_id, "papers/fixture.pdf")
    assert entry["index_config"] == {
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "dimension": 4,
        "chunk_size": settings.rag_chunk_size,
        "chunk_overlap": settings.rag_chunk_overlap,
    }
    assert entry["title"] == "Fixture Research"
    assert second.indexed == 0
    assert second.skipped == 1
    assert count_after_first == count_after_second == 3
    assert collection_metadata["amrb_workspace_id"] == settings.workspace_id
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


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("embedding_model", "different-model", "model"),
        ("rag_chunk_size", 999, "chunk_size"),
        ("rag_chunk_overlap", 25, "chunk_overlap"),
        ("embedding_dimension", 8, "dimension"),
    ],
)
def test_index_profile_mismatch_blocks_mixed_reads_and_writes(workspace, field, value, expected):
    settings, _ = workspace
    entry = {
        "paper_id": "paper-id",
        "index_config": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimension": 4,
            "chunk_size": settings.rag_chunk_size,
            "chunk_overlap": settings.rag_chunk_overlap,
        },
    }
    changed = settings.model_copy(update={field: value})

    with pytest.raises(ValueError, match=expected):
        validate_index_compatibility({"papers/example.pdf": entry}, changed)


def test_mixed_recorded_dimensions_block_index_compatibility(workspace):
    settings, _ = workspace
    first = {
        "paper_id": "paper-one",
        "index_config": {"dimension": 4, "model": settings.embedding_model},
    }
    second = {
        "paper_id": "paper-two",
        "index_config": {"dimension": 8, "model": settings.embedding_model},
    }

    with pytest.raises(ValueError, match="different vector dimensions"):
        validate_index_compatibility(
            {"papers/one.pdf": first, "papers/two.pdf": second},
            settings,
        )


def test_embedding_404_has_actionable_configuration_hint(workspace, make_pdf):
    settings, _ = workspace
    make_pdf(settings.papers_dir / "fixture.pdf")

    class MissingEmbeddingEndpoint(RuntimeError):
        status_code = 404

    class FailingEmbedding:
        def get_text_embedding_batch(self, texts):
            raise MissingEmbeddingEndpoint("Error code: 404")

    try:
        ingest_papers(settings, FailingEmbedding())
    except RuntimeError as exc:
        assert "EMBEDDING_BASE_URL" in str(exc)
        assert "EMBEDDING_MODEL" in str(exc)
    else:
        raise AssertionError("expected missing embedding endpoint error")


def test_embedding_401_has_actionable_configuration_hint(workspace, make_pdf):
    settings, _ = workspace
    make_pdf(settings.papers_dir / "fixture.pdf")

    class InvalidKey(RuntimeError):
        status_code = 401

    class FailingEmbedding:
        def get_text_embedding_batch(self, texts):
            raise InvalidKey("Error code: 401")

    try:
        ingest_papers(settings, FailingEmbedding())
    except RuntimeError as exc:
        assert "EMBEDDING_API_KEY" in str(exc)
        assert "EMBEDDING_BASE_URL" in str(exc)
    else:
        raise AssertionError("expected invalid embedding key error")


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

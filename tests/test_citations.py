from paper_rag.models import PaperHit


def test_paper_hit_has_required_fields_and_readable_citation():
    hit = PaperHit(
        paper_id="e8a1f6b3d7c9a012",
        title="Hindsight: Learning from Experience",
        file_name="hindsight.pdf",
        page=5,
        score=0.892,
        text="Memory consolidation updates long-term observations.",
    )

    assert hit.paper_id
    assert hit.title or hit.file_name
    assert hit.page == 5
    assert hit.text
    assert hit.citation == "Hindsight: Learning from Experience, p. 5"
    assert hit.model_dump()["citation"] == hit.citation

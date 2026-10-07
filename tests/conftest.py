from pathlib import Path

import pymupdf
import pytest
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

from paper_rag.config import Settings


class FixtureEmbedding(BaseEmbedding):
    _batch_calls: int = PrivateAttr(default=0)

    def __init__(self) -> None:
        super().__init__(embed_dim=4)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        concepts = (
            ("update", "modif", "chang"),
            ("retriev", "select", "relevant"),
            ("store", "across", "interactions"),
            ("evaluation", "benchmark", "measure"),
        )
        return [float(any(word in lowered for word in group)) for group in concepts]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    def get_text_embedding_batch(self, texts, **kwargs):
        self._batch_calls += 1
        return super().get_text_embedding_batch(texts, **kwargs)


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Settings, FixtureEmbedding]:
    (tmp_path / "papers").mkdir()
    return (
        Settings(_env_file=None, workspace_root=tmp_path),
        FixtureEmbedding(),
    )


def create_fixture_pdf(path: Path, title: str = "Fixture Research") -> None:
    pdf = pymupdf.open()
    contents = [
        "Agent memory systems store information across interactions.",
        "Memory retrieval selects relevant stored information.",
        "Memory update modifies previously retained information.",
    ]
    for content in contents:
        page = pdf.new_page()
        page.insert_text((72, 72), content)
    pdf.set_metadata({"title": title})
    pdf.save(path)
    pdf.close()


@pytest.fixture
def make_pdf():
    return create_fixture_pdf

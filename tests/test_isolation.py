import json
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from qdrant_client import QdrantClient

from paper_rag.config import initialize_workspace_config
from paper_rag.ingest import ingest_papers
from paper_rag.retrieval import get_paper_pages, list_papers, search_papers
from paper_rag import server


class _MemoryRouteClient:
    urls = []
    banks = {}

    def __init__(self, url, **_kwargs):
        self.urls.append(url)
        self.bank_id = urlsplit(url).path.strip("/").split("/")[-1]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def call_tool(self, name, arguments):
        if name == "retain":
            self.banks.setdefault(self.bank_id, []).append(arguments["content"])
            text = "retained"
        else:
            text = "\n".join(self.banks.get(self.bank_id, []))
        return SimpleNamespace(
            is_error=False,
            content=[SimpleNamespace(type="text", text=text)],
        )


@pytest.mark.asyncio
async def test_two_workspaces_isolate_papers_and_memory_targets(
    tmp_path,
    monkeypatch,
    make_pdf,
    workspace,
):
    _, embedding = workspace
    roots = [tmp_path / "linguistics", tmp_path / "ecology"]
    settings = [
        initialize_workspace_config(root, name=name, discipline=name)[0]
        for root, name in zip(roots, ("Linguistics", "Ecology"), strict=True)
    ]
    shared_qdrant = QdrantClient(":memory:")
    for current, title in zip(settings, ("Language Change", "Wetland Recovery"), strict=True):
        current.papers_dir.mkdir(parents=True, exist_ok=True)
        make_pdf(current.papers_dir / "paper.pdf", title=title)
        ingest_papers(current, embedding, shared_qdrant)

    assert settings[0].resolved_qdrant_path != settings[1].resolved_qdrant_path
    assert settings[0].qdrant_collection != settings[1].qdrant_collection
    assert settings[0].hindsight_bank_id != settings[1].hindsight_bank_id
    for current, title in zip(settings, ("Language Change", "Wetland Recovery"), strict=True):
        inventory = list_papers(current)
        assert [paper.title for paper in inventory] == [title]
        hits = search_papers(
            "memory update",
            settings=current,
            qdrant_client=shared_qdrant,
            embedding_model=embedding,
        )
        assert hits
        assert {hit.title for hit in hits} == {title}
        pages = get_paper_pages(inventory[0].paper_id, 1, settings=current)
        assert pages[0].title == title

    _MemoryRouteClient.urls = []
    _MemoryRouteClient.banks = {}
    monkeypatch.setattr(server, "Client", _MemoryRouteClient)
    for current, title in zip(settings, ("Language Change", "Wetland Recovery"), strict=True):
        monkeypatch.setattr(server, "_BOUND_SETTINGS", current)
        result = await server.mcp.call_tool(
            "retain",
            {"content": f"My judgment for {title}.", "context": "research"},
        )
        assert not result.is_error

    recalled = []
    for current in settings:
        monkeypatch.setattr(server, "_BOUND_SETTINGS", current)
        for tool, arguments in (
            ("recall", {"query": "my prior judgment"}),
            ("reflect", {"query": "research direction"}),
        ):
            result = await server.mcp.call_tool(tool, arguments)
            assert not result.is_error
            recalled.append(json.dumps(result.model_dump(mode="json", by_alias=True)))

    assert len(_MemoryRouteClient.urls) == 6
    expected_banks = [
        settings[0].hindsight_bank_id,
        settings[1].hindsight_bank_id,
        settings[0].hindsight_bank_id,
        settings[0].hindsight_bank_id,
        settings[1].hindsight_bank_id,
        settings[1].hindsight_bank_id,
    ]
    actual_banks = [urlsplit(url).path.strip("/").split("/")[-1] for url in _MemoryRouteClient.urls]
    assert actual_banks == expected_banks
    assert "Language Change" in recalled[0] and "Wetland Recovery" not in recalled[0]
    assert "Wetland Recovery" in recalled[2] and "Language Change" not in recalled[2]
    shared_qdrant.close()

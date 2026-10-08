import json

import pytest

from paper_rag import server
from paper_rag.config import Settings
from paper_rag.models import PaperBrief, PaperHit, PaperPage


@pytest.mark.asyncio
async def test_mcp_exposes_paper_and_memory_tools(monkeypatch):
    expected_names = {
        "search_papers",
        "list_papers",
        "get_paper_pages",
        "retain",
        "recall",
        "reflect",
    }
    tools = await server.mcp.list_tools()
    assert {tool.name for tool in tools} == expected_names
    assert all(tool.output_schema for tool in tools)
    descriptions = " ".join(tool.description or "" for tool in tools).casefold()
    assert "agent memory" not in descriptions
    assert "amrb hindsight bank" not in descriptions
    assert "workspace knowledge" in descriptions

    hit = PaperHit(
        paper_id="0123456789abcdef",
        title="Fixture Research",
        file_name="fixture.pdf",
        page=3,
        score=0.9,
        text="Memory update modifies previously retained information.",
    )
    paper = PaperBrief(
        paper_id=hit.paper_id,
        title=hit.title,
        file_name=hit.file_name,
    )
    page = PaperPage(
        paper_id=hit.paper_id,
        title=hit.title,
        file_name=hit.file_name,
        page=3,
        citation=hit.citation,
        text=hit.text,
    )
    monkeypatch.setattr(
        server,
        "_BOUND_SETTINGS",
        Settings(_env_file=None, workspace_id="bound-workspace", hindsight_bank_id="bound-bank"),
    )
    monkeypatch.setattr(server, "retrieve_papers", lambda **kwargs: [hit])
    monkeypatch.setattr(server, "read_papers", lambda settings=None: [paper])
    monkeypatch.setattr(server, "read_paper_pages", lambda **kwargs: [page])

    remote_calls = []

    async def fake_hindsight(name, arguments):
        remote_calls.append((name, arguments))
        return json.dumps({"tool": name, "bank": "agent-memory-research"})

    monkeypatch.setattr(server, "_call_hindsight", fake_hindsight)

    results = [
        await server.mcp.call_tool("search_papers", {"query": "memory update"}),
        await server.mcp.call_tool("list_papers", {}),
        await server.mcp.call_tool(
            "get_paper_pages",
            {"paper_id": hit.paper_id, "start_page": 3},
        ),
        await server.mcp.call_tool(
            "retain",
            {"content": "My durable research judgment.", "context": "research"},
        ),
        await server.mcp.call_tool("recall", {"query": "my prior judgment"}),
        await server.mcp.call_tool("reflect", {"query": "my research direction"}),
    ]
    serialized = [json.dumps(result.model_dump(mode="json", by_alias=True)) for result in results]

    assert all(not result.is_error for result in results)
    assert "Fixture Research, p. 3" in serialized[0]
    assert "fixture.pdf" in serialized[1]
    assert "Memory update modifies" in serialized[2]
    assert all(not result.is_error for result in results[3:])
    assert [name for name, _ in remote_calls] == ["retain", "recall", "reflect"]
    assert "agent-memory-research" in serialized[3]

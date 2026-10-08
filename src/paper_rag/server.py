import argparse
import os
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.server import MCPServer

from paper_rag.config import Settings, get_settings
from paper_rag.hindsight_setup import inspect_hindsight_bank
from paper_rag.models import PaperBrief, PaperHit, PaperPage
from paper_rag.retrieval import get_paper_pages as read_paper_pages
from paper_rag.retrieval import list_papers as read_papers
from paper_rag.retrieval import search_papers as retrieve_papers
from paper_rag.service import hindsight_healthcheck, hindsight_mcp_url

mcp = MCPServer("amrb")
_BOUND_SETTINGS: Settings | None = None


def _workspace_settings() -> Settings:
    return _BOUND_SETTINGS or get_settings()


@mcp.tool()
def search_papers(
    query: str,
    top_k: int | None = None,
    paper: str | None = None,
) -> list[PaperHit]:
    """Search PDFs indexed in the current workspace and return source passages as external evidence."""
    return retrieve_papers(
        query=query,
        top_k=top_k,
        paper=paper,
        settings=_workspace_settings(),
    )


@mcp.tool()
def list_papers() -> list[PaperBrief]:
    """List PDF documents recorded in the current workspace's retrieval index."""
    return read_papers(settings=_workspace_settings())


@mcp.tool()
def get_paper_pages(
    paper_id: str,
    start_page: int,
    end_page: int | None = None,
) -> list[PaperPage]:
    """Read up to ten consecutive pages from an indexed PDF. Page numbers refer to PDF page order."""
    return read_paper_pages(
        paper_id=paper_id,
        start_page=start_page,
        end_page=end_page,
        settings=_workspace_settings(),
    )


async def _call_hindsight(name: str, arguments: dict[str, Any]) -> str:
    settings = _workspace_settings()
    async with Client(hindsight_mcp_url(settings), read_timeout_seconds=30) as client:
        result = await client.call_tool(name, arguments)
    if result.is_error:
        message = "\n".join(item.text for item in result.content if item.type == "text")
        raise RuntimeError(f"Hindsight {name} failed: {message}")
    return "\n".join(item.text for item in result.content if item.type == "text")


@mcp.tool()
async def retain(
    content: str,
    context: str = "general",
    timestamp: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Save explicit, durable workspace knowledge such as decisions, hypotheses, concepts, and stable preferences. Keep source passages, temporary questions, and routine task details in their original context. When a decision changes, identify the current version and preserve useful history."""
    arguments: dict[str, Any] = {"content": content, "context": context}
    if timestamp is not None:
        arguments["timestamp"] = timestamp
    if tags is not None:
        arguments["tags"] = tags
    return await _call_hindsight("retain", arguments)


@mcp.tool()
async def recall(
    query: str,
    max_tokens: int = 4096,
    budget: str = "high",
    prefer_observations: bool = False,
    tags: list[str] | None = None,
) -> str:
    """Recall prior knowledge, decisions, and preferences from this workspace's memory bank."""
    arguments: dict[str, Any] = {
        "query": query,
        "max_tokens": max_tokens,
        "budget": budget,
        "prefer_observations": prefer_observations,
    }
    if tags is not None:
        arguments["tags"] = tags
    return await _call_hindsight("recall", arguments)


@mcp.tool()
async def reflect(
    query: str,
    context: str | None = None,
    budget: str = "low",
    max_tokens: int = 4096,
    tags: list[str] | None = None,
) -> str:
    """Synthesize durable knowledge from this workspace's memory bank and distinguish recalled content from current analysis."""
    arguments: dict[str, Any] = {
        "query": query,
        "budget": budget,
        "max_tokens": max_tokens,
    }
    if context is not None:
        arguments["context"] = context
    if tags is not None:
        arguments["tags"] = tags
    return await _call_hindsight("reflect", arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description="AMRB retrieval and memory MCP server")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=os.environ.get("AMRB_WORKSPACE"),
        help="Absolute or relative path to an initialized workspace",
    )
    arguments = parser.parse_args()
    if arguments.workspace is None:
        parser.error("MCP startup requires --workspace or AMRB_WORKSPACE")
    workspace = arguments.workspace.expanduser().resolve()
    os.environ["AMRB_WORKSPACE"] = str(workspace)
    settings = get_settings(workspace)
    if (
        not settings.hindsight_resource_accepted
        and not settings.legacy_data_compatibility
        and hindsight_healthcheck(settings)
    ):
        bank = inspect_hindsight_bank(settings)
        if bank.exists:
            parser.error(
                f"Hindsight bank {settings.hindsight_bank_id} has unknown workspace ownership. "
                "Run `amrb setup --adopt-existing-resources` before connecting this workspace."
            )
    global _BOUND_SETTINGS
    _BOUND_SETTINGS = settings
    profile_fields = (
        ("Domain", settings.discipline),
        ("Purpose", settings.research_focus),
        ("Retrieval scope", settings.source_scope),
        ("Durable memory", settings.memory_policy),
    )
    profile = "\n".join(
        f"{label}: {' '.join(value.split())}"
        for label, value in profile_fields
        if value.strip()
    )
    mcp._lowlevel_server.instructions = (
        f"Bound to workspace {settings.workspace_name} ({settings.workspace_id}). "
        f"Retrieval collection: {settings.qdrant_collection}. Memory bank: {settings.hindsight_bank_id}. "
        + (f"\nWorkspace profile:\n{profile}\n" if profile else "")
        + "Retrieved source content and durable workspace memory are separate sources."
    )
    mcp.run(transport="stdio")

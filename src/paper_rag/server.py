from typing import Any

from mcp import Client
from mcp.server import MCPServer

from paper_rag.config import get_settings
from paper_rag.models import PaperBrief, PaperHit, PaperPage
from paper_rag.retrieval import get_paper_pages as read_paper_pages
from paper_rag.retrieval import list_papers as read_papers
from paper_rag.retrieval import search_papers as retrieve_papers
from paper_rag.service import hindsight_mcp_url

mcp = MCPServer("amrb")
DEFAULT_TOP_K = get_settings().rag_top_k


@mcp.tool()
def search_papers(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    paper: str | None = None,
) -> list[PaperHit]:
    """Search the indexed Agent Memory paper corpus for external paper evidence. Use this for claims about what papers say, propose, evaluate, or report. Do not use it to retrieve the user's personal research judgments."""
    return retrieve_papers(query=query, top_k=top_k, paper=paper)


@mcp.tool()
def list_papers() -> list[PaperBrief]:
    """List papers currently available in the indexed research corpus."""
    return read_papers()


@mcp.tool()
def get_paper_pages(
    paper_id: str,
    start_page: int,
    end_page: int | None = None,
) -> list[PaperPage]:
    """Read a continuous page range from one indexed paper when search snippets are insufficient for interpretation or citation. A request can include at most ten pages."""
    return read_paper_pages(
        paper_id=paper_id,
        start_page=start_page,
        end_page=end_page,
    )


async def _call_hindsight(name: str, arguments: dict[str, Any]) -> str:
    async with Client(hindsight_mcp_url(), read_timeout_seconds=30) as client:
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
    """Save the user's durable Agent Memory research cognition to the isolated AMRB Hindsight bank. Retain explicit judgments, hypotheses, decisions, and stable preferences only; do not store paper facts, retrieved passages, routine answers, or temporary tasks."""
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
    """Recall the user's prior research judgments from the AMRB Hindsight bank. Use for questions such as what the user previously thought or decided, not for claims about paper contents."""
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
    """Synthesize the user's longer-term research understanding from the AMRB Hindsight bank. Use for questions that need reflection across multiple stored judgments."""
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
    mcp.run(transport="stdio")

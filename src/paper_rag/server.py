from mcp.server import MCPServer

from paper_rag.config import get_settings
from paper_rag.models import PaperBrief, PaperHit, PaperPage
from paper_rag.retrieval import get_paper_pages as read_paper_pages
from paper_rag.retrieval import list_papers as read_papers
from paper_rag.retrieval import search_papers as retrieve_papers

mcp = MCPServer("paper-rag")
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


def main() -> None:
    mcp.run(transport="stdio")

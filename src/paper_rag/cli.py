import logging
import re
from typing import Any

import typer
import yaml
from rich.console import Console

from paper_rag.config import get_settings
from paper_rag.ingest import ingest_papers
from paper_rag.retrieval import get_status, search_papers

app = typer.Typer(no_args_is_help=True, help="Maintain and inspect the local paper index.")
console = Console()
error_console = Console(stderr=True)


@app.callback()
def configure_logging(
    verbose: bool = typer.Option(False, "--verbose", help="Show informational logs."),
) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
        force=True,
    )


def _run(action: Any) -> Any:
    try:
        return action()
    except Exception as exc:
        error_console.print(str(exc) or type(exc).__name__)
        raise typer.Exit(code=1) from exc


@app.command()
def ingest() -> None:
    """Parse and incrementally index PDFs in papers/."""
    result = _run(ingest_papers)
    console.print(
        f"Papers discovered: {result.discovered} | indexed: {result.indexed} | "
        f"skipped: {result.skipped} | removed: {result.removed} | "
        f"chunks indexed: {result.chunks}"
    )


@app.command()
def status() -> None:
    """Show the paper and vector index status."""
    result = _run(get_status)
    console.print("Paper RAG")
    console.print(f"Papers       {result.papers}")
    console.print(f"Chunks       {result.chunks}")
    console.print(f"Vector DB    {result.vector_backend}")
    console.print(f"Collection   {result.collection}")
    console.print("Last ingest")
    if result.last_ingest:
        from datetime import datetime

        last_ingest = datetime.fromisoformat(result.last_ingest).astimezone()
        console.print(last_ingest.strftime("%Y-%m-%d %H:%M"))
    else:
        console.print("never")


@app.command()
def search(
    query: str = typer.Argument(..., help="Research question or terms to search for."),
    top_k: int | None = typer.Option(None, min=1, max=100),
    paper: str | None = typer.Option(None, help="Restrict search to a paper title, file, or ID."),
) -> None:
    """Inspect dense retrieval results and their page citations."""
    hits = _run(lambda: search_papers(query=query, top_k=top_k, paper=paper))
    console.print(f"Query: {query}")
    console.print(f"top_k: {top_k or get_settings().rag_top_k} | paper: {paper or 'all'}")
    if not hits:
        console.print("No matching paper chunks.")
        return
    for rank, hit in enumerate(hits, start=1):
        preview = re.sub(r"\s+", " ", hit.text).strip()
        if len(preview) > 320:
            preview = preview[:317].rstrip() + "..."
        console.print(
            f"{rank}  {hit.file_name}  p.{hit.page}  {hit.score:.3f}  {hit.title}"
        )
        console.print(f"   {preview}")


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _expected_is_retrieved(expected: str, hits: list[Any]) -> bool:
    label = _normalize_label(expected)
    return any(
        label in _normalize_label(hit.file_name)
        or label in _normalize_label(hit.title)
        for hit in hits
    )


@app.command()
def eval() -> None:
    """Run paper retrieval cases from evals/rag.yaml and report Recall@K."""
    eval_path = get_settings().workspace_root / "evals" / "rag.yaml"

    def run_eval() -> tuple[list[dict[str, Any]], int, int]:
        config = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
        cases = config.get("cases", [])
        if not cases:
            raise ValueError(f"Eval cases not found: {eval_path}")
        evaluated: list[dict[str, Any]] = []
        expected_total = retrieved_total = 0
        for case in cases:
            query = str(case["query"])
            expected = [str(label) for label in case["expected_papers"]]
            top_k = int(case.get("top_k", config.get("top_k", get_settings().rag_top_k)))
            hits = search_papers(query=query, top_k=top_k)
            matches = [_expected_is_retrieved(label, hits) for label in expected]
            expected_total += len(matches)
            retrieved_total += sum(matches)
            evaluated.append(
                {
                    "query": query,
                    "expected": expected,
                    "retrieved": [hit.file_name for hit in hits],
                    "passed": bool(matches) and all(matches),
                    "matches": matches,
                }
            )
        return evaluated, expected_total, retrieved_total

    evaluated, expected_total, retrieved_total = _run(run_eval)
    passed = sum(bool(case["passed"]) for case in evaluated)
    console.print("RAG Eval")
    console.print(f"Cases       {len(evaluated)}")
    console.print(f"Passed      {passed}")
    console.print(f"Recall@K    {retrieved_total / expected_total:.2f}")
    for case in evaluated:
        if case["passed"]:
            continue
        console.print("FAIL")
        console.print(f"Query: {case['query']}")
        console.print(f"Expected: {', '.join(case['expected'])}")
        console.print(f"Retrieved: {', '.join(case['retrieved']) or '(none)'}")


if __name__ == "__main__":
    app()

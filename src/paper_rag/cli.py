import json
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
    query: str = typer.Argument(
        ..., help="Question or terms to search for in indexed documents."
    ),
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
    return "".join(char for char in value.casefold() if char.isalnum())


def _expected_rank(expected: str, hits: list[Any]) -> int | None:
    label = _normalize_label(expected)
    if not label:
        return None
    for rank, hit in enumerate(hits, start=1):
        candidates = (hit.paper_id, hit.file_name, hit.title)
        if any(label in _normalize_label(candidate) for candidate in candidates):
            return rank
    return None


def _summarize_eval_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_count = len(cases)
    expected_count = sum(len(case["expected_ranks"]) for case in cases)
    retrieved_count = sum(
        rank is not None
        for case in cases
        for rank in case["expected_ranks"].values()
    )
    hit_cases = sum(case["retrieved_expected"] > 0 for case in cases)
    fully_retrieved_cases = sum(case["passed"] for case in cases)
    return {
        "cases": case_count,
        "fully_retrieved_cases": fully_retrieved_cases,
        "case_pass_rate": fully_retrieved_cases / case_count if case_count else 0.0,
        "hit_cases": hit_cases,
        "hit_rate": hit_cases / case_count if case_count else 0.0,
        "expected_papers": expected_count,
        "retrieved_expected_papers": retrieved_count,
        "recall": retrieved_count / expected_count if expected_count else 0.0,
        "mrr": (
            sum(case["reciprocal_rank"] for case in cases) / case_count
            if case_count
            else 0.0
        ),
    }


def _has_eval_failures(report: dict[str, Any]) -> bool:
    metrics = report["metrics"]
    return metrics["fully_retrieved_cases"] != metrics["cases"]


@app.command()
def eval(
    top_k: int | None = typer.Option(
        None, min=1, max=100, help="Override the benchmark's configured cutoff."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print machine-readable JSON."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit with status 1 if any expected paper is missed."
    ),
) -> None:
    """Evaluate paper retrieval quality on the curated cases in evals/rag.yaml."""
    eval_path = get_settings().workspace_root / "evals" / "rag.yaml"

    def run_eval() -> dict[str, Any]:
        config = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
        cases = config.get("cases", [])
        if not cases:
            raise ValueError(f"Eval cases not found: {eval_path}")
        cutoff = int(top_k or config.get("top_k", get_settings().rag_top_k))
        if cutoff < 1 or cutoff > 100:
            raise ValueError("eval top_k 必须在 1 到 100 之间。")

        evaluated: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, case in enumerate(cases, start=1):
            case_id = str(case.get("id") or f"case-{index}")
            if case_id in seen_ids:
                raise ValueError(f"重复的 eval case id: {case_id}")
            seen_ids.add(case_id)
            query = str(case["query"])
            expected = [str(label) for label in case["expected_papers"]]
            if not query.strip() or not expected:
                raise ValueError(f"Eval case {case_id} needs a query and expected_papers.")

            hits = search_papers(query=query, top_k=cutoff)
            expected_ranks = {
                label: _expected_rank(label, hits) for label in expected
            }
            found_ranks = [rank for rank in expected_ranks.values() if rank is not None]
            first_relevant_rank = min(found_ranks) if found_ranks else None
            reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            retrieved = [
                {
                    "rank": rank,
                    "paper_id": hit.paper_id,
                    "title": hit.title,
                    "file_name": hit.file_name,
                    "page": hit.page,
                    "score": hit.score,
                }
                for rank, hit in enumerate(hits, start=1)
            ]
            evaluated.append(
                {
                    "id": case_id,
                    "category": str(case.get("category", "uncategorized")),
                    "language": str(case.get("language", "unspecified")),
                    "query": query,
                    "expected_ranks": expected_ranks,
                    "retrieved_expected": len(found_ranks),
                    "reciprocal_rank": reciprocal_rank,
                    "passed": len(found_ranks) == len(expected_ranks),
                    "retrieved": retrieved,
                }
            )

        by_category = {
            category: _summarize_eval_cases(
                [case for case in evaluated if case["category"] == category]
            )
            for category in sorted({case["category"] for case in evaluated})
        }
        by_language = {
            language: _summarize_eval_cases(
                [case for case in evaluated if case["language"] == language]
            )
            for language in sorted({case["language"] for case in evaluated})
        }
        return {
            "top_k": cutoff,
            "metrics": _summarize_eval_cases(evaluated),
            "by_category": by_category,
            "by_language": by_language,
            "results": evaluated,
        }

    report = _run(run_eval)
    if json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        cutoff = report["top_k"]
        metrics = report["metrics"]
        console.print(f"RAG Eval (K={cutoff})")
        console.print(
            f"Cases {metrics['cases']} | all targets "
            f"{metrics['fully_retrieved_cases']}/{metrics['cases']} "
            f"({metrics['case_pass_rate']:.2f})"
        )
        console.print(
            f"Hit@{cutoff} {metrics['hit_rate']:.2f} | "
            f"Recall@{cutoff} {metrics['recall']:.2f} | "
            f"MRR@{cutoff} {metrics['mrr']:.2f}"
        )
        console.print("By category")
        for category, summary in report["by_category"].items():
            console.print(
                f"  {category}: Recall {summary['recall']:.2f} | "
                f"MRR {summary['mrr']:.2f} | cases {summary['cases']}"
            )
        console.print("By language")
        for language, summary in report["by_language"].items():
            console.print(
                f"  {language}: Recall {summary['recall']:.2f} | "
                f"MRR {summary['mrr']:.2f} | cases {summary['cases']}"
            )
        console.print("Cases")
        for case in report["results"]:
            expected = ", ".join(
                f"{paper}@{rank if rank is not None else '—'}"
                for paper, rank in case["expected_ranks"].items()
            )
            status = "PASS" if case["passed"] else "FAIL"
            console.print(
                f"  {status} {case['id']} [{case['category']}/{case['language']}] "
                f"expected rank: {expected}"
            )
            if not case["passed"]:
                titles = list(dict.fromkeys(item["file_name"] for item in case["retrieved"]))
                console.print(f"    retrieved: {', '.join(titles) or '(none)'}")

    if strict and _has_eval_failures(report):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

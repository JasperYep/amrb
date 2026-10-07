from __future__ import annotations

import json
import logging
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import typer
from rich.console import Console

from paper_rag.batch import add_papers_batch, paper_batch_status, wait_paper_batch
from paper_rag.config import get_settings
from paper_rag.health import get_system_status, run_doctor, _check_bank
from paper_rag.ops import (
    add_papers,
    list_paper_records,
    remove_paper,
    sync_papers,
)
from paper_rag.service import (
    hindsight_healthcheck,
    restart_service,
    service_is_active,
    service_logs,
    systemd_user_available,
)
from paper_rag.setup import setup_workspace

app = typer.Typer(no_args_is_help=True, help="Operations for the Agent Memory Research Workspace.")
paper_app = typer.Typer(no_args_is_help=True)
batch_app = typer.Typer(no_args_is_help=True)
service_app = typer.Typer(no_args_is_help=True)
open_app = typer.Typer(no_args_is_help=True)
app.add_typer(paper_app, name="paper")
paper_app.add_typer(batch_app, name="batch")
app.add_typer(service_app, name="service")
app.add_typer(open_app, name="open")
console = Console()
error_console = Console(stderr=True)


@app.callback()
def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s", force=True)


def _run(action: Any) -> Any:
    try:
        return action()
    except Exception as exc:
        message = " ".join((str(exc) or type(exc).__name__).split())
        error_console.print(message)
        raise typer.Exit(code=1) from exc


def _json(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    typer.echo(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _valid_web_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _batch_status_callback(progress_console: Console) -> Any:
    last_update: tuple[str, tuple[tuple[str, int], ...]] | None = None

    def report(
        local_id: str,
        status: str,
        counts: dict[str, int] | None,
    ) -> None:
        nonlocal last_update
        normalized_counts = tuple(sorted((counts or {}).items()))
        update = (status, normalized_counts)
        if update == last_update:
            return
        last_update = update
        summary = ""
        if counts:
            summary = f" · {counts.get('completed', 0)}/{counts.get('total', 0)} requests"
        progress_console.print(f"Batch {local_id}: {status}{summary}")

    return report


@app.command()
def setup() -> None:
    """Initialize the workspace and install the Hindsight user service."""
    result = _run(setup_workspace)
    console.print("AMRB setup\n")
    console.print(f"Paper RAG    {'Ready' if result.environment == 'ready' and result.qdrant == 'ready' else 'Degraded'}")
    console.print(f"Qdrant       {result.qdrant.title()}")
    console.print(f"Hindsight    {result.hindsight.title()}")
    console.print(f"Memory bank  {result.memory_bank.title()}")
    for note in result.notes:
        console.print(f"\n{note}")
    if "unavailable" in {
        result.qdrant,
        result.hindsight,
        result.memory_bank,
    }:
        raise typer.Exit(code=1)
    console.print("\nDone.")


@app.command()
def status(as_json: bool = typer.Option(False, "--json", help="Print stable JSON.")) -> None:
    """Show a lightweight workspace health summary."""
    result = _run(get_system_status)
    if as_json:
        _json(result)
    else:
        console.print("AMRB\n")
        console.print(f"Paper RAG    {result.paper_rag.title()}")
        console.print(f"Hindsight    {result.hindsight.title()}")
        console.print(f"Qdrant       {result.qdrant.title()}")
        console.print(f"\nPapers       {result.papers}")
        console.print(f"Chunks       {result.chunks}")
    if {result.paper_rag, result.hindsight, result.qdrant} != {"ready"}:
        raise typer.Exit(code=1)


@app.command()
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Print stable JSON."),
    verbose: bool = typer.Option(False, "--verbose", help="Include sanitized low-level error details."),
) -> None:
    """Run deeper dependency, index, and memory-bank checks."""
    report, details = _run(lambda: run_doctor(verbose=verbose))
    if as_json:
        data = report.model_dump(mode="json")
        if verbose:
            data["details"] = details
        _json(data)
    else:
        console.print("AMRB Doctor\n")
        labels = (
            ("Environment", report.environment),
            ("Embedding", report.embedding),
            ("Qdrant", report.qdrant),
            ("Paper index", report.paper_index),
            ("Hindsight", report.hindsight),
            ("Memory bank", report.memory_bank),
        )
        for label, component in labels:
            console.print(f"{label:<15} {component.status.upper()}")
            if component.message:
                console.print(f"  {component.message}")
        console.print(f"\nPapers          {report.papers}")
        console.print(f"Chunks          {report.chunks}")
        if verbose:
            for name, detail in details.items():
                console.print(f"\n{name}: {detail}")
    statuses = [
        report.environment.status,
        report.embedding.status,
        report.qdrant.status,
        report.paper_index.status,
        report.hindsight.status,
        report.memory_bank.status,
    ]
    if any(value != "ready" for value in statuses):
        raise typer.Exit(code=1)


@paper_app.command("add")
def paper_add(
    paths: list[str] = typer.Argument(..., help="One or more local PDF paths."),
    as_json: bool = typer.Option(False, "--json", help="Print stable JSON."),
    batch_mode: bool = typer.Option(
        False,
        "--batch",
        help="Use the provider's OpenAI-compatible Batch File API for offline indexing.",
    ),
) -> None:
    """Copy PDFs into papers/ and index them, optionally with Batch File."""
    source_paths = [Path(path) for path in paths]
    if batch_mode:
        progress_console = error_console if as_json else console
        status_update = _batch_status_callback(progress_console)

        def submitted(local_id: str, remote_id: str) -> None:
            progress_console.print(
                f"Batch submitted: {remote_id} · local job {local_id}. "
                f"Resume with `uv run amrb paper batch wait {local_id}`."
            )

        result = _run(
            lambda: add_papers_batch(
                source_paths,
                on_submitted=submitted,
                on_status=status_update,
            )
        )
    else:
        result = _run(lambda: add_papers(source_paths))
    _print_paper_add_result(result, as_json)


def _print_paper_add_result(result: Any, as_json: bool) -> None:
    if as_json:
        _json(result)
        if result.failed:
            raise typer.Exit(code=1)
        return
    for paper in result.added:
        console.print(f"Added       {paper.title}\n            {paper.file_name} · {paper.pages or '?'} pages · Indexed")
    for paper in result.updated:
        console.print(f"Updated     {paper.title}\n            {paper.file_name} · {paper.pages or '?'} pages · Indexed")
    for paper in result.skipped:
        console.print(f"Already indexed  {paper.file_name}")
    for failure in result.failed:
        console.print(f"Failed      {failure.source}")
        console.print(f"            {'File copied; index failed. ' if failure.copied else ''}{failure.message}")
    if result.failed:
        raise typer.Exit(code=1)


@batch_app.command("status")
def paper_batch_status_command(
    job_id: str = typer.Argument(..., help="Local AMRB Batch job ID."),
    as_json: bool = typer.Option(False, "--json", help="Print stable JSON."),
) -> None:
    """Inspect an AMRB Batch job."""
    status = _run(lambda: paper_batch_status(job_id))
    if as_json:
        _json(status)
    else:
        console.print(f"Local job  {status['job_id']}")
        if status.get("batch_id"):
            console.print(f"Provider   {status['batch_id']}")
        console.print(f"Status     {status.get('status', 'unknown')}")
        counts = status.get("request_counts")
        if counts:
            console.print(
                f"Requests   {counts.get('completed', 0)}/{counts.get('total', 0)} "
                f"completed · {counts.get('failed', 0)} failed"
            )


@batch_app.command("wait")
def paper_batch_wait_command(
    job_id: str = typer.Argument(..., help="Local AMRB Batch job ID."),
    as_json: bool = typer.Option(False, "--json", help="Print stable JSON."),
) -> None:
    """Wait for a submitted Batch job and apply its vectors to Qdrant."""
    progress_console = error_console if as_json else console
    status_update = _batch_status_callback(progress_console)

    result = _run(
        lambda: wait_paper_batch(job_id, on_status=status_update)
    )
    _print_paper_add_result(result, as_json)


@paper_app.command("sync")
def paper_sync(as_json: bool = typer.Option(False, "--json", help="Print stable JSON.")) -> None:
    """Incrementally reconcile papers/ with the vector index."""
    result = _run(sync_papers)
    if as_json:
        _json(result)
    else:
        console.print("Paper sync\n")
        console.print(f"Added      {result.added}")
        console.print(f"Updated    {result.updated}")
        console.print(f"Removed    {result.removed}")
        console.print(f"Unchanged  {result.unchanged}")
        if result.error:
            console.print(f"\nIndex error: {result.error}")
    if result.error:
        raise typer.Exit(code=1)


@paper_app.command("remove")
def paper_remove(identifier: str = typer.Argument(..., help="Paper filename, title, or paper_id.")) -> None:
    """Remove a paper file and its indexed vectors."""
    result = _run(lambda: remove_paper(identifier))
    if result is None:
        error_console.print(f"Paper not found: {identifier}")
        raise typer.Exit(code=1)
    console.print(f"Removed     {result.removed.title} ({result.removed.file_name})")


@paper_app.command("list")
def paper_list(as_json: bool = typer.Option(False, "--json", help="Print stable JSON.")) -> None:
    """List indexed papers and basic local metadata."""
    records = _run(list_paper_records)
    if as_json:
        _json({"papers": [record.model_dump(mode="json") for record in records], "count": len(records)})
        return
    console.print("Papers\n")
    for paper in records:
        pages = f"{paper.pages} pages" if paper.pages is not None else "pages unknown"
        console.print(f"{paper.title:<55} {pages}")
    console.print(f"\n{len(records)} papers")


def _service_status() -> dict[str, str]:
    settings = get_settings()
    managed = systemd_user_available() and service_is_active()
    api_ready = hindsight_healthcheck(settings)
    bank_health = _check_bank(settings) if api_ready else None
    return {
        "service": "running" if managed else ("unmanaged" if api_ready else "stopped"),
        "api": "ready" if api_ready else "unavailable",
        "bank": "ready" if bank_health and bank_health.status == "ready" else "unavailable",
    }


@service_app.command("status")
def service_status(as_json: bool = typer.Option(False, "--json", help="Print stable JSON.")) -> None:
    """Inspect Hindsight's managed service, API, and memory bank."""
    result = _run(_service_status)
    if as_json:
        _json(result)
    else:
        console.print("Hindsight\n")
        console.print(f"Service   {result['service'].title()}")
        console.print(f"API       {result['api'].title()}")
        console.print(f"Bank      {result['bank'].title()}")
    if result["api"] != "ready" or result["bank"] != "ready":
        raise typer.Exit(code=1)


@service_app.command("restart")
def service_restart() -> None:
    """Restart the managed Hindsight user service and wait for readiness."""
    _run(restart_service)
    console.print("Hindsight restarted and ready.")


@service_app.command("logs")
def service_log_lines(
    lines: int = typer.Option(100, min=1, max=1000),
) -> None:
    """Show a bounded tail of Hindsight service logs."""
    output = _run(lambda: service_logs(lines))
    typer.echo(output, nl=not output.endswith("\n"))


@open_app.command("hindsight")
def open_hindsight() -> None:
    """Open the configured Hindsight control plane in the system browser."""
    settings = get_settings()
    if not settings.hindsight_ui_url:
        error_console.print("Set HINDSIGHT_UI_URL to the official Hindsight UI address.")
        raise typer.Exit(code=1)
    if not _valid_web_url(settings.hindsight_ui_url):
        error_console.print("HINDSIGHT_UI_URL must be an http:// or https:// URL.")
        raise typer.Exit(code=1)
    if not hindsight_healthcheck(settings):
        error_console.print("Hindsight API is not ready. Run `amrb service status` first.")
        raise typer.Exit(code=1)
    if not _run(lambda: webbrowser.open(settings.hindsight_ui_url)):
        error_console.print("Could not open the system browser.")
        raise typer.Exit(code=1)
    console.print("Opened Hindsight.")


@open_app.command("qdrant")
def open_qdrant() -> None:
    """Open the configured official Qdrant dashboard in the system browser."""
    settings = get_settings()
    if not settings.qdrant_dashboard_url:
        error_console.print("Set QDRANT_DASHBOARD_URL to the official Qdrant dashboard address.")
        raise typer.Exit(code=1)
    if not _valid_web_url(settings.qdrant_dashboard_url):
        error_console.print("QDRANT_DASHBOARD_URL must be an http:// or https:// URL.")
        raise typer.Exit(code=1)
    if not _run(lambda: webbrowser.open(settings.qdrant_dashboard_url)):
        error_console.print("Could not open the system browser.")
        raise typer.Exit(code=1)
    console.print("Opened Qdrant.")


if __name__ == "__main__":
    app()

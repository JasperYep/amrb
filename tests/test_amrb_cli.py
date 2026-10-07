import json

from typer.testing import CliRunner

from paper_rag.amrb_cli import app
from paper_rag.config import Settings
from paper_rag.models import PaperAddResult, PaperFailure, PaperRecord
from paper_rag.setup import SetupResult
from paper_rag import amrb_cli


def test_paper_add_cli_reports_indexed_result(monkeypatch, tmp_path):
    monkeypatch.setattr(
        amrb_cli,
        "add_papers",
        lambda paths: PaperAddResult(
            added=[PaperRecord(paper_id="id", title="Example", file_name="example.pdf", pages=2)]
        ),
    )
    result = CliRunner().invoke(app, ["paper", "add", str(tmp_path / "example.pdf"), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["added"][0]["file_name"] == "example.pdf"


def test_paper_add_cli_returns_nonzero_for_index_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        amrb_cli,
        "add_papers",
        lambda paths: PaperAddResult(
            failed=[PaperFailure(source=str(tmp_path / "paper.pdf"), message="provider unavailable", copied=True)]
        ),
    )
    result = CliRunner().invoke(app, ["paper", "add", str(tmp_path / "paper.pdf"), "--json"])
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["failed"][0]["copied"] is True


def test_setup_cli_outputs_component_summary(monkeypatch):
    monkeypatch.setattr(
        amrb_cli,
        "setup_workspace",
        lambda: SetupResult(
            python="ready",
            uv="ready",
            environment="ready",
            qdrant="ready",
            hindsight="ready",
            memory_bank="ready",
        ),
    )
    result = CliRunner().invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "Memory bank" in result.stdout
    assert "Done." in result.stdout


def test_open_hindsight_uses_configured_url_after_healthcheck(monkeypatch):
    settings = Settings(_env_file=None, hindsight_ui_url="https://hindsight.example/ui")
    opened = []
    monkeypatch.setattr(amrb_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(amrb_cli, "hindsight_healthcheck", lambda current: True)
    monkeypatch.setattr(amrb_cli.webbrowser, "open", lambda url: opened.append(url) or True)

    result = CliRunner().invoke(app, ["open", "hindsight"])

    assert result.exit_code == 0
    assert opened == ["https://hindsight.example/ui"]


def test_open_qdrant_uses_configured_dashboard_url(monkeypatch):
    settings = Settings(_env_file=None, qdrant_dashboard_url="https://qdrant.example/dashboard")
    opened = []
    monkeypatch.setattr(amrb_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(amrb_cli.webbrowser, "open", lambda url: opened.append(url) or True)

    result = CliRunner().invoke(app, ["open", "qdrant"])

    assert result.exit_code == 0
    assert opened == ["https://qdrant.example/dashboard"]

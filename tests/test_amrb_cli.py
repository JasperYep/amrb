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


def test_setup_cli_outputs_component_summary(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "amrb.toml").write_text(
        '[workspace]\nid = "test-workspace"\nname = "Test"\n[index]\ncollection = "test-papers"\n[memory]\nbank_id = "test-memory"\n',
        encoding="utf-8",
    )
    (workspace / ".env").write_text(
        "EMBEDDING_API_KEY=test-key\nEMBEDDING_BASE_URL=https://embedding.example/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AMRB_WORKSPACE", str(workspace))
    monkeypatch.setattr(
        amrb_cli,
        "setup_workspace",
        lambda settings, **kwargs: SetupResult(
            python="ready",
            uv="ready",
            environment="ready",
            qdrant="ready",
            hindsight="ready",
            memory_bank="ready",
        ),
    )
    monkeypatch.setattr(amrb_cli, "plan_workspace_client_files", lambda root, settings: ())
    monkeypatch.setattr(amrb_cli, "inspect_qdrant_resource", lambda settings: (False, 0, None))
    monkeypatch.setattr(amrb_cli, "hindsight_healthcheck", lambda settings=None: False)
    monkeypatch.setattr(
        amrb_cli,
        "verify_mcp_startup",
        lambda root, settings: {"tools": ["one", "two", "three", "four", "five", "six"]},
    )
    result = CliRunner().invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "Memory bank" in result.stdout
    assert "Done." in result.stdout


def test_setup_cli_failure_includes_saved_state_and_recovery(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "amrb.toml").write_text(
        '[workspace]\nid = "test-workspace"\nname = "Test"\n'
        '[index]\ncollection = "test-papers"\n'
        '[memory]\nbank_id = "test-memory"\n',
        encoding="utf-8",
    )
    (workspace / ".env").write_text(
        "EMBEDDING_API_KEY=workspace-test-key\n"
        "EMBEDDING_BASE_URL=https://embedding.example/v1\n"
        "HINDSIGHT_API_LLM_API_KEY=workspace-memory-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AMRB_WORKSPACE", str(workspace))
    monkeypatch.setattr(amrb_cli, "plan_workspace_client_files", lambda root, settings: ())
    monkeypatch.setattr(amrb_cli, "inspect_qdrant_resource", lambda settings: (False, 0, None))
    monkeypatch.setattr(amrb_cli, "hindsight_healthcheck", lambda settings=None: False)

    def fail_setup(settings, **kwargs):
        raise RuntimeError("service unavailable")

    monkeypatch.setattr(amrb_cli, "setup_workspace", fail_setup)
    result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 1
    assert "service unavailable" in result.output
    assert "Workspace files and any completed backend setup steps are preserved" in " ".join(result.output.split())
    assert "Rerun `amrb setup --workspace" in result.output
    assert "workspace-test-key" not in result.output
    assert "workspace-memory-key" not in result.output


def test_open_hindsight_uses_configured_url_after_healthcheck(monkeypatch):
    settings = Settings(_env_file=None, hindsight_ui_url="https://hindsight.example/ui")
    opened = []
    monkeypatch.setattr(amrb_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(amrb_cli, "hindsight_healthcheck", lambda current: True)
    monkeypatch.setattr(amrb_cli, "probe_web_ui", lambda url: "reachable")
    monkeypatch.setattr(amrb_cli.webbrowser, "open", lambda url: opened.append(url) or True)

    result = CliRunner().invoke(app, ["open", "hindsight"])

    assert result.exit_code == 0
    assert opened == ["https://hindsight.example/ui"]


def test_open_qdrant_uses_configured_dashboard_url(monkeypatch):
    settings = Settings(
        _env_file=None,
        qdrant_url="https://qdrant.example",
        qdrant_dashboard_url="https://qdrant.example/dashboard",
    )
    opened = []
    monkeypatch.setattr(amrb_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(amrb_cli.webbrowser, "open", lambda url: opened.append(url) or True)
    monkeypatch.setattr(amrb_cli, "probe_web_ui", lambda url: "reachable")

    result = CliRunner().invoke(app, ["open", "qdrant"])

    assert result.exit_code == 0
    assert opened == ["https://qdrant.example/dashboard"]

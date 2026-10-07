import json

from pydantic import SecretStr
from typer.testing import CliRunner

from paper_rag.config import create_qdrant_client
from paper_rag.health import get_system_status, run_doctor
from paper_rag.ingest import ingest_papers, write_manifest
from paper_rag.models import ComponentHealth, DoctorReport
from paper_rag import health
from paper_rag.amrb_cli import app


def test_doctor_reports_all_ready(workspace, make_pdf, monkeypatch):
    settings, embedding = workspace
    settings = settings.model_copy(
        update={
            "embedding_api_key": SecretStr("test-key"),
            "embedding_base_url": "https://embedding.example/v1",
        }
    )
    (settings.workspace_root / ".env").write_text("configured")
    make_pdf(settings.papers_dir / "paper.pdf")
    client = create_qdrant_client(settings)
    try:
        ingest_papers(settings, embedding, client)
    finally:
        client.close()

    monkeypatch.setattr(health, "get_settings", lambda: settings)
    monkeypatch.setattr(health, "create_embedding_model", lambda current: embedding)
    monkeypatch.setattr(health, "hindsight_healthcheck", lambda current=None: True)
    monkeypatch.setattr(health, "_check_bank", lambda current: ComponentHealth(status="ready"))

    report, details = run_doctor()
    assert not details
    assert report.environment.status == "ready"
    assert report.embedding.status == "ready"
    assert report.qdrant.status == "ready"
    assert report.paper_index.status == "ready"
    assert report.hindsight.status == "ready"
    assert report.memory_bank.status == "ready"
    assert report.papers == 1
    assert report.chunks == 3


def test_status_reports_hindsight_offline(workspace, monkeypatch):
    settings, _ = workspace
    monkeypatch.setattr(health, "hindsight_healthcheck", lambda current=None: False)
    status = get_system_status(settings)
    assert status.hindsight == "unavailable"
    assert status.qdrant == "ready"


def test_status_reports_qdrant_offline(workspace, monkeypatch):
    settings, _ = workspace
    settings.resolved_qdrant_path.mkdir(parents=True)

    def fail_client(current):
        raise RuntimeError("fixture offline")

    monkeypatch.setattr(health, "create_qdrant_client", fail_client)
    monkeypatch.setattr(health, "hindsight_healthcheck", lambda current=None: True)
    status = get_system_status(settings)
    assert status.qdrant == "unavailable"
    assert status.paper_rag == "degraded"


def test_doctor_reports_missing_embedding_configuration(workspace):
    settings, _ = workspace
    report, _ = run_doctor(settings)
    assert report.environment.status == "unavailable"
    assert report.embedding.status == "unavailable"


def test_doctor_reports_manifest_without_qdrant_index(workspace, make_pdf):
    settings, _ = workspace
    make_pdf(settings.papers_dir / "paper.pdf")
    write_manifest(
        settings.manifest_path,
        {
            "papers/paper.pdf": {
                "sha256": "fixture",
                "paper_id": "paper-id",
                "indexed_at": "2026-01-01T00:00:00+00:00",
                "title": "Fixture Research",
                "chunk_count": 3,
            }
        },
    )
    qdrant, chunks, _ = health._check_qdrant(settings)
    index = health._check_paper_index(settings, chunks, qdrant.status)
    assert qdrant.status == "degraded"
    assert index.status == "degraded"


def test_status_json_is_parseable_and_stable(monkeypatch):
    monkeypatch.setattr(
        "paper_rag.amrb_cli.get_system_status",
        lambda: health.SystemStatus(
            paper_rag="ready", hindsight="ready", qdrant="ready", papers=4, chunks=195
        ),
    )
    result = CliRunner().invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data == {
        "paper_rag": "ready",
        "hindsight": "ready",
        "qdrant": "ready",
        "papers": 4,
        "chunks": 195,
    }


def test_doctor_json_is_parseable_and_contains_every_check(monkeypatch):
    ready = ComponentHealth(status="ready")
    report = DoctorReport(
        environment=ready,
        embedding=ready,
        qdrant=ready,
        paper_index=ready,
        hindsight=ready,
        memory_bank=ready,
        papers=4,
        chunks=195,
    )
    monkeypatch.setattr("paper_rag.amrb_cli.run_doctor", lambda verbose=False: (report, {}))

    result = CliRunner().invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["environment"]["status"] == "ready"
    assert data["embedding"]["status"] == "ready"
    assert data["qdrant"]["status"] == "ready"
    assert data["paper_index"]["status"] == "ready"
    assert data["hindsight"]["status"] == "ready"
    assert data["memory_bank"]["status"] == "ready"
    assert data["papers"] == 4
    assert data["chunks"] == 195


def test_paper_list_json_is_parseable():
    runner = CliRunner()
    result = runner.invoke(app, ["paper", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert set(data) == {"papers", "count"}
    assert data["count"] == len(data["papers"])

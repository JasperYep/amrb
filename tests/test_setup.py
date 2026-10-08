import tomlkit
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from paper_rag.config import Settings, claim_qdrant_collection, update_workspace_config
from paper_rag.setup import (
    WorkspaceFileChange,
    plan_workspace_client_files,
    verify_mcp_startup,
    write_environment_values,
    write_workspace_client_files,
)


def test_workspace_client_merge_preserves_other_servers_and_unknown_fields(tmp_path, monkeypatch):
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    config_path = codex_dir / "config.toml"
    config_path.write_text(
        '''# keep this comment
[mcp_servers.other]
command = "other-server"
unknown_setting = "keep"

[mcp_servers.amrb]
command = "old-command"
args = ["old"]
user_extension = "keep-too"
''',
        encoding="utf-8",
    )
    agents = tmp_path / "AGENTS.md"
    agents.write_text("user research rules\n", encoding="utf-8")
    monkeypatch.setattr("paper_rag.setup.shutil.which", lambda name: "/opt/amrb-mcp" if name == "amrb-mcp" else "/opt/uv")
    settings = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        workspace_id="workspace-id",
        workspace_name="Biology",
        discipline="Biology",
    )

    changes = plan_workspace_client_files(tmp_path, settings)
    assert len(changes) == 1
    assert "command, args" in changes[0].summary
    write_workspace_client_files(changes, tmp_path)
    result = tomlkit.loads(config_path.read_text(encoding="utf-8"))

    assert result["mcp_servers"]["other"]["unknown_setting"] == "keep"
    assert result["mcp_servers"]["amrb"]["user_extension"] == "keep-too"
    assert result["mcp_servers"]["amrb"]["args"] == ["--workspace", str(tmp_path.resolve())]
    assert agents.read_text(encoding="utf-8") == "user research rules\n"


def test_workspace_client_plan_creates_generic_rules_only_if_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("paper_rag.setup.shutil.which", lambda name: "/opt/amrb-mcp" if name == "amrb-mcp" else "/opt/uv")
    settings = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        workspace_id="workspace-id",
        workspace_name="Ecology project",
        discipline="Ecology",
        research_focus="Wetland restoration",
    )

    changes = plan_workspace_client_files(tmp_path, settings)

    assert {change.path.name for change in changes} == {"AGENTS.md", "config.toml"}
    policy = next(change.content for change in changes if change.path.name == "AGENTS.md")
    assert "Wetland restoration" in policy
    assert "A question, tentative discussion" in policy
    assert "A paper fact" in policy


def test_workspace_env_is_atomic_and_owner_only(tmp_path):
    env_path = tmp_path / ".env"

    changed = write_environment_values(env_path, {"API_KEY": "secret-value", "BASE_URL": "https://example.test"})

    assert changed == ("API_KEY", "BASE_URL")
    assert env_path.stat().st_mode & 0o777 == 0o600
    assert "API_KEY=\"secret-value\"" in env_path.read_text(encoding="utf-8")


def test_qdrant_collection_owner_mismatch_is_rejected():
    client = QdrantClient(":memory:")
    client.create_collection(
        "shared",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        metadata={"amrb_workspace_id": "workspace-a"},
    )
    settings = Settings(
        _env_file=None,
        workspace_id="workspace-b",
        qdrant_collection="shared",
    )
    try:
        with pytest.raises(ValueError, match="belongs to workspace workspace-a"):
            claim_qdrant_collection(client, settings)
    finally:
        client.close()


def test_populated_unknown_qdrant_collection_requires_adoption():
    client = QdrantClient(":memory:")
    client.create_collection(
        "legacy",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    client.upsert(
        "legacy",
        [PointStruct(id="e5992fbb-3f9f-4d1c-b186-20d804bcfb27", vector=[0.1, 0.2], payload={})],
    )
    settings = Settings(
        _env_file=None,
        workspace_id="workspace-new",
        qdrant_collection="legacy",
    )
    try:
        with pytest.raises(ValueError, match="unknown ownership"):
            claim_qdrant_collection(client, settings)

        adopted = settings.model_copy(update={"qdrant_resource_accepted": True})
        exists, count, owner = claim_qdrant_collection(client, adopted)

        assert exists is True
        assert count == 1
        assert owner is None
        assert client.get_collection("legacy").config.metadata["amrb_workspace_id"] == "workspace-new"
    finally:
        client.close()


def test_workspace_env_does_not_replace_nonempty_existing_secrets(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("API_KEY=original-secret\nOTHER=value\n", encoding="utf-8")

    changed = write_environment_values(env_path, {"API_KEY": "replacement", "NEW_KEY": "new"})

    assert changed == ("NEW_KEY",)
    assert "API_KEY=original-secret" in env_path.read_text(encoding="utf-8")
    assert env_path.stat().st_mode & 0o777 == 0o600


def test_existing_workspace_env_is_made_private_even_when_no_values_change(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("API_KEY=already-set\n", encoding="utf-8")
    env_path.chmod(0o644)

    assert write_environment_values(env_path, {"API_KEY": "replacement"}) == ()
    assert env_path.read_text(encoding="utf-8") == "API_KEY=already-set\n"
    assert env_path.stat().st_mode & 0o777 == 0o600


def test_workspace_config_update_preserves_unmanaged_lines_and_values(tmp_path):
    config = tmp_path / "amrb.toml"
    config.write_text(
        'config_version = 1\n\n[workspace]\nid = "keep-id"\n\n[index]\ncollection = "keep-collection"\nchunk_size = 700\n',
        encoding="utf-8",
    )

    update_workspace_config(tmp_path, "index", {"embedding_dimension": 1024, "legacy_index_accepted": True})
    updated = config.read_text(encoding="utf-8")
    parsed = tomlkit.loads(updated)

    assert parsed["workspace"]["id"] == "keep-id"
    assert parsed["index"]["collection"] == "keep-collection"
    assert parsed["index"]["embedding_dimension"] == 1024
    assert parsed["index"]["legacy_index_accepted"] is True
    assert "chunk_size = 700" in updated


def test_client_file_write_failure_reports_completed_files_and_preserves_next_file(tmp_path, monkeypatch):
    agents = tmp_path / "AGENTS.md"
    codex_config = tmp_path / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    codex_config.write_text("existing config\n", encoding="utf-8")
    changes = (
        WorkspaceFileChange(agents, "new rules\n", "create rules", False),
        WorkspaceFileChange(codex_config, "new config\n", "update MCP", True),
    )
    original_replace = __import__("os").replace

    def fail_second(source, destination):
        if destination == codex_config:
            raise OSError("simulated client write failure")
        return original_replace(source, destination)

    monkeypatch.setattr("paper_rag.setup.os.replace", fail_second)

    with pytest.raises(RuntimeError, match="AGENTS.md") as error:
        write_workspace_client_files(changes, tmp_path)

    assert "rerun setup" in str(error.value)
    assert agents.read_text(encoding="utf-8") == "new rules\n"
    assert codex_config.read_text(encoding="utf-8") == "existing config\n"
    assert not list(codex_config.parent.glob(".config.toml.*.tmp"))


def test_mcp_initializes_with_explicit_workspace_from_other_cwd(tmp_path):
    settings = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        workspace_id="workspace-smoke",
        workspace_name="Smoke project",
        qdrant_collection="smoke-papers",
        hindsight_bank_id="smoke-memory",
    )
    (tmp_path / "amrb.toml").write_text(
        '''[workspace]
id = "workspace-smoke"
name = "Smoke project"
[index]
collection = "smoke-papers"
[memory]
bank_id = "smoke-memory"
resource_accepted = true
''',
        encoding="utf-8",
    )

    result = verify_mcp_startup(tmp_path, settings)

    assert result["workspace_id"] == "workspace-smoke"
    assert result["cwd_independent"] is True
    assert result["tools"] == [
        "get_paper_pages",
        "list_papers",
        "recall",
        "reflect",
        "retain",
        "search_papers",
    ]

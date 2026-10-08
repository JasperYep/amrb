import json
import shutil

from llama_index.embeddings import openai

from paper_rag.config import (
    Settings,
    create_embedding_model,
    derived_resource_ids,
    copied_workspace_origin,
    find_workspace_root,
    get_settings,
    initialize_workspace_config,
    resolved_configuration,
)


def test_openai_compatible_model_and_base_url_are_passed_to_embedding(monkeypatch):
    captured = {}

    class FakeEmbedding:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, "OpenAIEmbedding", FakeEmbedding)
    settings = Settings(
        _env_file=None,
        embedding_model="qwen3-embedding:8b",
        embedding_api_key="test-key",
        embedding_base_url="https://embedding.example/v1",
    )

    create_embedding_model(settings)

    assert captured["api_key"] == "test-key"
    assert captured["api_base"] == settings.embedding_base_url
    assert captured["model_name"] == settings.embedding_model
    assert captured["embed_batch_size"] == 20


def test_embedding_base_url_env_takes_precedence_over_legacy_alias(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example/compatible-mode/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")

    settings = Settings(_env_file=None)

    assert settings.embedding_base_url == "https://embedding.example/compatible-mode/v1"


def test_openai_base_url_remains_supported_as_legacy_alias(monkeypatch):
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")

    settings = Settings(_env_file=None)

    assert settings.embedding_base_url == "https://legacy.example/v1"


def test_embedding_api_key_env_takes_precedence_over_legacy_alias(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")

    settings = Settings(_env_file=None)

    assert settings.embedding_api_key.get_secret_value() == "embedding-key"


def test_openai_api_key_remains_supported_as_legacy_alias(monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")

    settings = Settings(_env_file=None)

    assert settings.embedding_api_key.get_secret_value() == "legacy-key"
    assert settings.openai_api_key.get_secret_value() == "legacy-key"


def test_hindsight_llm_settings_are_agent_agnostic_defaults():
    settings = Settings(_env_file=None)

    assert settings.hindsight_api_llm_provider == "openai"
    assert settings.hindsight_api_llm_model == "gpt-4o-mini"


def test_workspace_settings_bind_toml_and_workspace_dotenv(tmp_path, monkeypatch):
    (tmp_path / "amrb.toml").write_text(
        '''config_version = 1
[workspace]
id = "workspace-a"
name = "Cognitive Science"
[papers]
directory = "sources/papers"
[index]
collection = "cog_papers"
qdrant_path = ".state/qdrant"
[memory]
bank_id = "cog-memory"
''',
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "EMBEDDING_API_KEY=dotenv-secret\nEMBEDDING_MODEL=dotenv-model\nQDRANT_COLLECTION=ignored-old-value\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    settings = get_settings(tmp_path)

    assert settings.workspace_root == tmp_path.resolve()
    assert settings.workspace_id == "workspace-a"
    assert settings.workspace_name == "Cognitive Science"
    assert settings.papers_dir == tmp_path / "sources/papers"
    assert settings.qdrant_collection == "cog_papers"
    assert settings.hindsight_bank_id == "cog-memory"
    assert settings.embedding_model == "dotenv-model"
    assert settings.embedding_api_key.get_secret_value() == "dotenv-secret"


def test_process_environment_overrides_deployment_settings_but_not_data_identity(tmp_path, monkeypatch):
    (tmp_path / "amrb.toml").write_text(
        '[workspace]\nid = "workspace-a"\n[index]\ncollection = "toml-collection"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("QDRANT_COLLECTION", "process-collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "process-model")

    settings = get_settings(tmp_path)

    assert settings.qdrant_collection == "toml-collection"
    assert settings.embedding_model == "process-model"
    assert settings.workspace_root == tmp_path.resolve()


def test_workspace_discovery_walks_up_and_requires_initialization(tmp_path):
    root = tmp_path / "research"
    nested = root / "drafts" / "notes"
    nested.mkdir(parents=True)
    (root / "amrb.toml").write_text('[workspace]\nid = "id"\n', encoding="utf-8")

    assert find_workspace_root(nested) == root

    try:
        find_workspace_root(tmp_path)
    except FileNotFoundError as exc:
        assert "amrb setup --workspace" in str(exc)
    else:
        raise AssertionError("workspace lookup must not fall back to the installation directory")


def test_workspace_resource_names_are_stable_and_domain_neutral():
    first = derived_resource_ids("workspace-a")
    second = derived_resource_ids("workspace-a")

    assert first == second
    assert first[0].startswith("amrb_papers_")
    assert first[1].startswith("amrb-")
    assert "agent-memory" not in " ".join(first)


def test_workspace_initialization_persists_identity_and_derived_targets(tmp_path):
    workspace = tmp_path / "new-project"

    settings, created = initialize_workspace_config(
        workspace,
        name="New project",
        discipline="Linguistics",
        research_focus="Language acquisition",
    )
    repeated, created_again = initialize_workspace_config(
        workspace,
        name="Renamed",
        discipline="Other",
        research_focus="Different focus",
    )

    assert created is True
    assert created_again is False
    assert settings.workspace_id == repeated.workspace_id
    assert settings.workspace_name == repeated.workspace_name == "New project"
    assert settings.discipline == "Linguistics"
    assert settings.qdrant_collection.startswith("amrb_papers_")
    assert settings.hindsight_bank_id.startswith("amrb-")
    assert settings.legacy_index_accepted is False
    assert settings.workspace_bound_path == workspace.resolve()


def test_index_profile_in_toml_cannot_be_overridden_by_process_environment(tmp_path, monkeypatch):
    workspace = tmp_path / "profiled"
    initialized, _ = initialize_workspace_config(
        workspace,
        name="Profiled",
        embedding_model="e5-large",
        chunk_size=512,
        chunk_overlap=64,
    )
    monkeypatch.setenv("EMBEDDING_MODEL", "different-model")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "999")

    settings = get_settings(workspace)
    resolved = resolved_configuration(settings)

    assert settings.embedding_model == initialized.embedding_model == "e5-large"
    assert settings.rag_chunk_size == 512
    assert settings.rag_chunk_overlap == 64
    assert resolved["embedding_model_source"] == "amrb.toml"
    assert resolved["chunk_profile_source"] == "amrb.toml"


def test_legacy_workspace_requires_explicit_adoption_and_preserves_targets(tmp_path):
    (tmp_path / ".state").mkdir()
    (tmp_path / ".state" / "index.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "QDRANT_COLLECTION=old-collection\nQDRANT_PATH=.legacy/qdrant\n",
        encoding="utf-8",
    )

    try:
        initialize_workspace_config(tmp_path, name="Legacy")
    except ValueError as exc:
        assert "Explicitly choose" in str(exc)
    else:
        raise AssertionError("legacy workspace adoption must be explicit")

    settings, created = initialize_workspace_config(
        tmp_path,
        name="Legacy",
        adopt_legacy=True,
    )

    assert created is True
    assert settings.qdrant_collection == "old-collection"
    assert settings.qdrant_path == tmp_path / ".legacy/qdrant"
    assert settings.hindsight_bank_id == "agent-memory-research"
    assert settings.legacy_index_accepted is False


def test_copied_workspace_requires_an_explicit_identity_choice(tmp_path):
    original = tmp_path / "original"
    copy = tmp_path / "copy"
    settings, _ = initialize_workspace_config(original, name="Project")
    shutil.copytree(original, copy)

    assert copied_workspace_origin(copy) == original.resolve()
    try:
        get_settings(copy)
    except ValueError as exc:
        assert "--copied-identity reuse" in str(exc)
        assert "--copied-identity new" in str(exc)
    else:
        raise AssertionError("a copied workspace must not silently reuse its data identity")


def test_copied_workspace_can_reuse_identity_after_explicit_rebinding(tmp_path):
    from paper_rag.setup import resolve_copied_workspace_identity

    original = tmp_path / "original"
    copy = tmp_path / "copy"
    settings, _ = initialize_workspace_config(original, name="Project")
    shutil.copytree(original, copy)

    rebound = resolve_copied_workspace_identity(copy, "reuse")

    assert rebound.workspace_id == settings.workspace_id
    assert rebound.qdrant_collection == settings.qdrant_collection
    assert rebound.hindsight_bank_id == settings.hindsight_bank_id
    assert rebound.workspace_bound_path == copy.resolve()
    assert copied_workspace_origin(copy) is None


def test_copied_workspace_can_create_new_identity_without_losing_prior_index(tmp_path, monkeypatch):
    from paper_rag.setup import resolve_copied_workspace_identity

    monkeypatch.delenv("QDRANT_URL", raising=False)
    original = tmp_path / "original"
    copy = tmp_path / "copy"
    settings, _ = initialize_workspace_config(original, name="Project")
    original_index = original / ".state" / "index.json"
    original_index.parent.mkdir(parents=True)
    original_index.write_text(json.dumps({"papers/example.pdf": {"paper_id": "old-id"}}), encoding="utf-8")
    original_qdrant = settings.resolved_qdrant_path
    original_qdrant.mkdir(parents=True)
    (original_qdrant / "preserve-me").write_text("old vectors", encoding="utf-8")
    shutil.copytree(original, copy)
    copied_index = copy / ".state" / "index.json"

    rebound = resolve_copied_workspace_identity(copy, "new")

    archive = copy / ".state" / "copied-workspace" / settings.workspace_id / "index.json"
    assert rebound.workspace_id != settings.workspace_id
    assert rebound.qdrant_collection != settings.qdrant_collection
    assert rebound.hindsight_bank_id != settings.hindsight_bank_id
    assert rebound.resolved_qdrant_path != copy / ".state" / "qdrant"
    assert json.loads(copied_index.read_text(encoding="utf-8")) == {}
    assert json.loads(archive.read_text(encoding="utf-8"))["papers/example.pdf"]["paper_id"] == "old-id"
    assert (copy / ".state" / "qdrant" / "preserve-me").read_text(encoding="utf-8") == "old vectors"

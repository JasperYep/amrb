from llama_index.embeddings import openai

from paper_rag.config import Settings, create_embedding_model


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

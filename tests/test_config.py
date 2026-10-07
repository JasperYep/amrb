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
        openai_api_key="test-key",
        openai_base_url="https://embedding.example/v1",
    )

    create_embedding_model(settings)

    assert captured["api_key"] == "test-key"
    assert captured["api_base"] == "https://embedding.example/v1"
    assert captured["model_name"] == settings.embedding_model

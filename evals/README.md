# Retrieval evaluation

`rag.yaml` contains simple expected-paper retrieval cases. Run `uv run paper-rag eval` after adding the matching papers to `papers/` and indexing them. The CLI reports case pass rate and Recall@K; it does not call an LLM judge.

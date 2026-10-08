# Paper RAG retrieval evaluation

`rag.yaml` is a small, deterministic retrieval benchmark. Its 22 queries cover memory architectures, implicit and episodic memory, memory evaluation, and security/privacy. Five Chinese queries also provide a basic check of cross-language retrieval.

Run it after indexing the papers named by the cases:

```sh
uv run paper-rag eval
```

The report includes:

- **Case pass rate**: all expected papers were retrieved for a query.
- **Hit@K**: at least one expected paper was retrieved for a query.
- **Recall@K**: fraction of all expected papers retrieved.
- **MRR@K**: reciprocal rank of the first expected paper, averaged over queries.
- Recall and MRR grouped by topic and query language.

Ranks are positions in the returned chunk list; an expected paper's rank is the position of its first matching chunk. Expected labels are matched against paper IDs, titles, and filenames. Use distinctive labels to avoid accidental substring matches.

Useful options:

```sh
uv run paper-rag eval --top-k 4
uv run paper-rag eval --json
uv run paper-rag eval --strict
```

`--strict` returns a nonzero exit code if any expected paper is missed, which is useful in automation. This benchmark measures retrieval only; it does not judge answer correctness, citation quality, or whether retrieved evidence supports a generated answer. It uses no LLM judge.

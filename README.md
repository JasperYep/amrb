# Agent Memory Research Workspace

A local-first research workspace that connects Codex to a paper evidence index and Hindsight long-term research memory.

## Architecture

Codex is the agent runtime. Paper RAG uses LlamaIndex and Qdrant to return cited evidence from local PDFs. Hindsight stores the researcher's long-term judgments through its MCP server. Paper evidence and personal research memory remain separate.

## Setup

Requirements: Python 3.12 or newer, `uv`, Codex CLI, and an OpenAI embedding API key.

```bash
uv sync
cp .env.example .env
```

If the runtime refuses to read hard-linked package data during ingestion, recreate the environment with `UV_LINK_MODE=copy uv sync --locked --reinstall`.

Set `OPENAI_API_KEY` in `.env`. OpenAI-compatible BYOK endpoints can be selected with `OPENAI_BASE_URL`; set `EMBEDDING_MODEL` to a model supported by that endpoint. Qdrant uses local persistent storage in `.state/qdrant/` by default. To use a Qdrant server, set `QDRANT_URL` and optionally `QDRANT_API_KEY`.

## Add Papers

Copy PDF files into `papers/`, then build or update the index:

```bash
cp ~/Downloads/*.pdf papers/
uv run paper-rag ingest
uv run paper-rag status
uv run paper-rag search "memory consolidation"
```

Ingestion is incremental. Unchanged PDFs are skipped. The manifest and local vector database live under `.state/`.

## Start Hindsight

Hindsight's current local MCP endpoint is served by its API server. It uses an embedded database in `~/.hindsight/data/` by default. In a separate terminal, use the Codex account already signed in on this machine:

```bash
HINDSIGHT_API_LLM_PROVIDER=openai-codex \
HINDSIGHT_API_LLM_MODEL=gpt-6-luna \
uvx --from hindsight-api==0.10.2 hindsight-api --host 127.0.0.1
```

The `openai-codex` provider reads the current Codex auth file (by default `~/.codex/auth.json`) and needs no separate API key. `gpt-6-luna` is the model verified with this Codex account; for another account, choose a supported model from `codex debug models`. To use an OpenAI API key instead:

```bash
set -a
source .env
set +a
HINDSIGHT_API_LLM_PROVIDER=openai \
HINDSIGHT_API_LLM_API_KEY="$OPENAI_API_KEY" \
uvx --from hindsight-api==0.10.2 hindsight-api --host 127.0.0.1
```

The project connects to the single-bank endpoint `http://localhost:8888/mcp/agent-memory-research/`. See the [Hindsight installation](https://hindsight.vectorize.io/developer/installation), [model providers](https://hindsight.vectorize.io/developer/models), and [MCP server](https://hindsight.vectorize.io/developer/mcp-server) documentation for current provider and storage options.

After the server is running, configure the project's bank missions from `AGENTS.md`:

```bash
uv run python -m paper_rag.hindsight_setup
```

## Start Codex

From the project directory:

```bash
codex
```

On first launch, trust this folder so Codex loads the project-level `.codex/config.toml`. It registers Paper RAG as a required stdio MCP server and Hindsight as an optional HTTP MCP server. Use `/mcp` in the Codex TUI to check the connections and exposed tools. Paper RAG should be connected; Hindsight connects after its API server starts.

Start Hindsight before a session where long-term memory is needed. Paper RAG remains available when Hindsight is offline.

## Usage

Try these prompts in Codex:

- `列出当前论文库里的论文。` — uses `list_papers`.
- `Hindsight 的 memory consolidation 是怎么做的？` — uses `search_papers`.
- `记住：我认为 Hindsight 最值得关注的是 consolidation，而不是普通 semantic retrieval。` — uses Hindsight `retain`.
- In a later session: `我之前怎么看 Hindsight？` — uses Hindsight `recall`.
- `综合之前的阅读，我现在对 Agent Memory evaluation 的总体观点是什么？` — uses Hindsight `reflect`.
- `我的看法和 Hindsight 论文自己的 framing 一致吗？` — compares memory with paper evidence.

When a retrieved chunk is not enough, Codex can read up to ten consecutive pages with `get_paper_pages`. Research response and selective-retain behavior are defined in `AGENTS.md`.

## Maintenance

```bash
uv run paper-rag ingest
uv run paper-rag status
uv run paper-rag search "memory update"
uv run paper-rag eval
```

Only `.pdf` files directly inside `papers/` are indexed. The CLI does not download papers or expose ingestion as an agent tool.

## Tests

```bash
uv run pytest
```

Tests create a small three-page PDF in a temporary directory and use local Qdrant with a mock embedding model, so they do not require API credentials or real papers.

## Design Boundary

Codex handles reasoning and tool choice. Paper RAG reports what the indexed papers say. Hindsight retains and recalls the researcher's long-term understanding. The RAG service does not generate answers or write paper content into Hindsight.

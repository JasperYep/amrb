# AMRB Research Workspace MCP

[English](README.md) | [简体中文](README.zh-CN.md)

AMRB connects a research project to one MCP server for paper evidence and long term research memory. The MCP client remains the conversation and reasoning interface; AMRB does not run an agent loop or provide a chat UI.

```text
MCP client ── AMRB ──┬── Paper RAG / Qdrant: external paper evidence
                     └── Hindsight: researcher's long term judgments
```

Each workspace represents one research project. Its `amrb.toml` stores nonsecret project identity and data targets. Its `.env` stores API keys and deployment connection settings. Different workspaces use different collections and Hindsight banks by default.

## Install and initialize

Install AMRB once, then initialize a workspace in the research project directory:

```bash
git clone https://github.com/JasperYep/amrb.git
cd amrb
uv sync --locked
uv run amrb setup --workspace /absolute/path/to/research-project
```

When running from source, the equivalent command is:

```bash
uv --directory /absolute/path/to/amrb run amrb setup --workspace /absolute/path/to/research-project
```

Setup asks for the project name, field, optional research focus, embedding endpoint and key, and whether to use local or remote Qdrant. Optional Hindsight Control Plane and remote Qdrant Dashboard URLs can also be entered; setup probes configured links. API keys and URLs are entered in hidden terminal prompts, never in the MCP conversation. It creates `amrb.toml`, a user-only `.env`, a Codex MCP entry, and—only when absent—a generic `AGENTS.md` research policy. Existing `AGENTS.md`, other MCP entries, and unknown Codex config fields are preserved.

For scripts, provide required settings through process environment or the workspace `.env` and use:

```bash
uv --directory /absolute/path/to/amrb run amrb setup \
  --workspace /absolute/path/to/research-project \
  --name "Research project" --discipline "Your field" --non-interactive
```

If setup detects existing unconfigured papers or index data, explicitly pass `--adopt-legacy` after checking the configured targets. An old index without model and chunk metadata remains unverified until its configuration is inspected and confirmed with `--accept-legacy-index`. Setup preserves the old paper IDs, manifest, collection, bank, and shared Hindsight service.

Workspace configs record the path where their identity was initialized. If a config is copied or moved, setup asks whether to keep the existing collection and bank (`--copied-identity reuse`) or create a new identity (`--copied-identity new`). A new identity selects a separate collection, bank, and local Qdrant directory; it archives the copied manifest and leaves PDFs and prior vector files in place without reindexing. A populated collection or existing Hindsight bank with unknown ownership requires explicit confirmation with `--adopt-existing-resources`.

## MCP clients

Codex is the reference onboarding client. Setup writes a workspace-bound stdio command to `.codex/config.toml`; the command pins both the AMRB installation and workspace paths, so MCP startup does not depend on the client's current directory. Other MCP clients can launch the same server:

```bash
uv --directory /absolute/path/to/amrb run amrb-mcp --workspace /absolute/path/to/research-project
```

The MCP provides `search_papers`, `list_papers`, `get_paper_pages`, `retain`, `recall`, and `reflect`. Paper results identify the paper, retrieval method, and PDF page order. Hindsight tools are bound to the workspace bank at server startup; they do not accept a project or bank selector on each call.

## Inspect and maintain

```bash
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/research-project status --json
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/research-project doctor --json
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/research-project paper list --json
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/research-project paper add ~/Downloads/paper.pdf
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/research-project paper sync
```

`paper list` shows each PDF's title, page count, stable paper ID, and whether it is indexed, not indexed, changed since indexing, missing, failed, or unverified. JSON output is intended for scripts. `status` and `doctor` show workspace, bank, collection, paper, chunk, and service state.

Hindsight Control Plane is the official interface for inspecting memory. Qdrant Dashboard is an infrastructure view for remote Qdrant and does not replace the paper inventory or retrieval results. These vendor interfaces may permit writes or deletion; opening them does not make them read only. A local embedded Qdrant store has no server Dashboard to open.

AMRB does not automatically import PDFs, retain conversation history, or turn retrieved paper evidence into personal memory. Workspace collaboration rules govern when to use Paper RAG and Hindsight; they do not enforce write authorization at the MCP server.

## Configuration

`amrb.toml` contains the workspace ID, project description, paper directory, embedding model and chunking profile, Qdrant collection, and Hindsight bank. The workspace ID is generated once; renaming the project does not change data identity. Relative data paths resolve from the workspace root.

`.env` contains keys and service connection values and is excluded from version control. Generated secret files use owner-only permissions. Workspace identity and the recorded embedding/chunk profile come from `amrb.toml`; deployment values resolve as process environment, workspace `.env`, then defaults. AMRB does not load an `.env` from its installation directory.

Hindsight is a shared user service. A new workspace reuses a healthy service and gets a separate bank. Setup does not rewrite service credentials, restart a healthy shared service, or replace missions in an existing bank.

If a bank already exists, setup shows any mission differences and asks before updating them. In scripts, use `--update-missions` only after reviewing the proposed workspace missions. Service logs redact configured keys and service URLs. If a setup step fails, its message reports what was saved and how to resume.

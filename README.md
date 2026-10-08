# AMRB — Agent Memory & Retrieval Bridge

English | [简体中文](README.zh-CN.md)

An agent-agnostic MCP layer for retrieval and long-term memory.

AMRB connects an MCP client to a workspace that combines PDF retrieval with persistent memory. The client handles conversation and reasoning; AMRB supplies retrieval, memory, and workspace operations.

```text
MCP client ── AMRB ──┬── PDF retrieval / Qdrant
                     └── Long-term memory / Hindsight
```

Each workspace represents a project or knowledge area. It has its own retrieval collection and Hindsight memory bank. Setup records a compact profile covering its domain, purpose, source scope, and durable memory policy, then creates a stable `AGENTS.md` when the workspace has no existing one.

## Set up

Install AMRB once, then initialize the target workspace:

```bash
git clone https://github.com/JasperYep/amrb.git
cd amrb
uv sync --locked
uv run amrb setup --workspace /absolute/path/to/workspace
```

From another directory, use:

```bash
uv --directory /absolute/path/to/amrb run amrb setup --workspace /absolute/path/to/workspace
```

Ask the agent: “Set up AMRB and connect it to this agent.”

Setup collects the project name, domain, workspace purpose, retrieval scope, memory policy, embedding model and endpoint, API key, and Qdrant location. It configures a shared Hindsight service when needed. Endpoint URLs are visible in the terminal and setup review; API keys use hidden prompts. Setup writes `amrb.toml`, a user-only `.env`, and the selected client configuration. It creates `AGENTS.md` only when the workspace does not already have one.

The saved workspace profile is also included in the MCP server's initialization instructions, so connected agents receive the same context through their client.

Codex configuration is written by default. For another MCP client, pass `--skip-codex-config`; setup prints the exact `command` and `args` to add to the active client's settings. The same stdio server works with any MCP-compatible agent.

On macOS, AMRB manages Hindsight with a user LaunchAgent. On Linux, it uses `systemd --user`. The service reads credentials from the owner-only `~/.config/amrb/hindsight.env` file.

For scripted setup, provide required settings through the process environment or workspace `.env`:

```bash
uv --directory /absolute/path/to/amrb run amrb setup \
  --workspace /absolute/path/to/workspace \
  --name "Project name" --domain "Project domain" \
  --purpose "Workspace purpose" \
  --source-scope "PDF collection to retrieve from" \
  --memory-policy "Durable knowledge to retain" --non-interactive
```

## Retrieval and memory

The retrieval layer indexes PDFs with Qdrant. Its MCP tools search the collection, list indexed documents, and read consecutive PDF pages. Search results include the source and PDF page order.

Hindsight stores durable workspace knowledge in a separate bank. `retain` records explicit decisions, hypotheses, concepts, and preferences. `recall` retrieves prior knowledge; `reflect` synthesizes its history. Workspace instructions define how the active agent uses these tools.

AMRB supports one workspace per project or knowledge area. Workspace identity and profile live in `amrb.toml`; API keys and service connection settings live in `.env`. The generated workspace policy stays stable between ordinary sessions, keeping the agent's context prefix consistent. When the workspace purpose changes, update the profile in `amrb.toml` and its instructions in `AGENTS.md`.

## Connect other MCP clients

Run the server directly with the target workspace:

```bash
uv --directory /absolute/path/to/amrb run amrb-mcp --workspace /absolute/path/to/workspace
```

Setup also prints a machine-readable launch configuration. Add its `command` and `args` to the MCP settings used by the active client, then reconnect the client.

## Inspect and maintain

```bash
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/workspace status --json
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/workspace doctor --json
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/workspace paper list --json
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/workspace paper add ~/Downloads/document.pdf
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/workspace paper sync
uv --directory /absolute/path/to/amrb run amrb --workspace /absolute/path/to/workspace service status --json
```

`paper list` reports each PDF's title, page count, stable ID, and index state. `status` and `doctor` report workspace identity, collection, memory bank, document and chunk counts, and service health.

Hindsight Control Plane displays memory entries. Qdrant Dashboard displays remote vector infrastructure. Setup can record these URLs and check reachability. Local embedded Qdrant stores vectors in the workspace and has no server dashboard.

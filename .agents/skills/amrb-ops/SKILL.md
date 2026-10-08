---
name: amrb-ops
description: >
  Manage a local AMRB retrieval and memory workspace.
  Use when the user asks to add, remove, sync, inspect, diagnose,
  repair, start, restart, or open infrastructure for the AMRB document
  collection or Hindsight service.
---

# AMRB Operations

Use this skill for workspace operations. Use the AMRB MCP for retrieval and long-term memory queries. Prefer the stable `amrb` CLI over manually reproducing its internal shell steps. In this repository run it as `uv run amrb ...` to use the project environment.

## Add papers

When the user asks to add one or more local PDFs:

1. Resolve the requested paths and expand an explicitly requested directory or date filter.
2. Confirm the selected files are PDFs.
3. Run `uv run amrb paper add <paths...>`.
4. Report papers added, already indexed, and failures. If indexing failed after a copy, say that the file was copied but is not indexed.

For an explicitly requested large offline ingest, use `uv run amrb paper add --batch <paths...>` only when the configured embedding endpoint is known to support the OpenAI-compatible Batch File API. OpenAI-compatible embeddings alone do not imply Batch support. The command prints a local job ID; use `uv run amrb paper batch status <job-id>` to inspect it and `uv run amrb paper batch wait <job-id>` to resume waiting and apply results. The user's local `.state/` must be preserved until the job is applied.

Do not manually copy files and run ingestion when `amrb paper add` can do the complete operation.

## Sync

For requests such as “同步论文库”, “更新索引”, or “检查新论文”, run:

```bash
uv run amrb paper sync --json
```

Report added, updated, removed, and unchanged counts. A sync may remove indexed files that are no longer present in `papers/`; make sure that is what the user asked for before running it.

## Remove

When the user explicitly asks to remove a paper, run:

```bash
uv run amrb paper remove <filename-or-title-or-paper-id>
```

Do not remove unrelated files. For an ambiguous identifier, list matching papers and ask which one to remove.

## List papers

For workspace inventory or counts, run `uv run amrb paper list --json`. Do not also call Paper RAG `list_papers` for the same metadata request.

## System status and diagnosis

- For simple health questions, run `uv run amrb status --json`.
- For troubleshooting or a vague failure, run `uv run amrb doctor --json`.
- Do not run deep diagnostics for every ordinary request.

## Hindsight service

If Hindsight appears unavailable:

1. Run `uv run amrb service status --json`.
2. If a managed service exists, run `uv run amrb service restart` and check status again.
3. Inspect `uv run amrb service logs` only if restart does not solve the problem.

## Infrastructure UIs

- When the user asks to inspect Memory visually, run `uv run amrb open hindsight`.
- When the user asks to inspect vectors or Qdrant, run `uv run amrb open qdrant`.

Do not build alternative local dashboards.

## Knowledge queries

Use the AMRB MCP for source retrieval and durable workspace memory. Use this skill for workspace operations and infrastructure maintenance.

## Output

Keep operational responses short. Prefer “已加入 3 篇论文并完成索引。” over reproducing command logs. Mention only actionable failures.

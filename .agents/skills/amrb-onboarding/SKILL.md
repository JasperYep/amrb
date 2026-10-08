---
name: amrb-onboarding
description: Deploy or repair an AMRB retrieval and memory workspace end to end, install its dependencies, connect the active MCP client, and verify the result. Use when the user asks an agent to set up AMRB, including on a new machine.
---

# AMRB onboarding

Treat a setup request as permission to complete the deployment in the available environment. Do not stop at giving the user commands when you can run them. Infer the workspace from the current project when possible; ask only for missing nonsecret choices that cannot be inferred.

## Install and configure

1. Locate an existing AMRB checkout and `uv`. Reuse the current checkout when available. If AMRB is absent and the user asked to deploy it, install the official source in a stable user-owned directory without overwriting an existing checkout. Install `uv` if needed using its official installation method. Run `uv sync --locked` from the AMRB checkout.
2. Infer the workspace path and project name from the current project. Establish a compact profile: domain, purpose, source scope, and what durable knowledge belongs in memory. Infer what the project context supports, ask one concise grouped question for missing details, and confirm the profile once before saving it. Include local or remote Qdrant in the same question when it cannot be inferred. Never ask for API keys in chat.
3. For a new workspace, run `uv --directory <absolute-amrb-checkout> run amrb setup --workspace <absolute-workspace-path>` with `--name`, `--domain`, `--purpose`, `--source-scope`, and `--memory-policy` from the confirmed profile. For an existing workspace, use its saved profile and omit those metadata flags. Add `--skip-codex-config` when the active client is not Codex. Use an interactive PTY so the user can type secrets into hidden terminal prompts. Endpoint URLs are visible; API keys remain hidden. Never put secrets in command arguments, environment values supplied by the agent, config previews, MCP startup arguments, logs, or Hindsight Memory.
4. Let the CLI preview its writes and handle configuration. The confirmed profile seeds the workspace configuration, generated `AGENTS.md`, and Hindsight missions. Keep this profile and its instructions stable across ordinary sessions so the agent's context prefix stays consistent. Change them only when the user explicitly updates the workspace profile. Before adopting a directory with existing `.env`, papers, or index data, show the CLI summary and require the user's explicit choice. If a legacy index lacks an embedding profile, review its model, vector dimension, chunk size, and overlap before using `--accept-legacy-index`. If a copied workspace config is bound to another path, ask whether to reuse its collection and bank (`--copied-identity reuse`) or create a new identity (`--copied-identity new`). For a populated resource with unknown ownership, require explicit adoption. Preview Hindsight mission changes before using `--update-missions`.

## Connect the active agent

AMRB uses the same stdio MCP server with different agents. The CLI configures Codex by default. For another client, pass `--skip-codex-config`; use the printed `MCP launch` command and arguments to add AMRB to the active client's native MCP settings. Preserve unrelated servers and unknown settings. Each client loads its own configuration and workspace instructions.

Use the client’s supported settings API or configuration file, then refresh/reconnect that client if needed. If the current agent cannot edit or reload its own MCP settings, finish the workspace deployment and give the user the exact remaining client-specific action and launch values.

## Verify and report

After setup, check `amrb status --json` and `amrb paper list --json` using the AMRB checkout and workspace paths. Confirm the workspace ID, Hindsight bank, Qdrant collection, MCP startup, service state, and paper inventory. Report any component that remains unavailable or unverified and its recovery command. Do not add PDFs or write test memories unless the user requested them.

The CLI owns configuration validation and file changes. Preserve existing `AGENTS.md`, unrelated client configuration, workspace data, and shared Hindsight service settings. Do not reproduce file-writing logic in this skill.

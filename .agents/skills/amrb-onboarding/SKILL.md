---
name: amrb-onboarding
description: Initialize or repair an AMRB research workspace and connect an MCP client. Use when the user wants to set up AMRB in a research project, create workspace configuration, or review onboarding changes.
---

# AMRB onboarding

AMRB is a workspace tool, not an agent runtime. Ask only for nonsecret project details in chat: the workspace path, project name, field, optional research focus, and local or remote Qdrant preference.

Run `amrb setup --workspace <absolute-path>` from the AMRB installation. For a source checkout, use `uv --directory <absolute-amrb-install-path> run amrb setup --workspace <absolute-workspace-path>`. Let the CLI collect API keys in hidden terminal prompts. Never ask for, paste, or pass secrets through the agent conversation or command arguments.

Before adopting data in a directory without `amrb.toml`, inspect the CLI's summary and require an explicit user choice. If setup finds a legacy index without a recorded embedding profile, review the displayed model, vector dimension, chunk size, and overlap with the user. Pass `--accept-legacy-index` only after the user confirms they match the existing index.

If `amrb.toml` is bound to another path, explain that the workspace may be a copy or a move and ask whether it should reuse the same collection and bank (`--copied-identity reuse`) or create a new identity (`--copied-identity new`). A new identity archives the copied index manifest, keeps the PDFs and old local Qdrant data, and does not reindex. If a populated Qdrant collection or Hindsight bank has unknown ownership, show its identity and require explicit adoption. Existing Hindsight mission changes must be previewed and confirmed before using `--update-missions`.

Optional Hindsight Control Plane and remote Qdrant Dashboard URLs can be entered in the hidden CLI prompts. The CLI checks reachability; these vendor UIs can permit edits or deletion. Local embedded Qdrant has no Dashboard.

The CLI owns configuration validation and file changes. It previews paths and managed settings, preserves existing `AGENTS.md` and unrelated Codex MCP entries, and keeps workspace settings separate from secrets. Do not reproduce its config-writing logic in this skill.

After setup, check `amrb status --json` and `amrb paper list --json`. Report the workspace ID, bank and collection, paper inventory state, and any setup or verification steps that remain. Do not add PDFs or write test memories unless the user requested them.

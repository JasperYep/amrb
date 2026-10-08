from __future__ import annotations

import json
import logging
import os
import sys
import uuid
import webbrowser
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import typer
from rich.console import Console

from paper_rag.batch import add_papers_batch, paper_batch_status, wait_paper_batch
from paper_rag.config import (
    Settings,
    copied_workspace_origin,
    derived_resource_ids,
    find_workspace_root,
    get_settings,
    initialize_workspace_config,
    load_settings,
    load_settings,
    redact_sensitive_text,
)
from paper_rag.health import get_system_status, run_doctor, _check_bank, _check_embedding
from paper_rag.hindsight_setup import inspect_hindsight_bank
from paper_rag.ingest import load_manifest
from paper_rag.ops import (
    add_papers,
    list_paper_records,
    remove_paper,
    sync_papers,
)
from paper_rag.service import (
    hindsight_healthcheck,
    redact_service_text,
    restart_service,
    service_is_active,
    service_logs,
    service_paths,
    systemd_user_available,
)
from paper_rag.ui import probe_web_ui
from paper_rag.setup import (
    accept_existing_hindsight_resource,
    accept_existing_qdrant_resource,
    accept_legacy_index,
    inspect_qdrant_resource,
    inspect_legacy_index,
    plan_workspace_client_files,
    setup_workspace,
    resolve_copied_workspace_identity,
    verify_mcp_startup,
    write_environment_values,
    write_workspace_client_files,
)

app = typer.Typer(no_args_is_help=True, help="Research workspace operations.")
paper_app = typer.Typer(no_args_is_help=True)
batch_app = typer.Typer(no_args_is_help=True)
service_app = typer.Typer(no_args_is_help=True)
open_app = typer.Typer(no_args_is_help=True)
app.add_typer(paper_app, name="paper")
paper_app.add_typer(batch_app, name="batch")
app.add_typer(service_app, name="service")
app.add_typer(open_app, name="open")
console = Console()
error_console = Console(stderr=True)


@app.callback()
def configure_logging(
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        help="Use this initialized research workspace (defaults to the nearest amrb.toml).",
    ),
) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s", force=True)
    if workspace is not None:
        os.environ["AMRB_WORKSPACE"] = str(workspace.expanduser().resolve())


def _run(action: Any, *, recovery: str | None = None) -> Any:
    try:
        return action()
    except Exception as exc:
        message = " ".join((str(exc) or type(exc).__name__).split())
        secrets = [
            os.environ.get(name, "")
            for name in (
                "EMBEDDING_API_KEY",
                "OPENAI_API_KEY",
                "QDRANT_API_KEY",
                "HINDSIGHT_API_LLM_API_KEY",
            )
        ]
        try:
            settings = get_settings()
            message = redact_sensitive_text(message, settings)
            secrets.extend(
                secret.get_secret_value()
                for secret in (
                    settings.embedding_api_key,
                    settings.openai_api_key,
                    settings.qdrant_api_key,
                    settings.hindsight_api_llm_api_key,
                )
                if secret
            )
            secrets.extend(
                value
                for value in (
                    settings.embedding_base_url,
                    settings.qdrant_url,
                    settings.hindsight_api_llm_base_url,
                )
                if value
            )
        except Exception:
            workspace_root = os.environ.get("AMRB_WORKSPACE")
            if workspace_root:
                try:
                    settings = Settings(
                        _env_file=Path(workspace_root) / ".env",
                        _env_file_encoding="utf-8",
                        workspace_root=Path(workspace_root),
                    )
                    message = redact_sensitive_text(message, settings)
                    secrets.extend(
                        secret.get_secret_value()
                        for secret in (
                            settings.embedding_api_key,
                            settings.openai_api_key,
                            settings.qdrant_api_key,
                            settings.hindsight_api_llm_api_key,
                        )
                        if secret
                    )
                    secrets.extend(
                        value
                        for value in (
                            settings.embedding_base_url,
                            settings.qdrant_url,
                            settings.hindsight_api_llm_base_url,
                        )
                        if value
                    )
                except Exception:
                    pass
        for secret in secrets:
            if secret:
                message = message.replace(secret, "[REDACTED]")
        message = redact_service_text(message)
        if recovery:
            message += " " + recovery
        error_console.print(message)
        raise typer.Exit(code=1) from exc


def _json(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    typer.echo(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _print_resolved_configuration(configuration: dict[str, str]) -> None:
    labels = (
        ("Workspace root", "workspace_root"),
        ("Workspace source", "workspace_source"),
        ("Paper directory", "papers_directory"),
        ("Paper directory source", "papers_directory_source"),
        ("Qdrant backend", "qdrant_backend"),
        ("Qdrant target", "qdrant_target"),
        ("Qdrant target source", "qdrant_target_source"),
        ("Collection source", "qdrant_collection_source"),
        ("Embedding model", "embedding_model"),
        ("Embedding model source", "embedding_model_source"),
        ("Vector dimension", "embedding_dimension"),
        ("Embedding endpoint", "embedding_endpoint"),
        ("Embedding endpoint source", "embedding_endpoint_source"),
        ("Embedding key source", "embedding_api_key_source"),
        ("Chunk profile source", "chunk_profile_source"),
        ("Memory bank source", "hindsight_bank_source"),
        ("Hindsight UI", "hindsight_ui"),
        ("Hindsight UI status", "hindsight_ui_status"),
        ("Qdrant UI", "qdrant_ui"),
        ("Qdrant UI status", "qdrant_ui_status"),
    )
    for label, key in labels:
        if key in configuration:
            console.print(f"{label:<23} {configuration[key]}")


def _valid_web_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _workspace_has_legacy_data(root: Path) -> bool:
    return (
        (root / ".env").is_file()
        or (root / ".state" / "index.json").is_file()
        or any((root / "papers").glob("*.pdf"))
    )


def _setup_workspace_config(
    root: Path,
    *,
    name: str | None,
    discipline: str | None,
    research_focus: str | None,
    non_interactive: bool,
    adopt_legacy: bool,
    copied_identity: str | None,
) -> tuple[Settings, bool, bool, str | None]:
    """Initialize config after collecting metadata and any missing local settings."""
    config_exists = (root / "amrb.toml").is_file()
    copy_action: str | None = None
    current: Settings | None = None
    copied_from = copied_workspace_origin(root) if config_exists else None
    if copied_from is not None:
        identity_was_explicit = copied_identity is not None
        if copied_identity is None:
            if non_interactive:
                raise ValueError(
                    f"Workspace config is bound to {copied_from}. Choose --copied-identity reuse "
                    "or --copied-identity new explicitly."
                )
            console.print(
                f"This workspace config is bound to {copied_from}. Reusing keeps its existing "
                "Qdrant collection and Hindsight bank; creating a new identity selects new resources "
                "and archives the copied index manifest."
            )
            copied_identity = typer.prompt("Copied workspace identity [reuse/new]", default="reuse")
        copied_identity = copied_identity.strip().casefold()
        current = load_settings(root, allow_copied_workspace=True)
        new_workspace_id: str | None = None
        if copied_identity == "reuse":
            console.print(
                "Copied workspace identity preview:\n"
                f"  Rebind: {copied_from} → {root}\n"
                f"  Keep collection: {current.qdrant_collection}\n"
                f"  Keep Hindsight bank: {current.hindsight_bank_id}",
                markup=False,
            )
        elif copied_identity == "new":
            new_workspace_id = str(uuid.uuid4())
            new_collection, new_bank = derived_resource_ids(new_workspace_id)
            new_qdrant_path = (
                current.qdrant_path
                if current.qdrant_url
                else root / ".state" / f"qdrant-{new_collection.removeprefix('amrb_papers_')}"
            )
            archive = current.state_dir / "copied-workspace" / current.workspace_id
            console.print(
                "Copied workspace identity preview:\n"
                f"  New workspace ID: {new_workspace_id}\n"
                f"  New collection: {new_collection}\n"
                f"  New Hindsight bank: {new_bank}\n"
                f"  Local Qdrant path: {new_qdrant_path}\n"
                f"  Archive copied index manifest under: {archive}\n"
                "  Keep PDFs and old local vectors; do not reindex automatically.",
                markup=False,
            )
            if not identity_was_explicit and not no_prompts and not typer.confirm(
                "Apply this new workspace identity?", default=False
            ):
                raise ValueError("Copied workspace identity change canceled; existing files were left unchanged.")
        else:
            raise ValueError("--copied-identity must be either 'reuse' or 'new'.")
        current = resolve_copied_workspace_identity(
            root,
            copied_identity,
            new_workspace_id=new_workspace_id,
        )
        copy_action = copied_identity
    elif copied_identity is not None:
        raise ValueError("--copied-identity applies only when amrb.toml is bound to another path.")

    if config_exists:
        if name or discipline or research_focus:
            raise ValueError("Workspace metadata is persistent; edit amrb.toml explicitly to rename or retarget it.")
        current = current or load_settings(root)
        is_legacy = False
        selected_name = current.workspace_name
        selected_discipline = current.discipline
        selected_focus = current.research_focus
    else:
        is_legacy = _workspace_has_legacy_data(root)
        if non_interactive and not name:
            raise ValueError("Non-interactive setup requires --name when creating a workspace.")
        if is_legacy and not adopt_legacy:
            if non_interactive:
                raise ValueError("Existing AMRB data requires explicit --adopt-legacy confirmation.")
            console.print(
                f"Existing .env, papers, or index data was found in {root}. "
                "Adoption keeps the configured data targets in place."
            )
            adopt_legacy = typer.confirm("Adopt this existing workspace data?", default=False)
            if not adopt_legacy:
                raise ValueError("Workspace initialization canceled; existing files were left unchanged.")
        if non_interactive:
            selected_name = name or ""
            selected_discipline = discipline or ""
            selected_focus = research_focus or ""
        else:
            selected_name = name or typer.prompt("Research project name", default=root.name)
            selected_discipline = discipline if discipline is not None else typer.prompt(
                "Discipline or research area", default="", show_default=False
            )
            selected_focus = research_focus if research_focus is not None else typer.prompt(
                "Research focus (optional)", default="", show_default=False
            )
        current = Settings(
            _env_file=root / ".env",
            _env_file_encoding="utf-8",
            workspace_root=root,
        )

    env_values: dict[str, str] = {}
    selected_embedding_model = current.embedding_model
    if non_interactive:
        missing = []
        if not current.embedding_api_key or not current.embedding_api_key.get_secret_value():
            missing.append("EMBEDDING_API_KEY")
        if not current.embedding_base_url:
            missing.append("EMBEDDING_BASE_URL")
        if missing:
            raise ValueError(
                "Non-interactive setup is missing required settings: " + ", ".join(missing)
            )
    else:
        if not current.embedding_base_url:
            env_values["EMBEDDING_BASE_URL"] = typer.prompt(
                "OpenAI-compatible embedding endpoint",
                default="https://api.openai.com/v1",
                hide_input=True,
            )
        if not current.embedding_api_key or not current.embedding_api_key.get_secret_value():
            selected_embedding_model = typer.prompt("Embedding model", default=current.embedding_model)
            api_key = typer.prompt("Embedding API key", hide_input=True)
            if not api_key:
                raise ValueError("Embedding API key is required.")
            env_values["EMBEDDING_API_KEY"] = api_key

    if not current.qdrant_url and not non_interactive:
        mode = typer.prompt("Qdrant storage [local/remote]", default="local").strip().casefold()
        if mode not in {"local", "remote"}:
            raise ValueError("Choose either local or remote Qdrant storage.")
        if mode == "remote":
            remote_url = typer.prompt("Qdrant URL (input hidden)", hide_input=True)
            remote_key = typer.prompt("Qdrant API key (optional, input hidden)", hide_input=True, default="")
            if not remote_url:
                raise ValueError("A Qdrant URL is required for remote storage.")
            env_values["QDRANT_URL"] = remote_url
            if remote_key:
                env_values["QDRANT_API_KEY"] = remote_key
            if not current.qdrant_dashboard_url:
                dashboard = typer.prompt(
                    "Qdrant Dashboard URL (optional; hidden input)",
                    hide_input=True,
                    default="",
                )
                if dashboard:
                    if not _valid_web_url(dashboard):
                        raise ValueError("Qdrant Dashboard URL must be an http:// or https:// URL.")
                    env_values["QDRANT_DASHBOARD_URL"] = dashboard

    if current.qdrant_url and not current.qdrant_dashboard_url and not non_interactive:
        dashboard = typer.prompt(
            "Qdrant Dashboard URL (optional; hidden input)",
            hide_input=True,
            default="",
        )
        if dashboard:
            if not _valid_web_url(dashboard):
                raise ValueError("Qdrant Dashboard URL must be an http:// or https:// URL.")
            env_values["QDRANT_DASHBOARD_URL"] = dashboard

    if not current.hindsight_ui_url and not non_interactive:
        hindsight_ui = typer.prompt(
            "Hindsight Control Plane URL (optional; hidden input)",
            hide_input=True,
            default="",
        )
        if hindsight_ui:
            if not _valid_web_url(hindsight_ui):
                raise ValueError("Hindsight Control Plane URL must be an http:// or https:// URL.")
            env_values["HINDSIGHT_UI_URL"] = hindsight_ui

    shared_service = service_paths()
    configure_shared_service = (
        not hindsight_healthcheck(current)
        and not shared_service.env_file.exists()
        and not shared_service.unit_file.exists()
        and not current.hindsight_api_llm_api_key
        and not current.openai_api_key
    )
    if configure_shared_service:
        if non_interactive:
            raise ValueError("First-time Hindsight deployment requires HINDSIGHT_API_LLM_API_KEY.")
        provider = typer.prompt("Hindsight LLM provider", default=current.hindsight_api_llm_provider)
        model = typer.prompt("Hindsight LLM model", default=current.hindsight_api_llm_model)
        base_url = typer.prompt("Hindsight LLM endpoint (optional)", default=current.hindsight_api_llm_base_url or "", show_default=False)
        key = typer.prompt("Hindsight LLM API key", hide_input=True)
        if not key:
            raise ValueError("Hindsight LLM API key is required for a first-time service deployment.")
        env_values.update({
            "HINDSIGHT_API_LLM_PROVIDER": provider,
            "HINDSIGHT_API_LLM_MODEL": model,
            "HINDSIGHT_API_LLM_API_KEY": key,
        })
        if base_url:
            env_values["HINDSIGHT_API_LLM_BASE_URL"] = base_url

    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "amrb.toml"
    planned_files = [str(config_path)] if not config_exists else []
    if env_values:
        planned_files.append(f"{root / '.env'} (keys: {', '.join(sorted(env_values))}; values hidden)")
    if configure_shared_service:
        planned_files.append(
            f"{shared_service.env_file} (owner-only shared Hindsight service environment; affects all workspaces)"
        )
        if systemd_user_available():
            planned_files.append(f"{shared_service.unit_file} (managed shared Hindsight service)")
    if planned_files:
        console.print("Setup will write:")
        for path in planned_files:
            console.print(f"  {path}")
        if not non_interactive and not typer.confirm("Write these workspace and service settings?", default=False):
            raise ValueError("Workspace setup canceled; no files were written.")

    changed_keys: tuple[str, ...] = ()
    try:
        if env_values:
            changed_keys = write_environment_values(root / ".env", env_values)
        if not config_exists:
            current, _ = initialize_workspace_config(
                root,
                name=selected_name,
                discipline=selected_discipline,
                research_focus=selected_focus,
                embedding_provider=current.embedding_provider,
                embedding_model=selected_embedding_model,
                chunk_size=current.rag_chunk_size,
                chunk_overlap=current.rag_chunk_overlap,
                adopt_legacy=adopt_legacy,
            )
        else:
            current = load_settings(root)
    except Exception as exc:
        completed = ".env keys: " + ", ".join(changed_keys) if changed_keys else "none"
        raise RuntimeError(
            f"Workspace setup stopped after updating {completed}. Existing files were preserved; "
            "rerun `amrb setup` after correcting the reported issue."
        ) from exc
    return current, not config_exists, is_legacy, copy_action


def _batch_status_callback(progress_console: Console) -> Any:
    last_update: tuple[str, tuple[tuple[str, int], ...]] | None = None

    def report(
        local_id: str,
        status: str,
        counts: dict[str, int] | None,
    ) -> None:
        nonlocal last_update
        normalized_counts = tuple(sorted((counts or {}).items()))
        update = (status, normalized_counts)
        if update == last_update:
            return
        last_update = update
        summary = ""
        if counts:
            summary = f" · {counts.get('completed', 0)}/{counts.get('total', 0)} requests"
        progress_console.print(f"Batch {local_id}: {status}{summary}")

    return report


@app.command()
def setup(
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        help="Initialize or configure this research workspace.",
    ),
    name: str | None = typer.Option(None, "--name", help="Project name for a new workspace."),
    discipline: str | None = typer.Option(None, "--discipline", help="Field or research area."),
    research_focus: str | None = typer.Option(None, "--research-focus", help="Short optional research focus."),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Never prompt; missing values fail."),
    adopt_legacy: bool = typer.Option(False, "--adopt-legacy", help="Explicitly preserve data in an existing unconfigured directory."),
    copied_identity: str | None = typer.Option(None, "--copied-identity", help="For a copied workspace choose 'reuse' or 'new'."),
    confirm_legacy_index: bool = typer.Option(False, "--accept-legacy-index", help="Confirm that the current model and chunk settings match an existing unverified index."),
    adopt_existing_resources: bool = typer.Option(False, "--adopt-existing-resources", help="Confirm that this workspace may use an existing Qdrant collection or Hindsight bank with unknown ownership."),
    update_missions: bool = typer.Option(False, "--update-missions", help="Apply the generated mission changes to an existing Hindsight bank."),
    verify_embedding: bool = typer.Option(False, "--verify-embedding", help="Send one minimal embedding request; provider usage may be billable."),
) -> None:
    """Initialize or validate workspace configuration and research services."""
    explicit_workspace = workspace or (
        Path(os.environ["AMRB_WORKSPACE"]) if os.environ.get("AMRB_WORKSPACE") else None
    )
    if explicit_workspace is not None:
        root = explicit_workspace.expanduser().resolve()
    else:
        try:
            root = find_workspace_root()
        except FileNotFoundError:
            root = Path.cwd().resolve()
    os.environ["AMRB_WORKSPACE"] = str(root)
    no_prompts = non_interactive or not sys.stdin.isatty()
    current, created, adopted_legacy, copy_action = _run(
        lambda: _setup_workspace_config(
            root,
            name=name,
            discipline=discipline,
            research_focus=research_focus,
            non_interactive=no_prompts,
            adopt_legacy=adopt_legacy,
            copied_identity=copied_identity,
        )
    )
    notes: list[str] = []
    if created:
        notes.append(f"Created workspace configuration at {root / 'amrb.toml'}.")
    if adopted_legacy:
        notes.append("Existing workspace files and configured data targets were adopted in place.")
    if copy_action == "reuse":
        notes.append("Copied workspace was rebound to this path; its existing collection and bank identity were retained.")
    elif copy_action == "new":
        notes.append(
            "Copied workspace received new Qdrant and Hindsight identities. Its prior manifest was archived under "
            f"{current.state_dir / 'copied-workspace'}; papers were left in place and were not reindexed."
        )

    manifest = load_manifest(current.manifest_path)
    has_unverified = any(
        entry.get("paper_id") and not isinstance(entry.get("index_config"), dict)
        for entry in manifest.values()
    ) and not current.legacy_index_accepted
    if has_unverified:
        inspection = _run(lambda: inspect_legacy_index(current))
        console.print(
            "Existing index summary: "
            f"{inspection.papers} papers, {inspection.qdrant_chunks} chunks, "
            f"vector dimension {inspection.vector_dimension}; configured embedding "
            f"{current.embedding_provider}/{current.embedding_model}, "
            f"chunk size {current.rag_chunk_size}, overlap {current.rag_chunk_overlap}."
        )
        should_accept = confirm_legacy_index
        if not should_accept and not no_prompts:
            console.print(f"Will record the confirmation in {root / 'amrb.toml'}.")
            should_accept = typer.confirm(
                "Confirm these settings match the existing index and preserve its IDs and data?",
                default=False,
            )
        if should_accept:
            current = _run(lambda: accept_legacy_index(current, inspection))
            notes.append("Confirmed the existing index profile; paper files, IDs, and vectors were not rewritten.")
        elif no_prompts:
            error_console.print(
                "Existing index is unverified. Review its model, dimension, and chunk settings, "
                "then rerun with --accept-legacy-index."
            )
            raise typer.Exit(code=1)
        else:
            notes.append("Existing index remains unverified; Paper RAG search and indexing stay disabled until confirmed.")

    collection_exists, collection_points, collection_owner = _run(
        lambda: inspect_qdrant_resource(current)
    )
    if collection_owner and collection_owner != current.workspace_id:
        error_console.print(
            f"Qdrant collection '{current.qdrant_collection}' belongs to workspace "
            f"{collection_owner}; this workspace is {current.workspace_id}."
        )
        raise typer.Exit(code=1)
    if (
        collection_exists
        and collection_points
        and not collection_owner
        and not current.qdrant_resource_accepted
        and not current.legacy_index_accepted
        and not current.legacy_data_compatibility
    ):
        accepted = adopt_existing_resources
        if not accepted and not no_prompts:
            console.print(
                f"Collection {current.qdrant_collection} contains {collection_points} points and has no "
                "workspace ownership marker. Confirm that this workspace may adopt it."
            )
            accepted = typer.confirm("Adopt this existing Qdrant collection?", default=False)
        if accepted:
            current = _run(lambda: accept_existing_qdrant_resource(current))
            notes.append("Confirmed ownership of the existing populated Qdrant collection.")
        else:
            error_console.print(
                "The populated Qdrant collection has unknown ownership. Review the target and rerun "
                "with --adopt-existing-resources to confirm it."
            )
            raise typer.Exit(code=1)

    update_hindsight_missions = update_missions
    if hindsight_healthcheck(current):
        bank_plan = _run(lambda: inspect_hindsight_bank(current))
        if bank_plan.exists:
            if not current.hindsight_resource_accepted and not current.legacy_data_compatibility:
                accepted = adopt_existing_resources
                if not accepted and not no_prompts:
                    console.print(
                        f"Hindsight bank {bank_plan.bank_id} already exists and has no saved "
                        "workspace ownership confirmation. Setup will preserve its current missions."
                    )
                    accepted = typer.confirm("Use this existing Hindsight bank?", default=False)
                if not accepted:
                    error_console.print(
                        "The existing Hindsight bank has unknown ownership. Confirm it with "
                        "--adopt-existing-resources before setup."
                    )
                    raise typer.Exit(code=1)
                current = _run(lambda: accept_existing_hindsight_resource(current))
                notes.append("Confirmed ownership of the existing Hindsight bank; its missions were preserved.")

            if bank_plan.changes or bank_plan.unavailable_fields:
                console.print(f"Hindsight bank mission preview: {bank_plan.bank_id}")
                for change in bank_plan.changes:
                    console.print(f"  {change.field}", markup=False)
                    console.print(f"    Existing: {change.current}", markup=False)
                    console.print(f"    Proposed: {change.desired}", markup=False)
                if bank_plan.unavailable_fields:
                    console.print(
                        "  Current values could not be read for: "
                        + ", ".join(bank_plan.unavailable_fields)
                    )
                update_hindsight_missions = update_missions
                if not update_hindsight_missions and not no_prompts:
                    update_hindsight_missions = typer.confirm(
                        "Apply the generated mission changes to this bank?",
                        default=False,
                    )
                if update_hindsight_missions:
                    notes.append("Confirmed Hindsight mission updates for the existing bank.")
                else:
                    notes.append("Existing Hindsight missions will remain unchanged.")

    client_changes = _run(lambda: plan_workspace_client_files(root, current))
    if client_changes:
        console.print("Client setup preview:")
        for change in client_changes:
            console.print(f"  {change.path}: {change.summary}")
        if not no_prompts and not typer.confirm("Apply these client setup changes?", default=False):
            error_console.print("Client configuration was not changed; rerun setup to apply it.")
            raise typer.Exit(code=1)
        written = _run(lambda: write_workspace_client_files(client_changes, root))
        notes.append("Updated client files: " + ", ".join(str(path) for path in written))

    embedding_status = "unverified"
    should_verify_embedding = verify_embedding
    if not should_verify_embedding and not no_prompts:
        console.print("An embedding connectivity check sends one minimal request and may incur provider charges.")
        should_verify_embedding = typer.confirm("Verify the embedding endpoint now?", default=False)
    if should_verify_embedding:
        embedding_health, _ = _run(lambda: _check_embedding(current, verbose=False))
        embedding_status = embedding_health.status
        if embedding_health.message:
            notes.append(f"Embedding check: {embedding_health.message}")
    elif not current.embedding_api_key or not current.embedding_api_key.get_secret_value():
        embedding_status = "unavailable"
    else:
        notes.append("Embedding connectivity was not tested; rerun setup with --verify-embedding when ready.")

    if current.hindsight_ui_url:
        notes.append(f"Hindsight Control Plane: {probe_web_ui(current.hindsight_ui_url)}.")
    if current.qdrant_url:
        notes.append(
            "Qdrant Dashboard: "
            + (
                probe_web_ui(current.qdrant_dashboard_url)
                if current.qdrant_dashboard_url
                else "not configured (unavailable)"
            )
            + "."
        )

    recovery = (
        f"Workspace files and any completed backend setup steps are preserved. Rerun `amrb setup --workspace {root}`; "
        "check `amrb status --json` for the current state. Paper/vector data and Hindsight memory entries are preserved; "
        "explicitly approved setup metadata may already have been applied."
    )
    result = _run(
        lambda: setup_workspace(
            current,
            update_hindsight_missions=update_hindsight_missions,
        ),
        recovery=recovery,
    )
    result = replace(result, embedding=embedding_status)
    mcp_result = _run(
        lambda: verify_mcp_startup(root, current),
        recovery=(
            f"Workspace, client, and backend settings were preserved. Rerun `amrb setup --workspace {root}` "
            "after fixing the MCP launch issue."
        ),
    )
    console.print("AMRB setup\n")
    console.print(f"Workspace    {current.workspace_name} ({current.workspace_id})")
    console.print(f"Memory bank  {current.hindsight_bank_id}")
    console.print(f"Collection   {current.qdrant_collection}")
    console.print(f"MCP server   Ready ({len(mcp_result['tools'])} tools; workspace binding confirmed)")
    console.print(f"Embedding    {result.embedding.title()}")
    console.print(f"Paper RAG    {'Ready' if result.environment == 'ready' and result.qdrant == 'ready' else 'Degraded'}")
    console.print(f"Qdrant       {result.qdrant.title()}")
    console.print(f"Hindsight    {result.hindsight.title()}")
    console.print(f"Memory bank  {result.memory_bank.title()}")
    for note in (*notes, *result.notes):
        console.print(f"\n{note}")
    if "unavailable" in {
        result.qdrant,
        result.hindsight,
        result.memory_bank,
        result.embedding,
    }:
        raise typer.Exit(code=1)
    console.print("\nDone.")


@app.command()
def status(as_json: bool = typer.Option(False, "--json", help="Print stable JSON.")) -> None:
    """Show a lightweight workspace health summary."""
    result = _run(get_system_status)
    if as_json:
        _json(result)
    else:
        console.print("AMRB\n")
        console.print(f"Workspace    {result.workspace_name} ({result.workspace_id})")
        console.print(f"Memory bank  {result.memory_bank}")
        console.print(f"Collection   {result.collection}")
        console.print(f"Paper RAG    {result.paper_rag.title()}")
        console.print(f"Hindsight    {result.hindsight.title()}")
        console.print(f"Qdrant       {result.qdrant.title()}")
        console.print(f"\nPapers       {result.papers}")
        console.print(f"Chunks       {result.chunks}")
        _print_resolved_configuration(result.configuration)
    if {result.paper_rag, result.hindsight, result.qdrant} != {"ready"}:
        raise typer.Exit(code=1)


@app.command()
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Print stable JSON."),
    verbose: bool = typer.Option(False, "--verbose", help="Include sanitized low-level error details."),
    verify_embedding: bool = typer.Option(False, "--verify-embedding", help="Send one minimal embedding request; provider usage may be billable."),
) -> None:
    """Run deeper dependency, index, and memory-bank checks."""
    report, details = _run(lambda: run_doctor(verbose=verbose, verify_embedding=verify_embedding))
    if as_json:
        data = report.model_dump(mode="json")
        if verbose:
            data["details"] = details
        _json(data)
    else:
        console.print("AMRB Doctor\n")
        console.print(f"Workspace       {report.workspace_name} ({report.workspace_id})")
        console.print(f"Memory bank     {report.memory_bank_id}")
        console.print(f"Collection      {report.collection}")
        _print_resolved_configuration(report.configuration)
        labels = (
            ("Environment", report.environment),
            ("Embedding", report.embedding),
            ("Qdrant", report.qdrant),
            ("Paper index", report.paper_index),
            ("Hindsight", report.hindsight),
            ("Memory bank", report.memory_bank),
        )
        for label, component in labels:
            console.print(f"{label:<15} {component.status.upper()}")
            if component.message:
                console.print(f"  {component.message}")
        console.print(f"\nPapers          {report.papers}")
        console.print(f"Chunks          {report.chunks}")
        if verbose:
            for name, detail in details.items():
                console.print(f"\n{name}: {detail}")
    statuses = [
        report.environment.status,
        report.embedding.status,
        report.qdrant.status,
        report.paper_index.status,
        report.hindsight.status,
        report.memory_bank.status,
    ]
    if any(value != "ready" for value in statuses):
        raise typer.Exit(code=1)


@paper_app.command("add")
def paper_add(
    paths: list[str] = typer.Argument(..., help="One or more local PDF paths."),
    as_json: bool = typer.Option(False, "--json", help="Print stable JSON."),
    batch_mode: bool = typer.Option(
        False,
        "--batch",
        help="Use the provider's OpenAI-compatible Batch File API for offline indexing.",
    ),
) -> None:
    """Copy PDFs into papers/ and index them, optionally with Batch File."""
    source_paths = [Path(path) for path in paths]
    if batch_mode:
        progress_console = error_console if as_json else console
        status_update = _batch_status_callback(progress_console)

        def submitted(local_id: str, remote_id: str) -> None:
            progress_console.print(
                f"Batch submitted: {remote_id} · local job {local_id}. "
                f"Resume with `uv run amrb paper batch wait {local_id}`."
            )

        result = _run(
            lambda: add_papers_batch(
                source_paths,
                on_submitted=submitted,
                on_status=status_update,
            )
        )
    else:
        result = _run(lambda: add_papers(source_paths))
    _print_paper_add_result(result, as_json)


def _print_paper_add_result(result: Any, as_json: bool) -> None:
    if as_json:
        _json(result)
        if result.failed:
            raise typer.Exit(code=1)
        return
    for paper in result.added:
        console.print(f"Added       {paper.title}\n            {paper.file_name} · {paper.pages or '?'} pages · Indexed")
    for paper in result.updated:
        console.print(f"Updated     {paper.title}\n            {paper.file_name} · {paper.pages or '?'} pages · Indexed")
    for paper in result.skipped:
        console.print(f"Already indexed  {paper.file_name}")
    for failure in result.failed:
        console.print(f"Failed      {failure.source}")
        console.print(f"            {'File copied; index failed. ' if failure.copied else ''}{failure.message}")
    if result.failed:
        raise typer.Exit(code=1)


@batch_app.command("status")
def paper_batch_status_command(
    job_id: str = typer.Argument(..., help="Local AMRB Batch job ID."),
    as_json: bool = typer.Option(False, "--json", help="Print stable JSON."),
) -> None:
    """Inspect an AMRB Batch job."""
    status = _run(lambda: paper_batch_status(job_id))
    if as_json:
        _json(status)
    else:
        console.print(f"Local job  {status['job_id']}")
        if status.get("batch_id"):
            console.print(f"Provider   {status['batch_id']}")
        console.print(f"Status     {status.get('status', 'unknown')}")
        counts = status.get("request_counts")
        if counts:
            console.print(
                f"Requests   {counts.get('completed', 0)}/{counts.get('total', 0)} "
                f"completed · {counts.get('failed', 0)} failed"
            )


@batch_app.command("wait")
def paper_batch_wait_command(
    job_id: str = typer.Argument(..., help="Local AMRB Batch job ID."),
    as_json: bool = typer.Option(False, "--json", help="Print stable JSON."),
) -> None:
    """Wait for a submitted Batch job and apply its vectors to Qdrant."""
    progress_console = error_console if as_json else console
    status_update = _batch_status_callback(progress_console)

    result = _run(
        lambda: wait_paper_batch(job_id, on_status=status_update)
    )
    _print_paper_add_result(result, as_json)


@paper_app.command("sync")
def paper_sync(as_json: bool = typer.Option(False, "--json", help="Print stable JSON.")) -> None:
    """Incrementally reconcile papers/ with the vector index."""
    result = _run(sync_papers)
    if as_json:
        _json(result)
    else:
        console.print("Paper sync\n")
        console.print(f"Added      {result.added}")
        console.print(f"Updated    {result.updated}")
        console.print(f"Removed    {result.removed}")
        console.print(f"Unchanged  {result.unchanged}")
        if result.error:
            console.print(f"\nIndex error: {result.error}")
    if result.error:
        raise typer.Exit(code=1)


@paper_app.command("remove")
def paper_remove(identifier: str = typer.Argument(..., help="Paper filename, title, or paper_id.")) -> None:
    """Remove a paper file and its indexed vectors."""
    result = _run(lambda: remove_paper(identifier))
    if result is None:
        error_console.print(f"Paper not found: {identifier}")
        raise typer.Exit(code=1)
    console.print(f"Removed     {result.removed.title} ({result.removed.file_name})")


@paper_app.command("list")
def paper_list(as_json: bool = typer.Option(False, "--json", help="Print stable JSON.")) -> None:
    """List indexed papers and basic local metadata."""
    records = _run(list_paper_records)
    if as_json:
        _json({"papers": [record.model_dump(mode="json") for record in records], "count": len(records)})
        return
    console.print("Papers\n")
    for paper in records:
        pages = f"{paper.pages} pages" if paper.pages is not None else "pages unknown"
        identifier = paper.paper_id or "no paper ID"
        console.print(f"{paper.title:<48} {pages:<15} {paper.index_status:<14} {identifier}")
    console.print(f"\n{len(records)} papers")


def _service_status() -> dict[str, str]:
    settings = get_settings()
    managed = systemd_user_available() and service_is_active()
    api_ready = hindsight_healthcheck(settings)
    bank_health = _check_bank(settings) if api_ready else None
    return {
        "service": "running" if managed else ("unmanaged" if api_ready else "stopped"),
        "api": "ready" if api_ready else "unavailable",
        "bank": "ready" if bank_health and bank_health.status == "ready" else "unavailable",
    }


@service_app.command("status")
def service_status(as_json: bool = typer.Option(False, "--json", help="Print stable JSON.")) -> None:
    """Inspect Hindsight's managed service, API, and memory bank."""
    result = _run(_service_status)
    if as_json:
        _json(result)
    else:
        console.print("Hindsight\n")
        console.print(f"Service   {result['service'].title()}")
        console.print(f"API       {result['api'].title()}")
        console.print(f"Bank      {result['bank'].title()}")
    if result["api"] != "ready" or result["bank"] != "ready":
        raise typer.Exit(code=1)


@service_app.command("restart")
def service_restart() -> None:
    """Restart the managed Hindsight user service and wait for readiness."""
    _run(restart_service)
    console.print("Hindsight restarted and ready.")


@service_app.command("logs")
def service_log_lines(
    lines: int = typer.Option(100, min=1, max=1000),
) -> None:
    """Show a bounded tail of Hindsight service logs."""
    output = _run(lambda: service_logs(lines))
    typer.echo(output, nl=not output.endswith("\n"))


@open_app.command("hindsight")
def open_hindsight() -> None:
    """Open the configured Hindsight control plane in the system browser."""
    settings = get_settings()
    if not settings.hindsight_ui_url:
        error_console.print("Set HINDSIGHT_UI_URL to the official Hindsight UI address.")
        raise typer.Exit(code=1)
    if not _valid_web_url(settings.hindsight_ui_url):
        error_console.print("HINDSIGHT_UI_URL must be an http:// or https:// URL.")
        raise typer.Exit(code=1)
    if not hindsight_healthcheck(settings):
        error_console.print("Hindsight API is not ready. Run `amrb service status` first.")
        raise typer.Exit(code=1)
    ui_status = probe_web_ui(settings.hindsight_ui_url)
    if ui_status not in {"reachable", "reachable; authentication required"}:
        error_console.print(f"Hindsight Control Plane is {ui_status}.")
        raise typer.Exit(code=1)
    if not _run(lambda: webbrowser.open(settings.hindsight_ui_url)):
        error_console.print("Could not open the system browser.")
        raise typer.Exit(code=1)
    console.print("Opened Hindsight.")


@open_app.command("qdrant")
def open_qdrant() -> None:
    """Open the configured official Qdrant dashboard in the system browser."""
    settings = get_settings()
    if not settings.qdrant_url:
        error_console.print(
            "This workspace uses local embedded Qdrant, which has no server Dashboard. "
            "Use `amrb status` and `amrb paper list` to inspect it."
        )
        raise typer.Exit(code=1)
    if not settings.qdrant_dashboard_url:
        error_console.print("Set QDRANT_DASHBOARD_URL to the official Qdrant dashboard address.")
        raise typer.Exit(code=1)
    if not _valid_web_url(settings.qdrant_dashboard_url):
        error_console.print("QDRANT_DASHBOARD_URL must be an http:// or https:// URL.")
        raise typer.Exit(code=1)
    ui_status = probe_web_ui(settings.qdrant_dashboard_url)
    if ui_status not in {"reachable", "reachable; authentication required"}:
        error_console.print(f"Qdrant Dashboard is {ui_status}.")
        raise typer.Exit(code=1)
    if not _run(lambda: webbrowser.open(settings.qdrant_dashboard_url)):
        error_console.print("Could not open the system browser.")
        raise typer.Exit(code=1)
    console.print("Opened Qdrant.")


if __name__ == "__main__":
    app()

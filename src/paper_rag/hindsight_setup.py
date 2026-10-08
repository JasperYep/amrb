import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from paper_rag.config import Settings, get_settings, redact_sensitive_text
from paper_rag.service import hindsight_base_url, redact_service_text


def _section(markdown: str, heading: str) -> str:
    lines = markdown.splitlines()
    target = f"# {heading}"
    try:
        start = lines.index(target) + 1
    except ValueError as exc:
        raise ValueError(f"AGENTS.md is missing the '{heading}' section.") from exc
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("# ")),
        len(lines),
    )
    return "\n".join(lines[start:end]).strip()


def _hindsight_rest_json(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 8.0,
) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = Request(
        f"{hindsight_base_url()}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Hindsight bank request failed (HTTP {exc.code}).") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("Could not reach the shared Hindsight API for bank configuration.") from None
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("Hindsight returned an invalid bank configuration response.") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Hindsight returned an unexpected bank configuration response.")
    return value


@dataclass(frozen=True)
class MissionChange:
    field: str
    current: str | None
    desired: str


@dataclass(frozen=True)
class HindsightBankInspection:
    bank_id: str
    exists: bool
    name: str | None = None
    changes: tuple[MissionChange, ...] = ()
    unavailable_fields: tuple[str, ...] = ()


def _missions(settings: Settings) -> tuple[str, str, str]:
    workspace = settings.workspace_name.strip() or "AMRB workspace"
    domain = settings.discipline.strip() or "the workspace domain"
    focus = settings.research_focus.strip()
    focus_line = f"Workspace purpose: {focus}.\n\n" if focus else ""
    memory_policy = " ".join(settings.memory_policy.split())
    memory_policy_line = (
        f"Workspace memory policy: {memory_policy}.\n\n"
        if memory_policy
        else ""
    )
    retention_guidance = (
        f"Apply the workspace memory policy: {memory_policy}. "
        if memory_policy
        else "Retain explicit, durable judgments, hypotheses, decisions, concepts, and stable preferences. "
    )
    retain_mission = (
        f"Use this bank for durable knowledge associated with the {workspace} workspace. "
        f"The domain is {domain}.\n\n{focus_line}"
        f"{retention_guidance}"
        "Retain only knowledge the user states or confirms. "
        "Keep questions, temporary tasks, and source passages in their original context. "
        "When a new decision revises an older one, mark the new decision as current and preserve useful history."
    )
    reflect_mission = (
        f"Synthesize durable workspace knowledge for {workspace} ({domain}).\n\n"
        f"{focus_line}{memory_policy_line}"
        "Separate recalled knowledge from retrieved source evidence and the current synthesis. "
        "Identify meaningful changes in decisions and understanding over time."
    )
    observations_mission = (
        f"Consolidate durable knowledge for the {workspace} workspace ({domain}).\n\n"
        f"{focus_line}{memory_policy_line}"
        "Preserve attribution and meaningful changes over time. "
        "Distinguish user decisions from source content and tentative discussion."
    )
    return retain_mission, reflect_mission, observations_mission


def _bank_configuration(bank: dict[str, Any]) -> dict[str, Any]:
    config = bank.get("config") or bank.get("configuration") or bank.get("bank_config") or {}
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            return {}
    return config if isinstance(config, dict) else {}


def _inspect_hindsight_bank(settings: Settings) -> HindsightBankInspection:
    retain_mission, reflect_mission, observations_mission = _missions(settings)
    bank_id = settings.hindsight_bank_id
    query = urlencode({"q": bank_id, "limit": 100, "offset": 0})
    listed = _hindsight_rest_json("GET", f"/v1/default/banks?{query}")
    existing = next(
        (
            bank
            for bank in listed.get("banks", [])
            if isinstance(bank, dict) and str(bank.get("bank_id")) == bank_id
        ),
        None,
    )
    if existing is None:
        return HindsightBankInspection(bank_id=bank_id, exists=False)

    encoded_id = quote(bank_id, safe="")
    config_response = _hindsight_rest_json("GET", f"/v1/default/banks/{encoded_id}/config")
    config = config_response.get("config")
    if not isinstance(config, dict):
        config = _bank_configuration(existing)
    desired = {
        "mission": reflect_mission,
        "retain_mission": retain_mission,
        "reflect_mission": reflect_mission,
        "observations_mission": observations_mission,
    }
    changes: list[MissionChange] = []
    unavailable: list[str] = []
    for field, wanted in desired.items():
        actual = existing.get(field)
        if actual is None:
            actual = config.get(field)
        if actual is None:
            unavailable.append(field)
        elif actual != wanted:
            changes.append(MissionChange(field=field, current=str(actual), desired=wanted))
    return HindsightBankInspection(
        bank_id=bank_id,
        exists=True,
        name=str(existing.get("name") or "") or None,
        changes=tuple(changes),
        unavailable_fields=tuple(unavailable),
    )


async def inspect_hindsight_bank_async(settings: Settings | None = None) -> HindsightBankInspection:
    current = settings or get_settings()
    return _inspect_hindsight_bank(current)


def inspect_hindsight_bank(settings: Settings | None = None) -> HindsightBankInspection:
    return asyncio.run(inspect_hindsight_bank_async(settings))


async def configure_hindsight_bank(
    settings: Settings | None = None,
    *,
    update_existing: bool = False,
    adopt_existing: bool = False,
) -> str:
    current = settings or get_settings()
    bank_id = current.hindsight_bank_id
    retain_mission, reflect_mission, observations_mission = _missions(current)

    inspection = _inspect_hindsight_bank(current)
    encoded_id = quote(bank_id, safe="")
    config_updates = {
        "retain_mission": retain_mission,
        "reflect_mission": reflect_mission,
        "observations_mission": observations_mission,
    }
    if not inspection.exists:
        _hindsight_rest_json(
            "PUT",
            f"/v1/default/banks/{encoded_id}",
            body={
                "name": current.workspace_name,
                "mission": reflect_mission,
                **config_updates,
            },
        )
        action = "created"
    else:
        if not adopt_existing and not current.legacy_data_compatibility:
            raise ValueError(
                f"Hindsight bank '{bank_id}' already exists with unknown workspace ownership. "
                "Confirm adoption before using it."
            )
        if inspection.changes and not update_existing:
            return "needs-confirmation"
        if update_existing:
            _hindsight_rest_json(
                "PATCH",
                f"/v1/default/banks/{encoded_id}/config",
                body={"updates": config_updates},
            )
            action = "updated"
        else:
            return "unchanged"

    verified = _inspect_hindsight_bank(current)
    if not verified.exists:
        raise RuntimeError("Hindsight did not confirm the configured bank.")
    remaining = [change.field for change in verified.changes]
    if remaining:
        raise RuntimeError("Hindsight did not confirm the configured missions: " + ", ".join(remaining))
    return action


def main() -> None:
    try:
        result = asyncio.run(configure_hindsight_bank())
    except Exception as exc:
        message = str(exc)
        try:
            message = redact_sensitive_text(message, get_settings())
        except Exception:
            pass
        print(f"Could not configure the Hindsight bank: {redact_service_text(message)}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Hindsight bank: {get_settings().hindsight_bank_id} ({result})")


if __name__ == "__main__":
    main()

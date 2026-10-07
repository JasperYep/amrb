import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import Client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HINDSIGHT_MCP_URL = "http://127.0.0.1:8888/mcp/"
BANK_ID = "agent-memory-research"


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


async def _call(client: Client, name: str, arguments: dict[str, Any]) -> str:
    result = await client.call_tool(name, arguments)
    if result.is_error:
        message = "\n".join(item.text for item in result.content if item.type == "text")
        raise RuntimeError(f"Hindsight {name} failed: {message}")
    return "\n".join(item.text for item in result.content if item.type == "text")


async def configure_hindsight_bank() -> None:
    policy = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    memory_policy = _section(policy, "Memory")
    evidence_policy = _section(policy, "证据与判断")
    retain_mission = (
        "Use this bank for the user's durable Agent Memory research cognition. "
        "Apply this retention policy:\n\n" + memory_policy
    )
    reflect_mission = (
        "Synthesize the user's evolving long-term understanding of Agent Memory. "
        "Keep remembered judgments distinct from verified paper evidence. "
        "Apply this response distinction:\n\n" + evidence_policy
    )
    observations_mission = (
        "Consolidate durable Agent Memory research cognition while preserving the user's attribution "
        "and meaningful changes over time. Apply this retention policy:\n\n" + memory_policy
    )

    async with Client(HINDSIGHT_MCP_URL, read_timeout_seconds=30) as client:
        listed = await _call(client, "list_banks", {"query": BANK_ID})
        if BANK_ID not in listed:
            await _call(
                client,
                "create_bank",
                {"bank_id": BANK_ID, "name": "Agent Memory Research", "mission": reflect_mission},
            )
        await _call(
            client,
            "update_bank",
            {
                "bank_id": BANK_ID,
                "config_updates": {
                    "retain_mission": retain_mission,
                    "reflect_mission": reflect_mission,
                    "observations_mission": observations_mission,
                },
            },
        )
        bank = json.loads(await _call(client, "get_bank", {"bank_id": BANK_ID}))
        if bank.get("mission") != reflect_mission:
            raise RuntimeError("Hindsight did not confirm the configured reflect mission.")


def main() -> None:
    try:
        asyncio.run(configure_hindsight_bank())
    except Exception as exc:
        print(f"Could not configure the Hindsight bank: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Configured Hindsight bank: {BANK_ID}")


if __name__ == "__main__":
    main()

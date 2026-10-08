import asyncio

import pytest

from paper_rag.config import Settings
from paper_rag.hindsight_setup import (
    _missions,
    configure_hindsight_bank,
    inspect_hindsight_bank,
)


class _FakeHindsightREST:
    state = {}
    calls = []

    @classmethod
    def request(cls, method, path, body=None, **_kwargs):
        cls.calls.append((method, path, body))
        bank = cls.state.get("bank")
        bank_id = cls.state["bank_id"]
        if path.startswith("/v1/default/banks?"):
            return {
                "banks": [
                    {
                        "bank_id": bank_id,
                        "name": bank.get("name"),
                        "mission": bank.get("mission"),
                    }
                ]
                if bank
                else []
            }
        if path == f"/v1/default/banks/{bank_id}/config":
            if method == "GET":
                return {"bank_id": bank_id, "config": cls.state.get("config", {})}
            if method == "PATCH":
                cls.state.setdefault("config", {}).update(body["updates"])
                cls.state["bank"]["mission"] = body["updates"]["reflect_mission"]
                return {"bank_id": bank_id, "config": cls.state["config"]}
        if path == f"/v1/default/banks/{bank_id}":
            if method == "PUT":
                cls.state["bank"] = {
                    "bank_id": bank_id,
                    "name": body["name"],
                    "mission": body["mission"],
                }
                cls.state["config"] = {
                    key: body[key]
                    for key in ("retain_mission", "reflect_mission", "observations_mission")
                }
                return cls.state["bank"]
        raise AssertionError(f"unexpected request: {method} {path}")


def _fake_rest(monkeypatch, settings, bank, config=None):
    _FakeHindsightREST.state = {
        "bank_id": settings.hindsight_bank_id,
        "bank": bank,
        "config": config or {},
    }
    _FakeHindsightREST.calls = []
    monkeypatch.setattr("paper_rag.hindsight_setup._hindsight_rest_json", _FakeHindsightREST.request)


def test_existing_hindsight_bank_mission_diff_is_inspectable(monkeypatch):
    settings = Settings(_env_file=None, hindsight_bank_id="project-memory", workspace_name="Project")
    _fake_rest(
        monkeypatch,
        settings,
        {
            "bank_id": settings.hindsight_bank_id,
            "name": settings.workspace_name,
            "mission": "old reflect mission",
        },
        {
            "retain_mission": "old retain mission",
            "reflect_mission": "old reflect mission",
            "observations_mission": "old observations mission",
        },
    )

    plan = inspect_hindsight_bank(settings)

    assert plan.exists is True
    assert plan.name == "Project"
    assert {change.field for change in plan.changes} == {
        "mission",
        "retain_mission",
        "reflect_mission",
        "observations_mission",
    }
    assert not plan.unavailable_fields
    assert not any(
        method == "GET" and path == f"/v1/default/banks/{settings.hindsight_bank_id}"
        for method, path, _ in _FakeHindsightREST.calls
    )


def test_existing_hindsight_missions_change_only_after_explicit_confirmation(monkeypatch):
    settings = Settings(_env_file=None, hindsight_bank_id="project-memory", workspace_name="Project")
    _fake_rest(
        monkeypatch,
        settings,
        {
            "bank_id": settings.hindsight_bank_id,
            "name": settings.workspace_name,
            "mission": "old mission",
        },
        {"retain_mission": "old retain", "observations_mission": "old observations"},
    )

    deferred = asyncio.run(configure_hindsight_bank(settings, adopt_existing=True))
    assert deferred == "needs-confirmation"
    assert not any(method == "PATCH" for method, _, _ in _FakeHindsightREST.calls)

    updated = asyncio.run(
        configure_hindsight_bank(settings, adopt_existing=True, update_existing=True)
    )
    assert updated == "updated"
    retain, reflect, observations = _missions(settings)
    assert _FakeHindsightREST.state["bank"]["mission"] == reflect
    assert _FakeHindsightREST.state["config"] == {
        "retain_mission": retain,
        "reflect_mission": reflect,
        "observations_mission": observations,
    }


def test_existing_hindsight_bank_requires_explicit_adoption(monkeypatch):
    settings = Settings(_env_file=None, hindsight_bank_id="project-memory")
    _fake_rest(
        monkeypatch,
        settings,
        {"bank_id": settings.hindsight_bank_id, "mission": "existing"},
    )

    with pytest.raises(ValueError, match="unknown workspace ownership"):
        asyncio.run(configure_hindsight_bank(settings))


def test_missing_hindsight_bank_is_created_with_workspace_missions(monkeypatch):
    settings = Settings(_env_file=None, hindsight_bank_id="new-memory", workspace_name="New project")
    _fake_rest(monkeypatch, settings, None)

    result = asyncio.run(configure_hindsight_bank(settings))

    retain, reflect, observations = _missions(settings)
    assert result == "created"
    assert _FakeHindsightREST.state["bank"]["name"] == settings.workspace_name
    assert _FakeHindsightREST.state["config"] == {
        "retain_mission": retain,
        "reflect_mission": reflect,
        "observations_mission": observations,
    }
    assert any(method == "PUT" for method, _, _ in _FakeHindsightREST.calls)

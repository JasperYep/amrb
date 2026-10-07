import subprocess

import pytest

from paper_rag.config import Settings
from paper_rag import service


def _success(args, **kwargs):
    return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n", stderr="")


def test_install_user_service_writes_private_env_and_resolves_uvx(tmp_path, monkeypatch):
    paths = service.service_paths(tmp_path)
    paths.config_dir.mkdir(parents=True)
    paths.env_file.write_text("CUSTOM_SETTING=keep\nHINDSIGHT_API_LLM_MODEL=custom-model\n")
    commands = []

    def runner(args, **kwargs):
        commands.append(args)
        return _success(args, **kwargs)

    monkeypatch.setattr(service, "systemd_user_available", lambda runner=runner: True)
    monkeypatch.setattr(service, "service_is_active", lambda runner=runner: False)
    monkeypatch.setattr(service, "hindsight_healthcheck", lambda settings=None: False)
    monkeypatch.setattr(service.shutil, "which", lambda name: "/opt/uv tools/uvx" if name == "uvx" else None)

    settings = Settings(
        _env_file=None,
        hindsight_api_llm_provider="openai",
        hindsight_api_llm_base_url="https://llm.example/v1",
        OPENAI_API_KEY="test-openai-key",
    )
    changed = service.install_user_service(settings, tmp_path, runner)

    assert changed
    assert paths.env_file.stat().st_mode & 0o777 == 0o600
    assert "CUSTOM_SETTING=keep" in paths.env_file.read_text()
    assert "HINDSIGHT_API_LLM_MODEL=custom-model" in paths.env_file.read_text()
    assert 'HINDSIGHT_API_LLM_API_KEY="test-openai-key"' in paths.env_file.read_text()
    assert 'HINDSIGHT_API_LLM_BASE_URL="https://llm.example/v1"' in paths.env_file.read_text()
    unit = paths.unit_file.read_text()
    assert "hindsight-api==0.10.2" in unit
    assert "/opt/uv tools/uvx" in unit
    assert "API_KEY" not in unit
    assert ["systemctl", "--user", "daemon-reload"] in commands
    assert ["systemctl", "--user", "enable", "--now", service.HINDSIGHT_SERVICE_NAME] in commands


def test_manual_command_sources_private_env_without_exposing_key(tmp_path):
    settings = Settings(
        _env_file=None,
        hindsight_api_llm_provider="openai",
        OPENAI_API_KEY="test-openai-key",
    )
    env_file = service.prepare_service_environment(settings, tmp_path)

    command = service.manual_hindsight_command(settings, tmp_path)

    assert env_file.stat().st_mode & 0o777 == 0o600
    assert 'HINDSIGHT_API_LLM_API_KEY="test-openai-key"' in env_file.read_text()
    assert str(env_file) in command
    assert "test-openai-key" not in command
    assert "hindsight-api==" in command


def test_codex_provider_does_not_copy_unrelated_openai_key(tmp_path):
    settings = Settings(
        _env_file=None,
        hindsight_api_llm_provider="openai-codex",
        OPENAI_API_KEY="embedding-or-openai-key",
    )

    env_file = service.prepare_service_environment(settings, tmp_path)
    contents = env_file.read_text()

    assert "HINDSIGHT_API_LLM_API_KEY" not in contents
    assert "embedding-or-openai-key" not in contents


def test_install_is_idempotent_and_does_not_rewrite_env(tmp_path, monkeypatch):
    paths = service.service_paths(tmp_path)
    paths.config_dir.mkdir(parents=True)
    paths.env_file.write_text("HINDSIGHT_API_LLM_PROVIDER=custom-provider\n")
    commands = []

    def runner(args, **kwargs):
        commands.append(args)
        return _success(args, **kwargs)

    monkeypatch.setattr(service, "systemd_user_available", lambda runner=runner: True)
    monkeypatch.setattr(service, "service_is_active", lambda runner=runner: True)
    monkeypatch.setattr(service, "hindsight_healthcheck", lambda settings=None: True)
    monkeypatch.setattr(service.shutil, "which", lambda name: "/usr/local/bin/uvx")
    settings = Settings(_env_file=None)

    service.install_user_service(settings, tmp_path, runner)
    original_env = paths.env_file.read_text()
    commands.clear()
    changed = service.install_user_service(settings, tmp_path, runner)

    assert changed is False
    assert paths.env_file.read_text() == original_env
    assert not any(command[-1] == "daemon-reload" for command in commands)
    assert ["systemctl", "--user", "enable", "--now", service.HINDSIGHT_SERVICE_NAME] in commands


def test_install_reports_missing_systemd(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "systemd_user_available", lambda runner=subprocess.run: False)
    with pytest.raises(RuntimeError, match="systemd --user is unavailable"):
        service.install_user_service(Settings(_env_file=None), tmp_path)


def test_restart_waits_until_api_is_ready(monkeypatch):
    commands = []

    def runner(args, **kwargs):
        commands.append(args)
        return _success(args, **kwargs)

    ready = iter([False, True])
    monkeypatch.setattr(service, "systemd_user_available", lambda runner=runner: True)
    monkeypatch.setattr(service, "hindsight_healthcheck", lambda settings=None: next(ready))
    monkeypatch.setattr(service.time, "sleep", lambda delay: None)

    service.restart_service(runner=runner, timeout=1)
    assert commands == [["systemctl", "--user", "restart", service.HINDSIGHT_SERVICE_NAME]]


def test_logs_are_bounded_and_use_journalctl(monkeypatch):
    commands = []

    def runner(args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="recent log\n", stderr="")

    monkeypatch.setattr(service, "systemd_user_available", lambda runner=runner: True)
    assert service.service_logs(25, runner) == "recent log\n"
    assert commands == [["journalctl", "--user", "-u", service.HINDSIGHT_SERVICE_NAME,
                         "-n", "25", "--no-pager"]]
    with pytest.raises(ValueError, match="1 到 1000"):
        service.service_logs(0, runner)

from __future__ import annotations

import os
import re
import signal
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from paper_rag.config import Settings, get_settings

DEFAULT_HINDSIGHT_VERSION = "0.10.2"
HINDSIGHT_SERVICE_NAME = "amrb-hindsight.service"
DEFAULT_HINDSIGHT_HOST = "127.0.0.1"
HINDSIGHT_API_PORT = 8888
HINDSIGHT_BANK_ID = "agent-memory-research"
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ServicePaths:
    config_dir: Path
    env_file: Path
    unit_file: Path


def service_paths(home: Path | None = None) -> ServicePaths:
    root = home or Path.home()
    config_dir = root / ".config" / "amrb"
    return ServicePaths(
        config_dir=config_dir,
        env_file=config_dir / "hindsight.env",
        unit_file=root / ".config" / "systemd" / "user" / HINDSIGHT_SERVICE_NAME,
    )


def hindsight_base_url(settings: Settings | None = None) -> str:
    return f"http://127.0.0.1:{HINDSIGHT_API_PORT}"


def hindsight_mcp_url(settings: Settings | None = None) -> str:
    return f"{hindsight_base_url(settings)}/mcp/{HINDSIGHT_BANK_ID}/"


def hindsight_healthcheck(settings: Settings | None = None, timeout: float = 2.0) -> bool:
    url = f"{hindsight_base_url(settings)}/health"
    try:
        with urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def _run(
    args: Sequence[str],
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    return runner(args, capture_output=True, text=True, check=False)


def systemd_user_available(runner: Runner = subprocess.run) -> bool:
    if os.name != "posix" or shutil.which("systemctl") is None:
        return False
    result = _run(["systemctl", "--user", "show-environment"], runner)
    return result.returncode == 0


def service_is_active(runner: Runner = subprocess.run) -> bool:
    if not systemd_user_available(runner):
        return False
    return _run(
        ["systemctl", "--user", "is-active", "--quiet", HINDSIGHT_SERVICE_NAME], runner
    ).returncode == 0


def _env_value(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("Hindsight service configuration values must be single-line.")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ensure_env_file(paths: ServicePaths, settings: Settings) -> None:
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    defaults = {
        "HINDSIGHT_API_LLM_PROVIDER": settings.hindsight_api_llm_provider,
        "HINDSIGHT_API_LLM_MODEL": settings.hindsight_api_llm_model,
    }
    if settings.hindsight_api_llm_base_url:
        defaults["HINDSIGHT_API_LLM_BASE_URL"] = settings.hindsight_api_llm_base_url
    existing = paths.env_file.read_text(encoding="utf-8") if paths.env_file.exists() else ""
    present = {
        match.group(1)
        for line in existing.splitlines()
        if (match := re.match(r"\s*([A-Z0-9_]+)\s*=", line))
    }
    additions = [
        f"{key}={_env_value(value)}"
        for key, value in defaults.items()
        if key not in present
    ]
    api_key = settings.hindsight_api_llm_api_key
    if api_key is None and settings.hindsight_api_llm_provider.casefold() == "openai":
        api_key = settings.openai_api_key
    if "HINDSIGHT_API_LLM_API_KEY" not in present and api_key:
        secret = api_key.get_secret_value()
        if secret:
            additions.append(f"HINDSIGHT_API_LLM_API_KEY={_env_value(secret)}")
    if additions:
        separator = "" if not existing or existing.endswith("\n") else "\n"
        paths.env_file.write_text(
            existing + separator + "\n".join(additions) + "\n", encoding="utf-8"
        )
    paths.env_file.chmod(0o600)


def prepare_service_environment(
    settings: Settings | None = None,
    home: Path | None = None,
) -> Path:
    current = settings or get_settings()
    paths = service_paths(home)
    _ensure_env_file(paths, current)
    return paths.env_file


def _unit_contents(uvx: str) -> str:
    executable = '"' + uvx.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return f"""[Unit]
Description=AMRB Hindsight API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=%h/.config/amrb/hindsight.env
ExecStart={executable} --from hindsight-api=={DEFAULT_HINDSIGHT_VERSION} hindsight-api --host {DEFAULT_HINDSIGHT_HOST} --port {HINDSIGHT_API_PORT}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""


def _stop_existing_manual_server(
    proc_root: Path = Path("/proc"),
    timeout: float = 10.0,
) -> bool:
    if not hindsight_healthcheck():
        return False

    matching: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="ignore").split("\0")
        except (OSError, PermissionError):
            continue
        argv = [arg for arg in argv if arg]
        if (
            any(Path(arg).name == "hindsight-api" for arg in argv)
            and "--host" in argv
            and DEFAULT_HINDSIGHT_HOST in argv
        ):
            matching.append(int(entry.name))

    if not matching:
        raise RuntimeError(
            "Hindsight is already listening on port 8888, but its process is not a recognized "
            "Hindsight API command. Stop that process before enabling the managed service."
        )

    for pid in matching:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not hindsight_healthcheck():
            return True
        time.sleep(0.2)
    raise RuntimeError(
        "The existing Hindsight API did not stop after SIGTERM; the managed service was not started."
    )


def manual_hindsight_command(
    settings: Settings | None = None,
    home: Path | None = None,
) -> str:
    current = settings or get_settings()
    uvx = shutil.which("uvx") or "uvx"
    env_file = prepare_service_environment(current, home)
    command = (
        f"{shlex.quote(uvx)} --from hindsight-api=={DEFAULT_HINDSIGHT_VERSION} hindsight-api "
        f"--host {DEFAULT_HINDSIGHT_HOST} --port {HINDSIGHT_API_PORT}"
    )
    source = f"set -a; . {shlex.quote(str(env_file))}; set +a; exec"
    return f"{source} {command}"


def install_user_service(
    settings: Settings | None = None,
    home: Path | None = None,
    runner: Runner = subprocess.run,
) -> bool:
    current = settings or get_settings()
    if not systemd_user_available(runner):
        raise RuntimeError("systemd --user is unavailable")
    uvx = shutil.which("uvx")
    if uvx is None:
        raise RuntimeError("uvx was not found in PATH")

    paths = service_paths(home)
    prepare_service_environment(current, home)
    paths.unit_file.parent.mkdir(parents=True, exist_ok=True)
    contents = _unit_contents(uvx)
    previous = paths.unit_file.read_text(encoding="utf-8") if paths.unit_file.exists() else None
    changed = previous != contents
    if changed:
        temporary = paths.unit_file.with_suffix(paths.unit_file.suffix + ".tmp")
        temporary.write_text(contents, encoding="utf-8")
        os.replace(temporary, paths.unit_file)

    if changed:
        result = _run(["systemctl", "--user", "daemon-reload"], runner)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "systemd daemon-reload failed")

    active = service_is_active(runner)
    if not active and hindsight_healthcheck(current):
        _stop_existing_manual_server()
    if changed and active:
        enable_result = _run(
            ["systemctl", "--user", "enable", HINDSIGHT_SERVICE_NAME], runner
        )
        if enable_result.returncode != 0:
            raise RuntimeError(enable_result.stderr.strip() or "could not enable Hindsight")
        action = ["systemctl", "--user", "restart", HINDSIGHT_SERVICE_NAME]
    else:
        action = ["systemctl", "--user", "enable", "--now", HINDSIGHT_SERVICE_NAME]
    result = _run(action, runner)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "could not start the Hindsight service")
    return changed


def restart_service(
    runner: Runner = subprocess.run,
    timeout: float = 30.0,
    settings: Settings | None = None,
) -> None:
    if not systemd_user_available(runner):
        raise RuntimeError("systemd --user is unavailable")
    result = _run(["systemctl", "--user", "restart", HINDSIGHT_SERVICE_NAME], runner)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "could not restart the Hindsight service")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if hindsight_healthcheck(settings):
            return
        time.sleep(0.25)
    raise RuntimeError("Hindsight service restarted, but its API did not become ready.")


def service_logs(
    lines: int = 100,
    runner: Runner = subprocess.run,
) -> str:
    if lines < 1 or lines > 1000:
        raise ValueError("日志行数必须在 1 到 1000 之间。")
    if not systemd_user_available(runner):
        raise RuntimeError("systemd --user is unavailable; no managed service logs exist")
    result = _run(
        [
            "journalctl",
            "--user",
            "-u",
            HINDSIGHT_SERVICE_NAME,
            "-n",
            str(lines),
            "--no-pager",
        ],
        runner,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "could not read Hindsight service logs")
    return result.stdout

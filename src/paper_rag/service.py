from __future__ import annotations

import os
import plistlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from paper_rag.config import Settings, get_settings, redact_sensitive_text

DEFAULT_HINDSIGHT_VERSION = "0.10.2"
HINDSIGHT_SERVICE_NAME = "amrb-hindsight.service"
HINDSIGHT_LAUNCHD_LABEL = "com.amrb.hindsight"
DEFAULT_HINDSIGHT_HOST = "127.0.0.1"
HINDSIGHT_API_PORT = 8888
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ServicePaths:
    config_dir: Path
    env_file: Path
    unit_file: Path
    launch_agent_file: Path
    log_file: Path


def service_paths(home: Path | None = None) -> ServicePaths:
    root = home or Path.home()
    config_dir = root / ".config" / "amrb"
    return ServicePaths(
        config_dir=config_dir,
        env_file=config_dir / "hindsight.env",
        unit_file=root / ".config" / "systemd" / "user" / HINDSIGHT_SERVICE_NAME,
        launch_agent_file=root / "Library" / "LaunchAgents" / f"{HINDSIGHT_LAUNCHD_LABEL}.plist",
        log_file=root / "Library" / "Logs" / "amrb-hindsight.log",
    )


def hindsight_base_url(settings: Settings | None = None) -> str:
    return f"http://127.0.0.1:{HINDSIGHT_API_PORT}"


def hindsight_mcp_url(settings: Settings | None = None) -> str:
    current = settings or get_settings()
    return f"{hindsight_base_url(current)}/mcp/{current.hindsight_bank_id}/"


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


def launchd_user_available() -> bool:
    return sys.platform == "darwin" and shutil.which("launchctl") is not None


def service_is_active(runner: Runner = subprocess.run) -> bool:
    if systemd_user_available(runner):
        return _run(
            ["systemctl", "--user", "is-active", "--quiet", HINDSIGHT_SERVICE_NAME], runner
        ).returncode == 0
    if launchd_user_available():
        target = f"gui/{os.getuid()}/{HINDSIGHT_LAUNCHD_LABEL}"
        return _run(["launchctl", "print", target], runner).returncode == 0
    return False


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
    lines = existing.splitlines()
    present: dict[str, tuple[int, bool]] = {}
    for index, line in enumerate(lines):
        match = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        key, raw_value = match.groups()
        try:
            parsed = shlex.split(raw_value, comments=True, posix=True)
        except ValueError:
            parsed = [raw_value] if raw_value else []
        present[key] = (index, bool(parsed and parsed[0]))

    changed = False

    def set_value(key: str, value: str) -> None:
        nonlocal changed
        encoded = f"{key}={_env_value(value)}"
        current = present.get(key)
        if current and current[1]:
            return
        if current:
            index = current[0]
            lines[index] = encoded
        else:
            index = len(lines)
            lines.append(encoded)
        present[key] = (index, True)
        changed = True

    api_key = settings.hindsight_api_llm_api_key
    if api_key is None and settings.hindsight_api_llm_provider.casefold() == "openai":
        api_key = settings.openai_api_key
    if api_key:
        secret = api_key.get_secret_value()
        if secret:
            set_value("HINDSIGHT_API_LLM_API_KEY", secret)
    for key, value in defaults.items():
        set_value(key, value)
    if changed:
        content = "\n".join(lines).rstrip() + "\n"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=paths.config_dir,
                prefix=f".{paths.env_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                os.fchmod(temporary.fileno(), 0o600)
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, paths.env_file)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
    paths.env_file.chmod(0o600)


def prepare_service_environment(
    settings: Settings | None = None,
    home: Path | None = None,
) -> Path:
    current = settings or get_settings()
    paths = service_paths(home)
    _ensure_env_file(paths, current)
    return paths.env_file


def service_environment_has_llm_key(home: Path | None = None) -> bool:
    env_file = service_paths(home).env_file
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    has_key = False
    for line in lines:
        match = re.match(r"\s*(?:export\s+)?HINDSIGHT_API_LLM_API_KEY\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        try:
            values = shlex.split(match.group(1), comments=True, posix=True)
        except ValueError:
            has_key = False
        else:
            has_key = bool(values and values[0])
    return has_key


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


def _launch_agent_contents(env_file: Path, uvx: str, log_file: Path) -> bytes:
    return plistlib.dumps(
        {
            "Label": HINDSIGHT_LAUNCHD_LABEL,
            "ProgramArguments": [
                sys.executable,
                "-m",
                "paper_rag.service_runner",
                str(env_file),
                uvx,
            ],
            "WorkingDirectory": str(env_file.parents[2]),
            "RunAtLoad": True,
            "KeepAlive": True,
            "ThrottleInterval": 5,
            "StandardOutPath": str(log_file),
            "StandardErrorPath": str(log_file),
        },
        fmt=plistlib.FMT_XML,
        sort_keys=True,
    )


def _install_launch_agent(
    settings: Settings,
    paths: ServicePaths,
    uvx: str,
    runner: Runner,
) -> bool:
    paths.launch_agent_file.parent.mkdir(parents=True, exist_ok=True)
    paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    log_descriptor = os.open(paths.log_file, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    os.close(log_descriptor)
    paths.log_file.chmod(0o600)
    contents = _launch_agent_contents(paths.env_file, uvx, paths.log_file)
    previous = paths.launch_agent_file.read_bytes() if paths.launch_agent_file.exists() else None
    changed = previous != contents
    target = f"gui/{os.getuid()}/{HINDSIGHT_LAUNCHD_LABEL}"
    active = service_is_active(runner)

    if active and not changed:
        return False
    if not active and hindsight_healthcheck(settings):
        raise RuntimeError(
            "Hindsight is already running outside the AMRB LaunchAgent. Stop that process before enabling managed startup."
        )
    if active:
        result = _run(["launchctl", "bootout", target], runner)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "could not unload the existing Hindsight LaunchAgent")

    if changed:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=paths.launch_agent_file.parent,
                prefix=f".{paths.launch_agent_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                os.fchmod(temporary.fileno(), 0o600)
                temporary.write(contents)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, paths.launch_agent_file)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
    paths.launch_agent_file.chmod(0o600)

    result = _run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(paths.launch_agent_file)],
        runner,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "could not load the Hindsight LaunchAgent")
    return changed


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
    systemd_available = systemd_user_available(runner)
    launchd_available = not systemd_available and launchd_user_available()
    if not systemd_available and not launchd_available:
        raise RuntimeError("systemd --user is unavailable and macOS launchd is unavailable")
    uvx = shutil.which("uvx")
    if uvx is None:
        raise RuntimeError("uvx was not found in PATH")

    paths = service_paths(home)
    prepare_service_environment(current, home)
    if launchd_available:
        return _install_launch_agent(current, paths, uvx, runner)

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
    if systemd_user_available(runner):
        command = ["systemctl", "--user", "restart", HINDSIGHT_SERVICE_NAME]
    elif launchd_user_available():
        command = [
            "launchctl",
            "kickstart",
            "-k",
            f"gui/{os.getuid()}/{HINDSIGHT_LAUNCHD_LABEL}",
        ]
    else:
        raise RuntimeError("systemd --user is unavailable and macOS launchd is unavailable")
    result = _run(command, runner)
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
    settings: Settings | None = None,
) -> str:
    if lines < 1 or lines > 1000:
        raise ValueError("日志行数必须在 1 到 1000 之间。")
    if systemd_user_available(runner):
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
            message = redact_service_text(result.stderr.strip(), settings)
            raise RuntimeError(message or "could not read Hindsight service logs")
        return redact_service_text(result.stdout, settings)
    if launchd_user_available():
        log_file = service_paths().log_file
        try:
            output = "\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except OSError as exc:
            raise RuntimeError("No Hindsight LaunchAgent logs are available yet.") from exc
        return redact_service_text(output, settings)
    raise RuntimeError(
        "systemd --user is unavailable and macOS launchd is unavailable; no managed service logs exist"
    )


def redact_service_text(text: str, settings: Settings | None = None) -> str:
    output = text
    current = settings
    if current is None:
        try:
            current = get_settings()
        except (FileNotFoundError, ValueError):
            current = None
    if current is not None:
        output = redact_sensitive_text(output, current)

    sensitive_values = {
        value
        for name, value in os.environ.items()
        if value and any(token in name.casefold() for token in ("key", "token", "secret", "password"))
    }
    env_file = service_paths().env_file
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        name, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].replace("\\\"", "\"").replace("\\\\", "\\")
        if value and (
            any(token in name.casefold() for token in ("key", "token", "secret", "password"))
            or "url" in name.casefold()
        ):
            sensitive_values.add(value)
    for value in sorted(sensitive_values, key=len, reverse=True):
        output = output.replace(value, "[REDACTED]")
    return output

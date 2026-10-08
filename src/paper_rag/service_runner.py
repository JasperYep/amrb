from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

from paper_rag.service import DEFAULT_HINDSIGHT_HOST, DEFAULT_HINDSIGHT_VERSION, HINDSIGHT_API_PORT


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: service_runner <environment-file> <uvx-executable>")

    environment_file = Path(sys.argv[1])
    uvx = sys.argv[2]
    environment = os.environ.copy()
    for line_number, line in enumerate(environment_file.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        name, separator, value = stripped.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        try:
            parsed = shlex.split(value, comments=True, posix=True)
        except ValueError as exc:
            raise SystemExit(f"Invalid Hindsight environment value on line {line_number}.") from exc
        if len(parsed) > 1:
            raise SystemExit(f"Invalid Hindsight environment value on line {line_number}.")
        environment[name] = parsed[0] if parsed else ""

    os.execvpe(
        uvx,
        [
            uvx,
            "--from",
            f"hindsight-api=={DEFAULT_HINDSIGHT_VERSION}",
            "hindsight-api",
            "--host",
            DEFAULT_HINDSIGHT_HOST,
            "--port",
            str(HINDSIGHT_API_PORT),
        ],
        environment,
    )


if __name__ == "__main__":
    main()

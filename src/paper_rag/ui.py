from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit


def probe_web_ui(url: str | None, timeout: float = 2.0) -> str:
    if not url:
        return "not configured"
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "invalid URL"

    try:
        with urlopen(Request(url, method="HEAD"), timeout=timeout) as response:
            return "reachable" if 200 <= response.status < 400 else "unavailable"
    except HTTPError as exc:
        if exc.code in {401, 403}:
            return "reachable; authentication required"
        if exc.code != 405:
            return "unavailable"
    except (URLError, TimeoutError, OSError):
        return "unavailable"

    try:
        request = Request(url, headers={"Range": "bytes=0-0"}, method="GET")
        with urlopen(request, timeout=timeout) as response:
            return "reachable" if 200 <= response.status < 400 else "unavailable"
    except HTTPError as exc:
        if exc.code in {401, 403}:
            return "reachable; authentication required"
        return "unavailable"
    except (URLError, TimeoutError, OSError):
        return "unavailable"

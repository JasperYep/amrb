from urllib.error import HTTPError

from paper_rag.ui import probe_web_ui


class _Response:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_ui_probe_handles_unconfigured_invalid_and_reachable_urls(monkeypatch):
    assert probe_web_ui(None) == "not configured"
    assert probe_web_ui("file:///tmp/ui") == "invalid URL"
    monkeypatch.setattr("paper_rag.ui.urlopen", lambda request, timeout: _Response(200))

    assert probe_web_ui("https://ui.example") == "reachable"


def test_ui_probe_reports_authentication_and_head_fallback(monkeypatch):
    def auth_required(request, timeout):
        raise HTTPError(request.full_url, 401, "auth required", {}, None)

    monkeypatch.setattr("paper_rag.ui.urlopen", auth_required)
    assert probe_web_ui("https://ui.example") == "reachable; authentication required"

    calls = []

    def head_not_supported(request, timeout):
        calls.append(request.get_method())
        if request.get_method() == "HEAD":
            raise HTTPError(request.full_url, 405, "method not allowed", {}, None)
        return _Response(200)

    monkeypatch.setattr("paper_rag.ui.urlopen", head_not_supported)
    assert probe_web_ui("https://ui.example") == "reachable"
    assert calls == ["HEAD", "GET"]


def test_ui_probe_hides_network_error_details(monkeypatch):
    def fail(request, timeout):
        raise OSError("secret=never-print")

    monkeypatch.setattr("paper_rag.ui.urlopen", fail)
    assert probe_web_ui("https://ui.example") == "unavailable"

import re

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.realdata import REAL_DATA, changed, snapshot


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_DATA_DIR", str(tmp_path / "data"))
    before = snapshot(REAL_DATA)
    yield
    diff = changed(before, snapshot(REAL_DATA))
    assert not diff, f"this test changed the real data folder {REAL_DATA}: {diff}"


@pytest.fixture(autouse=True)
def fake_engines(monkeypatch):
    for kind in ("STT", "TTS", "LLM"):
        monkeypatch.setenv(f"LOCI_{kind}_ENGINE", "fake")


TEST_TOKEN = "test-token-" + "0123456789abcdef" * 2


def plain_page(html):
    """The page without its data-i18n markup: the attributes and the spans that only carry a key."""
    html = re.sub(r' data-i18n(?:-attr)?="[^"]*"', "", html)
    return re.sub(r"<span>(.*?)</span>", r"\1", html, flags=re.DOTALL)


@pytest.fixture(autouse=True)
def access_token(monkeypatch):
    """The app's access token is TEST_TOKEN, and a TestClient without a base_url is served at the
    loopback address and sends it; pass base_url or an X-Loci-Token header to a client to keep your own."""
    monkeypatch.setenv("LOCI_ACCESS_TOKEN", TEST_TOKEN)
    original = TestClient.__init__

    def init(self, app, *args, **kwargs):
        if not args and "base_url" not in kwargs:
            kwargs["base_url"] = "http://127.0.0.1:8770"
        headers = dict(kwargs.get("headers") or {})
        if not any(name.lower() == "x-loci-token" for name in headers):
            headers["X-Loci-Token"] = TEST_TOKEN
        kwargs["headers"] = headers
        original(self, app, *args, **kwargs)

    monkeypatch.setattr(TestClient, "__init__", init)


@pytest.fixture
def settings():
    return Settings.from_env()


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client

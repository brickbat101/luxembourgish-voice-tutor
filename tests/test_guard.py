from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.guard import TOKEN_FILE, load_token
from app.main import create_app
from scripts import smoke
from tests.conftest import TEST_TOKEN

HOME = "http://127.0.0.1:8770"
NO_TOKEN = {"X-Loci-Token": ""}


def bare_client(settings, **kwargs):
    return TestClient(create_app(settings), **kwargs)


def test_another_host_gets_421_with_the_address_to_open(settings):
    with bare_client(settings, base_url="http://testserver") as client:
        for response in (client.get("/"), client.post("/api/session"), client.get("/api/health")):
            assert response.status_code == 421
            assert response.json() == {
                "error": "this page is served at http://127.0.0.1:8770",
                "fix": "open http://127.0.0.1:8770",
            }
        evil = client.get("/", headers={"Host": "evil.com:8770"})
    assert evil.status_code == 421


def test_both_loopback_names_are_served(settings):
    with bare_client(settings, base_url="http://localhost:8770") as client:
        assert client.get("/api/health").status_code == 200
    with bare_client(settings, base_url="http://127.0.0.1:8771") as client:
        assert client.get("/api/health").status_code == 421


def test_the_port_of_the_settings_decides_the_host(settings):
    with bare_client(replace(settings, port=9001), base_url="http://127.0.0.1:9001") as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/", headers={"Host": "127.0.0.1:8770"}).status_code == 421


def test_an_api_route_needs_the_token_but_health_does_not(settings):
    with bare_client(settings, base_url=HOME, headers=NO_TOKEN) as client:
        for method, path in (
            ("POST", "/api/session"),
            ("GET", "/api/review/x"),
            ("GET", "/api/learner/anything"),
            ("GET", "/api/health/deeper"),
        ):
            response = client.request(method, path)
            assert response.status_code == 401, path
            assert response.json() == {"error": "no access token", "fix": "reload the page"}
        assert client.post("/api/session", headers={"X-Loci-Token": "wrong"}).status_code == 401
        assert client.get("/api/health").status_code == 200
        assert client.get("/").status_code == 200


PRONUNCIATION_ROUTES = (
    ("GET", "/api/pronunciation/flags", None),
    ("POST", "/api/pronunciation/flags", {"lang": "lb", "word": "Moien", "text": "Moien!"}),
    ("DELETE", "/api/pronunciation/flags/abcd1234", None),
    ("POST", "/api/pronunciation/words", {"lang": "lb", "word": "Moien", "respelling": "mojen"}),
)


@pytest.mark.parametrize(("method", "path", "body"), PRONUNCIATION_ROUTES)
def test_the_pronunciation_routes_need_the_host_the_origin_and_the_token(settings, method, path, body):
    token = {"X-Loci-Token": TEST_TOKEN}
    with bare_client(settings, base_url=HOME, headers=NO_TOKEN) as client:
        missing = client.request(method, path, json=body)
        assert missing.status_code == 401
        assert missing.json() == {"error": "no access token", "fix": "reload the page"}
        origin = client.request(method, path, json=body, headers={**token, "Origin": "http://evil.com"})
        assert origin.status_code == 403
        host = client.request(method, path, json=body, headers={**token, "Host": "evil.com:8770"})
        assert host.status_code == 421


def test_the_right_host_and_token_are_served(client):
    assert client.post("/api/session").status_code == 200
    assert client.get("/api/review/x").status_code in (200, 404, 422)
    assert client.headers["X-Loci-Token"] == TEST_TOKEN


def test_another_origin_gets_403_even_with_the_token(client):
    response = client.post("/api/session", headers={"Origin": "http://evil.com"})

    assert response.status_code == 403
    assert response.json() == {
        "error": "request from another origin",
        "fix": "open the page from http://127.0.0.1:8770",
    }
    assert client.post("/api/session", headers={"Origin": HOME}).status_code == 200
    assert client.post("/api/session", headers={"Origin": "http://localhost:8770"}).status_code == 200


def test_the_page_holds_the_token_and_index_html_redirects_to_it(client):
    page = client.get("/")

    assert page.status_code == 200
    assert f'<meta name="loci-token" content="{TEST_TOKEN}">' in page.text
    assert page.headers["cache-control"] == "no-store"
    redirect = client.get("/index.html", follow_redirects=False)
    assert redirect.status_code == 307 and redirect.headers["location"] == "/"
    assert TEST_TOKEN not in client.get("/style.css").text


def test_the_page_source_has_an_empty_token_meta_tag():
    source = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")

    assert source.count('<meta name="loci-token" content="">') == 1


def test_the_environment_token_wins_over_the_file(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / TOKEN_FILE).write_text("from-the-file", encoding="utf-8")

    assert load_token(data_dir, "from-the-environment") == ("from-the-environment", "")
    assert load_token(data_dir, None) == ("from-the-file", "")


def test_the_token_file_is_made_once_and_reused(settings, monkeypatch):
    monkeypatch.delenv("LOCI_ACCESS_TOKEN")
    path = settings.data_dir / TOKEN_FILE

    first = create_app(settings).state.access_token
    second = create_app(settings).state.access_token

    assert first == second == path.read_text(encoding="utf-8")
    assert len(first) >= 32 and first != TEST_TOKEN


def test_a_token_that_cannot_be_written_lives_in_memory_and_health_says_so(settings, monkeypatch):
    monkeypatch.delenv("LOCI_ACCESS_TOKEN")

    def refuse(self, *args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(Path, "write_text", refuse)
    token, warning = load_token(settings.data_dir, None)

    assert token and "could not be written" in warning and "read-only" in warning
    assert str(settings.data_dir / TOKEN_FILE) in warning
    with bare_client(settings, base_url=HOME) as client:
        health = client.get("/api/health").json()
        token = client.app.state.access_token
        assert client.post("/api/session", headers={"X-Loci-Token": token}).status_code == 200
    assert health["access"]["persisted"] is False
    assert "could not be written" in health["access"]["detail"]
    assert not (settings.data_dir / TOKEN_FILE).exists()


def test_health_says_a_persisted_token_is_persisted(client):
    assert client.get("/api/health").json()["access"] == {"persisted": True, "detail": ""}


def test_the_token_never_appears_in_the_settings(settings):
    assert TEST_TOKEN not in repr(settings)


def test_the_smoke_script_reads_the_environment_then_the_file(tmp_path, monkeypatch):
    assert smoke.read_token(tmp_path) == TEST_TOKEN
    monkeypatch.delenv("LOCI_ACCESS_TOKEN")
    with pytest.raises(SystemExit) as exit_info:
        smoke.read_token(tmp_path)
    message = str(exit_info.value)
    assert str(tmp_path / TOKEN_FILE) in message and "start the server once with run.ps1" in message
    (tmp_path / TOKEN_FILE).write_text("from-the-file\n", encoding="utf-8")
    assert smoke.read_token(tmp_path) == "from-the-file"


def test_the_page_sends_the_token_on_every_request():
    script = (Path(__file__).resolve().parent.parent / "static" / "app.js").read_text(encoding="utf-8")

    assert "meta[name=\"loci-token\"]" in script
    assert '"X-Loci-Token": TOKEN' in script
    assert script.count("fetch(") == 1

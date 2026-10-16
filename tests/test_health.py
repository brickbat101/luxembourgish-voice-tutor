from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.engines import factory
from app.engines.base import LLMEngine, STTEngine, TTSEngine
from app.main import create_app


def test_health_reports_three_ready_fake_engines_and_no_encoder(client):
    response = client.get("/api/health")

    body = response.json()
    assert response.status_code == 200
    assert set(body) == {"status", "engines", "lod", "learner", "access"}
    assert body["status"] == "ok"
    assert body["engines"] == {
        "stt": {"name": "fake", "ready": True, "detail": ""},
        "tts": {"name": "fake", "ready": True, "detail": ""},
        "llm": {"name": "fake", "ready": True, "detail": ""},
        "acceptability": {"name": "none", "ready": True, "detail": "no encoder installed"},
    }
    # Whether the LOD store is built depends on the data folder; tests/test_orchestrator.py covers both.
    assert set(body["lod"]) == {"ready", "detail"}
    assert set(body["learner"]) == {"ready", "detail", "path"}


def test_engines_are_built_onto_app_state(client):
    state = client.app.state

    assert isinstance(state.stt, STTEngine)
    assert isinstance(state.tts, TTSEngine)
    assert isinstance(state.llm, LLMEngine)


def test_missing_engine_module_is_reported_not_ready(settings, monkeypatch):
    monkeypatch.setitem(factory.ENGINES["stt"], "absent", "app.engines.no_such_module:AbsentSTT")

    with TestClient(create_app(replace(settings, stt_engine="absent"))) as client:
        response = client.get("/api/health")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "degraded"
    assert body["engines"]["stt"]["name"] == "absent"
    assert body["engines"]["stt"]["ready"] is False
    assert "app.engines.no_such_module" in body["engines"]["stt"]["detail"]
    assert body["engines"]["tts"]["ready"] is True
    assert body["engines"]["llm"]["ready"] is True


def test_unknown_engine_name_is_reported_not_ready(settings):
    with TestClient(create_app(replace(settings, llm_engine="nonsense"))) as client:
        llm = client.get("/api/health").json()["engines"]["llm"]

    assert llm["name"] == "nonsense"
    assert llm["ready"] is False
    assert "unknown llm engine 'nonsense'" in llm["detail"]


def test_engine_that_fails_to_warm_up_is_reported_not_ready(settings, monkeypatch):
    monkeypatch.setitem(factory.ENGINES["tts"], "broken", f"{__name__}:BrokenTTS")

    with TestClient(create_app(replace(settings, tts_engine="broken"))) as client:
        tts = client.get("/api/health").json()["engines"]["tts"]
        engine = client.app.state.tts

    assert tts == {"name": "broken", "ready": False, "detail": "OSError: voice file missing"}
    with pytest.raises(RuntimeError, match="voice file missing"):
        engine.synthesize([])


class BrokenTTS:
    name = "broken"

    def __init__(self, settings):
        pass

    def warmup(self):
        raise OSError("voice file missing")

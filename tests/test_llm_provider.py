import json
import logging
import sys
import types
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.content import load_practice
from app.engines import llm_provider, providers
from app.engines.base import TEMPLATE_WRAP_UP, Judgement, LLMEngine, LLMError, Segment, Turn, WrapUpText
from app.engines.factory import ENGINES, UnavailableEngine, build_engine
from app.engines.fake import FakeLLM
from app.engines.llm_claude_cli import (
    JSON_SCHEMA,
    JUDGE_PROMPT_PATH,
    JUDGE_SCHEMA,
    SYSTEM_PROMPT_PATH,
    WRAP_UP_PROMPT_PATH,
    WRAP_UP_SCHEMA,
)
from app.engines.llm_provider import CallRecord, ProviderLLM
from app.engines.providers import PROVIDERS_FILE, ROLES, FakeProvider, ProviderEntry, ProvidersConfig, RoleConfig
from app.main import create_app

TUTOR = load_practice().tutor
HISTORY = [Turn("user", "Moien, wéi geet et?")]
GOOD_REPLY = {
    "segments": [{"lang": "lb", "text": "Gutt!"}, {"lang": "en", "text": "Good, thanks."}],
    "new_phrases": [],
    "mode": "chat",
    "claims": [],
}
ALL_ON_FAKE = {"providers": {"fake": {"kind": "fake"}}, "roles": {role: {"provider": "fake"} for role in ROLES}}


def call(engine: ProviderLLM, role: str):
    """One call of `role` through the engine's public methods."""
    if role == "tutor_chat":
        return engine.reply(HISTORY, "chat", "")
    if role == "teacher_explain":
        return engine.reply(HISTORY, "teacher", "")
    if role == "judge":
        return engine.judge("Wat schafft Dir?", "What do you do?", "Ech schaffen.", "")
    return engine.wrap_up("The learner answered one question.")


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), data_dir=tmp_path / "data")


@pytest.fixture
def fake():
    return FakeProvider(entry=ProviderEntry("fake", "fake", model="fake-default"))


@pytest.fixture
def engine(settings, fake):
    """Every role on one fake provider."""
    config = ProvidersConfig(
        {"fake": ProviderEntry("fake", "fake", model="fake-default", timeout_s=12.0)},
        {role: RoleConfig("fake", effort="low") for role in ROLES},
    )
    return ProviderLLM(settings, providers={"fake": fake}, config=config)


# --- the contract and the routing ---


def test_the_provider_engine_is_an_llm_engine_and_is_registered(engine):
    assert isinstance(engine, LLMEngine)
    assert engine.name == "provider"
    assert ENGINES["llm"]["provider"] == "app.engines.llm_provider:ProviderLLM"


def test_the_default_engine_is_still_the_claude_cli():
    assert Settings().llm_engine == "claude_cli"


def test_each_role_reaches_its_own_provider_with_its_model_effort_and_the_entry_timeout(settings):
    entries = {
        "a": ProviderEntry("a", "fake", model="a-default", timeout_s=30.0),
        "b": ProviderEntry("b", "fake", model="b-default", timeout_s=60.0),
    }
    roles = {
        "tutor_chat": RoleConfig("a", "chat-model", "low"),
        "teacher_explain": RoleConfig("b", "teacher-model", "high"),
        "judge": RoleConfig("a", "", "medium"),
        "wrapup": RoleConfig("b", "wrapup-model", ""),
    }
    fakes = {entry_id: FakeProvider(entry=entry) for entry_id, entry in entries.items()}
    engine = ProviderLLM(settings, providers=fakes, config=ProvidersConfig(entries, roles))

    for role in ROLES:
        call(engine, role)

    a, b = fakes["a"].calls, fakes["b"].calls
    assert [c["params"]["model"] for c in a] == ["chat-model", "a-default"]
    assert [c["params"]["model"] for c in b] == ["teacher-model", "wrapup-model"]
    assert [c["params"]["effort"] for c in a + b] == ["low", "medium", "high", ""]
    assert [c["params"]["timeout_s"] for c in a + b] == [30.0, 30.0, 60.0, 60.0]
    assert {c["params"]["temperature"] for c in a + b} == {0.3}
    assert [c["schema"] for c in a] == [JSON_SCHEMA, JUDGE_SCHEMA]
    assert [c["schema"] for c in b] == [JSON_SCHEMA, WRAP_UP_SCHEMA]
    assert "mode: chat" in a[0]["user"] and "mode: teacher" in b[0]["user"]
    assert a[0]["system"] == b[0]["system"] == SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    assert a[1]["system"] == JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    assert b[1]["system"] == WRAP_UP_PROMPT_PATH.read_text(encoding="utf-8")
    assert [(r.role, r.provider_id, r.model) for r in engine.records] == [
        ("tutor_chat", "a", "chat-model"),
        ("teacher_explain", "b", "teacher-model"),
        ("judge", "a", "a-default"),
        ("wrapup", "b", "wrapup-model"),
    ]


def test_a_valid_first_answer_is_parsed_and_recorded_as_one_valid_attempt(engine, fake):
    fake.latency_ms = 7.0
    fake.queue.append(GOOD_REPLY)

    reply = engine.reply(HISTORY, "chat", "")

    assert reply.segments == [Segment("lb", "Gutt!"), Segment("en", "Good, thanks.")]
    assert len(fake.calls) == 1
    assert engine.records[-1] == CallRecord("tutor_chat", "fake", "fake-default", 7.0, 1, True, False, "")


# --- the repair call ---


@pytest.mark.parametrize(
    "bad, error",
    [
        ("not json", "the reply is not JSON"),
        ({key: value for key, value in GOOD_REPLY.items() if key != "claims"}, "does not match the schema"),
    ],
)
def test_a_bad_first_answer_gets_one_repair_call_quoting_the_error_and_the_answer(engine, fake, bad, error):
    fake.queue += [bad, GOOD_REPLY]

    reply = engine.reply(HISTORY, "chat", "")

    assert reply.segments[0] == Segment("lb", "Gutt!")
    first, second = fake.calls
    bad_text = bad if isinstance(bad, str) else json.dumps(bad, ensure_ascii=False)
    assert second["user"].startswith(first["user"] + "\n\n")
    assert error in second["user"]
    assert bad_text[:500] in second["user"]
    assert "only the JSON object" in second["user"]
    assert second["system"] == first["system"] and second["schema"] == first["schema"]
    record = engine.records[-1]
    assert (record.attempts, record.first_try_valid, record.fell_back, record.error) == (2, False, False, "")


def test_an_other_failure_gets_the_repair_call_too(engine, fake):
    fake.queue += [LLMError("other", "server said 500"), GOOD_REPLY]

    engine.reply(HISTORY, "chat", "")

    assert len(fake.calls) == 2
    assert "server said 500" in fake.calls[1]["user"]


def test_the_repair_quotes_at_most_500_characters_of_the_answer(engine, fake):
    fake.queue += ["x" * 2000, GOOD_REPLY]

    engine.reply(HISTORY, "chat", "")

    assert "x" * 500 in fake.calls[1]["user"]
    assert "x" * 501 not in fake.calls[1]["user"]


# --- the template ---


def expected_template(role: str):
    if role in ("tutor_chat", "teacher_explain"):
        mode = "chat" if role == "tutor_chat" else "teacher"
        segments = [Segment("lb", TUTOR["more_lb"]), Segment("en", llm_provider.SORRY_EN)]
        return segments, mode
    if role == "judge":
        return Judgement("good", TUTOR["ack_en"])
    return TEMPLATE_WRAP_UP


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("second", ["still not json", LLMError("timeout", "slow"), LLMError("other", "down")])
def test_two_bad_answers_give_the_template_of_the_role(engine, fake, role, second, caplog):
    fake.queue += ["not json", second]

    with caplog.at_level(logging.WARNING, logger="app.engines.llm_provider"):
        answer = call(engine, role)

    expected = expected_template(role)
    if isinstance(expected, tuple):
        assert (answer.segments, answer.mode, answer.new_phrases, answer.claims) == (*expected, [], [])
    else:
        assert answer == expected
    assert len(fake.calls) == 2
    record = engine.records[-1]
    kind = second.kind if isinstance(second, LLMError) else "schema"
    assert (record.role, record.attempts, record.first_try_valid, record.fell_back) == (role, 2, False, True)
    assert record.error == kind
    assert f"{role} says its template; provider fake failed twice, {kind}" in caplog.text


def test_the_templates_hold_the_curated_lines():
    assert TUTOR["more_lb"] == llm_provider.MORE_LB
    assert TUTOR["ack_en"] == llm_provider.ACK_EN
    assert FakeLLM.WRAP_UP is TEMPLATE_WRAP_UP


def test_without_the_practice_file_the_templates_use_the_built_in_lines(settings, fake, monkeypatch, tmp_path):
    monkeypatch.setattr(llm_provider, "PRACTICE_PATH", tmp_path / "missing.toml")
    config = ProvidersConfig({"fake": ProviderEntry("fake", "fake")}, {role: RoleConfig("fake") for role in ROLES})
    engine = ProviderLLM(settings, providers={"fake": fake}, config=config)

    assert engine.template_reply("chat").segments[0] == Segment("lb", "Kënnt Dir e bësse méi dozou soen?")
    assert engine.template_judgement() == Judgement("good", "Got it.")


def test_the_wrap_up_template_is_a_copy(engine, fake):
    fake.queue += ["no", "no"]

    text = engine.wrap_up("facts")
    text.words_en = "changed"

    assert TEMPLATE_WRAP_UP.words_en == "You found a word for everything we talked about."
    assert isinstance(text, WrapUpText)


# --- what is raised ---


@pytest.mark.parametrize("kind", ["auth", "timeout"])
@pytest.mark.parametrize("role", ROLES)
def test_auth_and_timeout_are_raised_after_one_call(engine, fake, kind, role):
    fake.queue.append(LLMError(kind, "no"))

    with pytest.raises(LLMError) as raised:
        call(engine, role)

    assert raised.value.kind == kind
    assert len(fake.calls) == 1
    record = engine.records[-1]
    assert (record.attempts, record.fell_back, record.error) == (1, False, kind)


def test_auth_on_the_repair_call_is_raised_and_never_ends_in_the_template(engine, fake):
    fake.queue += ["not json", LLMError("auth", "rejected")]

    with pytest.raises(LLMError) as raised:
        engine.judge("Wat?", "What?", "Jo.", "")

    assert raised.value.kind == "auth"
    assert len(fake.calls) == 2
    record = engine.records[-1]
    assert (record.attempts, record.fell_back, record.error) == (2, False, "auth")


def test_records_keep_the_last_thousand_calls(engine):
    for _ in range(1005):
        engine.wrap_up("facts")

    assert len(engine.records) == 1000


# --- ready, warmup and building the providers ---


def test_a_warmup_that_raises_leaves_the_engine_not_ready_with_the_message(engine, fake, monkeypatch):
    engine.warmup()
    assert engine.ready() and engine.detail == ""

    def broken():
        raise RuntimeError("the server is down")

    monkeypatch.setattr(fake, "warmup", broken)
    engine.warmup()

    assert engine.ready() is False
    assert "fake" in engine.detail and "the server is down" in engine.detail


def test_a_provider_that_says_it_is_not_ready_makes_the_engine_not_ready(engine, fake):
    fake.ready = lambda: False
    fake.detail = "not logged in"

    assert engine.ready() is False
    assert "fake: not logged in" in engine.detail


def test_a_role_whose_provider_is_missing_is_not_ready_and_raises(settings, fake):
    entries = {"fake": ProviderEntry("fake", "fake"), "gone": ProviderEntry("gone", "fake")}
    roles = {role: RoleConfig("fake") for role in ROLES} | {"judge": RoleConfig("gone")}
    engine = ProviderLLM(settings, providers={"fake": fake}, config=ProvidersConfig(entries, roles))

    assert engine.ready() is False
    assert "gone" in engine.detail
    with pytest.raises(LLMError) as raised:
        engine.judge("Wat?", "What?", "Jo.", "")
    assert raised.value.kind == "other"
    assert engine.records[-1].error == "other"


def write_providers(settings: Settings, content: dict) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / PROVIDERS_FILE).write_text(json.dumps(content), encoding="utf-8")


def test_the_engine_builds_its_providers_from_providers_json(settings):
    write_providers(settings, ALL_ON_FAKE)

    engine = ProviderLLM(settings)

    assert isinstance(engine.providers["fake"], FakeProvider)
    assert engine.providers["fake"].id == "fake"
    assert engine.ready()


def test_only_the_providers_a_role_uses_are_built(settings, monkeypatch):
    monkeypatch.setitem(providers.PROVIDER_KINDS, "anthropic", "app.engines.nowhere:Nothing")
    spare = {"spare": {"kind": "anthropic", "key_env": "X_KEY"}}
    write_providers(settings, {**ALL_ON_FAKE, "providers": {**ALL_ON_FAKE["providers"], **spare}})

    assert list(ProviderLLM(settings).providers) == ["fake"]


class Recorder:
    """Stands in for an adapter class; keeps what it was built with."""

    built: list[tuple] = []

    def __init__(self, *args, **kwargs):
        Recorder.built.append((args, kwargs))
        self.id = args[0].id
        self.capabilities = None


def test_http_kinds_get_their_entry_and_key_and_the_cli_kind_gets_the_settings(settings, monkeypatch):
    Recorder.built = []
    monkeypatch.setitem(sys.modules, "recorded_kinds", types.SimpleNamespace(Recorder=Recorder))
    for kind in ("openai_compat", "claude_cli"):
        monkeypatch.setitem(providers.PROVIDER_KINDS, kind, "recorded_kinds:Recorder")
    monkeypatch.setenv("X_TEST_KEY", "sk-planted-value")
    write_providers(
        settings,
        {
            "providers": {"local": {"kind": "openai_compat", "key_env": "X_TEST_KEY"}, "cli": {"kind": "claude_cli"}},
            "roles": {role: {"provider": "local" if role != "wrapup" else "cli"} for role in ROLES},
        },
    )

    engine = ProviderLLM(settings)

    (local_args, local_kwargs), (cli_args, cli_kwargs) = Recorder.built
    assert local_args[0].id == "local" and local_args[1] == "sk-planted-value" and local_kwargs == {}
    assert cli_args[0].id == "cli" and cli_args[1] is settings and cli_kwargs == {}
    assert "sk-planted-value" not in engine.detail
    assert "sk-planted-value" not in repr(engine.config)


def test_a_kind_whose_module_is_missing_makes_the_factory_give_an_unavailable_engine(settings, monkeypatch):
    monkeypatch.setitem(providers.PROVIDER_KINDS, "openai_compat", "app.engines.nowhere:Nothing")
    write_providers(
        settings,
        {"providers": {"local": {"kind": "openai_compat"}}, "roles": {role: {"provider": "local"} for role in ROLES}},
    )

    engine = build_engine("llm", replace(settings, llm_engine="provider"))

    assert isinstance(engine, UnavailableEngine)
    assert engine.name == "provider"
    assert "openai_compat" in engine.detail and "not available" in engine.detail


def test_a_malformed_providers_json_makes_the_factory_give_an_unavailable_engine(settings):
    write_providers(settings, {"providers": {"fake": {"kind": "fake"}}, "roles": {}})

    engine = build_engine("llm", replace(settings, llm_engine="provider"))

    assert isinstance(engine, UnavailableEngine)
    assert "roles: missing" in engine.detail


def test_the_factory_builds_and_warms_up_the_provider_engine(settings):
    write_providers(settings, ALL_ON_FAKE)

    engine = build_engine("llm", replace(settings, llm_engine="provider"))

    assert isinstance(engine, ProviderLLM)
    assert engine.ready()


# --- end to end ---


@pytest.fixture
def provider_client(fake_engines, monkeypatch, tmp_path):
    """The app with LOCI_LLM_ENGINE=provider and every role on one fake provider in a tmp data folder."""
    monkeypatch.setenv("LOCI_LLM_ENGINE", "provider")
    settings = replace(Settings.from_env(), data_dir=tmp_path / "data")
    write_providers(settings, ALL_ON_FAKE)
    with TestClient(create_app(settings)) as client:
        yield client


def test_a_chat_turn_a_judgement_and_a_wrap_up_run_end_to_end_on_a_fake_provider(provider_client):
    client = provider_client
    llm = client.app.state.llm
    assert llm.name == "provider"
    assert client.get("/api/health").json()["engines"]["llm"] == {"name": "provider", "ready": True, "detail": ""}
    session_id = client.post("/api/session").json()["session_id"]

    turn = client.post("/api/turn", data={"session_id": session_id, "text": "Moien!", "mode": "chat"})
    assert turn.status_code == 200, turn.text
    assert turn.json()["segments"][-1]["text"] == FakeProvider.REPLY_EN

    started = client.post("/api/practice/start", data={"session_id": session_id, "topic_id": "aarbecht"})
    assert started.status_code == 200, started.text
    answered = client.post(
        "/api/practice/answer", data={"session_id": session_id, "text": "Ech schaffen an engem Büro, zu Esch."}
    )
    assert answered.status_code == 200, answered.text
    stopped = client.post("/api/practice/stop", data={"session_id": session_id})
    assert stopped.status_code == 200, stopped.text

    schemas = [c["schema"] for c in llm.providers["fake"].calls]
    assert JSON_SCHEMA in schemas and schemas[-2:] == [JUDGE_SCHEMA, WRAP_UP_SCHEMA]
    assert [record.role for record in llm.records][-2:] == ["judge", "wrapup"]
    assert all(record.first_try_valid and not record.fell_back for record in llm.records)

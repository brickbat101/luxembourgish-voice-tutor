import json
from pathlib import Path

import pytest

from app.config import Settings
from app.engines import providers
from app.engines.base import LLMError
from app.engines.llm_claude_cli import (
    JSON_SCHEMA,
    JUDGE_SCHEMA,
    WRAP_UP_SCHEMA,
    parse_judgement,
    parse_reply,
    parse_wrap_up,
)
from app.engines.providers import (
    KINDS,
    PROVIDER_KINDS,
    PROVIDERS_FILE,
    ROLES,
    FakeProvider,
    LLMProvider,
    LLMResult,
    ProviderCaps,
    ProviderEntry,
    ProvidersConfig,
    RoleConfig,
    default_config,
    load_providers,
    resolve_key,
)

PARAMS = {"model": "", "timeout_s": 5.0, "effort": "low", "temperature": 0.3}

TWO_PROVIDERS = {
    "providers": {
        "local": {
            "kind": "openai_compat",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "gemma3:12b",
            "timeout_s": 30,
        },
        "cloud": {
            "kind": "anthropic",
            "base_url": "https://api.anthropic.com",
            "model": "claude-sonnet",
            "key_env": "X_API_KEY",
            "options": {"max_tokens": 1024},
        },
    },
    "roles": {
        "tutor_chat": {"provider": "local", "effort": "low"},
        "teacher_explain": {"provider": "cloud", "model": "claude-opus", "effort": "high"},
        "judge": {"provider": "cloud"},
        "wrapup": {"provider": "local", "model": "gemma3:4b"},
    },
}


def write(tmp_path, data) -> Path:
    path = tmp_path / PROVIDERS_FILE
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def with_change(change) -> dict:
    data = json.loads(json.dumps(TWO_PROVIDERS))
    change(data)
    return data


# --- the provider types ---


def test_the_fake_provider_is_an_llm_provider():
    fake = FakeProvider()

    assert isinstance(fake, LLMProvider)
    assert fake.id == "fake"
    assert fake.capabilities.json_schema is True and fake.capabilities.local is True
    assert fake.capabilities.cost_per_1k == 0.0


def test_an_object_without_complete_is_not_an_llm_provider():
    class NoComplete:
        id = "x"
        capabilities = ProviderCaps(False, False, 0, False, 0.0)

    assert not isinstance(NoComplete(), LLMProvider)


def test_provider_caps_are_frozen():
    caps = ProviderCaps(True, False, 8192, True, 0.0)

    with pytest.raises(AttributeError):
        caps.local = False


def test_llm_result_defaults():
    assert LLMResult() == LLMResult(data=None, text="", ms=0.0, model="", usage={})


# --- the fake provider ---


def test_fake_provider_plays_back_the_queue_in_order_dict_str_then_exception():
    fake = FakeProvider()
    fake.queue += [{"verdict": "fix"}, "not json", LLMError("timeout", "no reply")]

    first = fake.complete("sys", "user", JUDGE_SCHEMA, PARAMS)
    second = fake.complete("sys", "user", JUDGE_SCHEMA, PARAMS)
    with pytest.raises(LLMError, match="no reply"):
        fake.complete("sys", "user", JUDGE_SCHEMA, PARAMS)
    third = fake.complete("sys", "user", JUDGE_SCHEMA, PARAMS)

    assert first.data == {"verdict": "fix"} and json.loads(first.text) == {"verdict": "fix"}
    assert second.data is None and second.text == "not json"
    assert third.data["verdict"] == "good"
    assert fake.queue == []


def test_fake_provider_logs_every_call_with_its_prompts_schema_and_params():
    fake = FakeProvider()
    fake.queue.append(LLMError("auth", "no key"))

    with pytest.raises(LLMError):
        fake.complete("system text", "user text", None, PARAMS)
    fake.complete("s2", "u2", WRAP_UP_SCHEMA, {**PARAMS, "model": "m"})

    assert fake.calls == [
        {"system": "system text", "user": "user text", "schema": None, "params": PARAMS},
        {"system": "s2", "user": "u2", "schema": WRAP_UP_SCHEMA, "params": {**PARAMS, "model": "m"}},
    ]


def test_fake_provider_default_reply_parses_and_takes_the_mode_from_the_prompt():
    fake = FakeProvider()

    chat = fake.complete("sys", "Conversation so far:\nUser: hi\n\nmode: chat\n", JSON_SCHEMA, PARAMS)
    teacher = fake.complete("sys", "Conversation so far:\nUser: hi\n\nmode: teacher\n", JSON_SCHEMA, PARAMS)

    reply = parse_reply(chat.data)
    assert [(s.lang, s.text) for s in reply.segments] == [("lb", "Moien!"), ("en", "Hello! How was your day?")]
    assert reply.new_phrases == [{"lb": "Moien!", "en": "Hello!"}]
    assert reply.mode == "chat" and reply.claims == []
    assert parse_reply(teacher.data).mode == "teacher"


def test_fake_provider_default_judgement_and_wrap_up_parse():
    fake = FakeProvider()

    judgement = parse_judgement(fake.complete("sys", "answer", JUDGE_SCHEMA, PARAMS).data)
    wrap_up = parse_wrap_up(fake.complete("sys", "facts", WRAP_UP_SCHEMA, PARAMS).data)

    assert (judgement.verdict, judgement.reaction_en) == ("good", "Nice, that came across.")
    assert (judgement.quote, judgement.better_lb, judgement.why_en, judgement.claims) == ("", "", "", [])
    assert wrap_up.words_en == "You found a word for everything we talked about."
    assert wrap_up.grammar_en == "Your sentences held together."
    assert wrap_up.task_en == "I could follow you the whole way."


def test_fake_provider_reports_its_latency_and_the_model_asked_for():
    fake = FakeProvider(latency_ms=12.5)
    fake.queue.append("raw")

    queued = fake.complete("s", "u", None, {**PARAMS, "model": "gemma3:12b"})
    default = fake.complete("s", "u", JSON_SCHEMA, PARAMS)

    assert (queued.ms, queued.model) == (12.5, "gemma3:12b")
    assert (default.ms, default.model) == (12.5, "fake")


def test_fake_provider_built_from_an_entry_takes_its_id_and_model():
    fake = FakeProvider(entry=ProviderEntry("scripted", "fake", model="tiny"))

    assert fake.id == "scripted"
    assert fake.complete("s", "u", JSON_SCHEMA, PARAMS).model == "tiny"


# --- entries, roles and the registry ---


def test_kinds_roles_and_registry():
    assert KINDS == ("fake", "claude_cli", "openai_compat", "anthropic")
    assert ROLES == ("tutor_chat", "teacher_explain", "judge", "wrapup")
    assert PROVIDERS_FILE == "providers.json"
    assert set(PROVIDER_KINDS) == set(KINDS)
    assert PROVIDER_KINDS["fake"] == "app.engines.providers:FakeProvider"
    assert PROVIDER_KINDS["claude_cli"] == "app.engines.providers_http:ClaudeCliProvider"
    assert PROVIDER_KINDS["openai_compat"] == "app.engines.providers_http:OpenAICompatProvider"
    assert PROVIDER_KINDS["anthropic"] == "app.engines.providers_http:AnthropicProvider"


def test_the_fake_registry_entry_names_the_fake_provider_class():
    module_name, class_name = PROVIDER_KINDS["fake"].split(":")

    assert module_name == providers.__name__ and getattr(providers, class_name) is FakeProvider


@pytest.mark.parametrize(
    ("kind", "style"), [("openai_compat", "nested"), ("anthropic", "tool"), ("fake", "none"), ("claude_cli", "none")]
)
def test_an_entry_without_a_schema_style_takes_the_default_of_its_kind(kind, style):
    assert ProviderEntry("p", kind).schema_style == style


def test_an_entry_keeps_the_schema_style_it_names_whatever_its_url():
    entry = ProviderEntry("llama", "openai_compat", base_url="http://127.0.0.1:8080/v1", schema_style="flat")

    assert entry.schema_style == "flat"
    assert ProviderEntry("other", "openai_compat", base_url="http://127.0.0.1:8080/v1").schema_style == "nested"


def test_an_entry_rejects_an_unknown_kind_or_schema_style():
    with pytest.raises(ValueError, match="'ollama'"):
        ProviderEntry("p", "ollama")
    with pytest.raises(ValueError, match="'grammar'"):
        ProviderEntry("p", "openai_compat", schema_style="grammar")


def test_entries_and_roles_are_frozen():
    with pytest.raises(AttributeError):
        ProviderEntry("p", "fake").model = "x"
    with pytest.raises(AttributeError):
        RoleConfig("p").effort = "high"


# --- providers.json ---


def test_load_providers_reads_two_entries_and_four_roles(tmp_path):
    config = load_providers(write(tmp_path, TWO_PROVIDERS), Settings())

    assert isinstance(config, ProvidersConfig)
    assert list(config.providers) == ["local", "cloud"]
    assert config.providers["local"] == ProviderEntry(
        "local", "openai_compat", "http://127.0.0.1:11434/v1", "gemma3:12b", "", "nested", 30.0, {}
    )
    assert config.providers["cloud"] == ProviderEntry(
        "cloud", "anthropic", "https://api.anthropic.com", "claude-sonnet", "X_API_KEY", "tool", 90.0,
        {"max_tokens": 1024},
    )
    assert config.roles == {
        "tutor_chat": RoleConfig("local", "", "low"),
        "teacher_explain": RoleConfig("cloud", "claude-opus", "high"),
        "judge": RoleConfig("cloud", "", ""),
        "wrapup": RoleConfig("local", "gemma3:4b", ""),
    }


def test_an_entry_of_only_a_kind_is_enough(tmp_path):
    data = {"providers": {"f": {"kind": "fake"}}, "roles": {role: {"provider": "f"} for role in ROLES}}

    config = load_providers(write(tmp_path, data), Settings())

    assert config.providers["f"] == ProviderEntry("f", "fake")
    assert {role.provider for role in config.roles.values()} == {"f"}


def test_a_missing_file_gives_every_role_on_the_claude_cli_with_the_settings(tmp_path):
    config = load_providers(tmp_path / PROVIDERS_FILE, Settings())

    assert config == default_config(Settings())
    assert list(config.providers) == ["claude_cli"]
    entry = config.providers["claude_cli"]
    assert (entry.kind, entry.model, entry.timeout_s, entry.key_env) == ("claude_cli", "sonnet", 90.0, "")
    assert config.roles == {
        "tutor_chat": RoleConfig("claude_cli", "", "low"),
        "teacher_explain": RoleConfig("claude_cli", "", "low"),
        "judge": RoleConfig("claude_cli", "", "low"),
        "wrapup": RoleConfig("claude_cli", "", "low"),
    }


def test_the_default_config_follows_the_llm_settings():
    settings = Settings(llm_model="opus", llm_timeout_s=30.0, llm_effort_chat="medium", llm_effort_teacher="high")

    config = default_config(settings)

    assert (config.providers["claude_cli"].model, config.providers["claude_cli"].timeout_s) == ("opus", 30.0)
    assert [config.roles[role].effort for role in ROLES] == ["medium", "high", "high", "high"]


def test_a_fifth_role_is_refused_by_name(tmp_path):
    data = with_change(lambda d: d["roles"].update(scenario_npc={"provider": "local"}))

    with pytest.raises(ValueError, match="scenario_npc"):
        load_providers(write(tmp_path, data), Settings())


def test_a_missing_role_is_refused_by_name(tmp_path):
    data = with_change(lambda d: d["roles"].pop("judge"))

    with pytest.raises(ValueError, match="judge"):
        load_providers(write(tmp_path, data), Settings())


def test_a_role_naming_an_unknown_provider_is_refused_by_name(tmp_path):
    data = with_change(lambda d: d["roles"]["wrapup"].update(provider="nowhere"))

    with pytest.raises(ValueError, match="roles.wrapup.provider.*'nowhere'"):
        load_providers(write(tmp_path, data), Settings())


def test_an_unknown_kind_is_refused_by_name(tmp_path):
    data = with_change(lambda d: d["providers"]["local"].update(kind="ollama"))

    with pytest.raises(ValueError, match="providers.local.kind.*'ollama'"):
        load_providers(write(tmp_path, data), Settings())


def test_a_literal_key_in_the_file_is_refused_by_name_without_repeating_it(tmp_path):
    data = with_change(lambda d: d["providers"]["cloud"].update(key="sk-literal-secret"))

    with pytest.raises(ValueError, match=r"providers\.cloud\.key\b") as raised:
        load_providers(write(tmp_path, data), Settings())

    assert "sk-literal-secret" not in str(raised.value)


@pytest.mark.parametrize(
    ("change", "named"),
    [
        (lambda d: d["providers"]["local"].pop("kind"), "providers.local.kind"),
        (lambda d: d["providers"]["local"].update(model=3), "providers.local.model"),
        (lambda d: d["providers"]["local"].update(timeout_s=0), "providers.local.timeout_s"),
        (lambda d: d["providers"]["local"].update(options=[]), "providers.local.options"),
        (lambda d: d["providers"]["local"].update(keyenv="X"), "providers.local.keyenv"),
        (lambda d: d["providers"]["local"].update(schema_style="grammar"), "providers.local.schema_style"),
        (lambda d: d["roles"]["judge"].update(temperature=0.1), "roles.judge.temperature"),
        (lambda d: d["roles"].update(judge="cloud"), "roles.judge"),
        (lambda d: d.pop("roles"), "roles"),
        (lambda d: d.update(providers={}), "providers"),
        (lambda d: d.update(extra=1), "extra"),
    ],
)
def test_a_malformed_file_raises_value_error_naming_the_key(tmp_path, change, named):
    with pytest.raises(ValueError, match=named.replace(".", r"\.")):
        load_providers(write(tmp_path, with_change(change)), Settings())


def test_bad_json_and_duplicate_ids_are_refused(tmp_path):
    with pytest.raises(ValueError, match="not valid JSON"):
        load_providers(write(tmp_path, "{not json"), Settings())

    duplicated = '{"providers": {"a": {"kind": "fake"}, "a": {"kind": "fake"}}, "roles": {}}'
    with pytest.raises(ValueError, match="duplicate key 'a'"):
        load_providers(write(tmp_path, duplicated), Settings())


# --- keys ---


def test_resolve_key_reads_the_variable_the_entry_names(monkeypatch):
    monkeypatch.setenv("X_API_KEY", "secret")
    entry = ProviderEntry("cloud", "anthropic", key_env="X_API_KEY")

    assert resolve_key(entry) == "secret"
    assert resolve_key(entry, {"X_API_KEY": "other"}) == "other"


def test_resolve_key_falls_back_to_the_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FILE_ONLY_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setattr(providers, "ENV_FILE", env_file)
    monkeypatch.delenv("FILE_ONLY_API_KEY", raising=False)

    assert resolve_key(ProviderEntry("p", "openai_compat", key_env="FILE_ONLY_API_KEY")) == "from-file"


def test_resolve_key_is_empty_for_an_unset_or_unnamed_variable(tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.delenv("UNSET_API_KEY", raising=False)

    assert resolve_key(ProviderEntry("p", "openai_compat", key_env="UNSET_API_KEY")) == ""
    assert resolve_key(ProviderEntry("p", "openai_compat", key_env="UNSET_API_KEY"), {}) == ""
    assert resolve_key(ProviderEntry("p", "openai_compat"), {"": "x"}) == ""


def test_no_entry_or_config_repr_holds_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("X_API_KEY", "secret")
    entry = ProviderEntry("cloud", "anthropic", key_env="X_API_KEY")
    config = load_providers(write(tmp_path, TWO_PROVIDERS), Settings())

    assert resolve_key(entry) == "secret"
    assert "secret" not in repr(entry)
    assert "X_API_KEY" in repr(entry)
    assert "secret" not in repr(config)

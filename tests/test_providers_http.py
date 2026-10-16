import json
import subprocess
import sys

import httpx
import pytest

from app.config import ROOT, Settings
from app.engines.base import LLMError
from app.engines.llm_claude_cli import JSON_SCHEMA, child_env
from app.engines.providers import PROVIDER_KINDS, LLMProvider, ProviderEntry
from app.engines.providers_http import (
    AnthropicProvider,
    ClaudeCliProvider,
    OpenAICompatProvider,
    is_loopback,
    redact,
)
from tests.test_llm_claude_cli import AUTH_ERROR, CRASH, FAKE_EXE, REPLY, FakeRunner, done, envelope

KEY = "sk-test-0123456789abcdef"
KEY_ENV = "LOCI_TEST_PROVIDER_API_KEY"
PARAMS = {"model": "gemma3:12b", "timeout_s": 5.0, "effort": "low", "temperature": 0.3}
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


class Server:
    """A MockTransport handler: records each request and plays back the queued outcomes."""

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)


def chat(content, **extra) -> httpx.Response:
    body = {"model": "gemma3:12b", "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}
    return httpx.Response(200, json=body | extra)


def messages(*content, stop_reason="tool_use") -> httpx.Response:
    body = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet",
        "content": list(content),
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 120, "output_tokens": 30},
    }
    return httpx.Response(200, json=body)


def tool_use(data) -> dict:
    return {"type": "tool_use", "id": "toolu_01", "name": "reply", "input": data}


def openai(server: Server, key: str = "", **fields) -> OpenAICompatProvider:
    entry = ProviderEntry("local", "openai_compat", **({"base_url": "http://127.0.0.1:11434/v1/"} | fields))
    return OpenAICompatProvider(entry, key, client=server.client())


def anthropic(server: Server, key: str = KEY, **fields) -> AnthropicProvider:
    entry = ProviderEntry("cloud", "anthropic", **({"model": "claude-sonnet", "key_env": KEY_ENV} | fields))
    return AnthropicProvider(entry, key, client=server.client())


def error_of(provider, schema=SCHEMA) -> LLMError:
    with pytest.raises(LLMError) as raised:
        provider.complete("You are a tutor.", "Moien", schema, PARAMS)
    return raised.value


# --- the registry ---


@pytest.mark.parametrize("kind", ["claude_cli", "openai_compat", "anthropic"])
def test_the_registry_names_the_classes_of_this_module(kind):
    module_name, class_name = PROVIDER_KINDS[kind].split(":")

    assert module_name == "app.engines.providers_http"
    assert getattr(__import__(module_name, fromlist=[class_name]), class_name).kind == kind


def test_the_fake_path_does_not_import_httpx():
    code = (
        "import os, sys\n"
        "for kind in ('STT', 'TTS', 'LLM'):\n"
        "    os.environ[f'LOCI_{kind}_ENGINE'] = 'fake'\n"
        "from app.config import Settings\n"
        "from app.engines.factory import KINDS, build_engine\n"
        "from app.engines.providers import FakeProvider, load_providers\n"
        "import app.main\n"
        "settings = Settings.from_env()\n"
        "[build_engine(kind, settings) for kind in KINDS]\n"
        "FakeProvider().complete('s', 'u', None, {})\n"
        "loaded = {'httpx', 'app.engines.providers_http'} & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )

    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_the_providers_are_built_as_the_engine_builds_them(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCI_CLAUDE_EXE", FAKE_EXE)

    http = OpenAICompatProvider(ProviderEntry("local", "openai_compat"), KEY)
    cloud = AnthropicProvider(ProviderEntry("cloud", "anthropic"), "")
    cli = ClaudeCliProvider(ProviderEntry("cli", "claude_cli"), Settings(data_dir=tmp_path))

    assert (http.id, cloud.id, cli.id) == ("local", "cloud", "cli")
    assert KEY not in repr(http)
    http.close()
    cloud.close()


def test_the_http_providers_are_llm_providers():
    assert isinstance(openai(Server()), LLMProvider)
    assert isinstance(anthropic(Server()), LLMProvider)


# --- openai_compat: the request ---


def test_openai_compat_posts_the_chat_completion_without_a_key():
    server = Server(chat('{"answer": "Moien!"}'))

    openai(server, key_env=KEY_ENV).complete("You are a tutor.", "Moien", SCHEMA, PARAMS)

    request = server.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
    assert "authorization" not in request.headers
    assert server.body == {
        "model": "gemma3:12b",
        "messages": [{"role": "system", "content": "You are a tutor."}, {"role": "user", "content": "Moien"}],
        "stream": False,
        "temperature": 0.3,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "reply", "strict": True, "schema": SCHEMA},
        },
    }
    assert "effort" not in server.body


def test_openai_compat_sends_the_key_as_a_bearer_token():
    server = Server(chat('{"answer": "Moien!"}'))

    openai(server, KEY, key_env=KEY_ENV).complete("s", "u", SCHEMA, PARAMS)

    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_the_key_passed_in_is_used_and_the_environment_is_not_read(monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-from-the-environment")
    server = Server(chat("{}"), chat("{}"))

    openai(server, "", key_env=KEY_ENV).complete("s", "u", SCHEMA, PARAMS)
    openai(server, KEY, key_env=KEY_ENV).complete("s", "u", SCHEMA, PARAMS)

    assert "authorization" not in server.requests[0].headers
    assert server.requests[1].headers["authorization"] == f"Bearer {KEY}"


def test_the_flat_envelope_puts_the_schema_beside_the_type():
    server = Server(chat("{}"))

    openai(server, schema_style="flat").complete("s", "u", SCHEMA, PARAMS)

    assert server.body["response_format"] == {"type": "json_schema", "schema": SCHEMA}


def test_the_none_envelope_puts_the_schema_in_the_system_message():
    server = Server(chat("{}"))

    openai(server, schema_style="none").complete("You are a tutor.", "u", SCHEMA, PARAMS)

    assert "response_format" not in server.body
    system = server.body["messages"][0]["content"]
    assert system == (
        "You are a tutor.\n\nAnswer with exactly one JSON object of this shape: "
        + json.dumps(SCHEMA, separators=(",", ":"))
    )


def test_the_entry_model_and_timeout_stand_in_for_missing_params():
    seen = {}

    def handler(request):
        seen["timeout"] = request.extensions["timeout"]
        seen["body"] = json.loads(request.content)
        return chat("{}")

    entry = ProviderEntry("local", "openai_compat", base_url="http://localhost:1234/v1", model="qwen", timeout_s=12)
    provider = OpenAICompatProvider(entry, "", client=httpx.Client(transport=httpx.MockTransport(handler)))

    result = provider.complete("s", "u", SCHEMA, {"model": "", "effort": ""})

    assert seen["body"]["model"] == "qwen"
    assert "temperature" not in seen["body"]
    assert seen["timeout"]["read"] == 12.0
    assert result.model == "gemma3:12b"  # the server's own name for the model it ran


def test_openai_compat_refuses_the_tool_style():
    with pytest.raises(ValueError, match="schema_style"):
        openai(Server(), schema_style="tool")


# --- openai_compat: the reply ---


def test_json_content_becomes_data_with_the_usage():
    usage = {"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59}
    server = Server(chat('{"answer": "Moien!"}', usage=usage))

    result = openai(server).complete("s", "u", SCHEMA, PARAMS)

    assert result.data == {"answer": "Moien!"}
    assert result.text == '{"answer": "Moien!"}'
    assert result.usage == usage
    assert result.ms >= 0


def test_non_json_content_is_kept_as_text():
    server = Server(chat("Moien! How are you?"))

    result = openai(server).complete("s", "u", SCHEMA, PARAMS)

    assert result.data is None
    assert result.text == "Moien! How are you?"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
    ],
)
def test_a_reply_without_content_is_a_schema_error(response):
    assert error_of(openai(Server(response))).kind == "schema"


# --- the failures every HTTP provider shares ---


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_an_auth_error_naming_the_variable_and_never_the_key(status):
    echo = httpx.Response(status, json={"error": f"Incorrect API key provided: {KEY}"})

    error = error_of(openai(Server(echo), KEY, key_env=KEY_ENV))

    assert error.kind == "auth"
    assert f"set {KEY_ENV} in .env or the environment" in error.detail
    assert KEY not in error.detail and KEY not in str(error)
    assert "***" in error.detail
    assert "claude auth login" not in error.detail


def test_a_key_cut_by_the_body_limit_is_still_redacted():
    echo = httpx.Response(401, text="x" * 290 + KEY)

    error = error_of(openai(Server(echo), KEY, key_env=KEY_ENV))

    assert KEY[:10] not in error.detail


@pytest.mark.parametrize("status", [429, 500, 503, 404])
def test_other_statuses_are_other_errors_with_the_status_and_the_body(status):
    server = Server(httpx.Response(status, text=f"slow down {KEY} " + "x" * 400))

    error = error_of(openai(server, KEY, key_env=KEY_ENV))

    assert error.kind == "other"
    assert f"HTTP {status}" in error.detail
    assert "slow down ***" in error.detail
    assert KEY not in error.detail
    assert error.detail.count("x") <= 300


def test_a_server_that_is_not_there_is_an_other_error():
    server = Server(httpx.ConnectError("[WinError 10061] connection refused"))

    error = error_of(openai(server))

    assert error.kind == "other"
    assert error.detail.startswith("http://127.0.0.1:11434/v1 is not answering: ConnectError")


@pytest.mark.parametrize("exc", [httpx.ReadTimeout("timed out"), httpx.ConnectTimeout("timed out")])
def test_a_timeout_is_a_timeout_error(exc):
    error = error_of(openai(Server(exc)))

    assert error.kind == "timeout"
    assert "5 s" in error.detail


def test_an_entry_without_a_base_url_is_an_other_error_without_a_request():
    server = Server()

    error = error_of(openai(server, base_url=""))

    assert error.kind == "other" and "base_url" in error.detail
    assert server.requests == []


def test_redact_replaces_the_key_and_leaves_text_without_one():
    assert redact(f"Bearer {KEY} rejected", KEY) == "Bearer *** rejected"
    assert redact("nothing to hide", "") == "nothing to hide"


def test_no_provider_repr_holds_the_key():
    provider = openai(Server(), KEY, key_env=KEY_ENV)

    assert repr(provider) == (
        "OpenAICompatProvider(id='local', kind='openai_compat', base_url='http://127.0.0.1:11434/v1', "
        f"model='', key_env='{KEY_ENV}')"
    )
    assert KEY not in repr(anthropic(Server()))


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://127.0.0.1:11434/v1", True),
        ("http://localhost:1234/v1", True),
        ("http://[::1]:8080", True),
        ("https://openrouter.ai/api/v1", False),
        ("http://192.168.1.20:11434/v1", False),
        ("", False),
    ],
)
def test_a_loopback_base_url_is_local(url, local):
    assert is_loopback(url) is local


def test_caps_follow_the_entry():
    assert openai(Server()).capabilities.local is True
    assert openai(Server(), schema_style="none").capabilities.json_schema is False
    remote = openai(Server(), base_url="https://openrouter.ai/api/v1")
    assert remote.capabilities.local is False and remote.capabilities.json_schema is True
    assert anthropic(Server()).capabilities.local is False


# --- anthropic ---


def test_anthropic_posts_a_strict_auto_tool_with_the_key_headers():
    """Current Claude models reject a forced tool_choice (type tool or any) with HTTP 400."""
    server = Server(messages(tool_use({"answer": "Moien!"})))

    anthropic(server).complete("You are a tutor.", "Moien", SCHEMA, PARAMS | {"model": "claude-opus"})

    request = server.requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == KEY
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["content-type"] == "application/json"
    assert "authorization" not in request.headers
    assert server.body == {
        "model": "claude-opus",
        "max_tokens": 1024,
        "system": "You are a tutor.\n\nAnswer only by calling the reply tool, exactly once, with no other text.",
        "messages": [{"role": "user", "content": "Moien"}],
        "tools": [
            {"name": "reply", "description": "The tutor's structured answer.", "input_schema": SCHEMA, "strict": True}
        ],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    }
    assert server.body["tool_choice"]["type"] not in ("tool", "any")


def test_anthropic_takes_max_tokens_and_base_url_from_the_entry():
    server = Server(messages(tool_use({})))

    anthropic(server, base_url="http://127.0.0.1:9000/v1", options={"max_tokens": 300}).complete(
        "s", "u", SCHEMA, PARAMS
    )

    assert str(server.requests[0].url) == "http://127.0.0.1:9000/v1/messages"
    assert server.body["max_tokens"] == 300


def test_the_tool_use_input_becomes_data():
    text = {"type": "text", "text": "Here is my answer."}
    server = Server(messages(text, tool_use({"answer": "Moien!"})))

    result = anthropic(server).complete("s", "u", SCHEMA, PARAMS)

    assert result.data == {"answer": "Moien!"}
    assert result.model == "claude-sonnet"
    assert result.usage == {"input_tokens": 120, "output_tokens": 30}


@pytest.mark.parametrize(
    "response",
    [
        messages(tool_use({"answer": "Mo"}), stop_reason="max_tokens"),
        messages({"type": "text", "text": "I can't help with that."}, stop_reason="refusal"),
        messages({"type": "text", "text": '{"answer": "Moien!"}'}, stop_reason="end_turn"),
    ],
)
def test_a_cut_refused_or_toolless_reply_is_a_schema_error(response):
    assert error_of(anthropic(Server(response))).kind == "schema"


def test_anthropic_rejects_a_bad_key_as_auth():
    server = Server(httpx.Response(401, json={"type": "error", "error": {"message": f"invalid x-api-key {KEY}"}}))

    error = error_of(anthropic(server))

    assert error.kind == "auth"
    assert KEY_ENV in error.detail and KEY not in str(error)


def test_anthropic_with_the_none_style_reads_the_text():
    server = Server(messages({"type": "text", "text": '{"answer": "Moien!"}'}, stop_reason="end_turn"))

    result = anthropic(server, schema_style="none").complete("s", "u", SCHEMA, PARAMS)

    assert "tools" not in server.body
    assert "Answer with exactly one JSON object" in server.body["system"]
    assert result.data == {"answer": "Moien!"}


# --- claude_cli ---


@pytest.fixture
def make_cli(monkeypatch, tmp_path):
    def make(*outcomes, drop=()) -> tuple[ClaudeCliProvider, FakeRunner]:
        monkeypatch.setenv("LOCI_CLAUDE_EXE", FAKE_EXE)
        runner = FakeRunner(*outcomes)
        settings = Settings(data_dir=tmp_path / "data", llm_model="sonnet", llm_timeout_s=90.0)
        entry = ProviderEntry("claude_cli", "claude_cli", model="sonnet")
        return ClaudeCliProvider(entry, settings, runner=runner, drop=drop), runner

    return make


def cli_error(provider) -> LLMError:
    with pytest.raises(LLMError) as raised:
        provider.complete("You are a tutor.", "Moien", JSON_SCHEMA, PARAMS)
    return raised.value


def test_the_cli_provider_is_an_llm_provider(make_cli):
    provider, _ = make_cli()

    assert isinstance(provider, LLMProvider)
    assert provider.id == "claude_cli"
    assert provider.capabilities.local is False and provider.capabilities.json_schema is True


def test_the_cli_runs_once_with_the_model_effort_and_timeout_of_the_call(make_cli):
    provider, runner = make_cli(envelope())

    result = provider.complete("You are a tutor.", "Moien", JSON_SCHEMA, PARAMS | {"model": "opus", "effort": "high"})

    assert len(runner.calls) == 1
    argv, kwargs = runner.calls[0]
    assert argv[argv.index("--model") + 1] == "opus"
    assert argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--system-prompt") + 1] == "You are a tutor."
    assert json.loads(argv[argv.index("--json-schema") + 1]) == JSON_SCHEMA
    assert kwargs["input"] == "Moien"
    assert kwargs["timeout"] == 5.0
    assert "Moien" not in argv
    assert result.data == REPLY
    assert result.model == "opus"
    assert result.ms >= 0


def test_an_empty_effort_passes_no_flag_and_an_empty_model_takes_the_entry(make_cli):
    provider, runner = make_cli(envelope())

    provider.complete("s", "u", JSON_SCHEMA, {"model": "", "effort": "", "timeout_s": 5.0})

    argv, _ = runner.calls[0]
    assert "--effort" not in argv
    assert argv[argv.index("--model") + 1] == "sonnet"


def test_each_call_takes_its_own_params(make_cli):
    provider, runner = make_cli(envelope(), envelope())

    provider.complete("s", "u", JSON_SCHEMA, PARAMS | {"model": "haiku", "effort": "low"})
    provider.complete("s", "u", JSON_SCHEMA, PARAMS | {"model": "opus", "effort": "max"})

    (first, _), (second, _) = runner.calls
    assert (first[first.index("--model") + 1], first[first.index("--effort") + 1]) == ("haiku", "low")
    assert (second[second.index("--model") + 1], second[second.index("--effort") + 1]) == ("opus", "max")
    assert provider.settings.llm_model == "sonnet"


@pytest.mark.parametrize(
    ("outcome", "kind"),
    [
        (AUTH_ERROR, "auth"),
        (CRASH, "other"),
        (done("Segmentation fault"), "schema"),
        (subprocess.TimeoutExpired(["claude"], 5.0), "timeout"),
    ],
)
def test_the_cli_errors_map_to_their_kinds_without_a_retry(make_cli, outcome, kind):
    provider, runner = make_cli(outcome, envelope())

    assert cli_error(provider).kind == kind
    assert len(runner.calls) == 1


def test_without_drop_the_cli_leaves_out_the_key_env_names_of_the_providers_file(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCI_CLAUDE_EXE", FAKE_EXE)
    monkeypatch.setenv("MISTRAL_KEY", "mistral")
    settings = Settings(data_dir=tmp_path / "data")
    settings.data_dir.mkdir()
    entries = {
        "cli": {"kind": "claude_cli"},
        "mistral": {"kind": "openai_compat", "base_url": "https://api.mistral.ai/v1", "key_env": "MISTRAL_KEY"},
    }
    roles = {role: {"provider": "cli"} for role in ("tutor_chat", "teacher_explain", "judge", "wrapup")}
    (settings.data_dir / "providers.json").write_text(json.dumps({"providers": entries, "roles": roles}))
    runner = FakeRunner(envelope())

    provider = ClaudeCliProvider(ProviderEntry("cli", "claude_cli"), settings, runner=runner)
    provider.complete("s", "u", JSON_SCHEMA, PARAMS)

    assert provider.cli.env_drop == ("MISTRAL_KEY",)
    assert "MISTRAL_KEY" not in {key.upper() for key in runner.calls[0][1]["env"]}


def test_a_malformed_providers_file_drops_no_extra_names(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCI_CLAUDE_EXE", FAKE_EXE)
    settings = Settings(data_dir=tmp_path)
    (tmp_path / "providers.json").write_text("{not json")

    provider = ClaudeCliProvider(ProviderEntry("cli", "claude_cli"), settings, runner=FakeRunner())

    assert provider.cli.env_drop == ()


def test_the_cli_runs_without_the_provider_keys(make_cli, monkeypatch):
    monkeypatch.setenv("MISTRAL_KEY", "mistral")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    provider, runner = make_cli(envelope(), drop=("MISTRAL_KEY",))

    provider.complete("s", "u", JSON_SCHEMA, PARAMS)

    env = runner.calls[0][1]["env"]
    assert env == child_env(drop=("MISTRAL_KEY",))
    assert not {"MISTRAL_KEY", "OPENAI_API_KEY"} & {key.upper() for key in env}


def test_cli_warmup_raises_when_not_logged_in_and_passes_when_it_is(make_cli):
    provider, runner = make_cli(done(json.dumps({"loggedIn": False})), done(json.dumps({"loggedIn": True})))

    with pytest.raises(LLMError) as raised:
        provider.warmup()
    assert raised.value.kind == "auth"
    assert "claude auth login" in provider.detail

    provider.warmup()
    assert provider.detail == ""
    assert runner.calls[0][0][1:] == ["auth", "status"]

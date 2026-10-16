"""LLM providers that reach a server: OpenAI-compatible and Anthropic over httpx, and the Claude CLI.

Imported only through `providers.PROVIDER_KINDS`, so the fake path never loads httpx. A key is passed
in by the engine (`resolve_key`) and never logged or shown by `repr`: every failure is an `LLMError`
whose detail has the key replaced by "***".
"""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import subprocess
import time
from collections.abc import Iterable, Mapping
from urllib.parse import urlsplit

import httpx

from app.config import Settings
from app.engines.base import LLMError
from app.engines.llm_claude_cli import ClaudeCliLLM
from app.engines.providers import PROVIDERS_FILE, LLMResult, ProviderCaps, ProviderEntry, load_providers

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MAX_TOKENS = 1024
TOOL_NAME = "reply"
TOOL_DESCRIPTION = "The tutor's structured answer."
SCHEMA_PROMPT = "Answer with exactly one JSON object of this shape: {schema}"
# Current Claude models reject a forced tool_choice (type "tool" or "any") with HTTP 400; the
# tool is strict and asked for in the prompt instead, and a reply without it is a schema error.
TOOL_PROMPT = f"Answer only by calling the {TOOL_NAME} tool, exactly once, with no other text."
TOOL_CHOICE = {"type": "auto", "disable_parallel_tool_use": True}
BODY_CHARS = 300
# Stop reasons of an Anthropic reply that cut the tool input short or withheld it.
INCOMPLETE_STOPS = ("max_tokens", "refusal")
CLI_MAX_CTX = 200_000


def redact(text: str, key: str) -> str:
    """`text` with `key` replaced by "***"; the Authorization and x-api-key header values hold it."""
    return text.replace(key, "***") if key else text


def compact(schema: Mapping) -> str:
    return json.dumps(schema, separators=(",", ":"))


def is_loopback(base_url: str) -> bool:
    host = urlsplit(base_url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class HttpProvider:
    """What the HTTP adapters share: the client, one POST and the mapping of its failures."""

    kind = ""
    default_base_url = ""
    styles: tuple[str, ...] = ()

    def __init__(self, entry: ProviderEntry, key: str = "", *, client: httpx.Client | None = None) -> None:
        if entry.schema_style not in self.styles:
            raise ValueError(
                f"providers.{entry.id}.schema_style: {self.kind} takes {', '.join(self.styles)}, "
                f"not {entry.schema_style!r}"
            )
        self.id = entry.id
        self.entry = entry
        self.base_url = (entry.base_url or self.default_base_url).rstrip("/")
        self.client = client or httpx.Client()
        self._key = key  # never in repr or an error detail

    def __repr__(self) -> str:
        entry = self.entry
        return (
            f"{type(self).__name__}(id={entry.id!r}, kind={entry.kind!r}, base_url={self.base_url!r}, "
            f"model={entry.model!r}, key_env={entry.key_env!r})"
        )

    def close(self) -> None:
        self.client.close()

    def _settings(self, params: Mapping) -> tuple[str, float]:
        """The model and timeout of one call: the params', else the entry's."""
        return params.get("model") or self.entry.model, float(params.get("timeout_s") or self.entry.timeout_s)

    def _post(self, path: str, headers: dict, body: dict, timeout_s: float) -> tuple[dict, float]:
        """The JSON body of a 2xx answer and the ms it took; LLMError for anything else."""
        if not self.base_url:
            raise LLMError("other", f"providers.{self.id}.base_url is empty")
        start = time.perf_counter()
        try:
            response = self.client.post(f"{self.base_url}{path}", headers=headers, json=body, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise LLMError("timeout", f"{self.id} gave no reply within {timeout_s:g} s") from exc
        except httpx.HTTPError as exc:
            detail = f"{self.base_url} is not answering: {type(exc).__name__}: {exc}"
            raise LLMError("other", redact(detail, self._key)) from None
        ms = (time.perf_counter() - start) * 1000
        status = response.status_code
        # Redacted before the cut, so no part of a key the server echoed is left at the end.
        text = redact(response.text, self._key)[:BODY_CHARS]
        if status in (401, 403):
            fix = (
                f"set {self.entry.key_env} in .env or the environment"
                if self.entry.key_env
                else f"name the variable holding its key as key_env of providers.{self.id}"
            )
            raise LLMError("auth", f"{self.id} rejected the key (HTTP {status}: {text}); {fix}")
        if not 200 <= status < 300:
            raise LLMError("other", f"{self.id} answered HTTP {status}: {text}")
        try:
            data = response.json()
        except ValueError:
            data = None
        if not isinstance(data, dict):
            raise LLMError("schema", f"{self.id} answered with no JSON object: {text!r}")
        return data, ms


class OpenAICompatProvider(HttpProvider):
    """`POST {base_url}/chat/completions` of Ollama, LM Studio, llama.cpp, OpenRouter, Mistral and the like.

    The schema goes in the envelope `entry.schema_style` names: nested, flat, or none (in the prompt).
    """

    kind = "openai_compat"
    styles = ("nested", "flat", "none")

    def __init__(self, entry: ProviderEntry, key: str = "", *, client: httpx.Client | None = None) -> None:
        super().__init__(entry, key, client=client)
        options = entry.options
        self.capabilities = ProviderCaps(
            json_schema=entry.schema_style != "none",
            streaming=False,
            max_ctx=int(options.get("max_ctx", 8192)),
            local=is_loopback(self.base_url),
            cost_per_1k=float(options.get("cost_per_1k", 0.0)),
        )

    def request(self, system: str, user: str, schema: dict | None, params: Mapping) -> dict:
        """The body of one call; `effort` is not an OpenAI-compatible field and is left out."""
        model, _ = self._settings(params)
        style = self.entry.schema_style
        if schema is not None and style == "none":
            system = f"{system}\n\n{SCHEMA_PROMPT.format(schema=compact(schema))}"
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
        }
        if params.get("temperature") is not None:
            body["temperature"] = params["temperature"]
        if schema is not None and style == "nested":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": TOOL_NAME, "strict": True, "schema": schema},
            }
        elif schema is not None and style == "flat":
            body["response_format"] = {"type": "json_schema", "schema": schema}
        return body

    def complete(self, system: str, user: str, schema: dict | None, params: Mapping) -> LLMResult:
        model, timeout_s = self._settings(params)
        key = self._key
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        data, ms = self._post("/chat/completions", headers, self.request(system, user, schema, params), timeout_s)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        if not isinstance(content, str):
            raise LLMError("schema", f"{self.id} answered without message content")
        try:
            parsed = json.loads(content)
        except ValueError:
            parsed = None
        usage = data.get("usage")
        return LLMResult(parsed, content, ms, data.get("model") or model, usage if isinstance(usage, dict) else {})


class AnthropicProvider(HttpProvider):
    """`POST {base_url}/messages` of the Anthropic API; the schema is a strict tool, `tool_use.input` the answer."""

    kind = "anthropic"
    default_base_url = ANTHROPIC_BASE_URL
    styles = ("tool", "none")

    def __init__(self, entry: ProviderEntry, key: str = "", *, client: httpx.Client | None = None) -> None:
        super().__init__(entry, key, client=client)
        options = entry.options
        self.max_tokens = int(options.get("max_tokens", ANTHROPIC_MAX_TOKENS))
        self.capabilities = ProviderCaps(
            json_schema=entry.schema_style == "tool",
            streaming=False,
            max_ctx=int(options.get("max_ctx", 200_000)),
            local=is_loopback(self.base_url),
            cost_per_1k=float(options.get("cost_per_1k", 0.0)),
        )

    def request(self, system: str, user: str, schema: dict | None, params: Mapping) -> dict:
        model, _ = self._settings(params)
        tool = schema is not None and self.entry.schema_style == "tool"
        if schema is not None and not tool:
            system = f"{system}\n\n{SCHEMA_PROMPT.format(schema=compact(schema))}"
        elif tool:
            system = f"{system}\n\n{TOOL_PROMPT}"
        body = {
            "model": model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if tool:
            body["tools"] = [
                {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_schema": schema, "strict": True}
            ]
            body["tool_choice"] = dict(TOOL_CHOICE)
        return body

    def complete(self, system: str, user: str, schema: dict | None, params: Mapping) -> LLMResult:
        model, timeout_s = self._settings(params)
        key = self._key
        headers = {"anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
        if key:
            headers["x-api-key"] = key
        body = self.request(system, user, schema, params)
        data, ms = self._post("/messages", headers, body, timeout_s)
        stop = data.get("stop_reason")
        if stop in INCOMPLETE_STOPS:
            raise LLMError("schema", f"{self.id} stopped early: stop_reason {stop}")
        blocks = data.get("content")
        blocks = [block for block in blocks if isinstance(block, dict)] if isinstance(blocks, list) else []
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        model = data.get("model") or model
        if "tools" in body:
            found = next((block for block in blocks if block.get("type") == "tool_use"), None)
            if found is None or not isinstance(found.get("input"), dict):
                raise LLMError("schema", f"{self.id} answered without a {TOOL_NAME} tool_use block")
            return LLMResult(found["input"], json.dumps(found["input"], ensure_ascii=False), ms, model, usage)
        text = "".join(str(block.get("text", "")) for block in blocks if block.get("type") == "text")
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        return LLMResult(parsed, text, ms, model, usage)


def key_envs(settings: Settings) -> tuple[str, ...]:
    """The `key_env` names of the providers file in `settings.data_dir`; () when it cannot be read."""
    try:
        config = load_providers(settings.data_dir / PROVIDERS_FILE, settings)
    except (OSError, ValueError):
        return ()
    return tuple(entry.key_env for entry in config.providers.values() if entry.key_env)


class ClaudeCliProvider:
    """The Claude CLI as a provider: one `claude -p` run per `complete`, with no retry of its own.

    `drop` names variables kept from the CLI, so a key exported for another provider never moves it
    off the user's login; None takes the `key_env` names of `settings.data_dir / PROVIDERS_FILE`.
    """

    kind = "claude_cli"

    def __init__(
        self,
        entry: ProviderEntry,
        settings=None,
        *,
        runner=subprocess.run,
        drop: Iterable[str] | None = None,
    ) -> None:
        if settings is None:
            settings = Settings.from_env()
        if drop is None:
            drop = key_envs(settings)
        self.id = entry.id
        self.entry = entry
        self.settings = settings
        self.cli = ClaudeCliLLM(settings, runner)
        self.cli.env_drop = tuple(drop)
        self.capabilities = ProviderCaps(
            json_schema=True, streaming=False, max_ctx=CLI_MAX_CTX, local=False, cost_per_1k=0.0
        )

    def __repr__(self) -> str:
        return f"ClaudeCliProvider(id={self.id!r}, kind={self.kind!r}, model={self.entry.model!r})"

    @property
    def detail(self) -> str:
        return self.cli.detail

    def warmup(self) -> None:
        """`claude auth status`; LLMError when the CLI is missing or not logged in."""
        self.cli.warmup()
        if not self.cli.ready():
            raise LLMError("auth" if self.cli.exe else "other", self.cli.detail)

    def complete(self, system: str, user: str, schema: dict | None, params: Mapping) -> LLMResult:
        effort = params.get("effort") or ""
        model = params.get("model") or self.entry.model or self.settings.llm_model
        self.cli.settings = dataclasses.replace(
            self.settings,
            llm_model=model,
            llm_effort_chat=effort,
            llm_effort_teacher=effort,
            llm_timeout_s=float(params.get("timeout_s") or self.entry.timeout_s),
        )
        cli_schema = compact(schema if schema is not None else {"type": "object"})
        start = time.perf_counter()
        data = self.cli._call(user, "chat", system, cli_schema, parse=lambda found: found)
        ms = (time.perf_counter() - start) * 1000
        return LLMResult(data, json.dumps(data, ensure_ascii=False), ms, model)

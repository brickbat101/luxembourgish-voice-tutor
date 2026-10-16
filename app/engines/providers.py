"""LLM providers: one `complete` call to one server, and the providers.json that names them per role.

Keys never live here: an entry names the environment variable that holds its key (`key_env`) and
`resolve_key` reads it at call time. Standard library and dotenv only.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dotenv import dotenv_values

if TYPE_CHECKING:
    from app.config import Settings

PROVIDERS_FILE = "providers.json"  # in settings.data_dir
# ROOT/.env, the file Settings.from_env reads.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

KINDS = ("fake", "claude_cli", "openai_compat", "anthropic")
SCHEMA_STYLES = ("nested", "flat", "tool", "none")
DEFAULT_SCHEMA_STYLE = {"fake": "none", "claude_cli": "none", "openai_compat": "nested", "anthropic": "tool"}
ROLES = ("tutor_chat", "teacher_explain", "judge", "wrapup")

# kind -> "module:Class", imported only when an entry of that kind is built.
PROVIDER_KINDS: dict[str, str] = {
    "fake": "app.engines.providers:FakeProvider",
    "claude_cli": "app.engines.providers_http:ClaudeCliProvider",
    "openai_compat": "app.engines.providers_http:OpenAICompatProvider",
    "anthropic": "app.engines.providers_http:AnthropicProvider",
}

ENTRY_FIELDS = ("kind", "base_url", "model", "key_env", "schema_style", "timeout_s", "options")
ROLE_FIELDS = ("provider", "model", "effort")
# Fields that would hold a key itself rather than the name of the variable holding it.
KEY_FIELDS = ("key", "api_key")


@dataclass(frozen=True)
class ProviderCaps:
    json_schema: bool  # the server constrains its output to a JSON schema
    streaming: bool
    max_ctx: int  # context window in tokens
    local: bool  # runs on this machine
    cost_per_1k: float  # USD per 1000 tokens, 0.0 when free


@dataclass
class LLMResult:
    data: object | None = None  # the parsed JSON object, when there is one
    text: str = ""  # the raw text of the answer
    ms: float = 0.0
    model: str = ""
    usage: dict = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """One server. `complete` raises only `LLMError`; `params` holds model, timeout_s, effort, temperature.

    A provider may also have `warmup()`, which the engine calls when present.
    """

    id: str
    capabilities: ProviderCaps

    def complete(self, system: str, user: str, schema: dict | None, params: Mapping) -> LLMResult: ...


@dataclass(frozen=True)
class ProviderEntry:
    id: str
    kind: str  # one of KINDS
    base_url: str = ""
    model: str = ""  # the model a role gets when it names none
    key_env: str = ""  # name of the environment variable holding the key, "" for none
    schema_style: str = ""  # one of SCHEMA_STYLES; "" takes the kind's default
    timeout_s: float = 90.0
    options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"providers.{self.id}.kind: unknown kind {self.kind!r}; known: {', '.join(KINDS)}")
        if not self.schema_style:
            object.__setattr__(self, "schema_style", DEFAULT_SCHEMA_STYLE[self.kind])
        if self.schema_style not in SCHEMA_STYLES:
            raise ValueError(
                f"providers.{self.id}.schema_style: unknown style {self.schema_style!r}; "
                f"known: {', '.join(SCHEMA_STYLES)}"
            )


@dataclass(frozen=True)
class RoleConfig:
    provider: str  # a ProviderEntry id
    model: str = ""  # "" is the entry's model
    effort: str = ""


@dataclass(frozen=True)
class ProvidersConfig:
    providers: dict[str, ProviderEntry]
    roles: dict[str, RoleConfig]  # exactly ROLES


def default_config(settings: Settings) -> ProvidersConfig:
    """Every role on the Claude CLI with the LOCI_LLM_* settings: what the app does with no providers.json."""
    entry = ProviderEntry("claude_cli", "claude_cli", model=settings.llm_model, timeout_s=settings.llm_timeout_s)
    efforts = {role: settings.llm_effort_teacher for role in ROLES} | {"tutor_chat": settings.llm_effort_chat}
    roles = {role: RoleConfig(entry.id, effort=efforts[role]) for role in ROLES}
    return ProvidersConfig({entry.id: entry}, roles)


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen[key] = value
    return seen


def _object(value: object, where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where} is not an object")
    return value


def _string(raw: dict, key: str, where: str) -> str:
    value = raw.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{where}.{key} is not a string")
    return value


def _unknown_fields(raw: dict, known: tuple[str, ...], where: str) -> None:
    for key in raw:
        if key in KEY_FIELDS:
            # The value is not repeated: it may be a key.
            raise ValueError(f"{where}.{key}: keys never go in {PROVIDERS_FILE}; name the variable in key_env")
        if key not in known:
            raise ValueError(f"{where}.{key}: unknown field; known: {', '.join(known)}")


def _entry(entry_id: str, raw: object) -> ProviderEntry:
    where = f"providers.{entry_id}"
    raw = _object(raw, where)
    _unknown_fields(raw, ENTRY_FIELDS, where)
    if "kind" not in raw:
        raise ValueError(f"{where}.kind is missing")
    timeout_s = raw.get("timeout_s", 90.0)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not timeout_s > 0:
        raise ValueError(f"{where}.timeout_s is not a positive number")
    return ProviderEntry(
        id=entry_id,
        kind=_string(raw, "kind", where),
        base_url=_string(raw, "base_url", where),
        model=_string(raw, "model", where),
        key_env=_string(raw, "key_env", where),
        schema_style=_string(raw, "schema_style", where),
        timeout_s=float(timeout_s),
        options=dict(_object(raw.get("options", {}), f"{where}.options")),
    )


def _role(role: str, raw: object, providers: Mapping[str, ProviderEntry]) -> RoleConfig:
    where = f"roles.{role}"
    raw = _object(raw, where)
    _unknown_fields(raw, ROLE_FIELDS, where)
    provider = _string(raw, "provider", where)
    if provider not in providers:
        raise ValueError(f"{where}.provider: unknown provider {provider!r}; known: {', '.join(providers)}")
    return RoleConfig(provider, _string(raw, "model", where), _string(raw, "effort", where))


def load_providers(path: Path, settings: Settings) -> ProvidersConfig:
    """The providers and roles in `path`, or `default_config(settings)` when there is no such file.

    Anything malformed raises ValueError naming the key at fault.
    """
    if not path.is_file():
        return default_config(settings)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except ValueError as exc:  # json.JSONDecodeError is one
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from None
    raw = _object(raw, path.name)
    _unknown_fields(raw, ("providers", "roles"), path.name)

    entries = _object(raw.get("providers"), "providers")
    if not entries:
        raise ValueError("providers is empty")
    providers = {entry_id: _entry(entry_id, value) for entry_id, value in entries.items()}

    given = _object(raw.get("roles"), "roles")
    for role in given:
        if role not in ROLES:
            raise ValueError(f"roles.{role}: unknown role; known: {', '.join(ROLES)}")
    missing = [role for role in ROLES if role not in given]
    if missing:
        raise ValueError(f"roles: missing {', '.join(missing)}")
    roles = {role: _role(role, given[role], providers) for role in ROLES}
    return ProvidersConfig(providers, roles)


def resolve_key(entry: ProviderEntry, environ: Mapping[str, str | None] | None = None) -> str:
    """The key of `entry`: `environ[entry.key_env]`, by default the process environment over ROOT/.env.

    "" when the entry names no variable or the variable is unset.
    """
    if not entry.key_env:
        return ""
    if environ is None:
        environ = {**dotenv_values(ENV_FILE), **os.environ}
    return environ.get(entry.key_env) or ""


MODE_LINE = re.compile(r"^mode: *(\w+) *$", re.MULTILINE)


class FakeProvider:
    """Scripted provider: answers from `queue`, else a valid default for the schema asked for."""

    REPLY_EN = "Hello! How was your day?"
    REACTION = "Nice, that came across."
    WRAP_UP = {
        "words_en": "You found a word for everything we talked about.",
        "grammar_en": "Your sentences held together.",
        "task_en": "I could follow you the whole way.",
    }

    def __init__(self, latency_ms: float = 0.0, *, entry: ProviderEntry | None = None) -> None:
        self.id = entry.id if entry else "fake"
        self.model = (entry.model if entry else "") or "fake"
        self.latency_ms = latency_ms  # the `ms` of every result
        self.capabilities = ProviderCaps(
            json_schema=True, streaming=False, max_ctx=32768, local=True, cost_per_1k=0.0
        )
        # Every call in order: {"system", "user", "schema", "params"}.
        self.calls: list[dict] = []
        # Outcomes of the next calls, first in first out: a dict is the parsed answer, a str the raw
        # text (so malformed JSON can be scripted), an exception is raised.
        self.queue: list[dict | str | Exception] = []

    def warmup(self) -> None:
        pass

    def complete(self, system: str, user: str, schema: dict | None, params: Mapping) -> LLMResult:
        self.calls.append({"system": system, "user": user, "schema": schema, "params": dict(params)})
        outcome = self.queue.pop(0) if self.queue else self._default(user, schema)
        if isinstance(outcome, Exception):
            raise outcome
        model = params.get("model") or self.model
        if isinstance(outcome, str):
            return LLMResult(None, outcome, self.latency_ms, model)
        return LLMResult(outcome, json.dumps(outcome, ensure_ascii=False), self.latency_ms, model)

    def _default(self, user: str, schema: dict | None) -> dict:
        required = set((schema or {}).get("required", ()))
        if "verdict" in required:
            empty = dict.fromkeys(("quote", "better_lb", "why_en"), "")
            return {"verdict": "good", "reaction_en": self.REACTION, **empty, "claims": []}
        if "words_en" in required:
            return dict(self.WRAP_UP)
        mode = MODE_LINE.search(user)
        return {
            "segments": [{"lang": "lb", "text": "Moien!"}, {"lang": "en", "text": self.REPLY_EN}],
            "new_phrases": [{"lb": "Moien!", "en": "Hello!"}],
            "mode": mode.group(1) if mode else "chat",
            "claims": [],
        }

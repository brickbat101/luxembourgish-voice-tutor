"""LLM engine that sends each role to the provider and model providers.json names for it.

Every answer is checked with the parse_* functions of the CLI engine. A "schema" or "other"
failure gets one repair call that quotes the error; when that fails too, the role's fixed
template is said instead. "auth" and "timeout" are raised at once, and "auth" never ends in
a template.
"""

from __future__ import annotations

import importlib
import json
import logging
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from app.config import Settings
from app.content import PRACTICE_PATH, load_practice
from app.engines.base import TEMPLATE_WRAP_UP, Judgement, LLMError, Segment, TutorReply, Turn, WrapUpText
from app.engines.llm_claude_cli import (
    JSON_SCHEMA,
    JUDGE_PROMPT_PATH,
    JUDGE_SCHEMA,
    SYSTEM_PROMPT_PATH,
    WRAP_UP_PROMPT_PATH,
    WRAP_UP_SCHEMA,
    build_judge_prompt,
    build_prompt,
    build_wrap_up_prompt,
    parse_judgement,
    parse_reply,
    parse_wrap_up,
)
from app.engines.providers import (
    PROVIDER_KINDS,
    PROVIDERS_FILE,
    LLMProvider,
    LLMResult,
    ProviderEntry,
    ProvidersConfig,
    load_providers,
    resolve_key,
)

log = logging.getLogger(__name__)

TEMPERATURE = 0.3
RAISED_KINDS = ("auth", "timeout")  # raised at once: no repair call, no template
MAX_RECORDS = 1000
REPAIR = (
    "Your previous answer did not match the required JSON shape.\n"
    "The error: {error}\n"
    "Your previous answer began: {output}\n"
    "Answer again with only the JSON object, nothing before or after it."
)
SORRY_EN = "Sorry, I lost my thread for a moment. Could you say that once more?"
# practice.toml's more_lb and ack_en, for when that file cannot be read.
MORE_LB = "Kënnt Dir e bësse méi dozou soen?"
ACK_EN = "Got it."


@dataclass
class CallRecord:
    """One call of a role, with its repair call when there was one."""

    role: str
    provider_id: str
    model: str
    ms: float = 0.0  # the providers' own time, summed over the attempts
    attempts: int = 0
    first_try_valid: bool = False
    fell_back: bool = False  # the template was said
    error: str = ""  # "" or the kind of the last failure


def build_provider(entry: ProviderEntry, settings: Settings) -> LLMProvider:
    """The provider of `entry`; its module is imported only now. ValueError when its kind cannot be built."""
    module_name, class_name = PROVIDER_KINDS[entry.kind].split(":")
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"providers.{entry.id}: kind {entry.kind!r} is not available: {exc}") from exc
    if entry.kind == "fake":
        return cls(entry=entry)
    if entry.kind == "claude_cli":
        return cls(entry, settings)
    return cls(entry, resolve_key(entry))


def _raw(result: LLMResult) -> str:
    if result.text:
        return result.text
    return "" if result.data is None else json.dumps(result.data, ensure_ascii=False)


def _parsed(result: LLMResult, parse: Callable):
    data = result.data
    if data is None:
        try:
            data = json.loads(result.text)
        except ValueError as exc:
            raise LLMError("schema", f"the reply is not JSON: {result.text[:200]!r}") from exc
    return parse(data)


class ProviderLLM:
    name = "provider"

    def __init__(
        self,
        settings: Settings,
        providers: Mapping[str, LLMProvider] | None = None,
        config: ProvidersConfig | None = None,
    ) -> None:
        """`config` defaults to data_dir/providers.json; `providers` (id -> provider) to one built per used entry."""
        self.settings = settings
        self.config = config or load_providers(settings.data_dir / PROVIDERS_FILE, settings)
        used = dict.fromkeys(role.provider for role in self.config.roles.values())
        if providers is None:
            providers = {entry_id: build_provider(self.config.providers[entry_id], settings) for entry_id in used}
        self.providers = dict(providers)
        self.missing = [entry_id for entry_id in used if entry_id not in self.providers]
        self.warmup_failures: list[str] = []
        self.records: deque[CallRecord] = deque(maxlen=MAX_RECORDS)
        self.system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        self.judge_prompt = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
        self.wrap_up_prompt = WRAP_UP_PROMPT_PATH.read_text(encoding="utf-8")
        try:
            tutor = load_practice(PRACTICE_PATH).tutor
            self.more_lb, self.ack_en = tutor["more_lb"], tutor["ack_en"]
        except (OSError, ValueError) as exc:
            log.warning("provider engine: the templates use built-in lines, %s: %s", PRACTICE_PATH.name, exc)
            self.more_lb, self.ack_en = MORE_LB, ACK_EN

    @property
    def detail(self) -> str:
        problems = [f"no provider {entry_id!r} was built" for entry_id in self.missing]
        problems += self.warmup_failures
        for entry_id, provider in self.providers.items():
            ready = getattr(provider, "ready", None)
            if ready is not None and not ready():
                problems.append(f"{entry_id}: {getattr(provider, 'detail', '') or 'not ready'}")
        return "; ".join(problems)

    def ready(self) -> bool:
        return not self.detail

    def warmup(self) -> None:
        """Calls each provider's own `warmup` when it has one; a failure goes into `detail`, never raises."""
        self.warmup_failures = []
        for entry_id, provider in self.providers.items():
            warmup = getattr(provider, "warmup", None)
            if warmup is None:
                continue
            try:
                warmup()
            except Exception as exc:
                self.warmup_failures.append(f"{entry_id}: warmup failed: {type(exc).__name__}: {exc}")

    def reply(self, history: list[Turn], mode: str, grounding: str) -> TutorReply:
        role = "tutor_chat" if mode == "chat" else "teacher_explain"
        prompt = build_prompt(history, mode, grounding)
        return self._ask(role, self.system_prompt, prompt, JSON_SCHEMA, parse_reply, self.template_reply(mode))

    def judge(self, question_lb: str, question_en: str, answer: str, grounding: str, extra: str = "") -> Judgement:
        prompt = build_judge_prompt(question_lb, question_en, answer, grounding, extra)
        return self._ask("judge", self.judge_prompt, prompt, JUDGE_SCHEMA, parse_judgement, self.template_judgement())

    def wrap_up(self, facts: str) -> WrapUpText:
        prompt = build_wrap_up_prompt(facts)
        template = replace(TEMPLATE_WRAP_UP)
        return self._ask("wrapup", self.wrap_up_prompt, prompt, WRAP_UP_SCHEMA, parse_wrap_up, template)

    def template_reply(self, mode: str) -> TutorReply:
        return TutorReply([Segment("lb", self.more_lb), Segment("en", SORRY_EN)], [], mode, [])

    def template_judgement(self) -> Judgement:
        return Judgement("good", self.ack_en)

    def _ask(self, role: str, system: str, user: str, schema: dict, parse: Callable, template):
        """What `parse` makes of the answer of the role's provider, after one repair call at most, else `template`."""
        route = self.config.roles[role]
        entry = self.config.providers[route.provider]
        params = {
            "model": route.model or entry.model,
            "timeout_s": entry.timeout_s,
            "effort": route.effort,
            "temperature": TEMPERATURE,
        }
        record = CallRecord(role, entry.id, params["model"])
        try:
            provider = self.providers.get(entry.id)
            if provider is None:
                raise LLMError("other", f"no provider {entry.id!r} was built for {role}")
            raw = ""
            try:
                result = self._complete(provider, system, user, schema, params, record)
                raw = _raw(result)
                answer = _parsed(result, parse)
                record.first_try_valid = True
                return answer
            except LLMError as error:
                if error.kind in RAISED_KINDS:
                    raise
                repair = f"{user}\n\n{REPAIR.format(error=(error.detail or str(error))[:300], output=raw[:500])}"
            try:
                return _parsed(self._complete(provider, system, repair, schema, params, record), parse)
            except LLMError as error:
                if error.kind == "auth":
                    raise
                record.fell_back, record.error = True, error.kind
                log.warning(
                    "provider engine: %s says its template; provider %s failed twice, %s: %s",
                    role,
                    entry.id,
                    error.kind,
                    error.detail[:300],
                )
                return template
        except LLMError as error:
            record.error = error.kind
            raise
        except Exception as exc:
            record.error = type(exc).__name__
            raise
        finally:
            self.records.append(record)

    @staticmethod
    def _complete(
        provider: LLMProvider, system: str, user: str, schema: dict, params: dict, record: CallRecord
    ) -> LLMResult:
        record.attempts += 1
        result = provider.complete(system, user, schema, params)
        record.ms += result.ms
        record.model = result.model or record.model
        return result

"""tutor-eval: the SCHEMA, LATENCY and LB-GEN suites over an LLM engine, the verdict and the report.

SCHEMA is the share of calls whose first answer passed the parse_* checks; LATENCY the total
time of a call (nothing streams); LB-GEN whether the "lb" segments of the replies are
Luxembourgish, by the share of their words LOD lists and, when the langid package is
installed, its language guess. The verdict uses the gates below; latency is shown, never gated.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from app.engines.base import LLMEngine, LLMError, Turn, TutorReply

PROMPTS_PATH = Path(__file__).with_name("evaluation_prompts.json")
ROLES = ("tutor_chat", "teacher_explain", "judge", "wrapup")
REPLY_ROLES = {"tutor_chat": "chat", "teacher_explain": "teacher"}  # role -> the mode of its reply call
UNREVIEWED = "unreviewed"

SCHEMA_GATE = 0.98
LB_GATE = 0.95
RESOLVABLE_GATE = 0.90
LB_THRESHOLD = 0.6  # LOD coverage from which a segment counts as Luxembourgish

# Apostrophes are not word characters, so the clitics d', 't, 'n and 's come off their word.
_WORD = re.compile(r"[^\W\d_]+")

CHAT = "chat"
EXPLANATIONS = "explanations"
CORRECTING = "correcting your Luxembourgish"
NOT_MEASURED = "LB-JUDGE not measured in this run"
RESOLVABLE_CAVEAT = (
    "LOD coverage below the gate; see README: names and forms LOD does not list count as misses, "
    "so this gate is not reliable and not part of the verdict"
)


class Lexicon(Protocol):
    def known(self, form: str) -> bool: ...


@dataclass(frozen=True)
class Prompt:
    id: str
    role: str  # one of ROLES
    args: dict  # the arguments of the role's engine call
    tags: tuple[str, ...] = ()


@dataclass
class Sample:
    """One engine call of a prompt and what the engine's CallRecord says of it."""

    prompt_id: str
    role: str
    tags: tuple[str, ...] = ()
    warmup: bool = False  # the call before the measured ones; no suite counts it
    ms: float = 0.0
    attempts: int = 0
    first_try_valid: bool = False
    fell_back: bool = False
    error: str = ""  # "" or the kind of the LLMError raised
    lb_segments: list[str] = field(default_factory=list)  # the "lb" texts of a reply


@dataclass(frozen=True)
class LbGen:
    lb_rate: float  # share of the lb segments that count as Luxembourgish
    n: int  # lb segments
    resolvable_rate: float  # mean LOD coverage over the lb segments
    by_tag: dict[str, tuple[float, int]] = field(default_factory=dict)  # tag -> (lb_rate, n)
    signals: tuple[str, ...] = ("LOD coverage",)
    reliable: bool = False  # of resolvable_rate
    caveat: str = RESOLVABLE_CAVEAT


@dataclass(frozen=True)
class Verdict:
    good_for: list[str]
    not_recommended: list[str]
    notes: list[str]


@dataclass(frozen=True)
class Results:
    schema: tuple[float, int]  # (rate, N)
    latency: tuple[float, float, int]  # (p50 ms, p90 ms, N)
    lb: LbGen
    verdict: Verdict
    errors: dict[str, int]  # LLMError kind -> calls
    fell_back: int  # calls that ended in the template
    unreviewed: bool  # the prompts are not native-reviewed
    schema_off: tuple[str, ...] | None = None  # the providers run without their schema; None: native


def _prompt(raw: object, review: str) -> Prompt:
    if not isinstance(raw, dict):
        raise ValueError(f"a prompt is not an object: {raw!r}")
    prompt_id, role, args, tags = raw.get("id"), raw.get("role"), raw.get("args"), raw.get("tags", [])
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError(f"a prompt has no id: {raw!r}")
    if role not in ROLES:
        raise ValueError(f"{prompt_id}: unknown role {role!r}; known: {', '.join(ROLES)}")
    if not isinstance(args, dict):
        raise ValueError(f"{prompt_id}.args is not an object")
    if role in REPLY_ROLES:
        if args.get("mode") != REPLY_ROLES[role]:
            raise ValueError(f"{prompt_id}: {role} takes mode {REPLY_ROLES[role]!r}")
        history = args.get("history")
        if not isinstance(history, list) or not history or history[-1].get("role") != "user":
            raise ValueError(f"{prompt_id}.history does not end in a user line")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError(f"{prompt_id}.tags is not a list of strings")
    if review == UNREVIEWED:
        tags = [*tags, UNREVIEWED]
    return Prompt(prompt_id, role, args, tuple(dict.fromkeys(tags)))


def load_prompts(path: Path = PROMPTS_PATH) -> list[Prompt]:
    """The prompts of `path`, a {"review", "prompts"} object; every prompt is tagged "unreviewed" when the set is."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("prompts"), list):
        raise ValueError(f"{Path(path).name} holds no prompts list")
    review = raw.get("review", "")
    prompts = [_prompt(item, review) for item in raw["prompts"]]
    ids = Counter(prompt.id for prompt in prompts)
    doubled = [prompt_id for prompt_id, count in ids.items() if count > 1]
    if doubled:
        raise ValueError(f"duplicate prompt ids: {', '.join(doubled)}")
    return prompts


def call(engine: LLMEngine, prompt: Prompt):
    """The engine's answer to `prompt`, through its public method for the role."""
    args = prompt.args
    if prompt.role in REPLY_ROLES:
        history = [Turn(turn["role"], turn["text"]) for turn in args["history"]]
        return engine.reply(history, args["mode"], args.get("grounding", ""))
    if prompt.role == "judge":
        return engine.judge(
            args["question_lb"], args["question_en"], args["answer"], args.get("grounding", ""), args.get("extra", "")
        )
    return engine.wrap_up(args["facts"])


def openai_compat_ids(engine) -> list[str]:
    """The providers of `engine` that put their schema in an OpenAI-compatible request envelope."""
    return [
        provider_id
        for provider_id, provider in engine.providers.items()
        if getattr(provider, "kind", "") == "openai_compat" and provider.entry.schema_style != "none"
    ]


@contextmanager
def schema_left_out(engine) -> Iterator[list[str]]:
    """For the block, each provider of `openai_compat_ids` is replaced by a copy whose style is "none"."""
    flipped = openai_compat_ids(engine)
    originals = {provider_id: engine.providers[provider_id] for provider_id in flipped}
    for provider_id, provider in originals.items():
        clone = copy.copy(provider)
        clone.entry = replace(provider.entry, schema_style="none")
        clone.capabilities = replace(provider.capabilities, json_schema=False)
        engine.providers[provider_id] = clone
    try:
        yield flipped
    finally:
        engine.providers.update(originals)


def _sample(engine, prompt: Prompt, warmup: bool) -> Sample:
    sample = Sample(prompt.id, prompt.role, prompt.tags, warmup)
    records = engine.records
    before = records[-1] if records else None
    try:
        answer = call(engine, prompt)
    except LLMError as error:
        sample.error = error.kind
        answer = None
    record = records[-1] if records and records[-1] is not before else None
    if record is not None:
        sample.ms, sample.attempts = record.ms, record.attempts
        sample.first_try_valid, sample.fell_back = record.first_try_valid, record.fell_back
    if isinstance(answer, TutorReply):
        sample.lb_segments = [segment.text for segment in answer.segments if segment.lang == "lb"]
    return sample


def run_suite(engine, prompts: Sequence[Prompt], schema_off: bool = False) -> list[Sample]:
    """One warm-up call of the first prompt (flagged `warmup`), then one call per prompt.

    `engine` keeps a `records` deque of CallRecord, as ProviderLLM does. An LLMError becomes the
    sample's `error`; anything else is raised. `schema_off` runs the providers of
    `openai_compat_ids` with the schema in the prompt instead of the request envelope.
    """
    if not prompts:
        return []
    if not schema_off:
        return [_sample(engine, prompts[0], True), *(_sample(engine, prompt, False) for prompt in prompts)]
    with schema_left_out(engine):
        return run_suite(engine, prompts)


def _measured(samples: Sequence[Sample]) -> list[Sample]:
    return [sample for sample in samples if not sample.warmup]


def schema_rate(samples: Sequence[Sample]) -> tuple[float, int]:
    """(share of the measured calls whose first answer was valid, N); a template or an error counts as invalid."""
    measured = _measured(samples)
    if not measured:
        return 0.0, 0
    return sum(sample.first_try_valid for sample in measured) / len(measured), len(measured)


def percentiles(values: Sequence[float]) -> tuple[float, float]:
    """(p50, p90) by nearest rank on the sorted values; (0.0, 0.0) for none."""
    ordered = sorted(values)
    if not ordered:
        return 0.0, 0.0

    def rank(percent: int) -> float:
        return ordered[max(1, -(-percent * len(ordered) // 100)) - 1]

    return rank(50), rank(90)


def latency(samples: Sequence[Sample]) -> tuple[float, float, int]:
    """(p50 ms, p90 ms, N) of the calls after the warm-up that gave an answer."""
    values = [sample.ms for sample in _measured(samples) if not sample.error]
    return *percentiles(values), len(values)


def lb_tokens(text: str) -> list[str]:
    """The words of `text`: letters only, with d', 't, 'n and 's split off; numerals are not words."""
    return _WORD.findall(text)


def lod_coverage(text: str, lod: Lexicon) -> float:
    """Share of the words of `text` that LOD lists; 0.0 for a text without words."""
    tokens = lb_tokens(text)
    if not tokens:
        return 0.0
    return sum(bool(lod.known(token)) for token in tokens) / len(tokens)


def is_lb(text: str, lod: Lexicon, lang_id: Callable[[str], str] | None = None) -> bool:
    """LOD coverage >= LB_THRESHOLD and, when `lang_id` is given, its guess is "lb"."""
    if lod_coverage(text, lod) < LB_THRESHOLD:
        return False
    return lang_id is None or lang_id(text) == "lb"


def default_lang_id() -> Callable[[str], str] | None:
    """langid's guess when the optional langid package is installed, else None."""
    try:
        import langid
    except ImportError:
        return None
    return lambda text: langid.classify(text)[0]


def _rate(flags: list[bool]) -> tuple[float, int]:
    return (sum(flags) / len(flags) if flags else 0.0), len(flags)


def lb_gen(samples: Sequence[Sample], lod: Lexicon, lang_id: Callable[[str], str] | None = None) -> LbGen:
    """LB-GEN over the "lb" segments of the measured replies; a template's segments are not the model's."""
    replies = [
        sample
        for sample in _measured(samples)
        if sample.role in REPLY_ROLES and not sample.error and not sample.fell_back
    ]
    flags, coverage, tagged = [], [], {}
    for sample in replies:
        for text in sample.lb_segments:
            flag = is_lb(text, lod, lang_id)
            flags.append(flag)
            coverage.append(lod_coverage(text, lod))
            for tag in sample.tags:
                tagged.setdefault(tag, []).append(flag)
    rate, n = _rate(flags)
    by_tag = {tag: _rate(tag_flags) for tag, tag_flags in tagged.items() if tag != UNREVIEWED}
    signals = ("LOD coverage",) if lang_id is None else ("LOD coverage", "langid")
    resolvable = sum(coverage) / len(coverage) if coverage else 0.0
    return LbGen(rate, n, resolvable, by_tag, signals)


def lb_passed(lb: LbGen) -> bool:
    return lb.n > 0 and lb.lb_rate >= LB_GATE


def verdict(schema: float, lb: LbGen) -> Verdict:
    """Chat and explanations are good for use when both LB-GEN's lb_rate and SCHEMA pass their gates."""
    notes = [NOT_MEASURED]
    passed = True
    if lb.n == 0:
        notes.append("LB-GEN: no Luxembourgish segment to score")
        passed = False
    elif lb.lb_rate < LB_GATE:
        notes.append(f"LB-GEN below the gate: {lb.lb_rate:.1%} of lb segments read as Luxembourgish")
        passed = False
    if schema < SCHEMA_GATE:
        notes.append(f"SCHEMA below the gate: {schema:.1%} valid on the first try")
        passed = False
    if lb.n and lb.resolvable_rate < RESOLVABLE_GATE:
        notes.append(lb.caveat)
    uses = [CHAT, EXPLANATIONS]
    good_for = uses if passed else []
    not_recommended = ([] if passed else uses) + [CORRECTING]
    return Verdict(good_for, not_recommended, notes)


def evaluate(
    samples: Sequence[Sample],
    lod: Lexicon,
    lang_id: Callable[[str], str] | None = None,
    schema_off: Sequence[str] | None = None,
) -> Results:
    """The three suites and the verdict over one run; `schema_off` names the providers run without schema."""
    measured = _measured(samples)
    schema = schema_rate(samples)
    lb = lb_gen(samples, lod, lang_id)
    return Results(
        schema=schema,
        latency=latency(samples),
        lb=lb,
        verdict=verdict(schema[0], lb),
        errors=dict(Counter(sample.error for sample in measured if sample.error)),
        fell_back=sum(sample.fell_back for sample in measured),
        unreviewed=any(UNREVIEWED in sample.tags for sample in samples),
        schema_off=None if schema_off is None else tuple(schema_off),
    )


def _percent(rate: float, n: int) -> str:
    return f"{rate * 100:.1f} %" if n else "n/a"


def _gate(passed: bool) -> str:
    return "pass" if passed else "below gate"


def _schema_mode(results: Results) -> str:
    if results.schema_off is None:
        return "schema: native"
    if not results.schema_off:
        return "schema off: no provider changed (only openai_compat entries put the schema in the request)"
    return f"schema off (in the prompt) for {', '.join(results.schema_off)}"


def format_report(
    results: Results, *, provider_id: str, model: str, n: int, date: str, signals: Sequence[str] | None = None
) -> str:
    """The text report: header, one line per suite with its N and gate, then the verdict."""
    rate, schema_n = results.schema
    p50, p90, latency_n = results.latency
    lb = results.lb
    prompt_set = f"{n} prompts" + (f" ({UNREVIEWED})" if results.unreviewed else "")
    lines = [
        f"tutor-eval  {date}  provider {provider_id}  model {model}  N={prompt_set}  {_schema_mode(results)}",
        f"SCHEMA  valid first try {_percent(rate, schema_n)}  (N={schema_n}, gate >= {SCHEMA_GATE * 100:.0f} %)"
        f"  {_gate(rate >= SCHEMA_GATE)}",
        (
            f"LATENCY p50 {p50:.0f} ms  p90 {p90:.0f} ms" if latency_n else "LATENCY p50 n/a  p90 n/a"
        ) + f"  (total call time, no streaming; N={latency_n}, warm-up dropped)",
        f"LB-GEN  lb segments {_percent(lb.lb_rate, lb.n)}  (N={lb.n}, gate >= {LB_GATE * 100:.0f} %)"
        f"  {_gate(lb_passed(lb))}"
        f"  LOD coverage {lb.resolvable_rate:.2f} (gate >= {RESOLVABLE_GATE:.2f}, not reliable: see README)",
    ]
    for tag, (tag_rate, tag_n) in sorted(lb.by_tag.items()):
        lines.append(f"        {tag} turns: lb segments {_percent(tag_rate, tag_n)}  (N={tag_n})")
    lines.append(f"signals: {', '.join(signals if signals is not None else lb.signals)}")
    errors = ", ".join(f"{count} {kind}" for kind, count in sorted(results.errors.items())) or "none"
    lines.append(f"errors: {errors}; templates said: {results.fell_back}")
    lines.append(f"Good for: {', '.join(results.verdict.good_for) or 'nothing in this run'}")
    lines.append(f"Not recommended for: {', '.join(results.verdict.not_recommended)}")
    lines += [f"note: {note}" for note in results.verdict.notes]
    return "\n".join(lines)


def to_json(
    results: Results, *, provider_id: str, model: str, n: int, date: str, signals: Sequence[str] | None = None
) -> dict:
    """The report as a JSON-ready dict."""
    rate, schema_n = results.schema
    p50, p90, latency_n = results.latency
    lb = results.lb
    return {
        "date": date,
        "provider": provider_id,
        "model": model,
        "prompts": n,
        "prompt_set": UNREVIEWED if results.unreviewed else "reviewed",
        "schema": {
            "valid_first_try": rate,
            "n": schema_n,
            "gate": SCHEMA_GATE,
            "passed": rate >= SCHEMA_GATE,
            "schema_off": None if results.schema_off is None else list(results.schema_off),
        },
        "latency": {
            "p50_ms": p50,
            "p90_ms": p90,
            "n": latency_n,
            "measure": "total call time, no streaming, warm-up dropped",
        },
        "lb_gen": {
            "lb_rate": lb.lb_rate,
            "n": lb.n,
            "gate": LB_GATE,
            "passed": lb_passed(lb),
            "threshold": LB_THRESHOLD,
            "lod_coverage": lb.resolvable_rate,
            "lod_coverage_gate": RESOLVABLE_GATE,
            "lod_coverage_reliable": lb.reliable,
            "by_tag": {tag: {"lb_rate": tag_rate, "n": tag_n} for tag, (tag_rate, tag_n) in lb.by_tag.items()},
        },
        "signals": list(signals if signals is not None else lb.signals),
        "errors": dict(results.errors),
        "templates_said": results.fell_back,
        "verdict": {
            "good_for": list(results.verdict.good_for),
            "not_recommended": list(results.verdict.not_recommended),
            "notes": list(results.verdict.notes),
        },
    }

"""tutor-eval: the suites, the verdict and the report on fakes, and scripts/tutor_eval.py on a tiny LOD store."""

import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from app import evaluation
from app.config import Settings
from app.engines.base import LLMError
from app.engines.llm_provider import ProviderLLM
from app.engines.providers import ROLES, FakeProvider, ProviderEntry, ProvidersConfig, RoleConfig
from app.engines.providers_http import OpenAICompatProvider
from app.evaluation import (
    CORRECTING,
    NOT_MEASURED,
    RESOLVABLE_CAVEAT,
    LbGen,
    Prompt,
    Sample,
    evaluate,
    format_report,
    is_lb,
    latency,
    lb_gen,
    lb_tokens,
    load_prompts,
    lod_coverage,
    percentiles,
    run_suite,
    schema_rate,
    to_json,
    verdict,
)
from app.lod.build import SCHEMA
from scripts import tutor_eval

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "tutor_eval.py"
LB_WORDS = "moien gutt ech sinn d kanner do wéi geet et mir ginn haut an de bësch".split()
GERMAN = "Ich bin heute sehr müde und gehe nach Hause"
GOOD_REPLY = {
    "segments": [{"lang": "lb", "text": "Gutt!"}, {"lang": "en", "text": "Good, thanks."}],
    "new_phrases": [],
    "mode": "chat",
    "claims": [],
}
GERMAN_REPLY = GOOD_REPLY | {"segments": [{"lang": "lb", "text": GERMAN}, {"lang": "en", "text": "Nice."}]}


class StubLod:
    def __init__(self, words=LB_WORDS) -> None:
        self.words = {word.casefold() for word in words}

    def known(self, form: str) -> bool:
        return form.casefold() in self.words


def chat_prompt(index: int, text: str = "I went to the market.", tags: tuple = ()) -> Prompt:
    args = {"history": [{"role": "user", "text": text}], "mode": "chat", "grounding": ""}
    return Prompt(f"chat-{index:02d}", "tutor_chat", args, tags)


def lod_store(folder: Path, words=LB_WORDS) -> Path:
    """A LOD store at folder/lod/lod.sqlite whose wordforms list `words` as lemmas."""
    path = folder / "lod" / "lod.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO wordforms VALUES (?, ?, 'lemma')", [(word, word) for word in words])
    db.commit()
    db.close()
    return path


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), data_dir=tmp_path / "data")


@pytest.fixture
def fake():
    return FakeProvider(latency_ms=5)


@pytest.fixture
def engine(settings, fake):
    config = ProvidersConfig({"fake": ProviderEntry("fake", "fake")}, {role: RoleConfig("fake") for role in ROLES})
    return ProviderLLM(settings, providers={"fake": fake}, config=config)


# --- LB-GEN scoring ---


def test_lb_tokens_split_the_clitics_and_drop_numerals():
    assert lb_tokens("d'Kanner sinn do.") == ["d", "Kanner", "sinn", "do"]
    assert lb_tokens("Ech hunn 3 Kanner, wéi geet’t?") == ["Ech", "hunn", "Kanner", "wéi", "geet", "t"]
    assert lb_tokens("42 !") == []


def test_lod_coverage_is_the_share_of_words_lod_lists():
    lod = StubLod()
    assert lod_coverage("d'Kanner sinn do.", lod) == 1.0
    assert lod_coverage("Ich arbeite heute", StubLod(["ech"])) == 0.0
    assert lod_coverage("Moien Welt", lod) == 0.5
    assert lod_coverage("", lod) == 0.0


def test_is_lb_needs_the_threshold_and_the_language_guess_when_one_is_given():
    lod = StubLod()
    assert is_lb("Moien, wéi geet et?", lod)
    assert not is_lb(GERMAN, lod)
    assert not is_lb("Moien, wéi geet et?", lod, lang_id=lambda text: "de")
    assert is_lb("Moien, wéi geet et?", lod, lang_id=lambda text: "lb")


def test_lb_gen_scores_the_lb_segments_of_the_measured_replies_only():
    samples = [
        Sample("w", "tutor_chat", warmup=True, lb_segments=[GERMAN]),
        Sample("a", "tutor_chat", ("unreviewed",), lb_segments=["Moien!", "Gutt!"]),
        Sample("b", "tutor_chat", ("german_negative", "unreviewed"), lb_segments=[GERMAN]),
        Sample("c", "teacher_explain", lb_segments=["Ech sinn midd"]),
        Sample("d", "tutor_chat", fell_back=True, lb_segments=[GERMAN]),
        Sample("e", "tutor_chat", error="timeout"),
        Sample("f", "judge"),
    ]

    lb = lb_gen(samples, StubLod())

    assert (lb.lb_rate, lb.n) == (0.75, 4)
    assert lb.by_tag == {"german_negative": (0.0, 1)}
    assert lb.resolvable_rate == pytest.approx((1 + 1 + 0 + 2 / 3) / 4)
    assert lb.signals == ("LOD coverage",) and lb.reliable is False
    assert lb_gen(samples, StubLod(), lang_id=lambda text: "lb").signals == ("LOD coverage", "langid")


# --- LATENCY ---


def test_percentiles_are_nearest_rank():
    assert percentiles([100, 10, 90, 20, 80, 30, 70, 40, 60, 50]) == (50, 90)
    assert percentiles([7]) == (7, 7)
    assert percentiles([]) == (0.0, 0.0)


def test_latency_drops_the_warm_up_and_the_failed_calls():
    samples = [Sample("w", "tutor_chat", warmup=True, ms=5000.0)]
    samples += [Sample(f"s{ms}", "tutor_chat", ms=float(ms)) for ms in range(10, 101, 10)]
    samples.append(Sample("x", "tutor_chat", ms=9000.0, error="timeout"))

    assert latency(samples) == (50.0, 90.0, 10)


# --- run_suite and SCHEMA ---


def test_run_suite_warms_up_once_then_calls_each_prompt_and_reads_the_records(engine, fake):
    judge = {"question_lb": "Wat schafft Dir?", "question_en": "What do you do?", "answer": "Ech schaffen."}
    prompts = [
        chat_prompt(1),
        Prompt("judge-01", "judge", judge),
        Prompt("wrapup-01", "wrapup", {"facts": "Questions the learner answered: 1"}),
    ]

    samples = run_suite(engine, prompts)

    assert [(s.prompt_id, s.role, s.warmup) for s in samples] == [
        ("chat-01", "tutor_chat", True),
        ("chat-01", "tutor_chat", False),
        ("judge-01", "judge", False),
        ("wrapup-01", "wrapup", False),
    ]
    assert len(fake.calls) == 4
    assert all(s.ms == 5 and s.attempts == 1 and s.first_try_valid and not s.error for s in samples)
    assert samples[1].lb_segments == ["Moien!"] and samples[2].lb_segments == []


def test_one_bad_first_answer_in_twenty_gives_a_schema_rate_of_095(engine, fake):
    # The warm-up takes the first outcome; the bad one is the third measured call.
    fake.queue += [GOOD_REPLY, GOOD_REPLY, GOOD_REPLY, "not json"]

    samples = run_suite(engine, [chat_prompt(index) for index in range(20)])

    assert schema_rate(samples) == (0.95, 20)
    bad = samples[3]
    assert (bad.attempts, bad.first_try_valid, bad.fell_back, bad.ms) == (2, False, False, 10)
    assert all(s.attempts == 1 for s in samples if s is not bad)


def test_an_llm_error_becomes_the_sample_error_and_counts_against_schema(engine, fake):
    fake.queue += [GOOD_REPLY, LLMError("timeout", "no reply")]

    samples = run_suite(engine, [chat_prompt(1), chat_prompt(2)])

    assert samples[1].error == "timeout" and not samples[1].first_try_valid
    assert schema_rate(samples) == (0.5, 2)
    assert evaluate(samples, StubLod()).errors == {"timeout": 1}


def test_a_template_counts_as_a_schema_failure(engine, fake):
    fake.queue += [GOOD_REPLY, "not json", "still not json"]

    samples = run_suite(engine, [chat_prompt(1)])

    assert samples[1].fell_back and samples[1].attempts == 2
    assert schema_rate(samples) == (0.0, 1)
    assert lb_gen(samples, StubLod()).n == 0


def test_schema_off_puts_the_schema_of_openai_compat_entries_in_the_prompt_for_the_run(settings):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        content = json.dumps(GOOD_REPLY)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})

    entry = ProviderEntry("local", "openai_compat", base_url="http://127.0.0.1:11434/v1", model="gemma")
    provider = OpenAICompatProvider(entry, client=httpx.Client(transport=httpx.MockTransport(handler)))
    config = ProvidersConfig({"local": entry}, {role: RoleConfig("local") for role in ROLES})
    engine = ProviderLLM(settings, providers={"local": provider}, config=config)

    assert evaluation.openai_compat_ids(engine) == ["local"]
    samples = run_suite(engine, [chat_prompt(1)], schema_off=True)

    assert all("response_format" not in body for body in bodies) and len(bodies) == 2
    assert "JSON" in bodies[0]["messages"][0]["content"]
    assert schema_rate(samples) == (1.0, 1)
    assert engine.providers["local"] is provider and provider.entry.schema_style == "nested"
    run_suite(engine, [chat_prompt(1)])
    assert "response_format" in bodies[-1]


# --- the verdict and the report ---


def passing_lb(**changes) -> LbGen:
    return replace(LbGen(lb_rate=0.97, n=40, resolvable_rate=0.92), **changes)


def test_verdict_passes_chat_and_explanations_but_never_correcting():
    result = verdict(0.99, passing_lb())

    assert result.good_for == ["chat", "explanations"]
    assert result.not_recommended == [CORRECTING]
    assert result.notes == [NOT_MEASURED]


def test_verdict_fails_on_german_in_the_lb_segments(engine, fake):
    fake.queue += [GOOD_REPLY] + [GERMAN_REPLY] * 3
    samples = run_suite(engine, [chat_prompt(index) for index in range(10)])

    lb = lb_gen(samples, StubLod())
    result = verdict(schema_rate(samples)[0], lb)

    assert lb.lb_rate == pytest.approx(0.7)
    assert result.good_for == []
    assert result.not_recommended == ["chat", "explanations", CORRECTING]
    assert any(note.startswith("LB-GEN below the gate") for note in result.notes)


def test_verdict_fails_on_schema():
    result = verdict(0.975, passing_lb())

    assert "chat" not in result.good_for and "chat" in result.not_recommended
    assert any(note.startswith("SCHEMA below the gate: 97.5%") for note in result.notes)


def test_low_lod_coverage_is_noted_but_does_not_decide():
    result = verdict(1.0, passing_lb(resolvable_rate=0.78))

    assert result.good_for == ["chat", "explanations"]
    assert RESOLVABLE_CAVEAT in result.notes


def test_no_lb_segment_is_no_pass():
    assert verdict(1.0, passing_lb(n=0, lb_rate=0.0)).good_for == []


def test_the_report_labels_every_suite_with_its_n_gate_and_the_date(engine):
    samples = run_suite(engine, [chat_prompt(index, tags=("unreviewed",)) for index in range(4)])
    results = evaluate(samples, StubLod())
    meta = {"provider_id": "fake", "model": "fake", "n": 4, "date": "2026-10-15"}

    text = format_report(results, **meta)

    lines = text.splitlines()
    assert lines[0] == "tutor-eval  2026-10-15  provider fake  model fake  N=4 prompts (unreviewed)  schema: native"
    assert lines[1] == "SCHEMA  valid first try 100.0 %  (N=4, gate >= 98 %)  pass"
    assert lines[2] == "LATENCY p50 5 ms  p90 5 ms  (total call time, no streaming; N=4, warm-up dropped)"
    assert lines[3].startswith("LB-GEN  lb segments 100.0 %  (N=4, gate >= 95 %)  pass  LOD coverage 1.00")
    assert "signals: LOD coverage" in lines
    assert "Good for: chat, explanations" in lines
    assert f"Not recommended for: {CORRECTING}" in lines
    assert f"note: {NOT_MEASURED}" in lines

    data = to_json(results, **meta)
    assert json.loads(json.dumps(data)) == data
    assert data["date"] == "2026-10-15" and data["prompt_set"] == "unreviewed"
    assert data["schema"] == {"valid_first_try": 1.0, "n": 4, "gate": 0.98, "passed": True, "schema_off": None}
    assert data["verdict"]["good_for"] == ["chat", "explanations"]


# --- the prompt set ---


def test_the_prompt_set_has_100_unreviewed_prompts_in_the_four_roles():
    raw = json.loads(evaluation.PROMPTS_PATH.read_text(encoding="utf-8"))
    prompts = load_prompts()

    assert raw["review"] == "unreviewed"
    assert len(prompts) == 100 and len({p.id for p in prompts}) == 100
    assert Counter(p.role for p in prompts) == {"tutor_chat": 70, "teacher_explain": 20, "judge": 5, "wrapup": 5}
    german = [p for p in prompts if "german_negative" in p.tags]
    assert len(german) == 10 and {p.role for p in german} == {"tutor_chat"}
    assert all("unreviewed" in p.tags for p in prompts)
    for prompt in prompts:
        if prompt.role == "tutor_chat":
            learner = [turn for turn in prompt.args["history"] if turn["role"] == "user"]
            assert 1 <= len(learner) <= 2 and prompt.args["grounding"] == ""
    # The first twenty, what a short live run sees, hold every role.
    assert {p.role for p in prompts[:20]} == set(ROLES)


def test_every_prompt_runs_on_the_fake_engine(engine):
    samples = run_suite(engine, load_prompts())

    assert len(samples) == 101 and not any(s.error for s in samples)
    assert schema_rate(samples) == (1.0, 100)


def test_load_prompts_refuses_a_malformed_set(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"review": "unreviewed", "prompts": [{"id": "x", "role": "npc", "args": {}}]}))
    with pytest.raises(ValueError, match="unknown role"):
        load_prompts(path)

    chat = {"id": "x", "role": "tutor_chat", "args": {"history": [{"role": "user", "text": "Hi"}], "mode": "chat"}}
    path.write_text(json.dumps({"review": "reviewed", "prompts": [chat, chat]}))
    with pytest.raises(ValueError, match="duplicate prompt ids: x"):
        load_prompts(path)

    path.write_text(json.dumps({"review": "reviewed", "prompts": [chat | {"role": "teacher_explain"}]}))
    with pytest.raises(ValueError, match="takes mode 'teacher'"):
        load_prompts(path)

    path.write_text(json.dumps({"review": "reviewed", "prompts": [chat]}))
    assert load_prompts(path)[0].tags == ()


# --- scripts/tutor_eval.py ---


def run_script(data_dir: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "LOCI_DATA_DIR": str(data_dir), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, encoding="utf-8", env=env, timeout=120
    )


def test_the_fake_run_prints_the_three_suites_and_passes(tmp_path):
    lod_store(tmp_path / "data")

    done = run_script(tmp_path / "data", "--fake", "--n", "10")

    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].startswith("tutor-eval  ") and "provider fake" in lines[0] and "N=10 prompts" in lines[0]
    assert lines[1] == "SCHEMA  valid first try 100.0 %  (N=10, gate >= 98 %)  pass"
    assert lines[2].startswith("LATENCY p50 5 ms  p90 5 ms")
    assert lines[3].startswith("LB-GEN  lb segments 100.0 %")
    assert "Good for: chat, explanations" in lines


def test_without_the_lod_store_the_run_is_skipped_with_exit_2(tmp_path):
    done = run_script(tmp_path / "data", "--fake", "--n", "10")

    assert done.returncode == 2
    assert done.stderr.strip() == tutor_eval.LOD_MISSING
    assert not (tmp_path / "data" / "lod").exists()


def test_a_provider_entry_takes_all_four_roles_with_the_model_given(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    lod_store(data_dir)
    providers = {
        "providers": {"cli": {"kind": "fake"}, "local": {"kind": "fake", "model": "small"}},
        "roles": {role: {"provider": "cli"} for role in ROLES},
    }
    (data_dir / "providers.json").write_text(json.dumps(providers), encoding="utf-8")
    monkeypatch.setenv("LOCI_DATA_DIR", str(data_dir))
    out = tmp_path / "out" / "eval.txt"

    code = tutor_eval.main(["--provider", "local", "--model", "big", "--n", "3", "--json", "--out", str(out)])

    printed = capsys.readouterr().out
    assert code == 0
    assert out.read_text(encoding="utf-8") == printed
    data = json.loads(printed[printed.index("{") :])
    assert (data["provider"], data["model"], data["prompts"]) == ("local", "big", 3)


def test_an_unknown_provider_id_exits_2(tmp_path, monkeypatch, capsys):
    lod_store(tmp_path / "data")
    monkeypatch.setenv("LOCI_DATA_DIR", str(tmp_path / "data"))

    assert tutor_eval.main(["--provider", "nope"]) == 2
    # Without a providers.json the default config has the Claude CLI entry only.
    assert capsys.readouterr().err.strip() == "no provider 'nope' in providers.json; known: claude_cli"


def test_one_entry_routes_every_role_and_keeps_the_roles_own_settings():
    entries = {"a": ProviderEntry("a", "fake", model="a1"), "b": ProviderEntry("b", "fake")}
    roles = {role: RoleConfig("b") for role in ROLES} | {"tutor_chat": RoleConfig("a", "a2", "low")}
    config = ProvidersConfig(entries, roles)

    routed = tutor_eval.one_entry(config, "a")
    assert list(routed.providers) == ["a"]
    assert routed.roles["tutor_chat"] == RoleConfig("a", "a2", "low")
    assert routed.roles["judge"] == RoleConfig("a")
    assert {role.model for role in tutor_eval.one_entry(config, "a", "m").roles.values()} == {"m"}

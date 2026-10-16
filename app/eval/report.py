"""The LB-JUDGE runner: each row through the shipped correction pipeline, then the unreviewed report.

A row goes answer_grounding -> llm.judge -> checked_judgement, as a typed answer to one fixed
question; the practice routes' notes (help used, picture, earlier reactions) are not part of
the call. The report counts false corrections on the correct arm (a fix or an english shown at
Verified or Likely), their one-sided exact 95% upper bound, recall on the synthetic errors and
the label mix. Nothing in it is reviewed by a native speaker.
"""

from __future__ import annotations

import datetime
import math
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field

from app.engines.base import Judgement, LLMError
from app.engines.fake import FakeLLM
from app.lod import checks
from app.practice import answer_grounding, checked_judgement

SHOWN = ("fix", "english")  # verdicts that show the learner a correction
GATED = ("verified", "likely")  # the tiers a false correction is counted at
MIX = ("verified", "likely", "unverified", "downgraded", "none")
CONFIDENCE = 0.95
REVIEW = "unreviewed"
REVIEWED_BY = "none"
SENTENCES = "ZLS dictionary examples (LOD, CC0) and single-token perturbations of them"


@dataclass
class Outcome:
    """What the pipeline made of one row of the set."""

    id: str
    label: str  # of the row: "correct" or "error"
    error_type: str = ""
    wrong: str = ""  # of an error row: the token put in
    judged: str = ""  # the model's verdict; "" when the call failed
    verdict: str = ""  # after the checks: what the learner would see
    tier: str = ""
    category: str = ""
    quote: str = ""
    better_lb: str = ""
    hit: bool = False  # an error row whose shown fix quotes `wrong`
    downgraded: bool = False  # the model's fix or english was not shown: the checks made it "good"
    alternative_lb: str = ""
    error: str = ""  # the LLMError kind; the row has no verdict
    seconds: float = 0.0  # of the judge call


@dataclass
class Report:
    correct: dict
    error: dict
    labels: dict  # over both arms: shown corrections per tier and per category
    errors: dict[str, int]  # failed calls per LLMError kind
    seconds: dict[str, float]  # per judge call: mean, p50, p90
    engine: str = ""
    model: str = ""
    date: str = ""
    seed: int | None = None
    set: str = ""
    sentences: str = SENTENCES
    review: str = REVIEW
    reviewed_by: str = REVIEWED_BY


def _plain(text: str) -> str:
    return " ".join(text.split()).casefold()


def quotes(quote: str, word: str) -> bool:
    """Whether `word` is one of the words of `quote`, whatever its case."""
    return bool(word) and word.casefold() in {part.casefold() for part in checks.words(quote)}


def oracle_judgement(answer: str, row: Mapping) -> Judgement | None:
    """The fix an ideal judge gives an error row: the word of `answer` that holds `wrong` (a
    clitic d' included) quoted as written, and `answer` with `gold` put back; None when
    `answer` does not hold `wrong`."""
    wrong, gold = row.get("wrong", ""), row.get("gold", "")
    for match in checks.WORD.finditer(answer):
        found = match.group()
        for part in checks.split_clitic(found):
            if part.casefold() != wrong.casefold():
                continue
            start = match.start() + found.replace("’", "'").find(part)
            put_back = gold[:1].upper() + gold[1:] if part[:1].isupper() else gold
            return Judgement(
                "fix",
                "Nearly there.",
                quote=found,
                better_lb=answer[:start] + put_back + answer[start + len(part) :],
                why_en=f"{part} should be {put_back}, so:",
                claims=[{"form": put_back, "lemma": ""}],
            )
    return None


class OracleJudge(FakeLLM):
    """A fake judge that knows the set: a fix of `wrong` on an error row, "good" on any other
    answer. A queued judgement or exception (FakeLLM.judge_queue) goes first."""

    name = "oracle"

    def __init__(self, rows: Iterable[Mapping], settings=None) -> None:
        super().__init__(settings)
        self.rows = {_plain(row["text"]): row for row in rows}

    def judge(self, question_lb: str, question_en: str, answer: str, grounding: str, extra: str = "") -> Judgement:
        queued = bool(self.judge_queue)
        judgement = super().judge(question_lb, question_en, answer, grounding, extra)
        row = self.rows.get(_plain(answer))
        if queued or row is None or row.get("label") != "error":
            return judgement
        return oracle_judgement(answer, row) or judgement


def judge_rows(
    rows: Sequence[Mapping],
    llm,
    lod,
    encoder,
    rules: str,
    question,
    clock: Callable[[], float] = time.perf_counter,
    progress: Callable[[int, int, Outcome], None] | None = None,
) -> list[Outcome]:
    """Each of `rows` as a typed answer to `question` (.lb, .en) through the pipeline, in order.

    An LLMError lands in the row's `error` and the run goes on; any other exception is raised.
    `progress` is called after each row with (rows done, rows, its outcome).
    """
    outcomes = []
    for done, row in enumerate(rows, 1):
        text = row["text"]
        outcome = Outcome(row["id"], row["label"], row.get("error_type", ""), row.get("wrong", ""))
        started = clock()
        try:
            judgement = llm.judge(question.lb, question.en, text, answer_grounding(lod, rules, text), "")
        except LLMError as error:
            outcome.error = error.kind
            judgement = None
        outcome.seconds = clock() - started
        if judgement is not None:
            correction = checked_judgement(judgement, text, lod, encoder, raw_transcript="")
            outcome.judged, outcome.verdict = judgement.verdict, correction.verdict
            outcome.tier, outcome.category = correction.label, correction.category
            outcome.quote, outcome.better_lb = correction.quote, correction.better_lb
            outcome.alternative_lb = correction.alternative_lb
            outcome.downgraded = judgement.verdict in SHOWN and correction.verdict not in SHOWN
            outcome.hit = correction.verdict == "fix" and quotes(correction.quote, outcome.wrong)
        outcomes.append(outcome)
        if progress is not None:
            progress(done, len(rows), outcome)
    return outcomes


def _log_binomial(k: int, n: int, p: float) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def binomial_cdf(x: int, n: int, p: float) -> float:
    """P(X <= x) for X ~ Binomial(n, p), 0 < p < 1."""
    return min(1.0, sum(math.exp(_log_binomial(k, n, p)) for k in range(x + 1)))


def upper_95(x: int, n: int, confidence: float = CONFIDENCE) -> float:
    """The one-sided exact (Clopper-Pearson) upper bound of a rate seen as `x` of `n`: the p at
    which P(X <= x) = 1 - confidence. 1.0 when x == n or n == 0."""
    if n <= 0 or x >= n:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2
        if binomial_cdf(x, n, middle) > 1 - confidence:
            low = middle
        else:
            high = middle
    return high


def _rate(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _shown(outcome: Outcome) -> bool:
    return outcome.verdict in SHOWN


def _bucket(outcome: Outcome) -> str:
    """Of MIX: the tier of a shown correction, else "downgraded" or "none"."""
    if _shown(outcome):
        return outcome.tier
    return "downgraded" if outcome.downgraded else "none"


def _mix(outcomes: Iterable[Outcome]) -> dict[str, int]:
    counts = Counter(_bucket(outcome) for outcome in outcomes)
    return {name: counts[name] for name in MIX}


def _recall(outcomes: list[Outcome]) -> dict:
    caught = sum(outcome.verdict == "fix" for outcome in outcomes)
    hit = sum(outcome.hit for outcome in outcomes)
    n = len(outcomes)
    return {"n": n, "caught": caught, "hit": hit, "recall": _rate(hit, n), "recall_any": _rate(caught, n)}


def _class(outcomes: list[Outcome], error_type: str) -> dict:
    mine = [outcome for outcome in outcomes if outcome.error_type == error_type]
    return {**_recall(mine), "tiers": _mix(mine)}


def _percentile(values: list[float], share: float) -> float:
    """The nearest-rank percentile of `values`; 0.0 for none."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def summarise(
    outcomes: Sequence[Outcome],
    engine: str = "",
    model: str = "",
    date: str = "",
    seed: int | None = None,
    set_name: str = "",
) -> Report:
    """The counts of `outcomes`. A row whose call failed is in `rows` and `errors` but not in `n`."""
    correct = [outcome for outcome in outcomes if outcome.label == "correct"]
    wrong = [outcome for outcome in outcomes if outcome.label == "error"]
    judged = [outcome for outcome in correct if not outcome.error]
    false = sum(_shown(outcome) and outcome.tier in GATED for outcome in judged)
    correct_arm = {
        "rows": len(correct),
        "n": len(judged),
        "errors": len(correct) - len(judged),
        "false_corrections": false,
        "unverified_corrections": sum(_shown(outcome) and outcome.tier == "unverified" for outcome in judged),
        "downgraded": sum(outcome.downgraded for outcome in judged),
        "fpr": _rate(false, len(judged)),
        "upper_95": upper_95(false, len(judged)),
        "verdicts": dict(Counter(outcome.verdict for outcome in judged)),
        "tiers": _mix(judged),
    }
    judged_errors = [outcome for outcome in wrong if not outcome.error]
    classes = sorted({outcome.error_type for outcome in wrong})
    error_arm = {
        "rows": len(wrong),
        **_recall(judged_errors),
        "errors": len(wrong) - len(judged_errors),
        "by_class": {name: _class(judged_errors, name) for name in classes},
        "verdicts": dict(Counter(outcome.verdict for outcome in judged_errors)),
        "tiers": _mix(judged_errors),
    }
    shown = [outcome for outcome in outcomes if _shown(outcome)]
    seconds = [outcome.seconds for outcome in outcomes]
    return Report(
        correct=correct_arm,
        error=error_arm,
        labels={
            "tiers": _mix(outcome for outcome in outcomes if not outcome.error),
            "categories": dict(Counter(outcome.category for outcome in shown)),
        },
        errors=dict(Counter(outcome.error for outcome in outcomes if outcome.error)),
        seconds={
            "mean": sum(seconds) / len(seconds) if seconds else 0.0,
            "p50": _percentile(seconds, 0.5),
            "p90": _percentile(seconds, 0.9),
        },
        engine=engine,
        model=model,
        date=date or datetime.date.today().isoformat(),
        seed=seed,
        set=set_name,
    )


def to_json(report: Report, outcomes: Sequence[Outcome] = ()) -> dict:
    """`report` as a JSON object, its headline first and every outcome last."""
    return {"headline": headline(report), **asdict(report), "outcomes": [asdict(outcome) for outcome in outcomes]}


def headline(report: Report) -> str:
    """The one line the figure is quoted by: always unreviewed and with its upper bound."""
    arm = report.correct
    x, n = arm["false_corrections"], arm["n"]
    if n:
        figures = f"{x}/{n} = {100 * x / n:.2f}%, 95% upper bound {100 * arm['upper_95']:.2f}%"
    else:
        figures = "no sentence was judged, so no upper bound"
    return (
        f"Unreviewed false-correction rate: {x} of {n} correct sentences received a Verified or Likely"
        f" correction ({figures}). Sentences: ZLS dictionary examples, not learner text, checked by no"
        f" native speaker; not the launch figure. Engine {report.engine}, model {report.model}, {report.date}."
    )


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def format_report(report: Report) -> str:
    """The headline, the counts of both arms, the recall per class and the label mix, as text."""
    correct, error = report.correct, report.error

    def judged(arm: dict) -> str:
        return f"{arm['n']} judged of {arm['rows']} rows, {arm['errors']} failed calls"

    def mix(tiers: dict) -> str:
        return ", ".join(f"{name} {tiers[name]}" for name in MIX)

    lines = [
        headline(report),
        "",
        f"correct arm ({REVIEW}): {judged(correct)}",
        f"  false corrections (Verified or Likely): {correct['false_corrections']}"
        f"  Unverified apart: {correct['unverified_corrections']}  downgraded to good: {correct['downgraded']}",
        f"error arm ({REVIEW}, synthetic errors): {judged(error)}",
        f"  recall (fix quotes the wrong word): {error['hit']} of {error['n']} = {_percent(error['recall'])}"
        f"  any fix: {error['caught']} = {_percent(error['recall_any'])}",
        "recall per class:",
        *(
            f"  {name}: {counts['hit']} of {counts['n']} = {_percent(counts['recall'])}"
            f"  (any fix {counts['caught']}; {mix(counts['tiers'])})"
            for name, counts in error["by_class"].items()
        ),
        "label mix:",
        f"  correct: {mix(correct['tiers'])}",
        f"  error: {mix(error['tiers'])}",
    ]
    if report.errors:
        lines.append("failed calls: " + ", ".join(f"{kind} {count}" for kind, count in sorted(report.errors.items())))
    seconds = report.seconds
    lines.append(f"seconds per call: mean {seconds['mean']:.2f}, p50 {seconds['p50']:.2f}, p90 {seconds['p90']:.2f}")
    lines.append(f"reviewed_by: {report.reviewed_by}")
    return "\n".join(lines)

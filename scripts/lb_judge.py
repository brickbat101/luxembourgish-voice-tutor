"""Build the LB-JUDGE set of correct and single-error Luxembourgish sentences (every row unreviewed), and run it.

    python scripts/lb_judge.py build
    python scripts/lb_judge.py build --n 500 --seed 7 --out data/eval/lb_judge.jsonl
    python scripts/lb_judge.py run --fake
    python scripts/lb_judge.py run --live --correct 150 --errors 50 --seed 7

`build` reads the LOD article dump and the LOD store (both from scripts/download_lod.py) and
writes --n correct rows (LOD example sentences) and --n error rows (one LOD form swapped:
verb person, auxiliary, adjective ending, noun gender). The same seed gives the same rows.
Paths default to <data_dir>; a set short of --n rows in either arm is not written.

`run` sends --correct and --errors rows of the set (drawn with --seed; default all) through
the practice correction pipeline and writes the unreviewed report to --report. --fake judges
with an oracle that knows every error, so it must show no false correction (exit 1 if it
does); --live judges with the Claude CLI and needs --yes above LIVE_ROWS rows. Exit code 2
when the run could not start.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.content import Question, load_practice  # noqa: E402
from app.engines.factory import build_engine  # noqa: E402
from app.engines.providers import PROVIDERS_FILE, load_providers  # noqa: E402
from app.eval.lb_judge import CLASSES, LodFacts, build_set, read_set, write_set  # noqa: E402
from app.eval.report import OracleJudge, format_report, judge_rows, summarise, to_json  # noqa: E402
from app.lod.build import ART_XML  # noqa: E402
from app.lod.store import LodStore  # noqa: E402
from app.orchestrator import RULES_PATH  # noqa: E402

DEFAULT_N = 500
DEFAULT_SEED = 7
LIVE_ROWS = 200  # the gate's budget of live judge calls; more needs --yes
PROGRESS_EVERY = 10
LIVE_ENGINE = "claude_cli"
EVAL_QUESTION = "lb-judge"  # the id of the one question every row answers


def missing(path: Path) -> SystemExit:
    return SystemExit(f"{path} is missing; run scripts/download_lod.py")


def build(args: argparse.Namespace) -> int:
    if not args.dump.is_file():
        raise missing(args.dump)
    if not args.store.is_file():
        raise missing(args.store)
    facts = LodFacts(args.store)
    correct, errors = build_set(args.dump, facts, args.n, args.seed)
    per_class = Counter(row["error_type"] for row in errors)
    print(f"correct {len(correct)} (lod_example)")
    print(f"error {len(errors)}: " + ", ".join(f"{name} {per_class[name]}" for name in CLASSES))
    if len(correct) < args.n or len(errors) < args.n:
        print(f"not written: fewer than {args.n} rows in an arm", file=sys.stderr)
        return 1
    write_set([*correct, *errors], args.out, args.seed, args.dump.name)
    print(f"wrote {args.out} (review: unreviewed)")
    return 0


def drawn(rows: list[dict], label: str, n: int | None, rng: random.Random) -> list[dict]:
    """`n` rows of `label` drawn with `rng` (all when n is None), in the order of the set."""
    pool = [row for row in rows if row["label"] == label]
    if n is None or n >= len(pool):
        return pool
    keep = set(rng.sample(range(len(pool)), n))
    return [row for index, row in enumerate(pool) if index in keep]


def live_engine(settings: Settings) -> tuple[str, str, str]:
    """(engine, model, problem) of the judge `--live` would use; `problem` is "" when it runs on the Claude CLI."""
    if settings.llm_engine == LIVE_ENGINE:
        return LIVE_ENGINE, settings.llm_model, ""
    wanted = f"--live runs the judge on {LIVE_ENGINE} only: set LOCI_LLM_ENGINE={LIVE_ENGINE}"
    if settings.llm_engine != "provider":
        return settings.llm_engine, "", f"{wanted} (it is {settings.llm_engine!r})"
    try:
        config = load_providers(settings.data_dir / PROVIDERS_FILE, settings)
    except ValueError as exc:
        return "provider", "", f"{PROVIDERS_FILE}: {exc}"
    route = config.roles["judge"]
    entry = config.providers[route.provider]
    if entry.kind != LIVE_ENGINE:
        return "provider", "", f"{wanted}, or route the judge role of {PROVIDERS_FILE} to a {LIVE_ENGINE} entry"
    return f"provider ({entry.id})", route.model or entry.model or settings.llm_model, ""


def run(args: argparse.Namespace, settings: Settings) -> int:
    if not args.set.is_file():
        raise SystemExit(f"{args.set} is missing; run scripts/lb_judge.py build")
    if not args.store.is_file():
        raise missing(args.store)
    rows = read_set(args.set)
    rng = random.Random(args.seed)
    picked = drawn(rows, "correct", args.correct, rng) + drawn(rows, "error", args.errors, rng)

    if args.live:
        engine, model, problem = live_engine(settings)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        if len(picked) > LIVE_ROWS and not args.yes:
            print(f"{len(picked)} rows is more than {LIVE_ROWS} live judge calls; add --yes", file=sys.stderr)
            return 2
        llm = build_engine("llm", settings)
        if not llm.ready():
            print(f"the judge is not ready: {getattr(llm, 'detail', '')}", file=sys.stderr)
            return 2
    else:
        engine, model, llm = "fake", OracleJudge.name, OracleJudge(picked)

    tutor = load_practice().tutor
    question = Question(EVAL_QUESTION, "general", tutor["eval_lb"], tutor["eval_en"])
    rules = RULES_PATH.read_text(encoding="utf-8").strip()
    encoder = build_engine("acceptability", settings)

    def progress(done: int, total: int, outcome) -> None:
        if done % PROGRESS_EVERY == 0 or done == total:
            print(f"judged {done} of {total}", flush=True)

    lod = LodStore(args.store)
    try:
        outcomes = judge_rows(picked, llm, lod, encoder, rules, question, progress=progress)
    finally:
        lod.close()

    report = summarise(outcomes, engine=engine, model=model, seed=args.seed, set_name=args.set.name)
    data = to_json(report, outcomes)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = format_report(report)
    if args.json:
        counts = {key: value for key, value in data.items() if key != "outcomes"}
        text += "\n" + json.dumps(counts, ensure_ascii=False, indent=2)
    print(text)
    print(f"wrote {args.report}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 1 if args.fake and report.correct["false_corrections"] else 0


def count(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return number


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    data = settings.data_dir
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="write the set from the LOD dump and store")
    build_parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"rows per arm (default {DEFAULT_N})")
    build_parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"draw seed (default {DEFAULT_SEED})")
    build_parser.add_argument("--out", type=Path, default=data / "eval" / "lb_judge.jsonl")
    build_parser.add_argument("--dump", type=Path, default=data / "lod" / "raw" / ART_XML)
    build_parser.add_argument("--store", type=Path, default=data / "lod" / "lod.sqlite")
    run_parser = commands.add_parser("run", help="judge the set and write the unreviewed report")
    judge = run_parser.add_mutually_exclusive_group(required=True)
    judge.add_argument("--fake", action="store_true", help="judge with the oracle (no calls out): the CI gate")
    judge.add_argument("--live", action="store_true", help=f"judge with the Claude CLI ({LIVE_ENGINE})")
    run_parser.add_argument("--correct", type=count, help="correct rows to draw (default: all)")
    run_parser.add_argument("--errors", type=count, help="error rows to draw (default: all)")
    run_parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"draw seed (default {DEFAULT_SEED})")
    run_parser.add_argument("--set", type=Path, default=data / "eval" / "lb_judge.jsonl")
    run_parser.add_argument("--report", type=Path, default=data / "eval" / "lb_judge_report.json")
    run_parser.add_argument("--store", type=Path, default=data / "lod" / "lod.sqlite")
    run_parser.add_argument("--json", action="store_true", help="print the report as JSON after the text")
    run_parser.add_argument("--out", type=Path, help="also write what is printed to this file")
    run_parser.add_argument("--yes", action="store_true", help=f"allow --live on more than {LIVE_ROWS} rows")
    args = parser.parse_args(argv)
    if args.command == "build" and args.n < 1:
        parser.error("--n must be at least 1")
    # The console code page may lack a character of a path.
    sys.stdout.reconfigure(errors="backslashreplace")
    if args.command == "run":
        return run(args, settings)
    return build(args)


if __name__ == "__main__":
    sys.exit(main())

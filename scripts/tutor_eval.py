"""Run the tutor-eval suites (SCHEMA, LATENCY, LB-GEN) on one provider and print the verdict.

    python scripts/tutor_eval.py --fake
    python scripts/tutor_eval.py --provider claude_cli --n 20
    python scripts/tutor_eval.py --provider ollama --model gemma3:12b --schema-off --json --out eval.json

--provider takes an entry of <data_dir>/providers.json (claude_cli works without the file) and
sends all four roles to it; every prompt is one call there, plus one warm-up call. LB-GEN
needs the LOD store <data_dir>/lod/lod.sqlite. Exit code 0 when the verdict is good for
chat, 1 when it is not, 2 when the run could not start.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.engines.llm_provider import ProviderLLM  # noqa: E402
from app.engines.providers import (  # noqa: E402
    PROVIDERS_FILE,
    ROLES,
    FakeProvider,
    ProviderEntry,
    ProvidersConfig,
    RoleConfig,
    load_providers,
)
from app.evaluation import (  # noqa: E402
    CHAT,
    PROMPTS_PATH,
    default_lang_id,
    evaluate,
    format_report,
    load_prompts,
    openai_compat_ids,
    run_suite,
    to_json,
)
from app.lod.store import LodStore  # noqa: E402

LOD_MISSING = "LB-GEN skipped: LOD store missing, run scripts/download_lod.py"
FAKE_LATENCY_MS = 5


def one_entry(config: ProvidersConfig, provider_id: str, model: str = "") -> ProvidersConfig:
    """`config` with all four roles on the entry `provider_id`; `model` overrides each role's model.

    A role that was on that entry keeps its model and effort.
    """
    entry = config.providers[provider_id]
    roles = {}
    for role in ROLES:
        own = config.roles[role]
        kept = own if own.provider == provider_id else RoleConfig(provider_id)
        roles[role] = replace(kept, model=model or kept.model)
    return ProvidersConfig({provider_id: entry}, roles)


def fake_engine(settings: Settings) -> ProviderLLM:
    entry = ProviderEntry("fake", "fake")
    config = ProvidersConfig({entry.id: entry}, {role: RoleConfig(entry.id) for role in ROLES})
    return ProviderLLM(settings, providers={entry.id: FakeProvider(latency_ms=FAKE_LATENCY_MS)}, config=config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fake", action="store_true", help="run on the scripted fake provider (no calls out)")
    source.add_argument("--provider", metavar="ID", help=f"the entry of {PROVIDERS_FILE} to run all roles on")
    parser.add_argument("--model", default="", help="the model for every role (default: the entry's or role's)")
    parser.add_argument("--n", type=int, help="run only the first N prompts (default: all)")
    parser.add_argument(
        "--schema-off", action="store_true", help="openai_compat entries put the schema in the prompt instead"
    )
    parser.add_argument("--json", action="store_true", help="print the results as JSON after the report")
    parser.add_argument("--out", type=Path, help="also write what is printed to this file")
    args = parser.parse_args(argv)
    if args.n is not None and args.n < 1:
        parser.error("--n must be at least 1")
    # The console code page may lack a character of a reply.
    sys.stdout.reconfigure(errors="backslashreplace")

    settings = Settings.from_env()
    lod_path = settings.data_dir / "lod" / "lod.sqlite"
    try:
        lod = LodStore(lod_path)
    except FileNotFoundError:
        print(LOD_MISSING, file=sys.stderr)
        return 2

    try:
        if args.fake:
            engine = fake_engine(settings)
        else:
            try:
                config = load_providers(settings.data_dir / PROVIDERS_FILE, settings)
            except ValueError as exc:
                print(f"{PROVIDERS_FILE}: {exc}", file=sys.stderr)
                return 2
            if args.provider not in config.providers:
                known = ", ".join(config.providers)
                print(f"no provider {args.provider!r} in {PROVIDERS_FILE}; known: {known}", file=sys.stderr)
                return 2
            try:
                engine = ProviderLLM(settings, config=one_entry(config, args.provider, args.model))
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
        engine.warmup()
        if not engine.ready():
            print(f"the provider is not ready: {engine.detail}", file=sys.stderr)
            return 2

        route = engine.config.roles["tutor_chat"]
        entry = engine.config.providers[route.provider]
        model = route.model or entry.model or (settings.llm_model if entry.kind == "claude_cli" else entry.kind)
        prompts = load_prompts(PROMPTS_PATH)[: args.n]
        lang_id = default_lang_id()
        samples = run_suite(engine, prompts, schema_off=args.schema_off)
        results = evaluate(samples, lod, lang_id, openai_compat_ids(engine) if args.schema_off else None)
    finally:
        lod.close()

    meta = {"provider_id": entry.id, "model": model, "n": len(prompts), "date": datetime.date.today().isoformat()}
    text = format_report(results, **meta)
    if args.json:
        text += "\n" + json.dumps(to_json(results, **meta), ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if CHAT in results.verdict.good_for else 1


if __name__ == "__main__":
    sys.exit(main())

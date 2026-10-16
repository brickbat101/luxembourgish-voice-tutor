"""LLM engine: one headless `claude -p` process per turn, tools off, schema-constrained JSON reply.

The CLI is spawned directly (list argv, no shell) and the per-turn prompt goes in on stdin,
so nothing the user said is parsed by cmd.exe or appears on a command line.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

from dotenv import dotenv_values

from app.config import ROOT
from app.engines.base import VERDICTS, Judgement, LLMError, Segment, TutorReply, Turn, WrapUpText

MAX_TURNS = 30
AUTH_STATUS_TIMEOUT_S = 20
RETRY_KINDS = ("other", "schema")
# The effort setting that, like an empty one, passes no --effort flag.
NO_EFFORT = "default"
# Never passed to `claude -p`: they point it at another endpoint or bill a key instead of the login.
DROPPED_VARIABLES = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
NOT_LOGGED_IN = "the claude CLI is not logged in"
FIX = "run `claude auth login` once in a terminal"
SYSTEM_PROMPT_PATH = ROOT / "app" / "prompts" / "tutor.md"
JUDGE_PROMPT_PATH = ROOT / "app" / "prompts" / "practice.md"
WRAP_UP_PROMPT_PATH = ROOT / "app" / "prompts" / "wrapup.md"
# Windows runs these through cmd.exe, which would interpret the arguments.
SHELL_SUFFIXES = (".cmd", ".bat")
# The program the npm shim `claude.cmd` starts, relative to the shim's folder.
NPM_EXE = Path("node_modules", "@anthropic-ai", "claude-code", "bin", "claude.exe")


def _objects(**properties: dict) -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": properties,
        },
    }


_TEXT = {"type": "string"}

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["segments", "new_phrases", "mode", "claims"],
    "properties": {
        "segments": {
            **_objects(lang={"type": "string", "enum": ["lb", "en"]}, text=_TEXT),
            "minItems": 1,
        },
        "new_phrases": _objects(lb=_TEXT, en=_TEXT),
        "mode": {"type": "string", "enum": ["chat", "teacher"]},
        "claims": _objects(form=_TEXT, lemma=_TEXT),
    },
}

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "reaction_en", "quote", "better_lb", "why_en", "claims"],
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "reaction_en": _TEXT,
        "quote": _TEXT,
        "better_lb": _TEXT,
        "why_en": _TEXT,
        "claims": _objects(form=_TEXT, lemma=_TEXT),
    },
}

WRAP_UP_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["words_en", "grammar_en", "task_en"],
    "properties": {"words_en": _TEXT, "grammar_en": _TEXT, "task_en": _TEXT},
}


def build_prompt(history: list[Turn], mode: str, grounding: str) -> str:
    """The stdin text of one call: the last MAX_TURNS turns, the mode and the grounding block."""
    lines = ["Conversation so far:"]
    for turn in history[-MAX_TURNS:]:
        speaker = "User" if turn.role == "user" else "Tutor"
        # One line per turn, so text inside a turn cannot pose as another line of the prompt.
        lines.append(f"{speaker}: {' '.join(turn.text.split())}")
    lines += ["", f"mode: {mode}"]
    if grounding.strip():
        lines += ["", "Material from the app:", grounding.strip()]
    lines += ["", "Reply to the last User line with the JSON object."]
    return "\n".join(lines)


def build_judge_prompt(question_lb: str, question_en: str, answer: str, grounding: str, extra: str = "") -> str:
    """The stdin text of one judge call: the question, the answer, `extra` and the grounding block."""
    # One line each, so text inside an answer cannot pose as another line of the prompt.
    lines = [
        f"The tutor asked: {' '.join(question_lb.split())}",
        f"In English: {' '.join(question_en.split())}",
        f"The learner answered: {' '.join(answer.split())}",
    ]
    if extra.strip():
        lines.append(extra.strip())
    if grounding.strip():
        lines += ["", "Material from the app:", grounding.strip()]
    lines += ["", "Give the JSON object for this answer."]
    return "\n".join(lines)


def build_wrap_up_prompt(facts: str) -> str:
    """The stdin text of one wrap-up call: `facts`, what the app knows of the talk."""
    return "\n".join(["What the app knows of the chat:", facts.strip(), "", "Give the JSON object for this chat."])


def child_env(environ: Mapping[str, str] | None = None, drop: Iterable[str] = ()) -> dict[str, str]:
    """The parent environment without the variables of a surrounding Claude Code session or API keys.

    A key in its environment would move `claude -p` off the user's own login. `drop` names more
    variables to leave out (the `key_env` names of providers.json); names compare upper-cased.
    """
    dropped = {name.upper() for name in drop}
    env = {}
    for key, value in (os.environ if environ is None else environ).items():
        name = key.upper()
        if name != "CLAUDE_CODE_OAUTH_TOKEN" and (
            name.startswith(("CLAUDECODE", "CLAUDE_CODE_"))
            or name in DROPPED_VARIABLES
            or name.endswith("_API_KEY")
            or name in dropped
        ):
            continue
        env[key] = value
    return env


def find_claude() -> tuple[str | None, str]:
    """(program to spawn, "") or (None, why not). Never a .cmd/.bat shim or cmd.exe."""
    override = {**dotenv_values(ROOT / ".env"), **os.environ}.get("LOCI_CLAUDE_EXE")
    if override:
        path = Path(override)
        if path.suffix.lower() in SHELL_SUFFIXES or path.stem.lower() == "cmd":
            return None, f"LOCI_CLAUDE_EXE must name claude.exe, not a shell wrapper: {override}"
        return override, ""
    found = shutil.which("claude.exe" if os.name == "nt" else "claude")
    if found:
        return found, ""
    shim = shutil.which("claude.cmd")
    if shim and (Path(shim).parent / NPM_EXE).is_file():
        return str(Path(shim).parent / NPM_EXE), ""
    return None, "the claude CLI was not found; install Claude Code or set LOCI_CLAUDE_EXE"


def _rows(data: dict, key: str, *fields: str) -> list[dict]:
    """`data[key]` as a list of objects with exactly the text `fields`; ValueError if it is not."""
    found = data[key]
    if not isinstance(found, list):
        raise ValueError(f"{key} is not a list")
    for row in found:
        if not isinstance(row, dict) or set(row) != set(fields):
            raise ValueError(f"a {key} item does not have exactly the keys {fields}")
        if not all(isinstance(value, str) for value in row.values()):
            raise ValueError(f"a {key} item has a value that is not text")
    return found


def parse_reply(data: object) -> TutorReply:
    """Check the model's object against JSON_SCHEMA; LLMError("schema") if it does not fit."""

    def rows(key: str, *fields: str) -> list[dict]:
        return _rows(data, key, *fields)

    try:
        if not isinstance(data, dict) or set(data) != set(JSON_SCHEMA["required"]):
            raise ValueError(f"expected an object with the keys {JSON_SCHEMA['required']}")
        segments = rows("segments", "lang", "text")
        if not segments or any(row["lang"] not in ("lb", "en") for row in segments):
            raise ValueError("segments is empty or has a lang other than lb, en")
        if data["mode"] not in ("chat", "teacher"):
            raise ValueError(f"mode is {data['mode']!r}")
        return TutorReply(
            segments=[Segment(row["lang"], row["text"]) for row in segments],
            new_phrases=rows("new_phrases", "lb", "en"),
            mode=data["mode"],
            claims=rows("claims", "form", "lemma"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMError("schema", f"the reply does not match the schema: {exc}") from exc


def parse_judgement(data: object) -> Judgement:
    """Check the model's object against JUDGE_SCHEMA; LLMError("schema") if it does not fit."""
    texts = ("reaction_en", "quote", "better_lb", "why_en")
    try:
        if not isinstance(data, dict) or set(data) != set(JUDGE_SCHEMA["required"]):
            raise ValueError(f"expected an object with the keys {JUDGE_SCHEMA['required']}")
        if data["verdict"] not in VERDICTS:
            raise ValueError(f"verdict is {data['verdict']!r}")
        if not all(isinstance(data[key], str) for key in texts):
            raise ValueError(f"one of {texts} is not text")
        return Judgement(
            verdict=data["verdict"],
            **{key: data[key] for key in texts},
            claims=_rows(data, "claims", "form", "lemma"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMError("schema", f"the judgement does not match the schema: {exc}") from exc


def parse_wrap_up(data: object) -> WrapUpText:
    """Check the model's object against WRAP_UP_SCHEMA; LLMError("schema") if it does not fit."""
    texts = WRAP_UP_SCHEMA["required"]
    try:
        if not isinstance(data, dict) or set(data) != set(texts):
            raise ValueError(f"expected an object with the keys {texts}")
        if not all(isinstance(data[key], str) for key in texts):
            raise ValueError(f"one of {texts} is not text")
        return WrapUpText(**{key: data[key] for key in texts})
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMError("schema", f"the wrap-up does not match the schema: {exc}") from exc


class ClaudeCliLLM:
    name = "claude_cli"
    # More variables `child_env` leaves out of the CLI's environment.
    env_drop: tuple[str, ...] = ()

    def __init__(self, settings, runner=subprocess.run) -> None:
        self.settings = settings
        self.runner = runner
        self.exe, self.detail = find_claude()
        self.logged_in = False
        self.system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        self.schema = json.dumps(JSON_SCHEMA, separators=(",", ":"))
        self.judge_prompt = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
        self.judge_schema = json.dumps(JUDGE_SCHEMA, separators=(",", ":"))
        self.wrap_up_prompt = WRAP_UP_PROMPT_PATH.read_text(encoding="utf-8")
        self.wrap_up_schema = json.dumps(WRAP_UP_SCHEMA, separators=(",", ":"))

    def ready(self) -> bool:
        return self.exe is not None and self.logged_in

    def warmup(self) -> None:
        """Ask `claude auth status` once; a missing CLI or login goes into `detail`, never raises."""
        if self.exe is None:
            return
        try:
            done = self.runner(
                [self.exe, "auth", "status"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=AUTH_STATUS_TIMEOUT_S,
                env=child_env(drop=self.env_drop),
            )
            self._set_logged_in(json.loads(done.stdout).get("loggedIn") is True)
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError) as exc:
            self._set_logged_in(False, f"`claude auth status` failed ({type(exc).__name__}: {exc})")

    def reply(self, history: list[Turn], mode: str, grounding: str) -> TutorReply:
        return self._call_twice(build_prompt(history, mode, grounding), mode)

    def judge(self, question_lb: str, question_en: str, answer: str, grounding: str, extra: str = "") -> Judgement:
        """What the tutor makes of one practice answer, at the effort of teacher mode."""
        prompt = build_judge_prompt(question_lb, question_en, answer, grounding, extra)
        return self._call_twice(prompt, "teacher", self.judge_prompt, self.judge_schema, parse_judgement)

    def wrap_up(self, facts: str) -> WrapUpText:
        """What the tutor says looking back on a practice talk, at the effort of teacher mode."""
        prompt = build_wrap_up_prompt(facts)
        return self._call_twice(prompt, "teacher", self.wrap_up_prompt, self.wrap_up_schema, parse_wrap_up)

    def _call_twice(self, prompt: str, mode: str, *task):
        """`_call`, and once more after a failure of one of RETRY_KINDS."""
        try:
            return self._call(prompt, mode, *task)
        except LLMError as error:
            if error.kind not in RETRY_KINDS:
                raise
        return self._call(prompt, mode, *task)

    def _set_logged_in(self, logged_in: bool, reason: str = NOT_LOGGED_IN) -> None:
        self.logged_in = logged_in
        self.detail = "" if logged_in else f"{reason} - {FIX}"

    def _argv(self, mode: str, system_prompt: str | None = None, schema: str | None = None) -> list[str]:
        """The command line of one call; without `system_prompt` and `schema`, those of a tutor reply."""
        settings = self.settings
        effort = settings.llm_effort_teacher if mode == "teacher" else settings.llm_effort_chat
        if effort.lower() == NO_EFFORT:
            effort = ""
        return [
            self.exe,
            "-p",
            "--output-format", "json",
            "--json-schema", self.schema if schema is None else schema,
            "--system-prompt", self.system_prompt if system_prompt is None else system_prompt,
            "--tools", "",
            "--safe-mode",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--permission-prompts", "none",
            "--disable-slash-commands",
            "--model", settings.llm_model,
            *(["--effort", effort] if effort else []),
        ]  # fmt: skip

    def _call(
        self, prompt: str, mode: str, system_prompt: str | None = None, schema: str | None = None, parse=parse_reply
    ):
        """One run of the CLI; what `parse` makes of the object it returns."""
        if self.exe is None:
            raise LLMError("other", self.detail)
        timeout_s = self.settings.llm_timeout_s
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            done = self.runner(
                self._argv(mode, system_prompt, schema),
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                cwd=self.settings.data_dir,
                env=child_env(drop=self.env_drop),
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMError("timeout", f"claude gave no reply within {timeout_s:g} s") from exc
        except OSError as exc:
            raise LLMError("other", f"could not start {self.exe}: {exc}") from exc

        try:
            envelope = json.loads(done.stdout)
        except (TypeError, ValueError):
            envelope = None
        if not isinstance(envelope, dict) or not {"result", "structured_output"} & set(envelope):
            if done.returncode:
                tail = (done.stderr or done.stdout or "").strip()[-500:]
                raise LLMError("other", f"claude exited with code {done.returncode}: {tail}")
            raise LLMError("schema", f"claude printed no JSON envelope: {str(done.stdout)[:200]!r}")

        result = envelope.get("result")
        if envelope.get("is_error"):
            if "authenticate" in str(result).lower():
                self._set_logged_in(False, str(result))
                raise LLMError("auth", f"{result} - {FIX}")
            raise LLMError("other", str(result or f"claude reported {envelope.get('subtype')}"))

        data = envelope.get("structured_output")
        if data is None:
            try:
                data = json.loads(result)
            except (TypeError, ValueError) as exc:
                raise LLMError("schema", f"the reply is not JSON: {str(result)[:200]!r}") from exc
        parsed = parse(data)
        self._set_logged_in(True)
        return parsed

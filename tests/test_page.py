"""Static checks of the page: static/index.html, static/style.css and static/app.js."""

import json
import re
import shutil
import subprocess

import pytest

from app.config import ROOT

STATIC = ROOT / "static"
NOTICE = (
    "You are talking to an AI tutor, not a person. The voices are computer-generated, "
    "and the Luxembourgish can contain mistakes."
)


@pytest.fixture(scope="module")
def page():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script():
    return (STATIC / "app.js").read_text(encoding="utf-8")


def test_the_page_says_at_the_top_that_the_tutor_is_an_ai(page):
    header = re.search(r"<header>(.*?)</header>", page, re.DOTALL).group(1)

    assert header.index("<h1") < header.index('id="ai-notice"')
    assert re.search(r'<p id="ai-notice"[^>]*>\s*' + re.escape(NOTICE) + r"\s*</p>", header)
    assert "hidden" not in re.search(r'<p id="ai-notice"[^>]*>', header).group(0)


def test_the_log_calls_the_tutor_an_ai_tutor(script):
    assert not re.search(r'"Tutor \(teacher\)"|tutorLine\("Tutor"|"who", "Tutor"', script)
    english = json.loads((STATIC / "i18n" / "en.json").read_text(encoding="utf-8"))
    assert english["log.tutor"] == "AI tutor" and english["log.tutor_teacher"] == "AI tutor (teacher)"
    assert script.count('t("log.tutor")') >= 2 and '"log.tutor_teacher" : "log.tutor"' in script


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_script_has_no_syntax_error():
    done = subprocess.run(["node", "--check", str(STATIC / "app.js")], capture_output=True, text=True, timeout=60)

    assert done.returncode == 0, done.stderr


@pytest.fixture(scope="module")
def style():
    return (STATIC / "style.css").read_text(encoding="utf-8")


def luminance(colour):
    channels = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(first, second):
    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def token(style, name):
    return re.search(rf"--{name}:\s*(#[0-9a-fA-F]{{6}});", style).group(1)


def test_the_page_has_one_main_a_talk_switch_and_distinct_labels(page):
    assert page.count("<main>") == 1 and page.count("</main>") == 1
    assert page.index("<main>") < page.index('class="layout"') < page.index("</main>")
    assert page.index('id="banner"') < page.index("<main>")
    assert '<input id="talk-toggle" type="checkbox" role="switch">' in page
    assert page.count('aria-label="Your message"') == 1
    assert 'aria-label="Message controls"' in page
    assert re.search(r'<p id="practice-english"[^>]*\blang="en"', page)


def test_the_borders_of_controls_have_three_to_one_against_the_page_and_the_panels(style):
    edge = token(style, "edge")

    assert contrast(edge, token(style, "page")) >= 3.0
    assert contrast(edge, token(style, "soft")) >= 3.0
    for selector in ("button", "textarea", r"\.voice fieldset"):
        assert re.search(r"^" + selector + r" \{[^}]*border: 1px solid var\(--edge\)", style, re.MULTILINE), selector


def test_no_state_is_shown_by_colour_alone(style):
    assert re.search(r"\.segment\.switched \{[^}]*border-bottom: 2px solid currentColor", style)
    assert re.search(r"\.segment\.speaking \{[^}]*outline: 2px solid var\(--ink\)", style)
    assert re.search(r'\[aria-pressed="true"\]::before \{[^}]*content: "\\2713 "', style)


def test_the_script_leaves_space_to_focused_controls_and_marks_the_language_of_its_parts(script):
    assert "function isInteractive(target)" in script
    assert script.count("isInteractive(event.target)") == 2
    assert 'event.key !== "Enter" && event.key !== " "' in script
    assert "node.lang = segment.lang;" in script
    assert 't("status.error_prefix") + text' in script
    # The seconds of a wait go into a hidden span once a second, not into the status.
    assert 'count.setAttribute("aria-hidden", "true")' in script and "setInterval(show, 1000)" in script


def function_source(script, name):
    start = script.index(f"function {name}(")
    return script[start : script.index("\n}\n", start) + 2]


def test_the_flag_button_is_built_in_the_tutor_line_and_never_inside_a_segment(script):
    tutor = function_source(script, "tutorLine")
    segment = function_source(script, "segmentNode")
    control = function_source(script, "appendFlagControl")

    assert "appendFlagControl(tutor, segments);" in tutor
    assert "appendSegments(tutor," in tutor and tutor.index("appendSegments(") < tutor.index("appendFlagControl(")
    assert "flag" not in segment.lower() and "wordsOf" not in segment
    assert 'element("button", "flag", t("flag.button"))' in control
    assert 'toggle.setAttribute("aria-expanded", "false");' in control
    assert 'toggle.setAttribute("aria-expanded", String(!row.hidden));' in control
    assert 'row.classList.add("flag-words");' in control and "line.append(" in control
    word = function_source(script, "wordButton")
    assert 'setAttribute("aria-pressed"' in word and "button.lang = lang;" in word
    # The look back puts it under its lines as well.
    assert script.count("appendFlagControl(") == 4


def test_the_words_of_a_line_are_its_runs_of_two_or_more_letters(script):
    assert "text.match(/\\p{L}{2,}/gu) || []" in function_source(script, "wordsOf")


LINES = [
    ("lb", "D'Kanner sinn ëmmer 3 Stonnen do, Lëtzebuergesch!"),
    ("lb", "Moien, wéi geet et dir haut? Ech heesche Stefan an ech schaffen zu Ettelbréck."),
    ("lb", "Dat kascht 5,50 € - an et ass d'Zäit fir e Kaffi."),
    ("en", "Hello, how are you today? I'm fine, thanks."),
    ("en", "The café in Ettelbruck - it's a naïve idea: 24 hours of Straße and Ärger!"),
    ("en", "a b c x2y 3d under_score"),
    ("en", ""),
]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_script_splits_a_line_into_the_runs_of_letters_the_engine_matches(script):
    from app.engines.tts_piper import LETTER

    source = function_source(script, "wordsOf")
    texts = [text for _, text in LINES]
    program = f"{source}\nconsole.log(JSON.stringify({json.dumps(texts)}.map(wordsOf)));"
    done = subprocess.run(["node", "-e", program], capture_output=True, text=True, encoding="utf-8", timeout=60)

    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [re.findall(f"{LETTER}{{2,}}", text) for text in texts]


def test_the_page_has_a_pronunciation_block_beside_data_progress_and_voice(page, script):
    block = re.search(r'<details id="pronunciation"[^>]*>(.*?)</details>', page, re.DOTALL).group(1)

    assert page.index('id="progress"') < page.index('id="pronunciation"') < page.index('id="voice"')
    assert '<summary data-i18n="pronunciation.summary">' in block
    assert '<ul id="pronunciation-flags">' in block
    assert 'pronunciationBlock.addEventListener("toggle"' in script
    assert "if (lastFlags) showFlags(lastFlags);" in function_source(script, "refreshLabels")
    row = function_source(script, "flagRow")
    assert "sounds.translate = false;" in row and "box.maxLength = MAX_RESPELLING_LENGTH;" in row
    assert 'const MAX_RESPELLING_LENGTH = 60;' in script
    play = function_source(script, "playFlag")
    assert "spokenBlob(flag.lang, flag.text," in play and "{ [flag.word]: respelling }" in play
    assert "await playOnSample(blob);" in play


def test_the_script_calls_the_server_only_through_one_fetch(script):
    assert script.count("fetch(") == 1
    assert script.count("isInteractive(event.target)") == 2

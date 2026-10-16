"""The listening part: the file of B1 recordings and their questions, its loader, the rendering at
two speeds and the check of the answers, the /api/listening routes, the page and the README."""

import base64
import dataclasses
import inspect
import io
import json
import re
import wave
from collections import Counter

import pytest
from fastapi.testclient import TestClient

from app import listening_api
from app.config import ROOT, Settings
from app.content.listening import (
    GENRES,
    ITEMS_PER_GENRE,
    LEVEL,
    LISTENING_PATH,
    OPTIONS_PER_QUESTION,
    QUESTIONS_PER_ITEM,
    ListeningItem,
    Speaker,
    Turn,
    load_listening,
)
from app.engines.base import VOICE_RANGES, WavAudio
from app.engines.fake import SAMPLE_RATE, FakeTTS
from app.listening import SPEED_LABELS, SPEEDS, TURN_GAP_S, check_answers, render_item
from app.main import create_app
from tests.conftest import plain_page
from tests.test_practice import in_lod, lb_words, real, real_path  # noqa: F401 (real and real_path are fixtures)

ITEMS = load_listening()
ITEM_IDS = ["strooss-bus", "wieder-fest", "gemeng-adress", "kaddo", "doheem-schaffen", "schlof"]
# Per genre: the questions of an item, its speakers, its words and its turns.
SHAPE = {
    "news": {"questions": 5, "speakers": 1, "words": (60, 110), "turns": (3, 5)},
    "conversation": {"questions": 4, "speakers": 2, "words": (120, 180), "turns": (4, 12)},
    "discussion": {"questions": 7, "speakers": 2, "words": (180, 260), "turns": (6, 14)},
}
# What a speaker may be called: a role, or one of the two first names LOD lists.
SPEAKER_NAMES = {
    "Spriecher",
    "Spriecherin",
    "Journalist",
    "Journalistin",
    "Reporter",
    "Reporterin",
    "Noper",
    "Verkeefer",
    "Verkeeferin",
    "Dokter",
    "Doktesch",
    "Client",
    "Cliente",
    "Beamten",
    "Beamtin",
    "Sekretärin",
    "Frënd",
    "Frëndin",
    "Mataarbechter",
    "Mataarbechterin",
    "Moderator",
    "Moderatorin",
    "Expert",
    "Expertin",
    "Jos",
    "Sam",
}
FIRST_PITCH, SECOND_PITCH = 0.0, -4.0


def listening_texts(items) -> list[tuple[str, str]]:
    """(where, text) of every Luxembourgish text of `items`: speaker names, turns, questions, options."""
    texts: list[tuple[str, str]] = []
    for item in items:
        texts += [(f"{item.id} speaker", speaker.name) for speaker in item.speakers]
        texts += [(f"{item.id} turn {number}", turn.lb) for number, turn in enumerate(item.turns, start=1)]
        for question in item.questions:
            texts.append((question.id, question.lb))
            texts += [(f"{question.id} option {'ABC'[index]}", option) for index, option in enumerate(question.options)]
    return texts


def question(question_id: str, options: str = '["Jo.", "Nee."]', answer: str = "0", why: str = "It says so.") -> str:
    return (
        f'[[item.question]]\nid = "{question_id}"\nlb = "Wat seet hien?"\nen = "What does he say?"\n'
        f"options = {options}\nanswer = {answer}\nwhy_en = \"{why}\"\n"
    )


def item(item_id: str = "a", genre: str = "news", questions: int = 4, rest: str = "") -> str:
    """A small item with every Luxembourgish word in LOD; `rest` is put in front of its questions."""
    head = (
        f'[[item]]\nid = "{item_id}"\ngenre = "{genre}"\ntitle_en = "A test item"\nlevel = "B1"\n'
        '[[item.speaker]]\nname = "Spriecher"\npitch = 0.0\n'
        '[[item.turn]]\nspeaker = "Spriecher"\nlb = "Moien."\n'
    )
    return head + rest + "".join(question(f"{item_id}-q{number}") for number in range(1, questions + 1))


def all_genres(skip: str = "") -> str:
    """Two items of every genre but `skip`, which gets one."""
    texts = []
    for genre in GENRES:
        for number in range(1, ITEMS_PER_GENRE + (genre != skip)):
            texts.append(item(f"{genre}-{number}", genre))
    return "".join(texts)


def written(tmp_path, text: str):
    path = tmp_path / "listening.toml"
    path.write_text(text, encoding="utf-8")
    return path


# --- the real file ---


def test_the_file_has_two_items_of_every_genre_with_five_four_and_seven_questions():
    assert (GENRES, ITEMS_PER_GENRE, QUESTIONS_PER_ITEM, OPTIONS_PER_QUESTION, LEVEL) == (
        ("news", "conversation", "discussion"),
        2,
        (4, 7),
        (2, 3),
        "B1",
    )
    assert [entry.id for entry in ITEMS] == ITEM_IDS
    assert Counter(entry.genre for entry in ITEMS) == {genre: 2 for genre in GENRES}
    for entry in ITEMS:
        assert entry.level == "B1", entry.id
        assert entry.title_en, entry.id
        assert len(entry.questions) == SHAPE[entry.genre]["questions"], entry.id
        assert [q.id for q in entry.questions] == [f"{entry.id}-q{n}" for n in range(1, len(entry.questions) + 1)]
    assert sum(len(entry.questions) for entry in ITEMS) == 32


def test_every_question_has_two_or_three_options_a_right_one_its_english_and_why():
    for entry in ITEMS:
        for asked in entry.questions:
            assert 2 <= len(asked.options) <= 3, asked.id
            assert len(set(asked.options)) == len(asked.options), asked.id
            assert 0 <= asked.answer < len(asked.options), asked.id
            assert asked.lb and asked.en and asked.why_en, asked.id
            assert all(option and "\n" not in option for option in asked.options), asked.id


def test_the_right_option_is_not_always_in_the_same_place():
    for entry in ITEMS:
        assert len({asked.answer for asked in entry.questions}) > 1, entry.id
    assert set(Counter(asked.answer for entry in ITEMS for asked in entry.questions)) == {0, 1, 2}


def test_every_item_has_the_speakers_words_and_turns_of_its_genre():
    for entry in ITEMS:
        shape = SHAPE[entry.genre]
        assert len(entry.speakers) == shape["speakers"], entry.id
        low, high = shape["words"]
        assert low <= entry.words <= high, (entry.id, entry.words)
        fewest, most = shape["turns"]
        assert fewest <= len(entry.turns) <= most, (entry.id, len(entry.turns))
        assert {turn.speaker for turn in entry.turns} == {speaker.name for speaker in entry.speakers}, entry.id
        assert all(turn.lb and "\n" not in turn.lb for turn in entry.turns), entry.id
        if len(entry.speakers) == 2:
            # A conversation goes back and forth.
            assert all(one.speaker != two.speaker for one, two in zip(entry.turns, entry.turns[1:])), entry.id


def test_the_first_speaker_has_the_voice_and_the_second_the_voice_four_semitones_lower():
    for entry in ITEMS:
        assert [speaker.pitch for speaker in entry.speakers] == [FIRST_PITCH, SECOND_PITCH][: len(entry.speakers)]
        assert entry.turns[0].speaker == entry.speakers[0].name, entry.id
    low, high = VOICE_RANGES["pitch"]
    assert low <= SECOND_PITCH < FIRST_PITCH <= high


def test_the_speakers_are_roles_and_no_first_names_that_lod_lacks():
    for entry in ITEMS:
        assert {speaker.name for speaker in entry.speakers} <= SPEAKER_NAMES, entry.id


def test_an_item_counts_its_words_joins_its_turns_and_finds_a_speaker_by_name():
    entry = ListeningItem(
        id="t",
        genre="conversation",
        title_en="T",
        level="B1",
        speakers=(Speaker("Noper", 0.0), Speaker("Frëndin", -4.0)),
        turns=(Turn("Noper", "Moien, wéi geet et?"), Turn("Frëndin", "Gutt, merci.")),
        questions=(),
    )

    assert entry.words == 6
    assert entry.text == "Moien, wéi geet et?\n\nGutt, merci."
    assert entry.speaker("Frëndin") == Speaker("Frëndin", -4.0)


def test_the_file_says_it_is_not_reviewed_and_takes_nothing_from_inll():
    header = LISTENING_PATH.read_text(encoding="utf-8").split("[[item]]")[0]

    assert "Not reviewed by a native speaker." in header
    assert "nothing is taken from INLL material" in header
    assert "tests/test_listening.py" in header


def test_the_file_keeps_the_words_of_a_test_out():
    # The words that tests/test_practice.py keeps out of the page and the practice content.
    unwanted = re.compile(r"\b(?:exams?|candidates?|marks?|scores?|pass(?:es|ed)?|timers?)\b", re.IGNORECASE)

    assert unwanted.findall(LISTENING_PATH.read_text(encoding="utf-8")) == []


def test_the_texts_address_the_listener_in_no_question():
    # A question speaks of the people of the recording, never to the listener as du.
    informal = re.compile(r"\b(?:du|dech|däin|deng|dengem|denger)\b", re.IGNORECASE)

    for entry in ITEMS:
        for asked in entry.questions:
            assert not informal.search(asked.lb), asked.id


def test_listening_texts_takes_every_name_turn_question_and_option():
    texts = listening_texts(ITEMS)

    assert len(texts) == sum(
        len(entry.speakers) + len(entry.turns) + sum(1 + len(asked.options) for asked in entry.questions)
        for entry in ITEMS
    )
    assert texts[0] == ("strooss-bus speaker", "Spriecherin")
    assert ("schlof-q7 option B", "Wann een méi wéi dräi Woche schlecht schléift.") in texts


@pytest.mark.models
def test_every_luxembourgish_word_of_the_listening_content_is_in_lod(real):
    texts = listening_texts(ITEMS)
    missing = [(where, word) for where, text in texts for word in lb_words(text) if not in_lod(real, word)]

    # Ten speaker names, the turns, 32 questions and their options.
    assert len(texts) >= 6 + 20 + 32 + 64
    assert missing == []


# --- a file that is wrong ---


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (
            item().replace('id = "a"', 'id = "A b"', 1),
            "item 'A b': the id may hold only a-z, 0-9 and single hyphens between them",
        ),
        (item(genre="radio"), "item 'a': unknown genre 'radio'; known: news, conversation, discussion"),
        (item().replace('level = "B1"', 'level = "B2"'), "item 'a': the level is 'B2', not 'B1'"),
        (item().replace('title_en = "A test item"\n', ""), "item 'a': `title_en` is missing"),
        (
            item().replace('speaker = "Spriecher"', 'speaker = "Sam"'),
            "item 'a', turn 1: the speaker 'Sam' is not one of the item's speakers",
        ),
        (
            item(rest='[[item.speaker]]\nname = "Noper"\npitch = -4.0\n'),
            "item 'a', speaker 'Noper': no turn is spoken by it",
        ),
        (
            item(rest='[[item.speaker]]\nname = "Spriecher"\npitch = -4.0\n'),
            "item 'a', speaker 'Spriecher': the name is used twice",
        ),
        (
            item().replace("pitch = 0.0", "pitch = 7.0"),
            "item 'a', speaker 'Spriecher': the pitch 7.0 is outside -6.0 to 6.0",
        ),
        (
            item().replace("pitch = 0.0", "pitch = -6.5"),
            "item 'a', speaker 'Spriecher': the pitch -6.5 is outside -6.0 to 6.0",
        ),
        (item().replace("pitch = 0.0", 'pitch = "low"'), "item 'a', speaker 'Spriecher': `pitch` is not a number"),
        (item().replace("pitch = 0.0", "pitch = true"), "item 'a', speaker 'Spriecher': `pitch` is not a number"),
        (item().replace('lb = "Moien."', 'lb = " "'), "item 'a', turn 1: `lb` is missing"),
        (item(questions=3), "item 'a': 3 questions, an item has 4 to 7"),
        (item(questions=8), "item 'a': 8 questions, an item has 4 to 7"),
        (item(questions=0), "item 'a': no [[question]] tables"),
        (
            item().replace(question("a-q2"), question("a-q2", options='["Jo."]')),
            "item 'a', question 'a-q2': `options` is not a list of 2 to 3 texts",
        ),
        (
            item().replace(question("a-q2"), question("a-q2", options='["Jo.", "Nee.", "Muer.", "Haut."]')),
            "item 'a', question 'a-q2': `options` is not a list of 2 to 3 texts",
        ),
        (
            item().replace(question("a-q2"), question("a-q2", options='"Jo."')),
            "item 'a', question 'a-q2': `options` is not a list of 2 to 3 texts",
        ),
        (
            item().replace(question("a-q2"), question("a-q2", options='["Jo.", " "]')),
            "item 'a', question 'a-q2': an option is empty or not a text",
        ),
        (
            item().replace(question("a-q2"), question("a-q2", answer="2")),
            "item 'a', question 'a-q2': `answer` is not an index of `options` (0 to 1)",
        ),
        (
            item().replace(question("a-q2"), question("a-q2", answer="-1")),
            "item 'a', question 'a-q2': `answer` is not an index of `options` (0 to 1)",
        ),
        (
            item().replace(question("a-q2"), question("a-q2", answer="true")),
            "item 'a', question 'a-q2': `answer` is not an index of `options` (0 to 1)",
        ),
        (
            item().replace(question("a-q2"), question("a-q2", why="")),
            "item 'a', question 'a-q2': `why_en` is missing",
        ),
        (
            item().replace(question("a-q2"), question("a-q2").replace('en = "What does he say?"\n', "")),
            "item 'a', question 'a-q2': `en` is missing",
        ),
        (
            item().replace('id = "a-q2"', 'id = "a_q2"'),
            "item 'a', question 'a_q2': the id may hold only a-z, 0-9 and single hyphens between them",
        ),
        (
            item().replace('id = "a-q2"\n', ""),
            "item 'a', question number 2: `id` is missing",
        ),
        (item() + item(), "item 'a': the id is used twice"),
        (item() + item("b").replace('id = "b-q3"', 'id = "a-q1"'), "item 'b', question 'a-q1': the id is used twice"),
        (item().replace('id = "a-q3"', 'id = "a-q1"'), "item 'a', question 'a-q1': the id is used twice"),
        (all_genres(skip="discussion"), "the file: 2 items of genre 'discussion' are needed, it has 1"),
        (item(), "the file: 2 items of genre 'news' are needed, it has 1"),
        ("", "the file: no [[item]] tables"),
        ('[[item]\nid = "a"\n', "not valid TOML: "),
    ],
)
def test_a_wrong_file_is_a_value_error_that_names_the_entry(tmp_path, text, error):
    with pytest.raises(ValueError) as raised:
        load_listening(written(tmp_path, text))

    assert str(raised.value).startswith(error) if error.endswith(": ") else str(raised.value) == error


def test_a_small_file_with_two_items_of_every_genre_loads(load_listening_text):
    text = all_genres()

    items = load_listening_text(text)

    assert [entry.id for entry in items] == [f"{genre}-{n}" for genre in GENRES for n in (1, 2)]
    assert items[0].questions[0].options == ("Jo.", "Nee.")
    assert items[0].speaker("Spriecher") == Speaker("Spriecher", 0.0)


def test_a_text_over_several_lines_is_one_line(load_listening_text):
    text = all_genres().replace('lb = "Moien."', 'lb = """\nMoien,\n   wéi geet et?"""', 1)

    assert load_listening_text(text)[0].turns[0].lb == "Moien, wéi geet et?"


@pytest.fixture
def load_listening_text(tmp_path):
    return lambda text: load_listening(written(tmp_path, text))


# --- rendering and checking ---

CONVERSATION = next(entry for entry in ITEMS if entry.id == "gemeng-adress")  # its last turn is the second speaker's


class Recorder:
    """A TTS engine that keeps the segments and the voice of every synthesize call."""

    def __init__(self, tts) -> None:
        self.tts = tts
        self.name = tts.name
        self.calls: list[tuple[list, dict | None]] = []

    def synthesize(self, segments, voice=None) -> WavAudio:
        self.calls.append((list(segments), voice))
        return self.tts.synthesize(segments, voice)


def wav_seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes)) as wav:
        return wav.getnframes() / wav.getframerate()


def test_the_speeds_are_rate_one_and_the_measured_three_quarters():
    assert SPEEDS == {"normal": 1.0, "slow": 0.6}
    assert SPEED_LABELS == {"normal": "1x", "slow": "0.75x"}
    assert TURN_GAP_S == 0.6


def test_an_item_is_one_call_per_turn_joined_with_a_gap_and_an_offset_per_turn():
    tts = Recorder(FakeTTS())

    audio = render_item(tts, CONVERSATION, "slow")

    turns = CONVERSATION.turns
    assert [[(segment.lang, segment.text) for segment in segments] for segments, _ in tts.calls] == [
        [("lb", turn.lb)] for turn in turns
    ]
    pitches = {speaker.name: speaker.pitch for speaker in CONVERSATION.speakers}
    assert [voice for _, voice in tts.calls] == [{"lb": {"rate": 0.6, "pitch": pitches[turn.speaker]}} for turn in turns]
    assert audio.sample_rate == SAMPLE_RATE
    tone, gap = FakeTTS.TONE_S, TURN_GAP_S
    assert [offset["index"] for offset in audio.segment_offsets] == list(range(len(turns)))
    assert [offset["speaker"] for offset in audio.segment_offsets] == [turn.speaker for turn in turns]
    assert {offset["lang"] for offset in audio.segment_offsets} == {"lb"}
    for number, offset in enumerate(audio.segment_offsets):
        assert offset["start_s"] == pytest.approx(number * (tone + gap), abs=1e-4)
        assert offset["end_s"] == pytest.approx(number * (tone + gap) + tone, abs=1e-4)
    assert wav_seconds(audio.wav_bytes) == pytest.approx(len(turns) * tone + (len(turns) - 1) * gap, abs=1e-4)


def test_the_learners_volume_stays_but_not_their_rate_and_pitch():
    tts = Recorder(FakeTTS())
    learner = {"lb": {"rate": 0.5, "pitch": 3.0, "volume": 1.5}, "en": {"rate": 1.2}}

    render_item(tts, CONVERSATION, "normal", learner)

    assert {json.dumps(voice, sort_keys=True) for _, voice in tts.calls} == {
        json.dumps({"lb": {"volume": 1.5, "rate": 1.0, "pitch": pitch}}, sort_keys=True) for pitch in (0.0, -4.0)
    }


def test_turns_of_different_sample_rates_are_not_joined():
    class Mixed(FakeTTS):
        def synthesize(self, segments, voice=None):
            audio = super().synthesize(segments, voice)
            if voice["lb"]["pitch"] != SECOND_PITCH:
                return audio
            with wave.open(io.BytesIO(audio.wav_bytes)) as wav:
                frames = wav.readframes(wav.getnframes())
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(frames)
            return WavAudio(buffer.getvalue(), 16000, audio.segment_offsets)

    with pytest.raises(ValueError, match=r"^turn 2 is \(1, 2, 16000\)"):
        render_item(Mixed(), CONVERSATION, "normal")


def test_every_right_choice_is_right_and_n_of_n():
    checked = check_answers(CONVERSATION, {question.id: question.answer for question in CONVERSATION.questions})

    assert set(checked) == {"questions", "right", "total", "transcript"}
    assert checked["right"] == checked["total"] == len(CONVERSATION.questions)
    assert [entry["right"] for entry in checked["questions"]] == [True] * len(CONVERSATION.questions)
    assert checked["transcript"] == [{"speaker": turn.speaker, "lb": turn.lb} for turn in CONVERSATION.turns]


def test_a_wrong_choice_is_named_with_why_and_a_missing_or_impossible_one_is_null():
    first, second, third, *rest = CONVERSATION.questions
    wrong = (first.answer + 1) % len(first.options)
    answers = {first.id: wrong, second.id: len(second.options), third.id: True, "elsewhere-q1": 0}
    answers.update({question.id: question.answer for question in rest})

    checked = check_answers(CONVERSATION, answers)

    assert checked["questions"][0] == {
        "id": first.id,
        "chosen": wrong,
        "answer": first.answer,
        "right": False,
        "why_en": first.why_en,
    }
    assert [(entry["chosen"], entry["right"]) for entry in checked["questions"][1:3]] == [(None, False), (None, False)]
    assert checked["right"] == len(rest)
    assert checked["total"] == len(CONVERSATION.questions)
    assert check_answers(CONVERSATION, {})["questions"][0]["chosen"] is None


# --- the routes ---


def play(client, item_id="gemeng-adress", speed="slow", **fields):
    return client.post("/api/listening/play", data={"item_id": item_id, "speed": speed, **fields})


def check(client, answers, item_id="gemeng-adress"):
    text = answers if isinstance(answers, str) else json.dumps(answers)
    return client.post("/api/listening/check", data={"item_id": item_id, "answers": text})


def test_the_items_route_lists_the_six_with_their_counts(client):
    response = client.get("/api/listening/items")

    assert response.status_code == 200
    listed = response.json()["items"]
    assert [entry["id"] for entry in listed] == ITEM_IDS
    for entry, known in zip(listed, ITEMS, strict=True):
        assert entry == {
            "id": known.id,
            "genre": known.genre,
            "title_en": known.title_en,
            "level": "B1",
            "questions": len(known.questions),
            "words": known.words,
        }


def test_play_slow_is_one_wav_an_offset_per_turn_and_the_second_speaker_lower(client):
    response = play(client)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"item", "audio_wav_base64", "sample_rate", "segment_offsets"}
    seconds = wav_seconds(base64.b64decode(body["audio_wav_base64"]))
    turns = len(CONVERSATION.turns)
    assert seconds == pytest.approx(turns * FakeTTS.TONE_S + (turns - 1) * TURN_GAP_S, abs=1e-3)
    assert [offset["speaker"] for offset in body["segment_offsets"]] == [turn.speaker for turn in CONVERSATION.turns]
    # The last turn is the second speaker's.
    assert client.app.state.tts.voice == {"lb": {"rate": 0.6, "pitch": -4.0}}
    assert body["item"] == {
        "id": "gemeng-adress",
        "genre": "conversation",
        "title_en": CONVERSATION.title_en,
        "level": "B1",
        "speed": "slow",
        "speed_label": "0.75x",
        "questions": [
            {"id": question.id, "lb": question.lb, "en": question.en, "options": list(question.options)}
            for question in CONVERSATION.questions
        ],
    }
    for word in ('"answer"', '"why_en"', '"transcript"', CONVERSATION.questions[0].why_en):
        assert word not in response.text


def test_play_normal_is_rate_one_and_the_learners_volume_reaches_the_engine(client):
    response = play(client, speed="normal", voice='{"lb": {"rate": 0.5, "volume": 1.5}}')

    assert response.status_code == 200
    assert response.json()["item"]["speed_label"] == "1x"
    assert client.app.state.tts.voice == {"lb": {"rate": 1.0, "volume": 1.5, "pitch": -4.0}}


def test_play_renders_every_call_afresh(client):
    client.app.state.tts = Recorder(client.app.state.tts)

    play(client)
    play(client, speed="normal")

    assert len(client.app.state.tts.calls) == 2 * len(CONVERSATION.turns)


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"speed": "fast"}, "unknown speed 'fast'; known: normal, slow"),
        ({"item_id": "nothing"}, f"unknown item 'nothing'; known: {', '.join(ITEM_IDS)}"),
        ({"voice": "{not json"}, "bad voice settings: not JSON: "),
        ({"voice": '{"lb": {"volume": "loud"}}'}, "bad voice settings: "),
    ],
)
def test_play_refuses_an_unknown_item_or_speed_and_a_bad_voice(client, fields, error):
    response = play(client, **fields)

    assert response.status_code == 400
    assert response.json()["error"].startswith(error)


def test_a_failing_engine_is_a_500_that_names_it(client):
    def broken(segments, voice=None):
        raise RuntimeError("no voice")

    client.app.state.tts.synthesize = broken

    response = play(client)

    assert response.status_code == 500
    assert response.json()["error"] == "tts engine (fake) failed: RuntimeError: no voice"


def test_check_with_every_answer_right_is_n_of_n_with_the_transcript(client):
    response = check(client, {question.id: question.answer for question in CONVERSATION.questions})

    assert response.status_code == 200
    body = response.json()
    assert body["item_id"] == "gemeng-adress"
    assert body["right"] == body["total"] == 4
    assert body["transcript"] == [{"speaker": turn.speaker, "lb": turn.lb} for turn in CONVERSATION.turns]


def test_check_names_a_wrong_answer_with_why_and_a_missing_one_as_null(client):
    first, second, *rest = CONVERSATION.questions
    answers = {first.id: (first.answer + 1) % len(first.options), **{q.id: q.answer for q in rest}}

    body = check(client, answers).json()

    assert body["questions"][0]["right"] is False
    assert body["questions"][0]["why_en"] == first.why_en
    assert body["questions"][1] == {
        "id": second.id,
        "chosen": None,
        "answer": second.answer,
        "right": False,
        "why_en": second.why_en,
    }
    assert (body["right"], body["total"]) == (2, 4)


@pytest.mark.parametrize(
    ("answers", "item_id", "error"),
    [
        ("{not json", "gemeng-adress", "bad answers: not JSON: "),
        ("[1, 2]", "gemeng-adress", "bad answers: not a JSON object"),
        ("{}", "nothing", f"unknown item 'nothing'; known: {', '.join(ITEM_IDS)}"),
    ],
)
def test_check_refuses_answers_that_are_no_object_and_an_unknown_item(client, answers, item_id, error):
    response = check(client, answers, item_id)

    assert response.status_code == 400
    assert response.json()["error"].startswith(error)


@pytest.mark.parametrize("missing", [False, True])
def test_the_app_starts_with_a_wrong_file_and_every_listening_route_says_why(tmp_path, settings, monkeypatch, missing):
    wrong = written(tmp_path, item())
    monkeypatch.setattr("app.main.LISTENING_PATH", tmp_path / "absent.toml" if missing else wrong)
    detail = "FileNotFoundError: " if missing else "the file: 2 items of genre 'news' are needed, it has 1"
    error = f"the listening content (app/content/listening.toml) cannot be used: {detail}"

    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health")
        topics = client.get("/api/practice/topics")
        responses = [client.get("/api/listening/items"), play(client), check(client, {})]

    assert (health.status_code, topics.status_code) == (200, 200)
    for response in responses:
        assert response.status_code == 503
        assert response.json()["error"].startswith(error)
        assert response.json()["fix"] == "fix the file and restart"


def test_the_listening_routes_are_plain_functions_and_need_no_session():
    routes = listening_api.router.routes

    assert [(route.path, sorted(route.methods)) for route in routes] == [
        ("/api/listening/items", ["GET"]),
        ("/api/listening/play", ["POST"]),
        ("/api/listening/check", ["POST"]),
    ]
    assert not any(inspect.iscoroutinefunction(route.endpoint) for route in routes)
    assert not any("session_id" in inspect.signature(route.endpoint).parameters for route in routes)


# --- the page ---


def test_the_page_has_the_listening_button_and_its_panel(client):
    html = plain_page(client.get("/").text)

    assert re.search(r'<button id="listening" type="button" aria-pressed="false"[^>]*>Listening</button>', html)
    assert '<div id="listening-panel" class="listening" hidden>' in html
    assert '<button id="listening-play-normal" type="button">Play 1x</button>' in html
    assert '<button id="listening-play-slow" type="button">Play 0.75x</button>' in html
    assert ">Check my answers</button>" in html
    assert ">Another item</button>" in html
    assert ">Leave listening</button>" in html
    assert "Not affiliated with INLL. Scripts and questions are not reviewed by a native speaker." in html
    # Right before Practice, in the row under the text box; Practice, Review and Teacher keep their order.
    assert re.search(r'Listening</button>\s*<button id="practice"', html)


def test_the_page_script_plays_checks_and_says_n_of_m_right():
    script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    section = script.split("// --- listening ---\n")[1].split("// --- events ---\n")[0]

    for route in ("items", "play", "check"):
        assert f'const LISTENING_{route.upper()} = "api/listening/{route}";' in section
    assert 'form.append("speed", speed);' in section
    assert 'form.append("answers", JSON.stringify(listening.chosen));' in section
    assert "loadAudio(played);" in section
    assert "reply = { offsets: played.segment_offsets, nodes: listening.nodes };" in section
    english = json.loads((ROOT / "static" / "i18n" / "en.json").read_text(encoding="utf-8"))
    assert 'listeningResult.textContent = t("listening.result", { right: checked.right, total: checked.total });' in section
    assert english["listening.result"] == "{right} of {total} right"
    assert 't(result.right ? "listening.chose_right" : "listening.chose_wrong", { chosen, key })' in section
    assert english["listening.chose_wrong"] == "You chose {chosen}: not this one - the answer is {key}"
    assert 'textButton(t("practice.help"), () => {' in section
    assert english["practice.help"] == "What was that in English?"
    assert 'input.type = "radio";' in section
    assert 'const OPTION_LETTERS = "ABC";' in section
    # Entering one of practice, review and listening leaves the others; nothing is said or typed in listening.
    assert "  leavePractice();\n  leaveReview();\n  listening = {" in section
    assert script.count("    leaveListening();\n") == 2
    assert "if (pressed || busy || acquiring || listening) return;" in script
    assert "async function send() {\n  if (listening) return;\n" in script


def test_the_page_says_right_and_wrong_in_words_and_never_in_red():
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    section = style.split("/* --- listening --- */")[1].split("/* --- progress --- */")[0]

    assert "--error" not in section and "red" not in section
    assert ".dock.listening-on .talk," in section


# --- the README ---


def test_the_readme_covers_listening_its_routes_and_the_voice_licence():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    text = " ".join(readme.split())
    section = " ".join(readme.split("\n## Listening\n")[1].split("\n## ")[0].split())

    for said in (
        "There are six items, two of each kind",
        "**Play 1x** or **Play 0.75x**",
        "You can play it again as often as you like",
        '"n of m right" and nothing else, and the transcript',
        "are not reviewed by a native speaker",
        "`app/content/listening.toml`",
        "To add an item",
        "`pytest -q tests/test_listening.py`",
    ):
        assert said in section, said
    for route in listening_api.router.routes:
        (method,) = route.methods
        assert f"| `{method} {route.path}` |" in readme, route.path
    for key in ("speed_label", "segment_offsets", "chosen", "why_en", "transcript", "words", "answers"):
        assert f"`{key}`" in readme, key
    licence = (
        "The listening audio is rendered with the `lb_LU-marylux-medium` voice, CC BY-NC-SA 4.0: it may not be "
        "sold or used commercially, and nothing rendered by the app is for sale."
    )
    assert licence in " ".join(readme.split("\n## Licences\n")[1].split("\n## ")[0].split())
    assert text.count(licence) == 1


# --- the real voice ---


@pytest.fixture(scope="module")
def piper(tmp_path_factory):
    from app.engines.tts_piper import PiperTTS

    settings = Settings.from_env()
    folder = settings.models_dir / "piper"
    for path in (folder / f"{settings.tts_voice_lb}.onnx", folder / f"{settings.tts_voice_lb}.onnx.json"):
        if not path.is_file():
            pytest.skip(f"no voice file {path}; run: python scripts/download_models.py --tts")
    tts = PiperTTS(dataclasses.replace(settings, data_dir=tmp_path_factory.mktemp("data")))
    tts.warmup()
    return tts


@pytest.mark.models
def test_a_real_conversation_is_slower_at_three_quarters_and_two_speakers_are_two_pitches(piper):
    normal_tts, slow_tts = Recorder(piper), Recorder(piper)

    normal = render_item(normal_tts, CONVERSATION, "normal")
    slow = render_item(slow_tts, CONVERSATION, "slow")

    normal_s, slow_s = wav_seconds(normal.wav_bytes), wav_seconds(slow.wav_bytes)
    print(f"\n[listening] {CONVERSATION.id}: {normal_s:.1f} s at 1x, {slow_s:.1f} s at 0.75x ({slow_s / normal_s:.2f})")
    assert 1.2 <= slow_s / normal_s <= 1.5
    assert len(normal.segment_offsets) == len(slow.segment_offsets) == len(CONVERSATION.turns)
    assert all(offset["end_s"] > offset["start_s"] for offset in slow.segment_offsets)
    for tts in (normal_tts, slow_tts):
        assert {voice["lb"]["pitch"] for _, voice in tts.calls} == {FIRST_PITCH, SECOND_PITCH}

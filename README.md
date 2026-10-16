# Luxembourgish voice tutor

A web app that runs on your own PC. You talk about your day; each reply opens with a short
Luxembourgish phrase and continues in English, and each language is spoken by its own voice.
Teacher mode explains how a word is formed or conjugated and checks every form it states
against the Lëtzebuerger Online Dictionnaire (LOD). The Luxembourgish phrases you meet are
saved between sessions. Practice mode is a talk the tutor leads in Luxembourgish, question by
question, on the topics and the photo task of the Sproochentest.

This app is not affiliated with INLL, and its feedback is not a prediction of a Sproochentest
result.

> **Written mostly by an AI.** Most of the code, tests and documentation here were written by
> Claude, Anthropic's AI model, working in Claude Code. The author set the direction and the
> design, steered the implementation, and reviewed and tested the result. Treat it like any
> code you did not write yourself: read it before you rely on it. The Luxembourgish it speaks
> and corrects was checked against the LOD dictionary, not by a native speaker.

Licensed under the GNU General Public License v3.0 or later: see [LICENSE](LICENSE) and
"Licences" below.

## Requirements

This is the setup the build was verified on:

- Windows 11, PowerShell.
- Python 3.14.
- Chrome or Edge.
- The Claude Code CLI, installed and logged in (`claude auth login`). The tutor's replies come
  from Claude through that login.
- About 6 GB of disk space: 2.5 GB for the Python packages, 3.4 GB for the models, 0.15 GB
  for the dictionary.
- An NVIDIA GPU is optional. Measured speech-recognition time per sentence: about 0.2 s on an
  RTX 5070 (float16), 4-6 s on the CPU (int8). The GPU libraries come with
  `requirements.txt`; without a usable GPU the app falls back to the CPU by itself.

ffmpeg and the CUDA toolkit are not needed.

## First-time setup

In PowerShell, in the repo root, in this order:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_models.py --stt --tts --lod-audio
python scripts/download_lod.py
claude auth login
```

- `download_models.py --stt` fetches the Whisper model (3.2 GB) into `models/whisper-lb`.
- `--tts` fetches the two Piper voices (about 63 MB each) into `models/piper`.
- `--lod-audio` fetches three short LOD example recordings (about 70 KB together) into
  `data/lod_audio`; the tests and `scripts/smoke.py` use them.
- `download_lod.py` fetches two LOD zips (about 12 MB), unpacks them (about 116 MB of XML in
  `data/lod/raw`) and builds the 21 MB store `data/lod/lod.sqlite`.
- `claude auth login` opens a browser once; `claude auth status` then reports `loggedIn: true`.

Each download skips what is already on disk. `models/` and `data/` are not in git.

The twelve photos of practice mode come with the repository, in `static/photos/`, so setup
downloads none. `python scripts/download_photos.py` fetches from Wikimedia Commons the ones
that are missing; `python scripts/download_photos.py --check` writes nothing and only compares
the licence Commons gives for each photo with the one recorded in `app/content/photos.toml`.

## Run

```powershell
.\run.ps1
```

Then open http://127.0.0.1:8770 in Chrome or Edge. Loading the models takes about ten seconds.
A banner at the top of the page says so when an engine or the dictionary store is not ready,
and gives the command that fixes it.

Using the page:

- Hold the **Hold to talk** button, or the space bar when the cursor is not in the text box,
  and speak. Release to stop. The first press asks for the microphone. When the cursor is on
  another control (a button, a tick box, a slider), the space bar does what that control does
  and does not record.
- **Talk button** in the Voice panel is a switch: on, the button and the space bar are pressed
  once to start recording and once more to stop, and the button reads **Press to talk**. Off
  (the default) is hold to talk. The choice is kept in this browser.
- What was heard appears in the text box. With **auto-send after speaking** ticked (the
  default) it is sent at once. Untick it to correct the text first, then press **Send** or
  Enter. You can also type a message instead of speaking.
- The reply is shown and spoken: the Luxembourgish part is marked LB. Click a part of the
  latest reply to hear it again. If the browser blocks automatic playback, a **Play** button
  appears.
- **Teacher** makes the next message a grammar question. Asking in words works too, for
  example "can you explain ...", "how do you say ...", "how do I conjugate ...". In a teacher
  reply each stated form is listed with "LOD" when the dictionary has it and "not verified in
  LOD" when it does not.
- **Luxembourgish mix**, next to Teacher, sets how much Luxembourgish a chat reply mixes into
  its English: see the next section.
- **Practice** opens practice mode: see "Practice" below.
- The side panel lists the phrases that are new in this session. Under it the **Data** block
  says what is kept on this PC, with **Export** and **Delete everything**: see "Export and
  delete" below.
- **Voice** in the side panel opens sliders for the speed, pitch and volume of each voice.
  **Play sample** speaks a fixed sentence with the current values; replies use them from the
  next message on. The values are kept in this browser; **Reset** returns to the settings below.

## Local access

The server answers only requests made to its own address, and only the page it serves can use
the API:

- The `Host` header must be `127.0.0.1:PORT` or `localhost:PORT`, else the answer is 421
  ("this page is served at http://127.0.0.1:PORT"; fix: open that address).
- An `Origin` header, when the request has one, must be `http://127.0.0.1:PORT` or
  `http://localhost:PORT`, else the answer is 403 ("request from another origin").
- Every route under `/api/` except `/api/health` needs the header `X-Loci-Token`, else the answer
  is 401 ("no access token"; fix: reload the page). `/api/health` is open.

The token is made once and kept in `data/access_token`; it is not in the settings, not in a URL
and not in a log line. The page gets it in a `loci-token` meta tag when the server serves `/`
(`/index.html` redirects there). If the file cannot be written, the token lives in memory until
the server stops, a tab that is open then needs a reload after the next start, and `/api/health`
says so under `access`. Set `LOCI_ACCESS_TOKEN` to use a token of your own instead
of the file. `scripts/smoke.py` reads the variable or the file; another client sends the header
itself.

## Luxembourgish mix

A chat reply can switch into Luxembourgish in the middle of its English, the way a bilingual
speaker does. What is switched is a phrase or a whole sentence, never a single word inside an
English sentence. The slider has four levels:

| Level | Name | A chat reply |
| --- | --- | --- |
| 0 | Opening phrase only | opens with one Luxembourgish phrase and continues in English |
| 1 | A phrase here and there | has one or two short Luxembourgish phrases after the opening phrase |
| 2 | Phrases and sentences | is Luxembourgish for about a third: phrases and whole short sentences |
| 3 | Mostly Luxembourgish | is mostly simple Luxembourgish sentences, with English where a beginner would be lost |

- The level applies to chat replies only; a teacher reply is the same at every level.
- From level 1 on the tutor is given the phrases you have met (the 40 seen last) and builds on
  them first. At levels 1 and 2 it adds at most one or two new phrases per reply.
- In the page the Luxembourgish inside such a reply is shown in colour and underlined by a solid line, without the LB tag.
  A dotted underline means the phrase has a translation: hold the mouse over it to read it.
- Each language is still spoken by its own voice. Inside a sentence the pause between the two
  voices is 0.08 s; after the end of a sentence it is 0.25 s.
- The level is kept in this browser. `LOCI_CHAT_MIX` sets the level the page starts with and
  the level of a request that names none.

## Practice

Practice mode is a talk in Luxembourgish that the tutor leads. It takes its content from the
speaking part of the Sproochentest, the Luxembourgish language test that INLL (Institut
national des langues Luxembourg) runs: a short warm-up, questions on one everyday topic, then
the description of a photo. It does not take its manner: nothing is marked, a clock runs only
when you pick the realistic setting and never ends the talk, and you can stop at any point. This app is not affiliated with INLL, and its feedback is not a prediction
of a Sproochentest result. The questions, the photo descriptions and the tutor's corrections
are not reviewed by a native speaker.

How a talk goes:

- **Start.** Press **Practice**, next to Send, and pick one of the twelve topics: Work, Family,
  Leisure, Food, Home, Travel, Health, Shopping, Commune and administration, School and
  children, Weather, Life in Luxembourg. The tutor opens the talk and asks the first question aloud, in
  Luxembourgish, before you have said anything. The question is also shown on a card above
  the talk button.
- **The warm-up.** The tutor names the topic, then asks two short personal questions drawn
  from a fixed list of six (your name, where you live, where you are from, the languages you
  speak, how you are, why you learn Luxembourgish); the card says "Warm-up - 1 of 2". Then the
  questions of the topic follow. The next talk of the same session draws two that were not
  asked yet. A warm-up answer is judged and kept like any other.
- **Relaxed or realistic.** Above the topics you pick **Relaxed** (the default) or
  **Realistic - a 10-minute clock runs while you talk**. In the realistic setting the card shows
  a clock that counts up from 0:00 when the talk opens; after ten minutes a line under it says
  "The ten minutes of the real talk are over - carry on as long as you like." The clock never
  ends the talk and nothing else changes: the same questions, help and corrections in both. The
  clock stops at the look back. The page remembers your choice in this browser, and every
  answer of the talk is kept with it (`setting`).
- **The English help.** **What was that in English?** on the card shows the question in
  English. The look back names the questions you wanted in English as something to go over
  again, not as a fault.
- **Answer, then confirm or edit.** Hold the talk button and answer aloud. What was heard
  appears in the text box and is not sent by itself: auto-send is off in practice mode.
  Change the text if it is not what you said, then press **That's what I said** or Enter.
  You can also type an answer. Only the text you confirm is looked at.
- **What comes back.** The tutor reacts in English in a sentence. When there is something to
  put right it gives one correction, the most important one: your own words, a short reason,
  and a better version in Luxembourgish. The word forms of the better version are listed with
  "LOD" or "not verified in LOD", as in a teacher reply. An answer in English gets the
  Luxembourgish for it. This takes a few seconds per answer (measured: 4 to 10 s), nearly all
  of it waiting for Claude.
- **The label of a correction.** Under every correction a label says in words how far the
  app could check it, then whether it is about **Words** or **Grammar**:
  - **Verified by LOD**: the form you said and the better form are forms of the same
    dictionary word, and every word the tutor added is in LOD. There is no acceptability
    encoder yet, so a Verified label rests on LOD alone: "Verified by LOD; no encoder
    installed".
  - **Likely**: every word the tutor added is in LOD, but LOD does not show that your form and
    the better one belong to one word (a different word, a missing verb, the order of the
    words), or the correction is of an answer in English, or of an answer that was heard and
    not edited.
  - **Unverified - may be wrong**: a word the tutor added is not in LOD. The correction is
    still said, but shown grey.

  Only the words the tutor added or changed are looked up: a word you said yourself and the
  tutor kept, such as the name of a town, does not make the label worse. Hover over the label
  for the facts it rests on.
- **Heard and not edited.** When you confirm what was heard without changing it, the
  correction is at most Likely and the label says "heard, not edited": the recogniser may have
  put the slip there, not you.
- **No correction where LOD accepts both.** When the tutor's better version only swaps a word
  for another spelling that LOD also lists (Problem and Probleem, or a form with or without the
  n of the n-rule), the answer counts as good: nothing is said about it, and the page shows the
  other spelling in grey as "Also possible", not spoken. A correction that only touches a name,
  a filler such as "ëëë" or a number is dropped the same way, without the "Also possible"
  line.
- **A short answer** (fewer than six words) is asked for a bit more, once per question.
- **"I don't know"** (or "no idea", "skip", "next", "keng Ahnung", "ech weess net") as the
  whole answer brings the question in English and a way to begin the answer; said a second
  time, it moves on to the next question.
- **The topic** has eight questions: two general ones, two about the past, two about the
  present, two about plans.
- **The photo.** After the topic the tutor shows three of twelve photos, those
  of the topic first when it has one that you have not talked about yet. Pick one; three
  questions about it follow: what you see, one question about that scene, and how you like it.
  The tutor does not see the picture: it works from a written description of it in
  `app/content/photos.toml`.
- **Stop.** **Stop here** ends the talk wherever it stands and brings the look back. A talk
  that runs to its end gets the look back by itself. **Leave practice** returns to the chat.

The look back on a talk:

- Three short comments of the tutor, shown and spoken in this order, under **Getting the
  message across**, **Words** and **Grammar** (INLL's criteria task completion, vocabulary and
  grammar in plain words): whether what you meant got across comes first. They come from one
  request to Claude; when it fails, the rest of the look back is still shown.
- **To take home**: every correction of the talk with its question, your words, the better
  version, its label and its LOD labels. This list is put together by the app from what was
  said in the talk, not by Claude. The first three better versions are spoken once more. An
  answer that counted as good with an "Also possible" spelling is not in the list.
- The questions you wanted in English, and the words of your answers that the dictionary does
  not list. Such a word was not checked; it need not be wrong.

What is judged and what is not:

- Judged: the text you confirmed, as an answer to the question. A word form the dictionary
  contradicts, a word that does not exist, a missing verb; whether the answer is in
  Luxembourgish; whether it answers the question.
- Not judged: pronunciation. The recogniser cannot tell your slip from its own mishearing, so
  nothing about how you sounded is said or kept as a verdict.
- Not judged: word order and case. LOD lists word forms; it cannot say whether a sentence is
  built right, so the tutor is told not to correct those on their own. Names are left alone.
- No marks, no points, no pass or fail, in the talk or in the look back.

Where your answers are kept, and how to delete them:

- Every answer you confirm is kept on this PC in `data/practice/`, in one folder per day and
  session, for example `data/practice/2026-10-13-0a1b2c3d/`. `answers.jsonl` there has one
  line per answer: the time, the topic, the setting of the talk (`setting`: `relaxed` or
  `realistic`) and the question, what was heard, the text you
  confirmed, whether you had the question in English, and what the tutor made of it, with
  the label, the category, whether the answer was typed or heard (`source`), the other
  spelling of an "Also possible" line, the facts the label rests on, and `disputed` (null
  until you tap **I think I was right**, see below). The
  recording of a spoken answer is next to it as it was recorded, `1.webm`, `2.webm` and so on
  by the number of the line. A typed answer has no recording. A skipped question is not kept.
- The Data block in the side panel shows how many answers and recordings are kept. Its
  **Delete everything** removes them with everything else the app keeps about you (see
  "Export and delete"); there is no delete of the answers alone, since their text is in the
  learner store too. A talk that is open goes on.
- **I think I was right.** Under the label of a correction of an answer that was kept, this
  button marks that answer: its line in `answers.jsonl` gets `disputed` with the time you
  tapped (`{"time": ..., "note": ""}`), and the button gives way to "Noted - you think you were
  right. Kept with this answer." A second tap on the same answer changes nothing. The
  correction itself stays as it was given, in the talk and in the look back, and nothing is
  spoken. The mark is kept on this PC only, with the answer (and in the learner store's copy
  of it, when it has one); it is not sent anywhere. **Export** in the Data block takes it along.
- When an answer cannot be written (a full or read-only disk, say), the talk goes on and a
  small note under the tutor's reply says that this answer was not saved.
- The app refuses to write to, count, export or delete a `practice` folder that is a link to
  another place, or that is not directly inside the data folder.

The content files, for a reviewer:

- `app/content/practice.toml` holds everything the tutor says by itself: the opening and
  closing lines, the line in front of the warm-up and of each stage, the warm-up questions,
  and the twelve topics with their questions. A warm-up question is a `[[warmup]]` entry with
  `id`, `lb`, `en` and an optional `starter`; there must be at least two. A topic question is
  one entry:

  ```toml
  [[topic.question]]
  id = "wunnen-1"
  stage = "general"
  lb = "Wou wunnt Dir?"
  en = "Where do you live?"
  starter = "Ech wunnen zu ..."
  ```

  `id` must be unique in the file, warm-up questions included. `stage` is `general`, `past`,
  `present` or `plans`. `lb`
  is the question as the tutor says it, `en` its English for the help button. `starter` is
  optional: the beginning of an answer, said after "I don't know".
- `app/content/photos.toml` holds the twelve photos: where each is from, its author and licence,
  the description the tutor works from, and the one question about that scene (`question_lb`,
  `question_en`). The description must be true to the picture. A photo may carry a `topic`,
  the id of a topic in `practice.toml`; the app logs a photo whose `topic` is no topic.
- Both files are plain text and say at the top how they are read. To correct a sentence, edit
  it and restart the app. `pytest -q tests/test_practice.py` then checks the shape of both
  files and looks up every Luxembourgish word in LOD. The app starts with a wrong file too:
  with a wrong `practice.toml` practice mode says which entry it is; with a wrong
  `photos.toml` or a missing image a talk ends after its topic and the server log says why.
- The instructions Claude gets for the corrections and for the look back are
  `app/prompts/practice.md` and `app/prompts/wrapup.md`.

## Listening

Press **Listening**, next to Practice, to hear a short recording in Luxembourgish and answer its
questions, as in the listening part of the Sproochentest. There are six items, two of each
kind: a news item read by one speaker, an everyday conversation and a longer discussion
between two. Each has four to seven questions with two or three options, A, B and C. Pick an
item, then **Play 1x** or **Play 0.75x**: the recording is read by the Luxembourgish voice,
and the second speaker of a conversation is the same voice four semitones lower, named before
each of its turns in the transcript. 1x is the voice's own speed (rate 1.0); 0.75x is rate
0.6, which measured about 0.75 times the speed. You can play it again as often as you like, at
either speed, before and after answering. **What was that in English?** under a question shows
it in English. Once every question has a choice, **Check my answers** shows for each the letter
you chose and "right", or "not this one" with the letter of the answer, and why; then a line
"n of m right" and nothing else, and the transcript, whose turns light up while it plays and
play again when pressed. **Another item** goes back to the list, **Leave listening** back to
the chat. Listening, practice and review share the place above the talk button: pressing one
leaves the others. Nothing of it is kept on this PC. This app is not affiliated with INLL; the
scripts and questions were written for it, take nothing from INLL material and are not
reviewed by a native speaker.

The content is `app/content/listening.toml`, read when the app starts; restart it after an
edit. An item is one `[[item]]` with an `id`, a `genre` (`news`, `conversation` or
`discussion`), a `title_en` and `level = "B1"`; then one `[[item.speaker]]` per speaker with a
`name`, shown before each of its turns, and a `pitch` in semitones (-6 to 6; the first speaker
is at 0, a second one at -4); then the script as `[[item.turn]]` entries, each with the
`speaker` who says it and its `lb`; then four to seven `[[item.question]]` entries in the order
of the recording, each with a unique `id`, the question as `lb` and `en`, two or three
`options`, the index of the right one as `answer` (from 0) and `why_en`, the words of the
script that give it away. To add an item, add such an entry and run
`pytest -q tests/test_listening.py`: it checks the shape of the file and looks up every
Luxembourgish word in LOD. With a wrong file the app still starts, and the listening routes
answer 503 with the entry that is wrong.

## Learner store

The app keeps your history in one SQLite database on this PC, `data/learner/learner.sqlite`
(in the folder `LOCI_DATA_DIR` names, if you set it): sessions, chat and teacher turns with how
long you spoke, the phrases you met, every practice answer (what was heard, the text you
confirmed, the verdict, the correction with its label and category, where its recording is),
the review items and every review. Everything is in `data/learner/`:

- `learner.sqlite`, with `learner.sqlite-wal` and `learner.sqlite-shm` next to it while the app
  runs;
- `backups/learner-YYYY-MM-DD.sqlite`, a copy made on each day the app starts; the last 7 are kept;
- `pre-migration/`, a copy of `data/phrases.json` and `data/practice/` as they were before the
  first import, written and checked by size before anything was imported.

On its first start the app copies `data/phrases.json` and the answers under `data/practice/`
into the database; a second start adds nothing. The originals stay where they are and are
still written as before, so everything that worked before still does. `/api/health` reports the
store under `learner`. A database that fails SQLite's `quick_check` is left exactly as it is:
the app runs without it, and `/api/health` says so and names the newest backup. To restore,
stop the app, copy that backup over `learner.sqlite` and delete `learner.sqlite-wal` and
`learner.sqlite-shm` if they are there. The app refuses a `learner` folder that is a link to
another place or that is not directly inside the data folder.

## Review

Press **Review**, next to Practice, to go over what you have met: the phrases of the chat and
the better versions of your practice answers. Review and practice share the place above the
talk button, so pressing one leaves the other. The card shows the English and, for a
correction, what you said last time; you say the Luxembourgish (hold to talk, then check the
text and press **That's what I said**) or type it. An answer that matches, once case,
punctuation and spaces are set aside or each word is a LOD form of the same word, is rated
Good on its own; the form is shown and spoken, and **I did not know it** turns the rating into
Again and goes to the next card. An answer that does not match shows the form beside what you
said, and you rate it yourself: **Again**, **Hard**, **Good** or **Easy**. The next card
follows; with nothing to review, the card says when the next one is due. At most 10 new
cards come each day (UTC). The text you check before **That's what I said** is the answer
that counts; what speech-to-text heard is kept beside it. The match rule takes the LOD forms
of a word as the same word only when the dictionary is installed; without it only the
normalised text counts.

The schedule is FSRS-6 with its 21 default weights, a desired retention of 0.9, learning steps
of 1 and 10 minutes and no fuzz. It is the app's own implementation, with no new dependency,
checked in a test against the values py-fsrs 6.3.2 gives for a fixed sequence of reviews. The
state of a card is built from its log of reviews, which is only added to. The routes are `GET /api/review/next` (`item` with `id`, `gloss`, `hint`,
`source`, `state`, never the form; `counts`; `next_due` when there is no item),
`POST /api/review/answer` (form: `item_id`, `text`, optional `raw_text`, `mode` `typed` or
`spoken`) and `POST /api/review/rate` (form: `item_id`, `rating` 1 to 4, `text`, `raw_text`,
`mode`, and `log_id` to replace an automatic Good).

## Progress

The **Progress** block in the side panel shows totals only, with no streaks and no goals:
minutes spoken this week (ISO weeks in UTC, from Monday) and in all, from the length of your
chat turns and practice answers; phrases met; review items in review, of all items, and how
many are due now; your practice corrections per week by category (Words, Grammar, Getting the
message across) for the last four weeks; and each practice topic with how many of its
questions you have answered. The category is a heuristic, not a judgement: a fix is Grammar
when a changed and a new word are forms of one LOD word, when it only adds or moves words, or
when it swaps forms of hunn, sinn, ginn or wäerten; any other fix, and an answer in English,
is Words. Minutes stay at 0 until the app keeps the length of what you say,
and the counts of chat turns and new practice answers grow once the app records them in the
learner store. The block is filled from `GET /api/learner/progress`: `minutes` (`week`,
`total`), `phrases`, `items` (`in_review`, `total`, `due`), `corrections_by_week` (four of
`week` as `YYYY-Www`, `words`, `grammar`, `task`, newest first) and `topics` (`id`, `name_en`,
`answered`, `questions`). Without the learner store it answers 503 and the block says why.

## Export and delete

The **Data** block in the side panel says what is kept on this PC: "Kept on this PC: N
phrases, M answers (K recordings), I review items" and the size of all the learner's files,
with the folders they are in. It is filled from `GET /api/learner/summary`.

**Export** saves one ZIP, `luxembourgish-tutor-export-YYYY-MM-DD.zip` (the date in UTC), from
`GET /api/learner/export`. It holds:

- `learner.json`: `exported_at`, `schema_version` and `tables`, every row of every table of the
  learner store (sessions, turns, phrases, answers with what was heard and what you confirmed,
  corrections, review items, every review);
- `learner.sqlite`: a copy of the database with the same rows, made with `VACUUM INTO`, which
  opens in any SQLite tool;
- `audio/practice/<folder>/<n>.webm`: every recording of your practice answers, as recorded;
- `originals/phrases.json` (and `originals/phrases.json.corrupt`, when there is one) and
  `originals/practice/<folder>/answers.jsonl`: the files the app still writes, as they are.

The backups and the pre-migration copy are not in the ZIP: they are older copies of the same
data, and the live files are under `originals/`. Without the learner store the export answers
503.

**Delete everything** asks first: "This deletes everything the app kept about you on this PC:
the learner database and its backups, the saved phrases and every practice recording. It is
the last copy. Copies made by a backup or sync tool outside the app are not touched." **No**
is chosen first; **Yes** sends `DELETE /api/learner`, which:

- empties every table of the learner store, truncates its write-ahead log, closes it and
  removes `data/learner/` with everything in it: `learner.sqlite`, `learner.sqlite-wal`,
  `learner.sqlite-shm`, the 7 backups and the pre-migration copy;
- removes `data/practice/` with every answer and recording;
- removes `data/phrases.json`, `data/phrases.json.corrupt` and `data/phrases.json.tmp`;
- then looks whether any of these is still there. When one is, it answers 500 and names it,
  and the Data block shows the names; press it again once the program holding the file is
  closed.

After the delete the app opens a new, empty learner store, so it goes on recording; the Data
block then shows zeros and the size of that empty database. A talk that is open goes on.
Deleting twice is fine. The LOD dictionary, the settings files and the rest of `data/` are not
the learner's and stay.

Copies outside the app are not deleted: a sync tool such as OneDrive, Windows File History or
your own backups may hold copies of `data/`, and the app cannot reach them. Whether this PC
syncs or backs up the project folder has not been checked. A deleted file's bytes can also stay
on the disk until they are written over.

## HTTP API

The page uses these routes, and another client (a later mobile app, for example) can call
them too. The server binds 127.0.0.1 only, so it is reachable from this PC only. Every `/api/`
route except `GET /api/health` needs the header `X-Loci-Token` (see "Local access").

| Route | Input | Answer |
| --- | --- | --- |
| `GET /api/health` | - | `status` (`ok` or `degraded`), `engines` (`stt`, `tts`, `llm`, `acceptability`, each with `name`, `ready`, `detail`), `lod` (`ready`, `detail`), `learner` (`ready`, `detail`, `path`), `access` (`persisted`, `detail`: whether the access token is kept in `data/access_token`) |
| `POST /api/session` | - | `session_id`; sessions live in memory until the server stops |
| `POST /api/stt` | multipart form: `audio` (file) | `text`, `language`, `duration_s`, `segments`, `stt_ms` |
| `POST /api/chat` | JSON: `session_id`, `text`, `mode` (`auto`, `chat` or `teacher`; default `auto`), `stt_ms` (default 0), `mix` (0 to 3; default the `LOCI_CHAT_MIX` setting) | a turn without audio: `heard`, `transcript`, `mode`, `mix` (the level the reply was made at; 0 in teacher mode), `segments` (`lang`, `text`, and `gloss` when an `lb` segment is a phrase of `new_phrases` or a saved one), `new_phrases`, `claims` (`form`, `lemma`, `verified`), `phrases_added`, `timings`, `shape_warning` |
| `POST /api/tts` | JSON: `segments` (list of `lang`, `text`); optional `voice`, and `respell` (`{word: respelling}`, up to 5 words, laid over the pronunciation table for this request only; nothing is saved; 400 for a word or respelling that breaks the rules) | a WAV file (22050 Hz, mono, 16 bit); the header `X-Segment-Offsets` holds the start and end of each segment as JSON |
| `POST /api/turn` | multipart form: `session_id`, and `audio` (file) or `text`; optional `mode`, `stt_ms`, `mix` | the turn as in `/api/chat` plus `audio_wav_base64`, `sample_rate`, `segment_offsets`. When no speech is heard: `heard` false, an empty `transcript` and `timings` only |
| `GET /api/mix` | - | `default` (the `LOCI_CHAT_MIX` setting), `levels` (the four mix levels, each with `value`, `name`, `hint`) |
| `GET /api/phrases` | - | `total`, `phrases` (the saved phrases, latest first) |
| `GET /api/practice/topics` | - | `topics`: the twelve topics, each with `id`, `name_lb`, `name_en`, `questions` (how many) |
| `POST /api/practice/start` | form: `session_id`, `topic_id`; optional `voice`, `setting` (`relaxed`, the default, or `realistic`; kept with every saved answer of the talk) | a practice turn (see below): the opening, the line of the warm-up and the first warm-up question. Makes no request to Claude. Starts the session's talk afresh; the warm-up questions the session's talks asked are drawn last. 400 for an unknown topic or setting |
| `POST /api/practice/answer` | multipart form: `session_id`, `text` (the confirmed answer); optional `raw_text` (what speech-to-text heard), `help_used` (`true` when the question was shown in English), `audio` (the recording; it is kept, not transcribed), `voice` | a practice turn: the tutor's reaction, the correction if any, then the next question, the three photos or the goodbye. One request to Claude, none for a skip |
| `POST /api/practice/photo` | form: `session_id`, `photo_id` (one of the three on offer); optional `voice` | a practice turn: the first question about the picked photo |
| `POST /api/practice/stop` | form: `session_id`; optional `voice` | the look back: `part` (`done`), `wrapup` (`answered`, `skipped`, `help_questions`, `corrections`, `unknown_words`, `photo_described`, `sentences` with `task_en`, `words_en`, `grammar_en`, and `llm_failed`; each of `corrections` has `question_number`, `question_lb`, `answer` and the `verdict`, `quote`, `better_lb`, `why_en`, `claims`, `label`, `category`, `source`, `checks` of the correction), `segments` (each with a `role`, the sentences in the order `task_en`, `words_en`, `grammar_en`), and the audio fields. One request to Claude; once per talk |
| `GET /api/practice/saved` | - | what is kept of the practice answers: `answers`, `bytes`, `sessions`, `folder` |
| `POST /api/practice/dispute` | form: `session_id`, `folder`, `line` (the `kept` of the answer's turn); optional `note` (up to 200 characters are kept) | marks that kept answer as one whose correction the learner thinks was wrong: `folder`, `line`, `disputed` (`time`, `note`). A second call keeps the first mark. 403 when `folder` is not one of the session's own, 404 when the answer is not kept on this PC |
| `GET /api/listening/items` | - | `items`: the six listening items, each with `id`, `genre` (`news`, `conversation` or `discussion`), `title_en`, `level`, `questions` (how many), `words` (of the script) |
| `POST /api/listening/play` | form: `item_id`, `speed` (`normal`, 1x, or `slow`, 0.75x); optional `voice` (only its `lb` volume is used) | `item` (`id`, `genre`, `title_en`, `level`, `speed`, `speed_label`, `questions` with `id`, `lb`, `en`, `options`; no answer, reason or transcript), and `audio_wav_base64`, `sample_rate`, `segment_offsets` as in `/api/turn`, one per turn of the script with its `speaker`. Rendered afresh on every call; no session needed. 400 for an unknown item or speed, 503 when the listening file cannot be used |
| `POST /api/listening/check` | form: `item_id`, `answers` (a JSON object: question id -> the index of the chosen option, from 0) | `item_id`, `questions` (each with `id`, `chosen`, null when missing or no option, `answer`, `right`, `why_en`), `right` and `total` (n of m), `transcript` (each turn with `speaker` and `lb`). No session needed; 400 for an unknown item or answers that are not a JSON object |
| `GET /api/review/next` | - | `item` (`id`, `gloss`, `hint`, `source`, `state`; never the form; null when nothing is to review), `counts` (`due`, `new_available`, `in_review`, `total`), `next_due` (when there is no item; null for never) |
| `POST /api/review/answer` | form: `item_id`, `text` (the confirmed answer); optional `raw_text`, `mode` (`typed` or `spoken`) | `matched`, `expected` (the form), and after a match `rating` (3, Good), `log_id`, `due`, `state`. A mismatch writes nothing |
| `POST /api/review/rate` | form: `item_id`, `rating` (1 Again, 2 Hard, 3 Good, 4 Easy); optional `text`, `raw_text`, `mode`, `log_id` (the automatic Good to replace) | `rating`, `log_id`, `due`, `state`, `superseded` |
| `GET /api/learner/progress` | - | the Progress block: `minutes`, `phrases`, `items`, `corrections_by_week`, `topics` (see "Progress") |
| `GET /api/learner/summary` | - | `phrases`, `answers`, `recordings`, `items`, `bytes` (of every learner file), `folders` (the learner folder, the practice folder, phrases.json) |
| `GET /api/learner/export` | - | the export ZIP (see "Export and delete"), `application/zip` with `Content-Disposition: attachment`; the page fetches it with the `X-Loci-Token` header and saves it |
| `DELETE /api/learner` | - | deletes everything the app keeps about you (see "Export and delete") and opens an empty learner store: `deleted` (true), `left` (empty). 500 naming what is still there. Deleting twice is fine |
| `GET /api/pronunciation/flags` | - | `flags`, the latest first: `id`, `lang`, `word`, `text` (the line), `flagged_at`, `phonemes`, `count`, and `respelling` (what the pronunciation table has for the word now; "" for none) |
| `POST /api/pronunciation/flags` | JSON: `lang`, `word` (one word of 2 to 40 letters), `text` (the line, up to 500 characters are kept) | the flag; a word flagged before is refreshed and counted again. 400 for a bad word |
| `DELETE /api/pronunciation/flags/{id}` | - | `dismissed` (the id); 404 for an unknown id |
| `POST /api/pronunciation/words` | JSON: `lang`, `word`, `respelling` (1 to 60 characters; no square bracket or one whole `[[ phonemes ]]` block) | `lang`, `word`, `respelling`; writes `data/pronunciations.json` and removes the flag of the word. 400 for a bad word or respelling, 409 (naming the file, left unchanged) when `data/pronunciations.json` cannot be read |
| `GET /` | - | the page |

The review and learner routes answer 503 without the learner store; `DELETE /api/learner`
works without it too. There is no route that deletes the practice answers alone: only
**Delete everything** deletes.

A practice turn is a JSON object with `part` (`warmup`, `interview`, `photo` or `done`), `kind`
(`question`, `more`, `again`, `choose` or `end`), `question` (`id`, `lb`, `en`, `stage`,
`index`, `total`; `stage` is `warmup` for a warm-up question; `index` and `total` count within
the part; null while a photo is picked and at the end), `segments` (`lang`, `text`, and
`gloss` where the content file gives the English), `correction` (`verdict`, one of `good`,
`fix`, `english`, `unclear`, with `quote`, `better_lb`, `why_en`, `claims`; `label`
(`verified`, `likely` or `unverified`), `category` (`words` or `grammar`), `source` (`typed`,
`heard-edited` or `heard-unedited`), these three empty but on a `fix` or an `english`;
`alternative_lb`, on a `good` only, the other spelling of an "Also possible" line, which is
not spoken; `checks`, the facts the label or the "Also possible" line rests on, one plain
sentence each; null when the
answer was not judged) and `feedback_error` (why not), `saved` (the fields of
`GET /api/practice/saved` once the answer is kept; null when the turn kept none) and
`save_error` (why not), `kept` (where that answer is kept: `folder` and `line`, for
`POST /api/practice/dispute`; null as `saved` is), `photos` (the three to pick from, on a `choose` turn; each with `id`,
`url`, `alt`, `caption`), `photo` (the picked one), and `audio_wav_base64`, `sample_rate`,
`segment_offsets` as in `/api/turn`. The talk is kept on the session, in memory.

A request the app cannot serve answers with an HTTP status of 400 or above and a JSON object
with `error` and `fix`. A request with missing or wrongly typed fields gets FastAPI's 422
answer with `detail` instead. An upload may be 25 MB at most.

## Tests

With the virtual environment active, in the repo root:

```powershell
pytest -q
pytest -q -m models -s
pytest -q -m live -s
python scripts/smoke.py
```

- `pytest -q` runs everything with fake engines. Tests marked `models` are part of it: they
  run when the downloaded files are present and are skipped when they are not. It makes no
  call to Claude.
- `pytest -q -m models -s` runs only the tests on the real Whisper model, Piper voices, LOD
  store and LOD recordings, and prints their timings. The end-to-end test sends a recording
  through the real speech engines with a fake tutor.
- `pytest -q -m live -s` calls outside services: the `claude` CLI (this uses your Claude
  subscription, about 25 requests, half of them for practice mode) and lod.lu (five
  requests). Its end-to-end test runs a spoken turn and a teacher turn through all three real
  engines.
- `python scripts/smoke.py` needs the server running in another window. It prints the engine
  states, sends `data/lod_audio/moien.m4a` as one turn (one request to Claude), prints the
  transcript and the reply, and writes the spoken reply to `data/last_reply.wav`. `--text
  "..."` sends typed text instead, `--audio` another recording, `--url` another address.

- `pytest -q -m "live and models" -s tests/test_practice_live.py` is the one test that needs
  both: it has the Luxembourgish voice say "Ech wunnt zu Esch." (a wrong verb form) and "Ech
  wunnen zu Esch.", sends both recordings through speech recognition, and has Claude judge
  the two transcripts as answers to "Wou wunnt Dir?" (two requests). It prints the
  transcripts, the verdicts and the timings. The recogniser can put the wrong form right or
  change another word, so the test checks the verdict against what the transcript holds: a
  transcript that still has "wunnt" must come back corrected, and "wunnen" must be left
  alone. In the one run of the build the wrong sentence was heard as "Ech wunnen zu Äech."
  and the right one as "Eech wunnen zu Eech.", so only the second check applied (see "Known
  limits"). The correction of the wrong form itself is checked on typed text, by the live
  test in `tests/test_practice.py`.

The tests never write to `data/phrases.json`, `data/practice/` or `data/learner/`, and never
delete there: they keep what they save in
a temporary folder. A turn through the running server, the smoke script included, does write
there.

### tutor-eval

`scripts/tutor_eval.py` measures how well one provider (see "Providers" under "Settings") does
the tutor's work, and prints for what it is and is not recommended. It needs the LOD store
and not the server. With the virtual environment active, in the repo root:

```powershell
python scripts/tutor_eval.py --fake
python scripts/tutor_eval.py --provider claude_cli --n 20
python scripts/tutor_eval.py --provider ollama --model gemma3:12b --schema-off --json --out data/tutor_eval_ollama.json
```

- `--fake` runs on fixed answers and calls nothing outside; the tests run it this way.
- `--provider ID` sends all four roles to that entry of `data/providers.json`. `claude_cli`
  works without the file, with the `LOCI_LLM_*` settings. Each prompt is one call there,
  two when the answer needs the second try, plus one warm-up call at the start that no suite
  counts. With the Claude CLI every call is a request on your subscription.
- `--model M` gives every role that model instead of the entry's.
- `--n N` runs only the first N prompts. The file has the chat prompts first (70), then the
  teacher prompts (20), then 5 practice answers to judge and 5 look backs, so a small N
  measures chat only.
- `--schema-off` has `openai_compat` entries describe the JSON shape in the prompt instead of
  sending it in the request; the report says which entries changed. A server that constrains
  its output to the shape passes SCHEMA with almost any model, so run once without and once
  with this flag.
- `--json` prints the results as JSON after the report; `--out FILE` also writes what is
  printed to that file. Put it under `data/`, which git ignores.
- Exit code 0 when the verdict is good for chat, 1 when it is not, 2 when the run could not
  start (no LOD store, a broken `providers.json`, an unknown id, or a provider that is not
  ready, such as a CLI that is not logged in).

The three suites:

- **SCHEMA**: the share of calls whose first answer had the right JSON shape, by the same
  checks the app makes. A second try or the fixed text counts as a miss. Gate: at least 98 %.
- **LATENCY**: p50 and p90 of the total time of a call, from request to complete answer,
  without the warm-up and without failed calls. Nothing streams, so this is not the time to
  the first word. Shown, not gated.
- **LB-GEN**: whether the Luxembourgish parts (the `lb` segments) of the chat and teacher
  replies are Luxembourgish. A segment counts when at least 0.6 of its words are forms LOD
  lists (words are letters only; `d'`, `'t`, `'n` and `'s` count as words of their own;
  numbers are not words) and, when the `langid` package is installed, langid also says `lb`;
  the report's `signals` line says which applied. Replies that ended in the fixed text are left
  out. Gate: at least 95 % of the segments. The mean LOD coverage of the segments is shown with
  its gate of 0.90, but that gate is not reliable and not part of the verdict: names and forms
  LOD does not list count as misses, and in the check made for this build even correct
  Luxembourgish sentences reached only 0.73 to 0.83. The replies to the 10 prompts in which
  the learner writes German-flavoured Luxembourgish get a line of their own.

The verdict: `Good for: chat, explanations` when SCHEMA and LB-GEN both pass, else `Good for:
nothing in this run` and `Not recommended for: chat, explanations, ...`. "Correcting your
Luxembourgish" is always under `Not recommended for`: whether a model's corrections are right
(LB-JUDGE) is not measured here. Each suite line gives its N, and the first line the date,
the provider, the model and the number of prompts.

The 100 prompts (`app/evaluation_prompts.json`) were written by a model for A1/A2 learners and
are not reviewed by a native speaker; the report says "unreviewed". The thresholds of LB-GEN
were set on 13 sentences, not validated by a native speaker either. A pass is a first sign,
not a proof that the Luxembourgish is good.

#### Measured

Claude CLI tier, 2026-10-16, model sonnet, effort low, N=20 of the unreviewed prompts (18 replies, 1 judgement, 1 wrap-up), one warm-up dropped: SCHEMA 100 % valid on the first try; LATENCY p50 5.1 s, p90 6.9 s total call time; LB-GEN 95.5 % of 22 Luxembourgish segments (gate 95 %), LOD coverage 0.95; no errors, no fixed text. Verdict: good for chat and explanations; not recommended for correcting your Luxembourgish (LB-JUDGE not measured). Command: `.venv\Scripts\python.exe scripts\tutor_eval.py --provider claude_cli --n 20 --json --out data\tutor_eval_claude_cli.json`.

### Correction quality (unreviewed)

LB-JUDGE is a set of sentences for measuring the practice corrections: 500 correct rows and
500 rows with one error, written to `data/eval/lb_judge.jsonl` (a header line, then one row
per sentence with `id`, `text`, `label`, `source`, `base_id`, `error_type`, for errors
`wrong` and `gold`, and `review`). The correct rows are LOD example sentences (ZLS dictionary,
CC0) with no register label, 3 to 10 words, no digit and every word in LOD, statements first.
An error row is such a sentence with one LOD form swapped for one that is not legal there:
**verb person** (`ech wunnen` -> `ech wunnt`), **auxiliary** (`ech sinn ... gaangen` -> `ech hunn
... gaangen`), **adjective ending** (`de gudde Mann` -> `de gutt Mann`) or **noun gender**
(`de Mann` -> `eng Mann`), 125 of each. No native speaker has checked a row: every row says
`review: unreviewed`. The set is biased: dictionary sentences are not learner speech; they come
from the same source as the LOD checks, which therefore know every word of them; and the errors
are synthetic single swaps, so recall on them says nothing about exam-relevant errors.

```powershell
python scripts/lb_judge.py build
python scripts/lb_judge.py build --n 500 --seed 7 --out data/eval/lb_judge.jsonl
```

It reads `data/lod/raw/new_lod-art.xml` and `data/lod/lod.sqlite` (from
`scripts/download_lod.py`) and takes a few seconds; the same seed gives the same rows. `data/`
is git-ignored, so the set lives on each PC.

`run` sends rows of the set through the practice correction pipeline as it ships (the LOD
facts for the sentence, the judge, the checks and the label), each as a typed answer to one
fixed question, `Sot mir eppes.` (`eval_lb` in `app/content/practice.toml`). What a talk adds
to a judge call (help used, the photo, earlier reactions) is not part of it.

```powershell
python scripts/lb_judge.py run --fake
python scripts/lb_judge.py run --live --correct 150 --errors 50 --seed 7
```

`--correct` and `--errors` draw that many rows of each arm with `--seed` (default: all of
them). The report goes to `data/eval/lb_judge_report.json` (`--report`), with every row's
outcome and `reviewed_by: "none"`; the console gets the headline, the counts per arm, the
recall per class and the label mix (`--json` adds the report, `--out` also writes what is
printed). What is counted:

- **False corrections**: correct sentences that were shown a fix or an English-answer
  correction labelled Verified or Likely. Unverified ones (greyed, "may be wrong") are counted
  apart, and so are fixes the checks turned back into "good" (`downgraded`). The 95% upper
  bound is the one-sided exact (Clopper-Pearson) bound of that count: at n = 500, 16 false
  corrections still keep it under 5%, 17 do not.
- **Recall**, on the synthetic errors only: `hit` is a fix that quotes the swapped word,
  `caught` any fix. It says nothing about the errors learners make.
- A judge call that fails (timeout, login) is counted under `errors` and left out of n.

`--fake` judges with an oracle that knows every error, so the run checks the pipeline, not a
model: it must show 0 false corrections (exit code 1 otherwise) and catch every error (recall
1.0 on this PC's set). It is the CI gate. `--live` runs the judge on the Claude CLI
(`LOCI_LLM_ENGINE=claude_cli`, or `provider` with the judge role on a `claude_cli` entry),
about 6 to 9 seconds a sentence; more than 200 sentences need `--yes`. Exit code 2 when a run
cannot start.

Live result (unreviewed), 2026-10-16, after the fix for Verified on a swap between two correct forms, Claude CLI, model sonnet, seed 7: 2 of 150 correct sentences received a Likely correction and none a Verified one (1.33 %, 95 % upper bound 4.14 %); recall 26 of 50 synthetic errors (52 %): verb person 12 of 13, auxiliary 8 of 14, noun gender 5 of 10, adjective ending 1 of 13. Sentences are ZLS dictionary examples and synthetic errors, checked by no native speaker; this is not the launch figure. Both false corrections rewrite a correct form into a more common one (the preterite "krute" into a perfect, the conditional "bräicht" into "brauch"). Mean 8.4 s per judgement (p90 13.4 s). The run before the fix gave 3 of 150, two of them Verified.

Known limits: the figure is measured on dictionary sentences, not on what learners say; no
native speaker has checked a row; nothing here is a published false-correction rate, and it
does not show that NFR-QUAL-1 is met.

## Settings

Environment variables, or the same names in a git-ignored `.env` file in the repo root; the
environment wins. A variable that is set but empty is ignored.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LOCI_PORT` | `8770` | Port of the server. `run.ps1` and `scripts/smoke.py` read it from the environment only, not from `.env` |
| `LOCI_HOST` | `127.0.0.1` | Read into the settings but not used: `run.ps1` always binds 127.0.0.1 |
| `LOCI_ACCESS_TOKEN` | made once in `data/access_token` | The token the API routes need in the header `X-Loci-Token`: see "Local access". Read from the environment only; it is not a setting and not in `.env` |
| `LOCI_MODELS_DIR` | `models` | Folder of the Whisper model and the voices; a relative path is taken from the repo root |
| `LOCI_DATA_DIR` | `data` | Folder of the LOD store, the LOD recordings, `phrases.json`, `last_reply.wav`, `practice/`, the practice answers kept on this PC (text and recordings), and `learner/`, the learner store; a relative path is taken from the repo root |
| `LOCI_STT_ENGINE` | `faster_whisper` | Speech-to-text engine; `fake` runs without a model |
| `LOCI_STT_MODEL` | `ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish` | The Hugging Face repository `download_models.py --stt` fetches. The app loads whatever is in `<models dir>/whisper-lb` |
| `LOCI_STT_DEVICE` | `auto` | `auto` (GPU, else CPU), `cuda` or `cpu` |
| `LOCI_TTS_ENGINE` | `piper` | Text-to-speech engine; `fake` plays tones |
| `LOCI_TTS_VOICE_LB` | `lb_LU-marylux-medium` | Piper voice for Luxembourgish segments |
| `LOCI_TTS_VOICE_EN` | `en_GB-cori-medium` | Piper voice for all other segments |
| `LOCI_TTS_RATE_LB` | `0.7` | Speaking rate of the Luxembourgish voice, 0.5 to 1.5: 1 is the voice's own speed, lower is slower. It is passed to Piper as a length scale, which the voices follow only in part: measured, 0.85 made Luxembourgish sentences about 6 % longer, 0.7 about 20 % and 0.5 about 50 % |
| `LOCI_TTS_RATE_EN` | `1.0` | The same for the English voice |
| `LOCI_TTS_PITCH_LB` | `0.0` | Pitch of the Luxembourgish voice in semitones, -6 to 6: negative is lower. A lower pitch also slows the speech a little (measured: about 5 % at -1.5) |
| `LOCI_TTS_PITCH_EN` | `-1.5` | The same for the English voice |
| `LOCI_TTS_VOLUME_LB` | `1.0` | Volume of the Luxembourgish voice, 0.2 to 2: 1 is unchanged. The voices already fill the 16-bit range, so above 1 the loudest parts are clipped |
| `LOCI_TTS_VOLUME_EN` | `1.0` | The same for the English voice |
| `LOCI_LLM_ENGINE` | `claude_cli` | Tutor engine: `claude_cli` (the Claude CLI, as before), `provider` (each role on the server `data/providers.json` names: see "Providers" below) or `fake` (one fixed reply) |
| `LOCI_LLM_MODEL` | `sonnet` | Passed to `claude --model` |
| `LOCI_LLM_TIMEOUT_S` | `90` | Seconds to wait for one answer of the CLI |
| `LOCI_LLM_EFFORT_CHAT` | `low` | Passed to `claude --effort` in chat mode: `low`, `medium`, `high`, `xhigh` or `max`. `default` (any case) passes no flag, which leaves the CLI's own level |
| `LOCI_LLM_EFFORT_TEACHER` | `low` | The same for teacher mode |
| `LOCI_ACCEPTABILITY_ENGINE` | `none` | Encoder that scores how acceptable a Luxembourgish sentence is, for the labels of corrections; `none` means no encoder is installed (corrections are then checked against LOD only) and `fake` is for tests |
| `LOCI_CHAT_MIX` | `0` | How much Luxembourgish a chat reply mixes into its English, 0 to 3: see "Luxembourgish mix". A value outside 0 to 3 counts as the nearest level |
| `LOCI_CLAUDE_EXE` | found on `PATH` | Full path of `claude.exe`, when it is not found by itself. Not a `.cmd` or `.bat` wrapper |

After changing a voice name, run `python scripts/download_models.py --tts` again.

With `LOCI_LLM_ENGINE=provider` and no `data/providers.json`, `LOCI_LLM_MODEL`,
`LOCI_LLM_TIMEOUT_S`, `LOCI_LLM_EFFORT_CHAT` and `LOCI_LLM_EFFORT_TEACHER` are the defaults of
the one built-in entry, `claude_cli`: every role goes to the Claude CLI with that model and
timeout, `tutor_chat` with the chat effort and the other three roles with the teacher effort.
Once the file exists, it decides; only a `claude_cli` entry without a `model` still takes
`LOCI_LLM_MODEL`.

### Providers

`LOCI_LLM_ENGINE=provider` sends each of the four roles of the tutor to its own server and
model. The roles are `tutor_chat` (chat replies), `teacher_explain` (teacher replies), `judge`
(the reaction and correction of a practice answer) and `wrapup` (the look back at the end of a
practice talk). Which server and model each role uses is set in `providers.json` in the data
folder (`data/providers.json`, or the folder `LOCI_DATA_DIR` names). Without the file every
role goes to the Claude CLI, as described above. The default, `LOCI_LLM_ENGINE=claude_cli`,
does not read the file.

The file has two objects. `providers` holds the servers, keyed by an id you choose; `roles`
names one of them for each of the four roles:

| Field | Meaning |
| --- | --- |
| `kind` | `claude_cli` (the Claude CLI and your login), `openai_compat` (any server with an OpenAI-style `/chat/completions`: Ollama, LM Studio, llama.cpp, OpenRouter, Mistral), `anthropic` (the Anthropic Messages API) or `fake` (fixed answers, for tests) |
| `base_url` | Address of the server, ending in `/v1` for `openai_compat`. For `anthropic` it must include `/v1` or be left out (then `https://api.anthropic.com/v1`) |
| `model` | The model a role gets when it names none |
| `key_env` | The name of the environment variable that holds the key, not the key; leave it out for a server that needs none |
| `schema_style` | How the reply's JSON shape is sent. `openai_compat`: `nested` (default, `response_format` with a nested `json_schema`, the OpenAI form, for Ollama, LM Studio and OpenRouter), `flat` (the schema directly under `response_format`, the form llama.cpp documents) or `none` (the shape is described in the prompt only). `anthropic`: `tool` (default, the shape is a strict tool asked for in the prompt) or `none`. `claude_cli` always passes it to `--json-schema` |
| `timeout_s` | Seconds to wait for one answer (default 90) |
| `options` | `max_ctx` and `cost_per_1k` (both kinds over HTTP), `max_tokens` (`anthropic`, default 1024) |

A role has `provider` (an id of `providers`), and optionally `model` (instead of the entry's)
and `effort`. Only the `claude_cli` kind uses `effort` (`claude --effort`); the others leave it
out. All four roles must be listed. An unknown field, a missing role or a role naming an
unknown provider stops the engine, and `/api/health` gives the field at fault in the `detail`
of the `llm` engine.

An example with every kind of server this build has an adapter for. Only `claude_cli` is used
here; any entry can serve any role:

```json
{
  "providers": {
    "claude_cli": { "kind": "claude_cli", "model": "sonnet", "timeout_s": 90 },
    "ollama": {
      "kind": "openai_compat",
      "base_url": "http://127.0.0.1:11434/v1",
      "model": "gemma3:12b",
      "schema_style": "nested"
    },
    "lmstudio": {
      "kind": "openai_compat",
      "base_url": "http://127.0.0.1:1234/v1",
      "model": "gemma-3-12b-it",
      "schema_style": "nested"
    },
    "llamacpp": {
      "kind": "openai_compat",
      "base_url": "http://127.0.0.1:8080/v1",
      "model": "gemma-3-12b-it",
      "schema_style": "flat"
    },
    "openrouter": {
      "kind": "openai_compat",
      "base_url": "https://openrouter.ai/api/v1",
      "model": "google/gemma-3-12b-it",
      "key_env": "OPENROUTER_API_KEY",
      "schema_style": "nested"
    },
    "anthropic": {
      "kind": "anthropic",
      "base_url": "https://api.anthropic.com/v1",
      "model": "claude-haiku-5-5",
      "key_env": "ANTHROPIC_API_KEY",
      "schema_style": "none",
      "options": { "max_tokens": 1024 }
    }
  },
  "roles": {
    "tutor_chat": { "provider": "claude_cli", "effort": "low" },
    "teacher_explain": { "provider": "claude_cli", "effort": "low" },
    "judge": { "provider": "claude_cli", "effort": "low" },
    "wrapup": { "provider": "claude_cli", "model": "sonnet", "effort": "low" }
  }
}
```

- Keys go in `.env` or the environment, under the name `key_env` gives (for example
  `OPENROUTER_API_KEY=...` in `.env`), never in this file. A field named `key` or `api_key`
  stops the engine with that message. A key is not logged, and an error that would show it
  has `***` in its place.
- The port of llama.cpp differs between builds: 8080 in older ones and in its Docker
  examples, 9931 in its current documentation. `llama-server --help` gives the default of
  yours; put that port in `base_url`.
- The model names above are examples: use the name your server lists. Ollama's is the one
  `ollama pull` took; llama.cpp serves the one model it was started with.
- `anthropic` with `schema_style` `tool` sends the shape as a strict tool with `tool_choice`
  `auto` and asks for it in the prompt, because Anthropic's documentation says Claude Opus 5.5,
  Sonnet 5.5 and Fable 5.1 reject a forced tool (`tool_choice` `tool` or `any`) with HTTP 400.
  A reply without the tool call counts as a schema error and is asked once more. Neither style
  was tried against the API.
- Every answer is checked against the same JSON shape as with the Claude CLI. When it does not
  fit, or the server fails, the engine asks once more, quoting the error. When that fails too,
  the role says a fixed text instead: a chat or teacher reply "Kënnt Dir e bësse méi dozou
  soen?" (from `app/content/practice.toml`, not reviewed by a native speaker) and "Sorry, I lost
  my thread for a moment. Could you say that once more?"; a practice answer gets "Got it." and
  no correction; the look back gets three fixed sentences. A rejected key or a timeout is shown
  as an error at once, without the second call.
- `claude -p` never gets a provider's key: the CLI is started without any variable whose name
  ends in `_API_KEY`, without `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN`, and, with the
  provider engine, without any variable a `key_env` of this file names. So it keeps using your
  own login.
- Nothing in this build ran a local server or used a key. The `openai_compat` and `anthropic`
  adapters are tested against fake HTTP answers only; whether a given server version takes
  the `nested` or `flat` envelope is open until one is tried.

Outside installs, all optional, none done by this build. For a local model:

```powershell
winget install Ollama.Ollama
ollama pull gemma3:12b
```

The download is several GB and was not tried here. For the language check of tutor-eval (see "tutor-eval" under
"Tests"), with the virtual environment active:

```powershell
pip install langid
```

## Pronunciation

Before a reply is spoken, the words listed in a pronunciation table are replaced, so a word a
voice says wrongly can be corrected. Each voice has its own table: `en` for the English voice,
`lb` for the Luxembourgish one. The table shipped with the app
(`app/engines/pronunciations.json`) makes the English voice say "Luxembourgish" with a hard g.
Your own entries go in `data/pronunciations.json`, in the same format:

```json
{
  "en": {
    "Luxembourgish": "Luxembourghish"
  },
  "lb": {}
}
```

- A word matches in any case, and only as a whole word: "Luxembourgish's" is replaced,
  "Luxembourgishness" is not. A key may be several words.
- The value is a respelling, which the voice reads like any text, or phonemes in double square
  brackets, which the voice speaks as written: `"[[ lˈʌksəmbˌɜːɡɪʃ ]]"`.
- Your entry wins over a shipped one for the same word; the value `""` switches a shipped
  entry off.
- To see the phonemes a voice makes of a text, with the virtual environment active, in the
  repo root (for the Luxembourgish voice, put its file name):

  ```powershell
  python -X utf8 -c "import sys; from piper import PiperVoice; print(*(''.join(s) for s in PiperVoice.load('models/piper/en_GB-cori-medium.onnx').phonemize(sys.argv[1])))" "Luxembourgish, Luxembourghish"
  ```

  This prints `lˈʌksɪmbˌɜːdʒɪʃ, lˈʌksɪmbˌɜːɡɪʃ`: the `dʒ` is the soft g, and the respelling
  has `ɡ` in its place.
- Try a respelling first. Phonemes in brackets have two measured limits: the word before
  them can get a stress it would not have otherwise ("in" did), and `'s` after them is read
  as the letter s.
- Save the file as UTF-8. An edit applies to the next reply; no restart is needed.
- A file that cannot be read (broken JSON, or another shape than above) is ignored: only the
  shipped table is used, and `/api/health` names the file and the problem in the `detail` of
  the `tts` engine.

Flag a word and fix it without editing a file. Under a tutor line, **Flag a word** shows a
button for each word of the line; a click flags that word. The **Pronunciation** block in the
side panel lists the flagged words, the latest first, each with the phonemes the voice made of
it when it was flagged, how often it was flagged, the line it stood in and the respelling the
table has for it now. Type a respelling in the box (up to 60 characters; plain text, or one
whole `[[ phonemes ]]` block), then:

- **Play** speaks the line again with that respelling. Nothing is saved. With the box empty it
  speaks the line as the voice says it now.
- **Save** writes the respelling into `data/pronunciations.json` (for the voice of the word's
  language) and takes the flag away. The next reply uses it; no restart is needed. A
  `data/pronunciations.json` that cannot be read is not written over: Save answers 409 and names
  the file.
- **Dismiss** takes the flag away and changes nothing else.

Flagging a word again refreshes its line and phonemes and counts one more; there is one flag per
word and language, 200 at most (the oldest goes first). The flags are in
`data/pronunciation_flags.json`. It holds sentences the tutor said, but it belongs to the app
like `data/pronunciations.json`: **Export** does not include it and **Delete everything** does
not remove it.

## Interface language

The **English / Français** control in the header sets the language of the page: `<html lang>`,
the title, every control and label, the statuses, the units and the headings of the Data,
Progress, Practice, Review and Listening parts. The choice is kept in this browser (`loci.lang`);
the first visit takes French for a browser set to French, else English. The texts are in
`static/i18n/en.json` and `static/i18n/fr.json`. The French is machine-translated and has not
been reviewed (the file says so in its `_note`).

What stays as it is: the turns already in the conversation, the tutor's replies and the server's
messages (`error` and `fix`, the names and hints of the mix levels, the English of questions,
topics and photo captions, the detail of the health banner). The **Export** button fetches
`api/learner/export` with the access token and saves the ZIP; it is not a link.

To add a language: copy `en.json` to `static/i18n/<code>.json` and translate every value (keep
the `{name}` placeholders, and the `_one` / `_other` pairs for counts), add an `<option>` to the
control in `static/index.html` and the code to `LANGS` in `static/app.js`.

The rule for strings: a string the page shows is a key in `en.json` and `fr.json`, applied with
`data-i18n` (or `data-i18n-attr` for an attribute) in `static/index.html` or with `t("key")`
in `static/app.js`. A string literal with letters and a space in `app.js` outside `t(...)` fails
`tests/test_i18n.py` unless it is a line of `tests/i18n_allow.txt`, which is for class lists and
selectors, not for shown text. The test also fails when a key is missing from one table, when
the English in the HTML differs from `en.json`, when a text with letters in the HTML has no
key, and when a key is used nowhere.

## Accessibility

The page was changed after an audit against WCAG 2.2 (checked in the code, not with a person
using assistive technology):

- Keyboard: Tab reaches every control and Space or Enter works it; on the talk button they
  record (held, or pressed once and again with the **Talk button** switch). A played-again part
  of a reply answers Space and Enter.
- Language: every Luxembourgish and English part of a reply, a claim, a new phrase and a topic
  carries a `lang` attribute, so a screen reader can switch voice.
- Colour is never the only cue: a status that is an error begins with "Error:", a pressed button
  shows a tick, the Luxembourgish inside a mixed reply has a solid line under it, and the part
  being spoken has an outline.
- The borders of buttons, the text box and the Voice panel are #767676, 3:1 against the page and
  the panels. The page has a `main` landmark; the status line reads a wait once, and the
  seconds counting beside it are hidden from a screen reader.
- Not done: no screen reader was run on the page, and the dotted underline of a gloss is still
  a cue by style only.

## AI notice and synthetic audio

The top of the page, under the title, always says: "You are talking to an AI tutor, not a
person. The voices are computer-generated, and the Luxembourgish can contain mistakes." The
tutor's lines in the conversation are labelled "AI tutor".

Every WAV the app makes carries a RIFF INFO comment (`ICMT`) saying the speech is synthetic:
"Synthetic speech generated by an AI text-to-speech system; not a recording of a person.", with
the software tag (`ISFT`) "Luxembourgish voice tutor (Piper TTS)". The text-to-speech engines
add it, so the audio of `/api/turn`, `/api/tts`, the practice routes, the listening items and
the voice samples all carry it, and so does `data/last_reply.wav` from `scripts/smoke.py`. To
read it:

```powershell
.venv\Scripts\python.exe -c "import sys;d=open(sys.argv[1],'rb').read();print(d[d.find(b'ICMT'):][:120])" data/last_reply.wav
```

## Privacy

- Your audio is transcribed and the replies are synthesized on this PC. No audio leaves it.
- The text of the conversation (the last 30 turns) and, in teacher mode, dictionary facts and
  a rules sheet are sent to Claude through your own Claude Code login. At a Luxembourgish mix
  level of 1 or more, the 40 saved phrases seen last are sent with each chat message too.
- In practice mode Claude gets, per answer: the question, the text you confirmed, whether you
  had the question in English, dictionary facts for the words of the answer, the rules sheet,
  the tutor's last four reactions and, for a photo question, the written description of the
  photo. For the look back it gets the questions and your answers of that talk with what the
  tutor made of each. It gets no recording and no picture.
- With `LOCI_LLM_ENGINE=provider`, the same text goes to the server `data/providers.json`
  names for each role instead: to this PC for a local server, to that company for OpenRouter,
  Anthropic or another server on the internet.
- The phrases you met are saved in `data/phrases.json` on this PC.
- Every practice answer you confirm is kept on this PC under `data/practice/` (in the folder
  `LOCI_DATA_DIR` names, if you set it): its text, what speech recognition heard, the tutor's
  reaction and correction, and the recording of your voice as the browser made it. These files
  hold your voice and what you said about your life.
- The learner store, `data/learner/`, holds the same and more: your sessions, chat and teacher
  turns with how long you spoke, the phrases, the practice answers and corrections, the review
  items and every review, with its 7 daily backups and the pre-migration copy (see "Learner
  store").
- None of this is sent anywhere, and none of it is in a git commit (`data/` is ignored). It
  stays until you delete it: **Export** in the Data block gives you all of it as one ZIP, and
  **Delete everything**, then **Yes**, removes the learner store with its backups and
  pre-migration copy, `data/practice/` and `data/phrases.json` in one go and checks that none
  is left (see "Export and delete"). It is the last copy. A backup or sync tool that covers the
  project folder (OneDrive, File History) copies these files like any other, and the app does
  not delete those copies; whether one runs on this PC has not been checked.
- Nothing else leaves the machine while the app runs. The LOD dictionary is a local copy.
  The download scripts and `pytest -m live` do contact Hugging Face, data.public.lu, lod.lu
  and, for the photos, commons.wikimedia.org.

## Licences

Copyright (C) 2026 brickbat101. This program is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version. It is
distributed WITHOUT ANY WARRANTY; see [LICENSE](LICENSE). The app uses piper-tts, which is
GPL-3.0-or-later. That licence covers this repository's code only: the models, the voices and
the dictionary data are downloaded separately and keep their own licences, listed below. The
Luxembourgish voice is non-commercial (CC BY-NC-SA 4.0), so audio made with it may not be used
commercially.

Python packages, as `pip show` reports them in the virtual environment. The first eleven are
the lines of `requirements.txt`; the last three are installed by them and do the speech work.

| Component | Version | Licence | Source |
| --- | --- | --- | --- |
| fastapi | 0.143.0 | MIT | https://github.com/fastapi/fastapi |
| uvicorn[standard] | 0.54.0 | BSD-3-Clause | https://uvicorn.dev/ |
| python-dotenv | 1.2.4 | BSD-3-Clause | https://pypi.org/project/python-dotenv/ |
| python-multipart | 0.0.32 | Apache-2.0 | https://github.com/Kludex/python-multipart |
| httpx | 0.28.1 | BSD-3-Clause | https://github.com/encode/httpx |
| pytest | 9.1.1 | MIT | https://docs.pytest.org/ |
| faster-whisper | 1.2.1 | MIT | https://github.com/SYSTRAN/faster-whisper |
| piper-tts | 1.8.0 | GPL-3.0-or-later | http://github.com/OHF-voice/piper1-gpl |
| nvidia-cublas-cu12 | 12.9.2.10 | NVIDIA proprietary (`LicenseRef-NVIDIA-Proprietary`) | https://developer.nvidia.com/cuda-zone |
| nvidia-cudnn-cu12 | 9.27.0.42 | NVIDIA proprietary (`LicenseRef-NVIDIA-Proprietary`) | https://developer.nvidia.com/cuda-zone |
| huggingface_hub | 2.2.0 | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| ctranslate2 | 4.8.2 | MIT | https://opennmt.net |
| av (PyAV) | 19.0.1 | BSD-3-Clause | https://pyav.basswood.io |
| onnxruntime | 1.31.0 | MIT | https://onnxruntime.ai |

Not checked: the other packages these pull in, and the FFmpeg libraries that the PyAV wheel
ships in `av.libs`, which have their own licences.

Models, data and the tutor:

| Component | Licence | Source and notes |
| --- | --- | --- |
| Whisper model `ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish` | MIT | https://huggingface.co/ZLSCompLing/Whisper-Large-V3-Turbo-Luxembourgish. Its model card (`models/whisper-lb/README.md`) says "License: MIT"; developed by the Zenter fir d'Lëtzebuerger Sprooch, fine-tuned from `openai/whisper-large-v3-turbo` |
| Piper voice `lb_LU-marylux-medium` | CC BY-NC-SA 4.0, non-commercial | https://huggingface.co/rhasspy/piper-voices. Its MODEL_CARD says "License: http://creativecommons.org/licenses/by-nc-sa/4.0/" and gives the dataset as https://github.com/marytts/marylux-data (the MaryLux data); it names no author. It also says "Finetuned from U.S. English lessac voice (medium quality)". The MODEL_CARD of that lessac voice gives as its licence https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html, the Blizzard 2013 Lessac licence, which allows research use only |
| Piper voice `en_GB-cori-medium` | public domain | https://huggingface.co/rhasspy/piper-voices. Its MODEL_CARD says "License: public domain"; the recordings are from https://librivox.org and the voice was trained from scratch |
| LOD dictionary data (the store in `data/lod`) | CC0 | Zenter fir d'Lëtzebuerger Sprooch, on data.public.lu: https://data.public.lu/fr/datasets/letzebuerger-online-dictionnaire-lod-linguistesch-daten/ and https://data.public.lu/fr/datasets/letzebuerger-online-dictionnaire-lod-flexiounstabellen/ |
| Three LOD example recordings (`data/lod_audio`) | assumed CC0 | Downloaded from lod.lu. They are assumed to fall under the same dataset licence as the dictionary data; this was not confirmed for the audio files |
| Claude | Anthropic's terms for your plan | Used through your own Claude Code login and subscription |

The listening audio is rendered with the `lb_LU-marylux-medium` voice, CC BY-NC-SA 4.0: it may
not be sold or used commercially, and nothing rendered by the app is for sale.

The twelve photos of practice mode (`static/photos/`), from Wikimedia Commons. The licence is
the one Commons names for the file; `scripts/download_photos.py --check` compares it again.

| Photo | Author | Licence | Source |
| --- | --- | --- | --- |
| `maart`: a vegetable stall at a street market | Joenomias Menno De Jong | CC0 | https://commons.wikimedia.org/wiki/File:Greengrocer_At_A_Street_Market_In_France_Dordogne_(242326619).jpeg |
| `mettegiessen`: a family at a meal | National Cancer Institute | Public domain | https://commons.wikimedia.org/wiki/File:Family_eating_meal.jpg |
| `busarret`: people waiting at a bus stop | Rakoon | CC0 | https://commons.wikimedia.org/wiki/File:Bialystok_poland_people_bus_station_august_2018.jpg |
| `park`: a picnic in a park | D Coetzee | CC0 | https://commons.wikimedia.org/wiki/File:People_having_picnic_in_park_(6968378538).jpg |
| `gare`: passengers boarding a train | Andy Li | CC0 | https://commons.wikimedia.org/wiki/File:Passengers_boarding_train_at_Bishops_Lydeard_station_2025-07-27.jpg |
| `terrass`: a café terrace | A. E. Crane, U.S. Department of Transportation | Public domain | https://commons.wikimedia.org/wiki/File:Grand_Rounds_Scenic_Byway_-_Late_Afternoon_Dining_at_an_Outdoor_Cafe_on_St._Anthony%27s_Heritage_Trail_-_NARA_-_7718743.jpg |
| `apdikt`: a pharmacy counter | Justin Carnahan, U.S. Air Force | Public domain | https://commons.wikimedia.org/wiki/File:The_31_MDG_pharmacy_takes_care_of_everyone_(8469053).jpg |
| `supermarche`: a supermarket aisle | Erik Calonius, U.S. Environmental Protection Agency | Public domain | https://commons.wikimedia.org/wiki/File:SHOPPING_FOR_GROCERIES_IN_A_WASHINGTON,_DISTRICT_OF_COLUMBIA_SUPERMARKET_-_NARA_-_555651.jpg |
| `gemengenhaus`: the town hall of Luxembourg City at dusk | Pierre Blaché | CC0 | https://commons.wikimedia.org/wiki/File:Luxembourg_City_Hall_(54540726309).jpg |
| `schoulhaff`: children in a school yard, 1941 | Irving Rusinow, U.S. Department of Agriculture | Public domain | https://commons.wikimedia.org/wiki/File:Taos_County,_New_Mexico._Children_play_in_the_Penasco_schoolyard._School_was_built_by_the_Catholic_._._._-_NARA_-_521831.jpg |
| `reen`: a shopping street in Ettelbruck in the rain | Alf van Beem | CC0 | https://commons.wikimedia.org/wiki/File:Ettelbruck_in_the_rain_with_sunshine.jpg |
| `stad`: a flea market on the Place d'Armes in Luxembourg City | Ipigott | CC0 | https://commons.wikimedia.org/wiki/File:Place_d%27Armes_Luxbg2.JPG |

These licences are about copyright. Several of the photos show people who can be recognised
(`stad`, `busarret`, `gare`, `apdikt`, `supermarche`, `schoulhaff`, and `mettegiessen` with two
children); the licences say nothing about the
rights of the people shown. That matters as soon as the app is shown beyond private use.

Two photos show lettering. In `apdikt` the monitor in front carries the logo of "elo", its
maker, and the Commons description of the file names the airman shown; the photo was kept by
the user's choice (2026-10-16). In `reen` the umbrella has a company's name printed on it.

The practice questions and the photo descriptions were written for this app, in the pattern
of INLL's published sample questions; none of INLL's questions, photos or recordings is in it.

This setup is for personal, non-commercial use. Do not redistribute the voices. For any
shared or commercial use the Luxembourgish voice must be replaced (it is a setting,
`LOCI_TTS_VOICE_LB`) and all licences checked again.

## Known limits

- No native speaker has reviewed the Luxembourgish this app produces.
- Forms are checked against LOD only. Anything LOD does not contain is labelled "not verified
  in LOD". The Luxembourgish of a chat reply is not checked at all: neither the opening phrase
  nor what a mix level switches in, so the higher the level, the more of a reply is unchecked.
- In teacher replies the model sometimes puts a Luxembourgish word inside an English
  sentence. The English voice then reads that word, with English pronunciation.
- How understandable the Luxembourgish voice is on short phrases has not been judged by a
  listener. In the build, speech recognition sometimes misread short phrases spoken by that
  voice.
- The microphone path was not used by a person during the build. Recorded files and typed
  text were.
- Speech recognition is forced to Luxembourgish, although a beginner mostly speaks English.
  In the one check made, an English sentence read by the synthetic English voice, "I went for
  a walk in the park and then cooked dinner for my family.", came back as "I went for a Walk
  in the Park and then cooked dinner fir my family." How well a real person's English through
  a microphone is recognised was not measured. Untick auto-send to correct the transcript
  before it is sent.
- Replies take a while. Measured: chat turns about 5-12 s; a teacher turn about 9 s at effort
  `low` (one turn measured) and 7-32 s with `LOCI_LLM_EFFORT_TEACHER=default`, nearly all of
  it waiting for Claude. Speech recognition and synthesis each took under a second on the GPU
  setup above. A reply that does not fit its mode or mix level is asked for once more, which
  doubles the wait: measured at mix levels 1 to 3, one reply in ten (10 turns).
- Practice mode: this app is not affiliated with INLL, and what the tutor says is not a
  prediction of a Sproochentest result. The questions, the photo descriptions and the
  corrections are not reviewed by a native speaker.
- Practice mode: the tutor misses real errors. In the build it let "e Hond" stand (it is "en
  Hond"), and when a learner described people sitting outside with drinks for the photo of a
  family meal indoors, it did not say that this does not fit the picture. A reaction without a
  correction does not mean the sentence was right.
- Practice mode: the tutor's own better version can be wrong where LOD cannot check it. LOD
  lists word forms, so the label "LOD" says that a form exists, not that it is the right word
  in that sentence: in the build the tutor wrote "an London" for a town. Word order, case and
  the choice of a preposition are not checked by anything. In the same way a correction
  labelled "Verified by LOD" says that LOD backs the change of form (your form and the better
  one are forms of one word), not that the whole better sentence is right.
- Practice mode: the questions are a fixed list, so the tutor can ask what you have just
  told it in the answer before.
- Practice mode: the tutor does not see the photo, only its written description. What you
  name that the description leaves out is neither confirmed nor corrected.
- Practice mode: in the look back the English voice reads the Luxembourgish words that Claude
  did not put in quotation marks.
- Practice mode: speech recognition can put a slip right before the tutor sees it. In the
  build the synthetic voice said "Ech wunnt zu Esch." (a wrong verb form) and the recogniser
  wrote "Ech wunnen zu Äech.": the wrong form was gone and the town was misheard. The tutor
  judges the text you confirm, so it cannot correct a mistake the transcript does not hold,
  and it may stumble over a word the recogniser got wrong. Read the transcript before you
  confirm it, and type what you really said if you want that looked at.
- Practice mode was built and tested with typed text and with the synthetic voice, not with a
  person at a microphone.
- Some of the twelve photos show people who can be recognised; their licences cover copyright only (see
  "Licences").
- With `LOCI_LLM_ENGINE=provider`, a reply whose two calls both failed is a fixed text (see
  "Providers"): it can appear in the chat, and a practice answer then gets "Got it." with no
  correction, whether the answer was right or not.
- No local server (Ollama, LM Studio, llama.cpp) and no API key was used in the build; those
  entries are tested against fake answers only. A server of `openai_compat` or `anthropic`
  that is not running shows up at the first reply, not in the banner.
- tutor-eval measures the total time of a call, not the time to the first word, because
  nothing streams. Its prompts are not reviewed by a native speaker, and its LOD-coverage
  gate is not reliable (see "tutor-eval").
- The wording of the AI notice and of the synthetic-audio comment is this build's own, not reviewed
  by anyone, and not legal advice. Whether a RIFF INFO comment is enough for Article 50(2) of
  the EU AI Act needs a lawyer.
- Not built: mobile apps, HTTPS, use from other devices, user accounts, pronunciation
  scoring, spaced repetition.

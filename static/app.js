"use strict";

// Every text from the server, the transcript or the tutor goes into the page through
// textContent or DOM nodes, never as HTML.

const MIMES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
const MIN_RECORDING_MS = 300;
const TOKEN = (document.querySelector('meta[name="loci-token"]') || {}).content || "";
const VOICE_STORE = "loci.voice";
const VOICE_LANGS = ["lb", "en"];
const VOICE_PARTS = ["rate", "pitch", "volume"];
const MIX_STORE = "loci.mix";
const TALK_STORE = "loci.talk";
const LANG_STORE = "loci.lang";
const LANGS = ["en", "fr"];
// The keys of the talk button's words in each mode: hold the button, or press it once and press it again.
const TALK_KEYS = {
  hold: {
    idle: "dock.talk_hold",
    live: "talk.live_hold",
    hint: "dock.hint_hold",
    ready: "talk.ready_hold",
    short: "talk.short_hold",
  },
  press: {
    idle: "dock.talk_toggle",
    live: "talk.live_toggle",
    hint: "dock.hint_toggle",
    ready: "talk.ready_toggle",
    short: "talk.short_toggle",
  },
};
// The practice routes, relative to the page.
const PRACTICE_TOPICS = "api/practice/topics";
const PRACTICE_START = "api/practice/start";
const PRACTICE_ANSWER = "api/practice/answer";
const PRACTICE_PHOTO = "api/practice/photo";
const PRACTICE_STOP = "api/practice/stop";
const PRACTICE_DISPUTE = "api/practice/dispute";
// Of the stop button when the talk is over and the look back on it has not arrived.
// How a talk runs, as the learner picks it; remembered in this browser.
const SETTING_STORE = "loci.practiceSetting";
const SETTINGS = ["relaxed", "realistic"];
// The clock of a realistic talk counts on past this many seconds; it never ends the talk.
const REAL_TALK_S = 600;
// The look back on a talk: a heading per sentence of the tutor, by the name the server gives it.
const LOOK_BACK_HEADINGS = [
  ["task_en", "lookback.task"],
  ["words_en", "lookback.words"],
  ["grammar_en", "lookback.grammar"],
];
// The label of a correction in words: the colour of its badge is never all that says it.
const LABEL_KEYS = { verified: "label.verified", likely: "label.likely", unverified: "label.unverified" };
// A better version that is only another way to say a good answer; it is not spoken.
// Under a labelled correction of an answer that is kept; the tap is noted on that answer, nothing is spoken.
// The Data block of the side panel: what is kept on this PC, its export and its delete.
const LEARNER_SUMMARY = "api/learner/summary";
const LEARNER_EXPORT = "api/learner/export";
const LEARNER_DELETE = "api/learner";
const EXPORT_FILE = "luxembourgish-tutor-export.zip"; // when the server names no file
// Review mode and the Progress block; the routes are relative to the page.
const REVIEW_NEXT = "api/review/next";
const REVIEW_ANSWER = "api/review/answer";
const REVIEW_RATE = "api/review/rate";
const LEARNER_PROGRESS = "api/learner/progress";
const FLAGS = "api/pronunciation/flags";
const FLAG_ITEM = "api/pronunciation/flags/";
const WORDS = "api/pronunciation/words";
const DASH = "\u2013"; // between the parts of a line of text
const MAX_WORD_LENGTH = 40; // letters of a word the server takes
const MAX_RESPELLING_LENGTH = 60;
const OVERRIDE_RATING = 1; // "I did not know it" in place of the automatic Good: Again
// The corrections per week by category, in the order of the Progress table.
const PROGRESS_COLUMNS = [
  ["words", "lookback.words"],
  ["grammar", "lookback.grammar"],
  ["task", "lookback.task"],
];

const byId = (id) => document.getElementById(id);
const banner = byId("banner");
const log = byId("log");
const statusLine = byId("status");
const playButton = byId("play");
const player = byId("player");
const talkButton = byId("talk");
const talkToggle = byId("talk-toggle");
const langSelect = byId("lang");
const talkHint = byId("talk-hint");
const transcript = byId("transcript");
const sendButton = byId("send");
const teacherButton = byId("teacher");
const autosend = byId("autosend");
const newPhrases = byId("new-phrases");
const noNew = byId("no-new");
const dataKept = byId("data-kept");
const dataWhere = byId("data-where");
const dataExport = byId("data-export");
const dataControls = byId("data-controls");
const voiceReset = byId("voice-reset");
const mixInput = byId("mix");
const mixValue = byId("mix-value");
const practiceButton = byId("practice");
const practicePanel = byId("practice-panel");
const practicePick = byId("practice-pick");
const practiceIntro = byId("practice-intro");
const practiceTopics = byId("practice-topics");
const practicePhotos = byId("practice-photos");
const practicePhoto = byId("practice-photo");
const practiceCard = byId("practice-card");
const practiceSetting = byId("practice-setting");
const practiceWhere = byId("practice-where");
const practiceClock = byId("practice-clock");
const practiceClockOver = byId("practice-clock-over");
const practiceQuestion = byId("practice-question");
const practiceEnglish = byId("practice-english");
const practiceHelp = byId("practice-help");
const practiceStop = byId("practice-stop");
const practiceLeave = byId("practice-leave");
const reviewButton = byId("review");
const reviewPanel = byId("review-panel");
const reviewCounts = byId("review-counts");
const reviewGloss = byId("review-gloss");
const reviewHint = byId("review-hint");
const reviewResult = byId("review-result");
const reviewExpected = byId("review-expected");
const reviewSaid = byId("review-said");
const reviewGood = byId("review-good");
const reviewRatings = byId("review-ratings");
const reviewOverride = byId("review-override");
const reviewNext = byId("review-next");
const reviewLeave = byId("review-leave");
const progressProblem = byId("progress-problem");
const progressFigures = byId("progress-figures");
const progressCorrections = byId("progress-corrections");
const progressTopics = byId("progress-topics");
const pronunciationBlock = byId("pronunciation");
const pronunciationProblem = byId("pronunciation-problem");
const pronunciationEmpty = byId("pronunciation-empty");
const pronunciationFlags = byId("pronunciation-flags");
const samplePlayer = new Audio();

// The interface language and its tables: {en, fr}, each null until it has loaded.
let uiLang = "en";
const tables = { en: null, fr: null };
let sessionId = null;
let busy = false; // a transcription or a turn is in flight
let pressed = false; // the talk button or the space bar is held
let acquiring = false; // the microphone permission is being asked
let stream = null;
let recorder = null;
let sttMs = 0; // of the transcript in the textarea
let spokenMs = 0; // how long the learner spoke the transcript in the textarea; 0 when it was typed
let counter = null;
let blobUrl = null;
let reply = null; // {offsets, nodes} of the reply the player holds
let stopAt = null; // where a replayed segment ends, in seconds
let speaking = false;
let afterSpeech = null; // what waits for the tutor to finish speaking; null when nothing does
let lastTimings = null;
let deleteAsking = false; // the delete control shows its question
let lastKept = null; // the last counts of the Data block
let lastProgress = null; // the last figures of the Progress block
let lastFlags = null; // the flags the server held at the last load; null until it has answered
const flagBoxes = new Map(); // flag id -> what is typed in its respelling box
let readyShown = false; // the status line says "ready" and nothing else is going on
let voiceInfo = null; // {defaults, ranges, samples} from the server; null until it has answered
let sampleUrl = null;
let mixLevels = null; // [{value, name, hint}] from the server; null until it has answered
// Practice mode, null outside it: {topic, question, picking, open, helpUsed, blob, heard, autosend}.
// `topic` and `question` are null until a topic is started; `picking` is true while the photos
// to pick from are shown; `open` is true from the start of a topic until the look back on the
// talk is shown; `blob` and `heard` are the recording behind the text in the textarea and what
// speech-to-text made of it.
let practice = null;
let topics = null; // [{id, name_lb, name_en, questions}] from the server; null until it has answered
// The clock of a realistic talk, null when none runs: {started, interval}; `started` is Date.now() at the start turn.
let clock = null;
// Review mode, null outside it; never on together with practice: {item, heard, answer, autosend}.
// `item` is the card shown ({id, gloss, hint, source, state}), null when nothing is to review;
// `heard` is what speech-to-text made of the text in the textarea; `answer` is null until the
// card is answered, then {text, heard, mode, matched, expected, logId}.
let review = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// --- interface language ---

// The text of `key` in the interface language, else in English, else the key; {name} is replaced
// from `params`.
function t(key, params) {
  const table = tables[uiLang] || tables.en || {};
  let text = typeof table[key] === "string" ? table[key] : (tables.en || {})[key];
  if (typeof text !== "string") text = key;
  return text.replace(/\{(\w+)\}/g, (whole, name) => (params && name in params ? String(params[name]) : whole));
}

// The text of a count: `key_one` or `key_other`, by the plural rules of the interface language.
function tp(key, n, params) {
  const form = new Intl.PluralRules(uiLang).select(n) === "one" ? "_one" : "_other";
  return t(key + form, { n, ...params });
}

// Sets the text and the attributes of every element of `root` that names a key.
function applyI18n(root = document) {
  for (const node of root.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  for (const node of root.querySelectorAll("[data-i18n-attr]")) {
    for (const pair of node.dataset.i18nAttr.split(";")) {
      const [name, key] = pair.split(":");
      if (name && key) node.setAttribute(name, t(key));
    }
  }
}

// A table that does not load stays null: the English in the HTML is shown.
async function loadTable(code) {
  try {
    tables[code] = await (await fetchOk("i18n/" + code + ".json")).json();
  } catch (_) {
    tables[code] = null;
  }
}

function storedLang() {
  try {
    return localStorage.getItem(LANG_STORE);
  } catch (_) {
    return null;
  }
}

function storeLang(code) {
  try {
    localStorage.setItem(LANG_STORE, code);
  } catch (_) {
    // Not saved.
  }
}

// The stored choice, else French for a browser in French, else English.
function pickLang() {
  const stored = storedLang();
  if (LANGS.includes(stored)) return stored;
  return (navigator.language || "").toLowerCase().startsWith("fr") ? "fr" : "en";
}

function setLang(code, persist) {
  uiLang = code;
  if (persist) storeLang(code);
  document.documentElement.lang = code;
  langSelect.value = code;
  applyI18n();
  refreshLabels();
}

// Writes again the texts that the script sets from its state, after the language has changed.
// The turns already in the log stay as they were written.
function refreshLabels() {
  if (!tables.en) return;
  showTalkMode();
  sendButton.textContent = sendLabel();
  transcript.placeholder = placeholderText();
  showStop();
  if (readyShown && !busy && !recorder) showReady();
  if (practice && practice.introKey) practiceIntro.textContent = t(practice.introKey);
  if (practice && practice.shown && !practiceCard.hidden) practiceWhere.textContent = whereText();
  if (review && !review.item) reviewNext.textContent = t("review.look_again");
  if (lastKept) showData(lastKept);
  if (dataControls.firstChild) showDelete(deleteAsking, false);
  if (lastProgress) showProgress(lastProgress);
  if (lastFlags) showFlags(lastFlags);
  if (voiceInfo) setVoice(null);
  if (mixLevels) showMix();
}

// --- status ---

function setStatus(text, fix, isError) {
  readyShown = false;
  statusLine.classList.toggle("error", Boolean(isError));
  statusLine.replaceChildren(element("div", "", isError ? t("status.error_prefix") + text : text));
  if (fix) statusLine.append(element("div", "fix", fix));
}

// The status "ready", with the timings of the last turn; written again when the language changes.
function showReady() {
  setStatus(withTimings(t("status.ready")));
  readyShown = true;
}

function showProblem(text, fix) {
  setStatus(text, fix, true);
}

function showError(error) {
  if (error.shown) showProblem(error.message, error.fix);
  else showProblem(error.name + ": " + error.message);
}

function seconds(ms) {
  return t("unit.seconds", { n: (ms / 1000).toFixed(1) });
}

function withTimings(word) {
  const timings = lastTimings;
  if (!timings) return word;
  return t("status.timings", {
    word,
    stt: seconds(timings.stt_ms),
    llm: seconds(timings.llm_ms),
    tts: seconds(timings.tts_ms),
    total: seconds(timings.total_ms),
  });
}

function startCounter(word) {
  stopCounter();
  const started = performance.now();
  // The word is written once; a screen reader hears it once, not with every second.
  setStatus(word);
  const count = element("span", "count");
  count.setAttribute("aria-hidden", "true");
  statusLine.firstChild.append(count);
  let shown = -1;
  const show = () => {
    const elapsed = Math.floor((performance.now() - started) / 1000);
    if (elapsed === shown) return;
    shown = elapsed;
    count.textContent = " " + t("unit.seconds", { n: elapsed });
  };
  show();
  counter = setInterval(show, 1000);
}

function stopCounter() {
  clearInterval(counter);
  counter = null;
}

function setBusy(value) {
  busy = value;
  sendButton.disabled = value;
  talkButton.disabled = value;
}

// --- server ---

function shownError(message, fix) {
  const error = new Error(message);
  error.shown = true;
  error.fix = fix || "";
  return error;
}

async function readJson(response) {
  try {
    return await response.json();
  } catch (_) {
    return null;
  }
}

// The response to a request the server served; anything else throws an error to show.
async function fetchOk(path, options) {
  let response;
  try {
    const headers = { ...((options || {}).headers || {}), "X-Loci-Token": TOKEN };
    response = await fetch(path, { ...options, headers });
  } catch (_) {
    throw shownError(t("error.unreachable"));
  }
  if (response.ok) return response;
  const body = await readJson(response);
  if (body && typeof body.error === "string") throw shownError(body.error, body.fix);
  throw shownError(t("error.http", { status: response.status }));
}

async function request(path, options) {
  const body = await readJson(await fetchOk(path, options));
  if (body === null) throw shownError(t("error.not_json"));
  return body;
}

function showHealth(health) {
  const lines = [];
  for (const [kind, engine] of Object.entries(health.engines || {})) {
    if (engine.ready) continue;
    lines.push(
      t("health.engine", {
        label: ENGINE_KEYS[kind] ? t(ENGINE_KEYS[kind]) : kind,
        name: engine.name,
        detail: engine.detail || t("health.no_detail"),
      })
    );
  }
  if (health.lod && !health.lod.ready) lines.push(t("health.no_lod"));
  banner.replaceChildren(...lines.map((line) => element("p", "", line)));
  banner.hidden = lines.length === 0;
}

function sizeLabel(bytes) {
  if (bytes < 1024 * 1024) return t("unit.kb", { n: Math.ceil(bytes / 1024) });
  return t("unit.mb", { n: (bytes / (1024 * 1024)).toFixed(1) });
}

function textButton(label, onClick) {
  const button = element("button", "", label);
  button.type = "button";
  button.addEventListener("click", onClick);
  return button;
}

// What is kept on this PC: {phrases, answers, recordings, items, bytes, folders}.
function showData(kept) {
  lastKept = kept;
  dataKept.textContent = t("data.kept", {
    phrases: kept.phrases,
    answers: kept.answers,
    recordings: kept.recordings,
    items: kept.items,
    size: sizeLabel(kept.bytes),
  });
  dataWhere.textContent = t("data.where", { folders: kept.folders.join(", ") });
}

// The delete control: the button, or the question it turns into when `asking`. The question
// opens on "No", so that a second Enter deletes nothing. The button stays when the counts fail,
// so that a learner store that does not open can still be deleted.
function showDelete(asking, focus = true) {
  deleteAsking = Boolean(asking);
  if (asking) {
    const no = textButton(t("data.no"), () => {
      showDelete(false);
      dataControls.firstChild.focus();
    });
    dataControls.replaceChildren(element("span", "", t("data.really_delete")), textButton(t("data.yes"), deleteEverything), no);
    if (focus) no.focus();
  } else {
    dataControls.replaceChildren(textButton(t("data.delete"), () => showDelete(true)));
  }
}

function showDataProblem(error) {
  dataWhere.textContent = error.fix ? error.message + " - " + error.fix : error.message;
}

// The block is a side note: a failure shows in its place and nowhere else.
async function refreshData() {
  try {
    showData(await request(LEARNER_SUMMARY));
  } catch (error) {
    dataKept.textContent = "-";
    showDataProblem(error);
  }
  if (!dataControls.firstChild) showDelete(false);
}

async function deleteEverything() {
  dataControls.replaceChildren();
  let problem = null;
  try {
    await request(LEARNER_DELETE, { method: "DELETE" });
  } catch (error) {
    problem = error;
  }
  // What is left is counted again, and the reason stays under the count.
  await refreshData();
  if (problem) showDataProblem(problem);
  refreshProgress();
  if (review) loadReviewCard();
}

// The name the server gives the file, or EXPORT_FILE.
function exportFileName(response) {
  const match = new RegExp('filename="([^"]+)"').exec(response.headers.get("Content-Disposition") || "");
  return match ? match[1] : EXPORT_FILE;
}

// Fetched rather than linked, so that the request can carry a header; then saved as a file.
async function exportData() {
  dataExport.disabled = true;
  try {
    const response = await fetchOk(LEARNER_EXPORT);
    const url = URL.createObjectURL(await response.blob());
    const link = element("a");
    link.href = url;
    link.download = exportFileName(response);
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    showDataProblem(error);
  } finally {
    dataExport.disabled = false;
  }
}

async function init() {
  uiLang = pickLang();
  await loadTable("en");
  if (uiLang !== "en") await loadTable(uiLang);
  setLang(uiLang, false);
  try {
    sessionId = (await request("/api/session", { method: "POST" })).session_id;
    showHealth(await request("/api/health"));
    await refreshData();
    await refreshProgress();
    await refreshFlags();
    await initVoice();
    await initMix();
    showReady();
  } catch (error) {
    showError(error);
  }
}

// --- recording ---

// The label of the send button and the placeholder of the text box follow the mode of the dock.
function sendLabel() {
  return t(review || practice ? "dock.confirm" : "dock.send");
}

function placeholderText() {
  if (review) return t("review.placeholder");
  if (practice) return t("practice.placeholder");
  return t("dock.placeholder");
}

// Whether the talk button is pressed once to start and once to stop, not held.
function toggleMode() {
  return talkToggle.checked;
}

// The words of the talk button's `part` (idle, live, hint, ready or short) in the mode it is in.
function talkText(part) {
  return t(TALK_KEYS[toggleMode() ? "press" : "hold"][part]);
}

// The button and the hint under it follow the mode.
function showTalkMode() {
  if (!recorder) talkButton.textContent = talkText("idle");
  talkHint.textContent = talkText("hint");
}

function storedTalk() {
  try {
    return localStorage.getItem(TALK_STORE) === "press";
  } catch (_) {
    return false;
  }
}

function storeTalk(press) {
  try {
    localStorage.setItem(TALK_STORE, press ? "press" : "hold");
  } catch (_) {
    // Not saved.
  }
}

// Press once to talk, press again to stop. While the microphone is being asked for, a press cancels.
function pressTalk() {
  if (recorder) {
    if (recorder.state === "recording") stopTalk();
  } else if (pressed) {
    stopTalk();
  } else {
    startTalk();
  }
}

function micMessage(error) {
  if (error.name === "NotAllowedError") return t("mic.blocked");
  if (error.name === "NotFoundError") return t("mic.missing");
  return error.name + ": " + error.message;
}

async function startTalk() {
  if (pressed || busy || acquiring || listening) return;
  pressed = true;
  if (stream && !stream.active) stream = null;
  if (!stream) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showProblem(t("mic.insecure"));
      return;
    }
    acquiring = true;
    setStatus(t("mic.waiting"));
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      showProblem(micMessage(error));
      return;
    } finally {
      acquiring = false;
    }
    // Answering the permission prompt ends the press that opened it.
    if (!pressed || busy) {
      setStatus(talkText("ready"));
      return;
    }
  }
  startRecording();
}

function startRecording() {
  if (typeof MediaRecorder === "undefined") {
    showProblem(t("mic.no_recorder"));
    return;
  }
  const mime = MIMES.find((type) => MediaRecorder.isTypeSupported(type));
  const chunks = [];
  const started = performance.now();
  let active;
  try {
    active = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    active.ondataavailable = (event) => {
      if (event.data.size) chunks.push(event.data);
    };
    active.onstop = () => {
      finishRecording(chunks, active.mimeType || mime || "audio/webm", performance.now() - started);
    };
    active.start();
  } catch (error) {
    showProblem(error.name + ": " + error.message);
    return;
  }
  recorder = active;
  player.pause(); // the tutor's voice must not end up in the recording
  samplePlayer.pause();
  talkButton.classList.add("recording");
  talkButton.textContent = talkText("live");
  startCounter(t("status.listening"));
}

function stopTalk() {
  pressed = false;
  if (recorder && recorder.state === "recording") recorder.stop();
}

function finishRecording(chunks, type, lengthMs) {
  recorder = null;
  talkButton.classList.remove("recording");
  talkButton.textContent = talkText("idle");
  stopCounter();
  if (lengthMs < MIN_RECORDING_MS || chunks.length === 0) {
    setStatus(talkText("short"));
    return;
  }
  transcribe(new Blob(chunks, { type }));
}

async function transcribe(blob) {
  const form = new FormData();
  form.append("audio", blob, "speech");
  let heard;
  setBusy(true);
  startCounter(t("status.transcribing"));
  try {
    heard = await request("/api/stt", { method: "POST", body: form });
  } catch (error) {
    showError(error);
    return;
  } finally {
    stopCounter();
    setBusy(false);
  }
  if (!heard.text.trim()) {
    setStatus(t("status.heard_nothing"));
    return;
  }
  transcript.value = heard.text;
  sttMs = heard.stt_ms;
  spokenMs = review ? 0 : Math.round((heard.duration_s || 0) * 1000);
  if (review) {
    review.heard = heard.text;
    setStatus(t("status.confirm", { button: t("dock.confirm") }));
    transcript.focus();
  } else if (practice) {
    practice.blob = blob;
    practice.heard = heard.text;
    setStatus(t("status.confirm", { button: t("dock.confirm") }));
    transcript.focus();
  } else if (autosend.checked) {
    send();
  } else {
    setStatus(t("status.edit_send", { button: t("dock.send") }));
    transcript.focus();
  }
}

// --- a turn ---

function setTeacher(on) {
  teacherButton.setAttribute("aria-pressed", String(on));
}

async function send() {
  if (listening) return;
  if (review) {
    sendReviewAnswer();
    return;
  }
  if (practice) {
    sendAnswer().then(refreshProgress);
    return;
  }
  const text = transcript.value.trim();
  if (busy || !text) return;
  const mode = teacherButton.getAttribute("aria-pressed") === "true" ? "teacher" : "auto";
  let turn;
  setBusy(true);
  startCounter(t(mode === "teacher" ? "status.thinking_teacher" : "status.thinking"));
  try {
    if (!sessionId) sessionId = (await request("/api/session", { method: "POST" })).session_id;
    const form = new FormData();
    form.append("session_id", sessionId);
    form.append("text", text);
    form.append("mode", mode);
    form.append("stt_ms", String(sttMs));
    form.append("duration_ms", String(spokenMs));
    const voice = voiceValues();
    if (voice) form.append("voice", JSON.stringify(voice));
    if (mixLevels) form.append("mix", mixInput.value);
    turn = await request("/api/turn", { method: "POST", body: form });
  } catch (error) {
    showError(error);
    return;
  } finally {
    stopCounter();
    setBusy(false);
  }
  if (!turn.heard) {
    setStatus(t("status.heard_nothing"));
    return;
  }
  transcript.value = "";
  sttMs = 0;
  spokenMs = 0;
  setTeacher(false);
  lastTimings = turn.timings;
  renderTurn(turn);
  addPhrases(turn);
  // The counts are a side note: a failure shows in the Data block, not over the reply's status.
  refreshData();
  loadAudio(turn);
  showReady();
  play(null, null);
}

// The parts of the reply before this one can no longer be played again.
function retireReply() {
  if (!reply) return;
  for (const node of reply.nodes) {
    // A look back has segments that are spoken and not shown.
    if (!node) continue;
    node.classList.remove("playable", "speaking");
    node.removeAttribute("role");
    node.removeAttribute("tabindex");
    // What stays is the gloss, when the segment has one.
    if (node.dataset.gloss) node.title = node.dataset.gloss;
    else node.removeAttribute("title");
  }
}

function userLine(text) {
  const user = element("p", "user");
  user.append(element("span", "who", t("log.you")), text);
  return user;
}

// The node of segment `index` of a reply: a click on it plays the segment again.
function segmentNode(segment, index, mixed) {
  // In a mixed chat reply the Luxembourgish after the opening phrase reads as part of the
  // sentence: it is marked by its style and has no tag.
  const switched = mixed && index > 0 && segment.lang === "lb";
  const node = element("span", "segment playable " + (switched ? "switched" : segment.lang === "lb" ? "lb" : "en"));
  node.lang = segment.lang;
  if (segment.lang === "lb" && !switched) node.append(element("span", "tag", t("log.lb_tag")));
  node.append(segment.text);
  node.setAttribute("role", "button");
  node.setAttribute("tabindex", "0");
  node.title = t("log.play_again");
  if (segment.gloss) {
    node.classList.add("glossed");
    node.dataset.gloss = segment.gloss;
    node.title = t("log.gloss_play", { gloss: segment.gloss });
  }
  node.addEventListener("click", () => replaySegment(node, index));
  node.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    replaySegment(node, index);
  });
  return node;
}

// Puts the nodes of the segments `indexes` of a reply at the end of `line`, and into `nodes`
// at their index.
function appendSegments(line, segments, indexes, nodes, mixed) {
  indexes.forEach((index, place) => {
    const segment = segments[index];
    // No space in front of a part that begins with the punctuation of the part before it.
    if (place > 0 && !/^[,.;:!?]/.test(segment.text)) line.append(" ");
    nodes[index] = segmentNode(segment, index, mixed);
    line.append(nodes[index]);
  });
}

// The tutor's line of a turn: {line, nodes}, with one node per segment that plays it again.
function tutorLine(label, segments, mixed) {
  const tutor = element("p", "tutor");
  tutor.append(element("span", "who", label));
  const nodes = [];
  appendSegments(tutor, segments, segments.map((_, index) => index), nodes, mixed);
  appendFlagControl(tutor, segments);
  return { line: tutor, nodes };
}

// The word forms of a reply, each with the label that says whether LOD lists it.
function claimList(claims) {
  const list = element("ul", "claims");
  for (const claim of claims) {
    const item = element("li");
    const form = element("span", "", claim.form);
    form.lang = "lb";
    item.append(form);
    if (claim.lemma) {
      const lemma = element("span", "", claim.lemma);
      lemma.lang = "lb";
      item.append(" (", lemma, ")");
    }
    item.append(
      claim.verified
        ? element("span", "tag verified", t("log.lod"))
        : element("span", "tag unverified", t("log.not_verified"))
    );
    list.append(item);
  }
  return list;
}

// The label of a practice correction in words, its category and where the answer came from;
// null when the correction has no label. The badge's tooltip lists what the label rests on.
function labelLine(correction) {
  const key = LABEL_KEYS[correction.label];
  if (!key) return null;
  const line = element("p", "label-line");
  const badge = element("span", "label label-" + correction.label, t(key));
  if (correction.checks.length) badge.title = correction.checks.join("; ");
  line.append(badge);
  const heading = LOOK_BACK_HEADINGS.find(([name]) => name === correction.category + "_en");
  if (heading) line.append(" " + t(heading[1]));
  if (correction.source === "heard-unedited") line.append(" - " + t("label.heard_unedited"));
  return line;
}

// The indexes of the segments that say the correction of a practice turn: from the line that
// leads into the learner's words (or into the Luxembourgish of an English answer) to the
// better version.
function correctionIndexes(segments, correction) {
  const lb = (text) => (segment, index) => segment.lang === "lb" && segment.text === text && index > 0;
  const first = correction.verdict === "fix" ? correction.quote : correction.better_lb;
  const start = segments.findIndex(lb(first));
  if (start < 1) return [];
  const end = segments.findIndex((segment, index) => index >= start && lb(correction.better_lb)(segment, index));
  if (end < start) return [];
  return segments.map((_, index) => index).filter((index) => index >= start - 1 && index <= end);
}

function renderTurn(turn) {
  retireReply();
  const article = element("article", "turn");
  article.append(userLine(turn.transcript));
  const mixed = turn.mode === "chat" && turn.mix > 0;
  const { line, nodes } = tutorLine(
    t(turn.mode === "teacher" ? "log.tutor_teacher" : "log.tutor"),
    turn.segments,
    mixed
  );
  article.append(line);

  if (turn.mode === "teacher" && turn.claims.length) article.append(claimList(turn.claims));
  if (turn.shape_warning) article.append(element("p", "warning", t("log.format_note", { note: turn.shape_warning })));

  log.append(article);
  article.scrollIntoView({ block: "nearest" });
  reply = { offsets: turn.segment_offsets, nodes };
}

function addPhrases(turn) {
  for (const lb of turn.phrases_added) {
    const phrase = turn.new_phrases.find((entry) => entry.lb === lb);
    const item = element("li");
    const strong = element("strong", "", lb);
    strong.lang = "lb";
    item.append(strong);
    if (phrase && phrase.en) {
      const english = element("span", "", phrase.en);
      english.lang = "en";
      item.append(" - ", english);
    }
    newPhrases.append(item);
  }
  noNew.hidden = newPhrases.children.length > 0;
}

// --- practice ---

async function enterPractice() {
  if (practice || busy) return;
  practice = {
    topic: null, question: null, picking: false, open: false, helpUsed: false, blob: null, heard: "",
    autosend: autosend.checked, introKey: null, shown: null,
  };
  // In practice the text always waits in the box until the learner confirms it.
  autosend.checked = false;
  autosend.disabled = true;
  setTeacher(false);
  teacherButton.disabled = true;
  practiceButton.setAttribute("aria-pressed", "true");
  sendButton.textContent = sendLabel();
  transcript.placeholder = placeholderText();
  showTopics("practice.pick_topic");
  practicePanel.hidden = false;
  if (topics) return;
  try {
    const listed = (await request(PRACTICE_TOPICS)).topics;
    topics = listed;
    if (practice) showTopics("practice.pick_topic");
  } catch (error) {
    showError(error);
  }
}

function leavePractice() {
  if (!practice) return;
  stopClock();
  autosend.checked = practice.autosend;
  autosend.disabled = false;
  teacherButton.disabled = false;
  practice = null;
  practiceButton.setAttribute("aria-pressed", "false");
  sendButton.textContent = sendLabel();
  transcript.placeholder = placeholderText();
  practicePanel.hidden = true;
}

function showTopics(introKey) {
  practice.introKey = introKey;
  practiceIntro.textContent = t(introKey);
  practiceTopics.replaceChildren(
    ...(topics || []).map((topic) => {
      const button = element("button");
      button.type = "button";
      const name = element("strong", "", topic.name_lb);
      name.lang = "lb";
      const english = element("span", "muted", topic.name_en);
      english.lang = "en";
      button.append(name, english);
      button.addEventListener("click", () => startTopic(topic));
      return button;
    })
  );
  practicePhotos.replaceChildren();
  showPhoto(null);
  practiceSetting.hidden = false;
  practiceCard.hidden = true;
  practicePick.hidden = false;
  showStop();
}

// The setting picked on the page: "relaxed" or "realistic".
function pickedSetting() {
  const checked = practiceSetting.querySelector('input[name="setting"]:checked');
  return checked ? checked.value : SETTINGS[0];
}

// The browser may refuse localStorage; the setting then lasts until the page is closed.
function storedSetting() {
  try {
    return localStorage.getItem(SETTING_STORE);
  } catch (_) {
    return null;
  }
}

function storeSetting(value) {
  try {
    localStorage.setItem(SETTING_STORE, value);
  } catch (_) {
    // Not saved.
  }
}

function initSetting() {
  const stored = storedSetting();
  const inputs = practiceSetting.querySelectorAll('input[name="setting"]');
  for (const input of inputs) {
    if (SETTINGS.includes(stored)) input.checked = input.value === stored;
    input.addEventListener("change", () => storeSetting(pickedSetting()));
  }
}

// Counts up from 0:00 at the start turn of a realistic talk; past ten minutes a line says the
// real talk would be over, and the talk goes on as before.
function startClock() {
  stopClock();
  clock = { started: Date.now(), interval: setInterval(showClock, 1000) };
  showClock();
}

function showClock() {
  if (!clock) return;
  const seconds = Math.floor((Date.now() - clock.started) / 1000);
  practiceClock.textContent = Math.floor(seconds / 60) + ":" + String(seconds % 60).padStart(2, "0");
  practiceClock.hidden = false;
  practiceClockOver.hidden = seconds < REAL_TALK_S;
}

function stopClock() {
  if (clock) clearInterval(clock.interval);
  clock = null;
  practiceClock.hidden = true;
  practiceClockOver.hidden = true;
}

// The stop button is there while the server holds a talk it has not looked back on.
function showStop() {
  practiceStop.hidden = !practice || !practice.open;
  practiceStop.textContent = t(practice && (practice.question || practice.picking) ? "practice.stop" : "practice.look_back");
}

// The image of `photo`: {id, url, alt, caption}. Its `alt` says what it shows.
function photoImage(photo) {
  const image = element("img");
  image.src = photo.url;
  image.alt = photo.alt;
  return image;
}

// The photos to pick from, each a button with its author and licence under it.
function showPhotos(photos) {
  practice.introKey = "practice.pick_photo";
  practiceIntro.textContent = t(practice.introKey);
  practiceSetting.hidden = true;
  practiceTopics.replaceChildren();
  practicePhotos.replaceChildren(
    ...photos.map((photo) => {
      const button = element("button");
      button.type = "button";
      button.append(photoImage(photo), element("span", "muted", photo.caption));
      button.addEventListener("click", () => choosePhoto(photo));
      return button;
    })
  );
  showPhoto(null);
  practiceCard.hidden = true;
  practicePick.hidden = false;
}

// The picked photo beside the question card; null takes it away.
function showPhoto(photo) {
  practicePhoto.hidden = !photo;
  if (!photo) practicePhoto.replaceChildren();
  // The same photo stays as it is, so that it does not flicker between its questions.
  else if (practicePhoto.dataset.photo !== photo.id) {
    practicePhoto.replaceChildren(photoImage(photo), element("figcaption", "muted", photo.caption));
  }
  practicePhoto.dataset.photo = photo ? photo.id : "";
}

// The line over the question: what it belongs to, and its place among the questions of the part.
function whereText() {
  const { question, about } = practice.shown;
  return t("practice.where", { about: about(), n: question.index + 1, total: question.total });
}

// `about` is a function that names what the question belongs to: the topic, or the picture.
function showQuestion(question, english, about) {
  practice.shown = { question, about };
  practiceWhere.textContent = whereText();
  practiceQuestion.textContent = question.lb;
  practiceEnglish.textContent = question.en;
  practiceEnglish.hidden = !english;
  practiceHelp.hidden = english;
  practicePick.hidden = true;
  practiceCard.hidden = false;
}

function showEnglish() {
  if (!practice || !practice.question) return;
  practice.helpUsed = true;
  practiceEnglish.hidden = false;
  practiceHelp.hidden = true;
}

function practiceForm() {
  const form = new FormData();
  form.append("session_id", sessionId);
  const voice = voiceValues();
  if (voice) form.append("voice", JSON.stringify(voice));
  return form;
}

// Posts `fill(form)` to a practice route; what the route answers, or null when it failed.
async function practicePost(path, word, fill) {
  setBusy(true);
  startCounter(word);
  try {
    if (!sessionId) sessionId = (await request("/api/session", { method: "POST" })).session_id;
    const form = practiceForm();
    fill(form);
    return await request(path, { method: "POST", body: form });
  } catch (error) {
    showError(error);
    return null;
  } finally {
    stopCounter();
    setBusy(false);
  }
}

// Posts `fill(form)` to a practice route; the tutor's lines, or null when it failed or practice was left.
async function practiceRequest(path, word, fill) {
  const turn = await practicePost(path, word, fill);
  return practice ? turn : null;
}

async function startTopic(topic) {
  if (busy || !practice) return;
  const setting = pickedSetting();
  const turn = await practiceRequest(PRACTICE_START, t("status.starting_talk"), (form) => {
    form.append("topic_id", topic.id);
    form.append("setting", setting);
  });
  if (!turn) return;
  if (setting === "realistic") startClock();
  else stopClock();
  practice.topic = topic;
  practice.open = true;
  transcript.value = "";
  showPracticeTurn(turn, null);
}

async function sendAnswer() {
  const text = transcript.value.trim();
  if (busy || !text) return;
  if (!practice.question) {
    setStatus(t(practice.picking ? "status.pick_picture" : "status.pick_topic"));
    return;
  }
  const talk = practice;
  const spoken = spokenMs;
  const turn = await practiceRequest(PRACTICE_ANSWER, t("status.one_moment"), (form) => {
    form.append("text", text);
    form.append("raw_text", talk.heard);
    form.append("help_used", String(talk.helpUsed));
    form.append("duration_ms", String(spoken));
    if (talk.blob) form.append("audio", talk.blob, "speech");
  });
  if (!turn) return;
  transcript.value = "";
  showPracticeTurn(turn, text);
}

async function choosePhoto(photo) {
  if (busy || !practice) return;
  const turn = await practiceRequest(PRACTICE_PHOTO, t("status.one_moment"), (form) => form.append("photo_id", photo.id));
  if (turn) showPracticeTurn(turn, null);
}

// The button that notes on the kept answer at `kept` ({folder, line}) that the learner thinks
// it was right; once noted, the words that say so take its place. A failure keeps the button.
function disputeLine(kept) {
  const line = element("p", "dispute-line");
  const button = textButton(t("dispute.label"), async () => {
    button.disabled = true;
    try {
      const form = practiceForm();
      form.append("folder", kept.folder);
      form.append("line", String(kept.line));
      await request(PRACTICE_DISPUTE, { method: "POST", body: form });
      line.classList.add("muted");
      line.replaceChildren(t("dispute.noted"));
    } catch (error) {
      showError(error);
      button.disabled = false;
    }
  });
  button.classList.add("dispute");
  line.append(button);
  return line;
}

// The tutor's lines in the log and aloud; `said` is the answer they follow, null for the opening
// and for the first question about a photo.
function showPracticeTurn(turn, said) {
  // A look back that waits for the tutor to finish is shown first: these lines are spoken over it.
  runAfterSpeech();
  const before = practice.question;
  const ended = !turn.question && turn.kind !== "choose";
  practice.question = turn.question;
  practice.picking = turn.kind === "choose";
  practice.blob = null;
  practice.heard = "";
  sttMs = 0;
  spokenMs = 0;
  lastTimings = null;
  if (practice.picking) {
    showPhotos(turn.photos);
  } else if (ended) {
    practice.topic = null;
    showTopics("practice.pick_another");
  } else {
    // After a skip the tutor has said the question in English; it stays open on the same question.
    const same = before && before.id === turn.question.id;
    const english = turn.kind === "again" || Boolean(same && practice.helpUsed);
    practice.helpUsed = english;
    const topic = practice.topic;
    let about = () => t("practice.topic_label", { lb: topic.name_lb, en: topic.name_en });
    if (turn.photo) about = () => t("practice.about_photo");
    else if (turn.part === "warmup") about = () => t("practice.warmup");
    showPhoto(turn.photo);
    showQuestion(turn.question, english, about);
  }
  showStop();
  retireReply();
  const article = element("article", "turn");
  if (said !== null) article.append(userLine(said));
  const { line, nodes } = tutorLine(t("log.tutor"), turn.segments, false);
  article.append(line);
  const correction = turn.correction;
  if (correction) {
    const labelled = labelLine(correction);
    // An unverified correction is still said, and shown grey: its label says it may be wrong.
    const doubtful = correction.label === "unverified";
    if (doubtful) {
      for (const index of correctionIndexes(turn.segments, correction)) nodes[index].classList.add("doubtful");
    }
    if (labelled) article.append(labelled);
    // The words of the tutor's better version, labelled as a teacher reply labels its forms.
    if (correction.claims.length) {
      const claims = claimList(correction.claims);
      if (doubtful) claims.classList.add("doubtful");
      article.append(claims);
    }
    if (correction.alternative_lb) {
      const also = element("p", "muted also", t("practice.also_possible"));
      also.append(shownLb(correction.alternative_lb));
      if (correction.checks.length) also.title = correction.checks.join("; ");
      article.append(also);
    }
    // Only a fix or an english has a label; only a kept answer can be disputed.
    if (labelled && turn.kept) article.append(disputeLine(turn.kept));
  }
  if (turn.feedback_error) article.append(element("p", "warning", t("practice.no_feedback", { detail: turn.feedback_error })));
  if (turn.saved) refreshData();
  if (turn.save_error) article.append(element("p", "warning", t("practice.saved_problem", { detail: turn.save_error })));
  log.append(article);
  article.scrollIntoView({ block: "nearest" });
  reply = { offsets: turn.segment_offsets, nodes };
  loadAudio(turn);
  showReady();
  play(null, null);
  // A talk that ran to its end is looked back on while the tutor says goodbye.
  if (ended) lookBack();
}

// Asks the server to end the talk where it stands and to look back on it. A talk that ran to
// its end has had its goodbye: the look back on it waits until that is spoken.
async function lookBack() {
  if (busy || !practice || !practice.open) return;
  const stopping = Boolean(practice.question || practice.picking);
  practiceStop.hidden = true;
  const turn = await practicePost(PRACTICE_STOP, t("status.looking_back"), () => {});
  if (!turn) {
    // The button asks again.
    showStop();
    return;
  }
  if (stopped()) showReady();
  else setStatus(t("status.speaking"));
  stopClock();
  if (practice) {
    practice.open = false;
    practice.topic = null;
    practice.question = null;
    practice.picking = false;
    practice.blob = null;
    practice.heard = "";
    transcript.value = "";
    spokenMs = 0;
    showTopics(stopping ? "practice.pick_after_stop" : "practice.pick_another");
  }
  if (stopping) showLookBack(turn);
  else whenQuiet(() => showLookBack(turn));
}

// A line of the look back that only shows a Luxembourgish text: it is not spoken.
function shownLb(text) {
  const node = element("span", "segment lb");
  node.lang = "lb";
  node.append(element("span", "tag", t("log.lb_tag")), text);
  return node;
}

// One thing the tutor put right: the question it belongs to, the learner's words, the better
// version with its LOD labels. `index` is the segment that says the better version again, or
// undefined when the look back does not say it.
function takeHome(correction, segments, index, nodes) {
  const item = element("div", "takehome");
  const about = element("p", "muted", t("lookback.question", { n: correction.question_number }));
  const asked = element("span", "", correction.question_lb);
  asked.lang = "lb";
  about.append(asked);
  const line = element("p");
  if (correction.verdict === "fix") {
    const why = correction.why_en.replace(/[\s.,;:]+$/, "");
    line.append(t("lookback.you_said"), shownLb(correction.quote), why ? " - " + why + ": " : " - ");
  } else {
    line.append(t("lookback.you_said_answer", { answer: correction.answer }));
  }
  if (index === undefined) line.append(shownLb(correction.better_lb));
  else appendSegments(line, segments, [index], nodes, false);
  item.append(about, line);
  const labelled = labelLine(correction);
  if (labelled) item.append(labelled);
  if (correction.claims.length) item.append(claimList(correction.claims));
  if (correction.label === "unverified") {
    line.classList.add("doubtful");
    for (const claims of item.querySelectorAll(".claims")) claims.classList.add("doubtful");
  }
  return item;
}

// The look back on a talk in the log and aloud: the tutor's sentences under their headings,
// everything the tutor put right, and what code knows of the talk. It stays in the log.
function showLookBack(turn) {
  const wrapup = turn.wrapup;
  const segments = turn.segments;
  const withRole = (role) => segments.map((_, index) => index).filter((index) => segments[index].role === role);
  const nodes = [];
  const article = element("article", "turn lookback");
  if (wrapup.answered) {
    const arrived = LOOK_BACK_HEADINGS.filter(([name]) => wrapup.sentences[name]);
    if (!arrived.length) article.append(element("p", "warning", t("lookback.no_comment")));
    for (const [name, headingKey] of arrived.length ? LOOK_BACK_HEADINGS : []) {
      article.append(element("h3", "", t(headingKey)));
      const line = element("p", wrapup.sentences[name] ? "" : "warning");
      if (wrapup.sentences[name]) {
        appendSegments(line, segments, withRole(name), nodes, false);
        appendFlagControl(line, withRole(name).map((index) => segments[index]));
      } else line.textContent = t("lookback.no_comment");
      article.append(line);
    }
    if (wrapup.corrections.length) article.append(element("h3", "", t("lookback.take_home")));
    const again = withRole("better");
    wrapup.corrections.forEach((correction, place) => {
      article.append(takeHome(correction, segments, again[place], nodes));
    });
    if (wrapup.help_questions.length) {
      article.append(element("p", "aside", t("lookback.help_questions", { list: wrapup.help_questions.join(", ") })));
    }
    if (wrapup.unknown_words.length) {
      article.append(element("p", "aside", t("lookback.unknown_words", { list: wrapup.unknown_words.join(", ") })));
    }
  }
  const closing = [...withRole("line"), ...withRole("end")];
  if (closing.length) {
    const tutor = element("p", "tutor");
    tutor.append(element("span", "who", t("log.tutor")));
    appendSegments(tutor, segments, closing, nodes, false);
    appendFlagControl(tutor, closing.map((index) => segments[index]));
    article.append(tutor);
  }
  retireReply();
  log.append(article);
  // From its first line: it is read from the top, and the dock covers what is nearest to it.
  article.scrollIntoView({ block: "start" });
  reply = { offsets: turn.segment_offsets, nodes };
  if (!segments.length) return;
  loadAudio(turn);
  // Nothing is played into a recording; the Play button plays it later.
  if (recorder) playButton.hidden = false;
  else play(null, null);
}

// --- audio ---

function loadAudio(turn) {
  const binary = atob(turn.audio_wav_base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  if (blobUrl) URL.revokeObjectURL(blobUrl);
  blobUrl = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
  player.src = blobUrl;
}

function play(start, end) {
  samplePlayer.pause();
  stopAt = end;
  if (start !== null) player.currentTime = start;
  playButton.hidden = true;
  player.play().catch((error) => {
    if (error.name === "NotAllowedError") {
      // The autoplay policy refused: the reader starts the audio with a click.
      playButton.hidden = false;
      setStatus(withTimings(t("status.ready_play")));
    } else if (error.name !== "AbortError") {
      showProblem(error.name + ": " + error.message);
    }
  });
}

function replaySegment(node, index) {
  if (!reply || reply.nodes[index] !== node || recorder) return;
  const offset = reply.offsets.find((entry) => entry.index === index);
  if (offset && offset.end_s > offset.start_s) play(offset.start_s, offset.end_s);
}

function stopped() {
  return player.paused || player.ended;
}

function tick() {
  if (!reply) return;
  const time = player.currentTime;
  if (stopAt !== null && time >= stopAt) {
    stopAt = null;
    player.pause();
  }
  for (const offset of reply.offsets) {
    const node = reply.nodes[offset.index];
    if (node) node.classList.toggle("speaking", !stopped() && time >= offset.start_s && time < offset.end_s);
  }
}

function frame() {
  tick();
  if (!stopped()) requestAnimationFrame(frame);
}

function playbackStarted() {
  speaking = true;
  if (!busy && !recorder) setStatus(withTimings(t("status.speaking")));
  requestAnimationFrame(frame);
}

function playbackStopped() {
  tick();
  if (speaking && !busy && !recorder) showReady();
  speaking = false;
  runAfterSpeech();
}

// Runs `run` once the tutor has finished speaking: at once when nothing plays.
function whenQuiet(run) {
  if (stopped()) run();
  else afterSpeech = run;
}

// Runs at once what waits for the tutor to finish speaking.
function runAfterSpeech() {
  const run = afterSpeech;
  afterSpeech = null;
  if (run) run();
}

// --- voice ---

function voiceInput(lang, part) {
  return byId("voice-" + lang + "-" + part);
}

function voiceLabel(part, value) {
  if (part === "pitch") return t("voice.semitones", { n: (value > 0 ? "+" : "") + value.toFixed(1) });
  if (part === "volume") return t("voice.percent", { n: Math.round(value * 100) });
  return t("voice.times", { n: value.toFixed(2) });
}

function showVoice(lang, part) {
  const value = Number(voiceInput(lang, part).value);
  byId("voice-" + lang + "-" + part + "-value").textContent = voiceLabel(part, value);
}

// Puts the numbers of `values` on the sliders; a slider clamps what is outside its range.
function setVoice(values) {
  for (const lang of VOICE_LANGS) {
    for (const part of VOICE_PARTS) {
      const value = values && values[lang] ? values[lang][part] : null;
      if (typeof value === "number" && Number.isFinite(value)) voiceInput(lang, part).value = String(value);
      showVoice(lang, part);
    }
  }
}

function voiceValues() {
  if (!voiceInfo) return null;
  const values = {};
  for (const lang of VOICE_LANGS) {
    values[lang] = {};
    for (const part of VOICE_PARTS) values[lang][part] = Number(voiceInput(lang, part).value);
  }
  return values;
}

// The browser may refuse localStorage; the sliders then last until the page is closed.
function storedVoice() {
  try {
    return JSON.parse(localStorage.getItem(VOICE_STORE));
  } catch (_) {
    return null;
  }
}

function storeVoice(values) {
  try {
    if (values) localStorage.setItem(VOICE_STORE, JSON.stringify(values));
    else localStorage.removeItem(VOICE_STORE);
  } catch (_) {
    // Not saved.
  }
}

async function initVoice() {
  const info = await request("/api/voice");
  for (const lang of VOICE_LANGS) {
    for (const part of VOICE_PARTS) {
      const input = voiceInput(lang, part);
      input.min = String(info.ranges[part][0]);
      input.max = String(info.ranges[part][1]);
      input.disabled = false;
    }
    byId("voice-" + lang + "-sample").disabled = false;
  }
  setVoice(info.defaults);
  setVoice(storedVoice());
  voiceReset.disabled = false;
  voiceInfo = info;
}

// The wav of `text` in `lang`, spoken with the voice of the sliders. `respell`, {word: respelling},
// is tried for this one request and kept nowhere.
async function spokenBlob(lang, text, respell) {
  const response = await fetchOk("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ segments: [{ lang, text }], voice: voiceValues(), respell }),
  });
  return response.blob();
}

// Plays `blob` on the player of the samples; the tutor's voice stops for it.
async function playOnSample(blob) {
  if (sampleUrl) URL.revokeObjectURL(sampleUrl);
  sampleUrl = URL.createObjectURL(blob);
  player.pause();
  samplePlayer.src = sampleUrl;
  await samplePlayer.play();
}

async function playSample(lang) {
  if (!voiceInfo || recorder) return;
  const button = byId("voice-" + lang + "-sample");
  button.disabled = true;
  try {
    const blob = await spokenBlob(lang, voiceInfo.samples[lang]);
    if (recorder) return;
    await playOnSample(blob);
  } catch (error) {
    if (error.name !== "AbortError") showError(error);
  } finally {
    button.disabled = false;
  }
}

// --- pronunciation flags ---

// The words of `text` the voice can be told to say otherwise: runs of two or more letters.
function wordsOf(text) {
  return text.match(/\p{L}{2,}/gu) || [];
}

// What makes two flags the same: the voice that speaks the segment (Luxembourgish, else English)
// and the word in lower case.
function flagKey(lang, word) {
  return (lang === "lb" ? "lb" : "en") + ":" + word.toLowerCase();
}

function flaggedKeys() {
  return new Set((lastFlags || []).map((flag) => flagKey(flag.lang, flag.word)));
}

// {lang, word, text} for each word of `segments` once, in the order they are said.
function flaggableWords(segments) {
  const seen = new Set();
  const words = [];
  for (const segment of segments) {
    for (const word of wordsOf(segment.text)) {
      const key = flagKey(segment.lang, word);
      if (word.length > MAX_WORD_LENGTH || seen.has(key)) continue;
      seen.add(key);
      words.push({ lang: segment.lang, word, text: segment.text });
    }
  }
  return words;
}

function wordButton({ lang, word, text }) {
  const button = textButton(word, () => flagWord(lang, word, text, button));
  button.lang = lang;
  button.dataset.key = flagKey(lang, word);
  button.setAttribute("aria-pressed", String(flaggedKeys().has(button.dataset.key)));
  return button;
}

// After the segments of a tutor line, never inside one: a button that opens a row of its words.
function appendFlagControl(line, segments) {
  const words = flaggableWords(segments);
  if (!words.length) return;
  const toggle = element("button", "flag", t("flag.button"));
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", "false");
  const row = element("div", "row");
  row.classList.add("flag-words");
  row.setAttribute("role", "group");
  row.setAttribute("aria-label", t("flag.words_label"));
  row.hidden = true;
  row.append(...words.map(wordButton));
  toggle.addEventListener("click", () => {
    row.hidden = !row.hidden;
    toggle.setAttribute("aria-expanded", String(!row.hidden));
  });
  line.append(" ", toggle, row);
}

async function flagWord(lang, word, text, button) {
  button.disabled = true;
  try {
    const flag = await request(FLAGS, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lang, word, text }),
    });
    setStatus(t("flag.flagged", { word: flag.word }));
    await refreshFlags();
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
  }
}

// Shows the word buttons in the log as flagged or not, after the flags have changed.
function showFlaggedWords() {
  const keys = flaggedKeys();
  for (const button of log.querySelectorAll("button[data-key]")) {
    button.setAttribute("aria-pressed", String(keys.has(button.dataset.key)));
  }
}

async function playFlag(flag, box, button) {
  if (recorder) return;
  button.disabled = true;
  try {
    const respelling = box.value.trim();
    const blob = await spokenBlob(flag.lang, flag.text, respelling ? { [flag.word]: respelling } : undefined);
    if (recorder) return;
    await playOnSample(blob);
  } catch (error) {
    if (error.name !== "AbortError") showError(error);
  } finally {
    button.disabled = false;
  }
}

// Saving and dismissing take the flag out of the list, and the focus with its buttons.
async function settleFlag(flag, button, send, doneKey) {
  button.disabled = true;
  try {
    await send();
    setStatus(t(doneKey, { word: flag.word }));
    flagBoxes.delete(flag.id);
    await refreshFlags();
    pronunciationBlock.querySelector("summary").focus();
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
  }
}

function saveFlag(flag, box, button) {
  const respelling = box.value.trim();
  if (!respelling) {
    showProblem(t("flag.need_respelling"));
    box.focus();
    return;
  }
  const body = JSON.stringify({ lang: flag.lang, word: flag.word, respelling });
  const send = () => request(WORDS, { method: "POST", headers: { "Content-Type": "application/json" }, body });
  settleFlag(flag, button, send, "flag.saved");
}

function dismissFlag(flag, button) {
  settleFlag(flag, button, () => request(FLAG_ITEM + flag.id, { method: "DELETE" }), "flag.dismissed");
}

// One flag of the Pronunciation block: the word, its phonemes, the line, the respelling box and
// the buttons that try, keep or drop it.
function flagRow(flag) {
  const item = element("li", "flagged");
  const word = element("strong", "", flag.word);
  word.lang = flag.lang;
  const language = t(flag.lang === "lb" ? "flag.lang_lb" : "flag.lang_en");
  const head = element("p", "head");
  head.append(word, " ", element("span", "muted", [language, DASH, tp("flag.times", flag.count)].join(" ")));

  const phonemes = element("p", "", t("flag.phonemes") + " ");
  const sounds = element("span", "phonemes", flag.phonemes || t("flag.no_phonemes"));
  if (flag.phonemes) sounds.translate = false;
  phonemes.append(sounds);

  const line = element("p", "line", flag.text);
  line.lang = flag.lang;
  item.append(head, phonemes, line);
  if (flag.respelling) {
    const current = element("p", "", t("flag.in_table") + " ");
    const respelled = element("span", "phonemes", flag.respelling);
    respelled.translate = false;
    current.append(respelled);
    item.append(current);
  }

  const label = element("label", "respelling", t("flag.respelling") + " ");
  const box = element("input");
  box.type = "text";
  box.maxLength = MAX_RESPELLING_LENGTH;
  box.autocomplete = "off";
  box.spellcheck = false;
  box.translate = false;
  box.value = flagBoxes.get(flag.id) || "";
  box.addEventListener("input", () => flagBoxes.set(flag.id, box.value));
  label.append(box);

  const buttons = element("div", "row");
  const play = textButton(t("flag.play"), () => playFlag(flag, box, play));
  const save = textButton(t("flag.save"), () => saveFlag(flag, box, save));
  const dismiss = textButton(t("flag.dismiss"), () => dismissFlag(flag, dismiss));
  buttons.append(play, save, dismiss);
  item.append(label, buttons);
  return item;
}

// The flags from the server, the latest first.
function showFlags(flags) {
  lastFlags = flags;
  pronunciationProblem.hidden = true;
  pronunciationEmpty.hidden = flags.length > 0;
  const ids = new Set(flags.map((flag) => flag.id));
  for (const id of flagBoxes.keys()) if (!ids.has(id)) flagBoxes.delete(id);
  pronunciationFlags.replaceChildren(...flags.map(flagRow));
  showFlaggedWords();
}

// The block is a side note: a failure shows inside it and nowhere else.
async function refreshFlags() {
  try {
    showFlags((await request(FLAGS)).flags);
  } catch (error) {
    pronunciationProblem.textContent = error.fix ? error.message + " - " + error.fix : error.message;
    pronunciationProblem.hidden = false;
  }
}

// --- Luxembourgish mix ---

function showMix() {
  const level = mixLevels.find((entry) => entry.value === Number(mixInput.value));
  mixValue.textContent = level ? level.name : "";
  mixValue.lang = "en";
  // On the label, so that the slider keeps "Luxembourgish mix" as its accessible name.
  mixInput.closest("label").title = level ? level.hint : "";
}

// The browser may refuse localStorage; the level then lasts until the page is closed.
function storedMix() {
  try {
    return localStorage.getItem(MIX_STORE);
  } catch (_) {
    return null;
  }
}

function storeMix(value) {
  try {
    localStorage.setItem(MIX_STORE, value);
  } catch (_) {
    // Not saved.
  }
}

async function initMix() {
  const info = await request("/api/mix");
  const values = info.levels.map((level) => level.value);
  mixInput.min = String(Math.min(...values));
  mixInput.max = String(Math.max(...values));
  const stored = Number(storedMix() || NaN);
  mixInput.value = String(values.includes(stored) ? stored : info.default);
  mixInput.disabled = false;
  mixLevels = info.levels;
  showMix();
}

// --- review ---

// Review takes the dock as practice does: the text waits in the box until the learner confirms it.
async function enterReview() {
  if (review || busy) return;
  leavePractice();
  review = { item: null, heard: "", answer: null, autosend: autosend.checked };
  autosend.checked = false;
  autosend.disabled = true;
  setTeacher(false);
  teacherButton.disabled = true;
  reviewButton.setAttribute("aria-pressed", "true");
  sendButton.textContent = sendLabel();
  transcript.placeholder = placeholderText();
  reviewGloss.textContent = "";
  reviewCounts.textContent = "";
  showReviewParts(null);
  reviewPanel.hidden = false;
  await loadReviewCard();
}

function leaveReview() {
  if (!review) return;
  autosend.checked = review.autosend;
  autosend.disabled = false;
  teacherButton.disabled = false;
  review = null;
  reviewButton.setAttribute("aria-pressed", "false");
  sendButton.textContent = sendLabel();
  transcript.placeholder = placeholderText();
  reviewPanel.hidden = true;
}

// Shows what belongs to `answer` of the card: nothing of it for null, the form with Good and the
// override after a match, the form beside the learner's text and the four ratings after a mismatch.
function showReviewParts(answer) {
  reviewResult.hidden = !answer;
  reviewGood.hidden = !answer || !answer.matched;
  reviewOverride.hidden = !answer || !answer.matched;
  reviewRatings.hidden = !answer || answer.matched;
  reviewNext.hidden = !answer || !answer.matched;
}

// Posts `form` to a review route; what the route answers, or null when it failed.
async function reviewPost(path, form) {
  setBusy(true);
  startCounter(t("status.one_moment"));
  try {
    return await request(path, { method: "POST", body: form });
  } catch (error) {
    showError(error);
    return null;
  } finally {
    stopCounter();
    setBusy(false);
  }
}

function dueText(due) {
  return new Date(due).toLocaleString(uiLang, { dateStyle: "medium", timeStyle: "short" });
}

// The card of `next`, from /api/review/next: {item, counts, next_due}.
function showReviewCard(next) {
  const item = next.item;
  review.item = item;
  review.heard = "";
  review.answer = null;
  transcript.value = "";
  sttMs = 0;
  reviewCounts.textContent = t("review.counts", { due: next.counts.due, fresh: next.counts.new_available });
  const later = next.next_due ? " " + t("review.next_due", { when: dueText(next.next_due) }) : "";
  reviewGloss.textContent = item ? item.gloss : t("review.nothing") + later;
  reviewHint.textContent = item && item.hint ? t("review.hint", { text: item.hint }) : "";
  reviewHint.hidden = !reviewHint.textContent;
  showReviewParts(null);
  // With nothing to review, the button asks again.
  reviewNext.hidden = Boolean(item);
  reviewNext.textContent = t(item ? "review.next" : "review.look_again");
  if (item) setStatus(t("review.say", { button: t("dock.confirm") }));
  else showReady();
}

// A store that is not open says so in the card, and in the status line.
function showReviewProblem(error) {
  reviewCounts.textContent = "";
  reviewGloss.textContent = error.fix ? error.message + " - " + error.fix : error.message;
  reviewHint.hidden = true;
  showReviewParts(null);
  reviewNext.textContent = t("review.look_again");
  reviewNext.hidden = false;
}

async function loadReviewCard() {
  if (busy || !review) return;
  const shown = review;
  let next;
  setBusy(true);
  startCounter(t("status.one_moment"));
  try {
    next = await request(REVIEW_NEXT);
  } catch (error) {
    if (review === shown) showReviewProblem(error);
    showError(error);
    return;
  } finally {
    stopCounter();
    setBusy(false);
  }
  if (review === shown) showReviewCard(next);
}

// The expected form in the Luxembourgish voice; the form is shown whether or not it is heard.
async function speakExpected(text) {
  if (recorder) return;
  try {
    const blob = await spokenBlob("lb", text);
    if (recorder || !review) return;
    await playOnSample(blob);
  } catch (error) {
    if (error.name !== "AbortError" && error.name !== "NotAllowedError") showError(error);
  }
}

async function sendReviewAnswer() {
  const text = transcript.value.trim();
  if (busy || !text) return;
  if (!review.item) {
    setStatus(t("review.nothing"));
    return;
  }
  if (review.answer) {
    setStatus(answerStatus(review.answer.matched));
    return;
  }
  const shown = review;
  const item = review.item;
  const answer = { text, heard: review.heard, mode: review.heard ? "spoken" : "typed" };
  const form = new FormData();
  form.append("item_id", String(item.id));
  form.append("text", text);
  form.append("raw_text", answer.heard);
  form.append("mode", answer.mode);
  const result = await reviewPost(REVIEW_ANSWER, form);
  if (!result || review !== shown || review.item !== item) return;
  review.answer = { ...answer, matched: result.matched, expected: result.expected, logId: result.log_id };
  review.heard = "";
  transcript.value = "";
  sttMs = 0;
  showReviewResult(review.answer);
  if (result.matched) refreshProgress();
  speakExpected(result.expected);
}

// What the status line says about an answered card.
function answerStatus(matched) {
  if (matched) return t("review.status_good", { next: t("review.next"), override: t("review.override") });
  return t("review.status_rate", {
    again: t("review.again"),
    hard: t("review.hard"),
    good: t("review.rate_good"),
    easy: t("review.easy"),
  });
}

function showReviewResult(answer) {
  reviewExpected.textContent = answer.expected;
  reviewSaid.textContent = answer.text;
  reviewNext.textContent = t("review.next");
  showReviewParts(answer);
  setStatus(answerStatus(answer.matched));
}

// The learner's rating of the answered card: with `logId`, in place of its automatic Good.
// The next card follows at once.
async function rateReview(rating, logId) {
  if (busy || !review || !review.answer) return;
  const shown = review;
  const answer = review.answer;
  const form = new FormData();
  form.append("item_id", String(review.item.id));
  form.append("rating", String(rating));
  form.append("text", answer.text);
  form.append("raw_text", answer.heard);
  form.append("mode", answer.mode);
  if (logId !== null) form.append("log_id", String(logId));
  const rated = await reviewPost(REVIEW_RATE, form);
  if (!rated || review !== shown) return;
  refreshProgress();
  await loadReviewCard();
}

// --- progress ---

function minutesText(minutes) {
  return t("unit.minutes", { n: minutes ? minutes.toFixed(1) : "0" });
}

function countCell(count) {
  return element("td", "count", count ? String(count) : "-");
}

// {minutes, phrases, items, corrections_by_week, topics} from the server, as plain totals.
function showProgress(progress) {
  lastProgress = progress;
  progressProblem.hidden = true;
  const items = progress.items;
  const minutes = progress.minutes;
  const figures = [
    [
      t("progress.minutes"),
      t("progress.minutes_value", { week: minutesText(minutes.week), total: minutesText(minutes.total) }),
    ],
    [t("progress.phrases"), String(progress.phrases)],
    [
      t("progress.items"),
      t("progress.items_value", { in_review: items.in_review, total: items.total, due: items.due }),
    ],
  ];
  progressFigures.replaceChildren(
    ...figures.flatMap(([term, value]) => [element("dt", "", term), element("dd", "", value)])
  );
  const head = element("tr");
  head.append(element("th", "", t("progress.week")), ...PROGRESS_COLUMNS.map(([, key]) => element("th", "", t(key))));
  const header = element("thead");
  header.append(head);
  const body = element("tbody");
  for (const week of progress.corrections_by_week) {
    const row = element("tr");
    row.append(element("td", "", week.week), ...PROGRESS_COLUMNS.map(([name]) => countCell(week[name])));
    body.append(row);
  }
  progressCorrections.replaceChildren(element("caption", "", t("progress.caption")), header, body);
  progressTopics.replaceChildren(
    ...progress.topics.map((topic) =>
      element("li", "", t("progress.topic_line", { name: topic.name_en, answered: topic.answered, questions: topic.questions }))
    )
  );
}

// The block is a side note: a failure shows inside it and nowhere else.
async function refreshProgress() {
  try {
    showProgress(await request(LEARNER_PROGRESS));
  } catch (error) {
    progressProblem.textContent = error.fix ? error.message + " - " + error.fix : error.message;
    progressProblem.hidden = false;
  }
}

// --- listening ---

// The listening routes, relative to the page; neither needs the session.
const LISTENING_ITEMS = "api/listening/items";
const LISTENING_PLAY = "api/listening/play";
const LISTENING_CHECK = "api/listening/check";
const GENRE_KEYS = { news: "genre.news", conversation: "genre.conversation", discussion: "genre.discussion" };
const ENGINE_KEYS = { stt: "engine.stt", tts: "engine.tts", llm: "engine.llm" };
const OPTION_LETTERS = "ABC";

const dock = document.querySelector(".dock");
const listeningButton = byId("listening");
const listeningPanel = byId("listening-panel");
const listeningPick = byId("listening-pick");
const listeningItemsBox = byId("listening-items");
const listeningCard = byId("listening-card");
const listeningTitle = byId("listening-title");
const listeningPlayNormal = byId("listening-play-normal");
const listeningPlaySlow = byId("listening-play-slow");
const listeningWait = byId("listening-wait");
const listeningQuestions = byId("listening-questions");
const listeningResult = byId("listening-result");
const listeningTranscript = byId("listening-transcript");
const listeningCheck = byId("listening-check");
const listeningAnother = byId("listening-another");
const listeningLeave = byId("listening-leave");

let listeningList = null; // [{id, genre, title_en, level, questions, words}] from the server; null until it has answered
// Listening mode, null outside it; never on together with practice or review:
// {item, questions, chosen, fieldsets, nodes, checked}. `item` is the entry of the list that is
// open, null while the list is shown; `questions` come with the first play; `chosen` maps a
// question id to the index of the option picked; `nodes` are the transcript's turns, one per
// turn of the recording, filled once the answers are checked.
let listening = null;

// Listening takes the dock without the talk button and the text box: nothing is said or typed.
async function enterListening() {
  if (listening || busy) return;
  leavePractice();
  leaveReview();
  listening = { item: null, questions: null, chosen: {}, fieldsets: {}, nodes: [], checked: false };
  setTeacher(false);
  teacherButton.disabled = true;
  dock.classList.add("listening-on");
  listeningButton.setAttribute("aria-pressed", "true");
  showListeningList();
  listeningPanel.hidden = false;
  if (listeningList) return;
  try {
    const listed = (await request(LISTENING_ITEMS)).items;
    listeningList = listed;
    if (listening && !listening.item) showListeningList();
  } catch (error) {
    showError(error);
  }
}

function leaveListening() {
  if (!listening) return;
  stopListeningAudio();
  listening = null;
  teacherButton.disabled = false;
  dock.classList.remove("listening-on");
  listeningButton.setAttribute("aria-pressed", "false");
  listeningPanel.hidden = true;
}

// The recording of the item stops, and its turns are no longer followed.
function stopListeningAudio() {
  if (listening && reply && reply.nodes === listening.nodes) {
    player.pause();
    reply = null;
  }
}

// The genre and the English title of a listening item.
function listeningHeading(entry) {
  const genre = GENRE_KEYS[entry.genre] ? t(GENRE_KEYS[entry.genre]) : entry.genre;
  return t("listening.title", { genre, title: entry.title_en });
}

function showListeningList() {
  stopListeningAudio();
  listening.item = null;
  listeningItemsBox.replaceChildren(
    ...(listeningList || []).map((entry) => {
      const button = element("button");
      button.type = "button";
      button.append(element("strong", "", listeningHeading(entry)), element("span", "muted", tp("listening.questions", entry.questions)));
      button.addEventListener("click", () => openListeningItem(entry));
      return button;
    })
  );
  listeningCard.hidden = true;
  listeningPick.hidden = false;
}

function openListeningItem(entry) {
  if (!listening) return;
  Object.assign(listening, { item: entry, questions: null, chosen: {}, fieldsets: {}, nodes: [], checked: false });
  listeningTitle.textContent = listeningHeading(entry);
  listeningWait.hidden = false;
  listeningQuestions.replaceChildren();
  listeningResult.hidden = true;
  listeningTranscript.replaceChildren();
  listeningTranscript.hidden = true;
  listeningCheck.hidden = true;
  listeningCheck.disabled = true;
  listeningAnother.hidden = false;
  listeningPick.hidden = true;
  listeningCard.hidden = false;
  setStatus(t("listening.ready"));
}

// Every play renders the recording afresh at `speed`: `normal` (1x) or `slow` (0.75x).
async function playListening(speed) {
  if (busy || !listening || !listening.item) return;
  const shown = listening;
  const item = listening.item;
  const form = new FormData();
  form.append("item_id", item.id);
  form.append("speed", speed);
  const voice = voiceValues();
  if (voice) form.append("voice", JSON.stringify(voice));
  let played;
  setBusy(true);
  listeningPlayNormal.disabled = listeningPlaySlow.disabled = true;
  startCounter(t("status.getting_recording"));
  try {
    played = await request(LISTENING_PLAY, { method: "POST", body: form });
  } catch (error) {
    showError(error);
    return;
  } finally {
    stopCounter();
    setBusy(false);
    listeningPlayNormal.disabled = listeningPlaySlow.disabled = false;
  }
  if (listening !== shown || listening.item !== item) return;
  setStatus(t(listening.checked ? "listening.done" : "listening.answer"));
  if (!listening.questions) showListeningQuestions(played.item.questions);
  if (listening.nodes.length !== played.segment_offsets.length) {
    listening.nodes = played.segment_offsets.map((offset) => listeningTurn(offset.index));
    listeningTranscript.replaceChildren(...listening.nodes);
  }
  loadAudio(played);
  reply = { offsets: played.segment_offsets, nodes: listening.nodes };
  play(0, null);
}

// A turn of the transcript; its text arrives with the check, and a press on it replays it.
function listeningTurn(index) {
  const node = element("li");
  node.addEventListener("click", () => replaySegment(node, index));
  return node;
}

// The questions, each a fieldset of its stem, the English help and the options A, B, C.
function showListeningQuestions(questions) {
  listening.questions = questions;
  listeningWait.hidden = true;
  listeningQuestions.replaceChildren(
    ...questions.map((question, number) => {
      const fieldset = element("fieldset", "listening-question");
      const legend = element("legend");
      const stem = element("span", "", number + 1 + ". " + question.lb);
      stem.lang = "lb";
      legend.append(stem);
      const english = element("p", "english", question.en);
      english.lang = "en";
      english.hidden = true;
      const help = textButton(t("practice.help"), () => {
        english.hidden = false;
        help.hidden = true;
      });
      fieldset.append(legend, help, english);
      question.options.forEach((option, index) => {
        const label = element("label", "option");
        const input = element("input");
        input.type = "radio";
        input.name = "listening-" + question.id;
        input.value = String(index);
        input.addEventListener("change", () => chooseOption(question.id, index));
        const text = element("span", "", option);
        text.lang = "lb";
        label.append(input, element("strong", "", OPTION_LETTERS[index]), text);
        fieldset.append(label);
      });
      listening.fieldsets[question.id] = fieldset;
      return fieldset;
    })
  );
  listeningCheck.hidden = false;
  listeningCheck.disabled = true;
}

function chooseOption(questionId, index) {
  if (!listening || listening.checked) return;
  listening.chosen[questionId] = index;
  listeningCheck.disabled = !listening.questions.every((question) => question.id in listening.chosen);
}

async function checkListening() {
  if (busy || !listening || !listening.questions || listening.checked) return;
  const shown = listening;
  const item = listening.item;
  const form = new FormData();
  form.append("item_id", item.id);
  form.append("answers", JSON.stringify(listening.chosen));
  let checked;
  setBusy(true);
  listeningCheck.disabled = true;
  try {
    checked = await request(LISTENING_CHECK, { method: "POST", body: form });
  } catch (error) {
    showError(error);
    listeningCheck.disabled = false;
    return;
  } finally {
    setBusy(false);
  }
  if (listening !== shown || listening.item !== item) return;
  listening.checked = true;
  showListeningResult(checked);
  setStatus(t("listening.done"));
}

// Words say which answers were right: "right", or "not this one" with the letter of the answer.
function showListeningResult(checked) {
  for (const result of checked.questions) {
    const fieldset = listening.fieldsets[result.id];
    if (!fieldset) continue;
    for (const input of fieldset.querySelectorAll("input")) input.disabled = true;
    const key = OPTION_LETTERS[result.answer];
    let said = t("listening.no_answer", { key });
    if (result.chosen !== null) {
      const chosen = OPTION_LETTERS[result.chosen];
      said = t(result.right ? "listening.chose_right" : "listening.chose_wrong", { chosen, key });
    }
    fieldset.append(element("p", "verdict", said), element("p", "muted", result.why_en));
  }
  listeningResult.textContent = t("listening.result", { right: checked.right, total: checked.total });
  listeningResult.hidden = false;
  checked.transcript.forEach((turn, index) => {
    let node = listening.nodes[index];
    if (!node) {
      node = listeningTurn(index);
      listening.nodes[index] = node;
      listeningTranscript.append(node);
    }
    const text = element("span", "", turn.lb);
    text.lang = "lb";
    node.replaceChildren(element("strong", "", turn.speaker + ": "), text);
    node.classList.add("playable");
  });
  listeningTranscript.hidden = false;
  listeningCheck.hidden = true;
}

// --- events ---

function isTextField(target) {
  if (!(target instanceof HTMLElement)) return false;
  if (target.tagName === "TEXTAREA" || target.isContentEditable) return true;
  return target.tagName === "INPUT" && !["checkbox", "radio", "button", "submit"].includes(target.type);
}

// A control that has its own use for the space bar.
function isInteractive(target) {
  if (!(target instanceof Element)) return false;
  return Boolean(target.closest('button, input, select, textarea, a[href], summary, [role="button"], [contenteditable]'));
}

talkToggle.checked = storedTalk();
talkToggle.addEventListener("change", () => {
  // The recording of the other mode ends first.
  stopTalk();
  storeTalk(toggleMode());
  showTalkMode();
});

talkButton.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  if (toggleMode()) pressTalk();
  else startTalk();
});
for (const type of ["pointerup", "pointerleave", "pointercancel"]) {
  talkButton.addEventListener(type, () => {
    if (!toggleMode()) stopTalk();
  });
}
talkButton.addEventListener("contextmenu", (event) => event.preventDefault());

// Space and Enter on the focused talk button: held in the hold mode, pressed in the other.
// preventDefault keeps the keys from also clicking the button or scrolling the page.
talkButton.addEventListener("keydown", (event) => {
  if (event.key !== " " && event.key !== "Enter") return;
  event.preventDefault();
  if (event.repeat) return;
  if (toggleMode()) pressTalk();
  else startTalk();
});
talkButton.addEventListener("keyup", (event) => {
  if (event.key !== " " && event.key !== "Enter") return;
  event.preventDefault();
  if (!toggleMode()) stopTalk();
});

// Outside a control and a text field the space bar is the talk button as well.
document.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || isInteractive(event.target) || isTextField(event.target)) return;
  event.preventDefault();
  if (event.repeat) return;
  if (toggleMode()) pressTalk();
  else startTalk();
});
document.addEventListener("keyup", (event) => {
  if (event.code !== "Space" || isInteractive(event.target) || isTextField(event.target)) return;
  event.preventDefault();
  if (!toggleMode()) stopTalk();
});
window.addEventListener("blur", stopTalk);

langSelect.addEventListener("change", async () => {
  if (!LANGS.includes(langSelect.value)) return;
  if (!tables[langSelect.value]) await loadTable(langSelect.value);
  setLang(langSelect.value, true);
});

sendButton.addEventListener("click", send);
transcript.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  send();
});
teacherButton.addEventListener("click", () => {
  setTeacher(teacherButton.getAttribute("aria-pressed") !== "true");
});
playButton.addEventListener("click", () => play(null, null));
// Practice, review and listening share the dock: entering one leaves the others.
practiceButton.addEventListener("click", () => {
  if (practice) {
    leavePractice();
  } else if (!busy) {
    leaveListening();
    leaveReview();
    enterPractice();
  }
});
reviewButton.addEventListener("click", () => {
  if (review) {
    leaveReview();
  } else if (!busy) {
    leaveListening();
    enterReview();
  }
});
listeningButton.addEventListener("click", () => {
  if (listening) leaveListening();
  else enterListening();
});
listeningLeave.addEventListener("click", leaveListening);
listeningPlayNormal.addEventListener("click", () => playListening("normal"));
listeningPlaySlow.addEventListener("click", () => playListening("slow"));
listeningCheck.addEventListener("click", checkListening);
listeningAnother.addEventListener("click", () => {
  if (listening) showListeningList();
});
reviewLeave.addEventListener("click", leaveReview);
dataExport.addEventListener("click", exportData);
reviewNext.addEventListener("click", loadReviewCard);
reviewOverride.addEventListener("click", () => {
  if (review && review.answer) rateReview(OVERRIDE_RATING, review.answer.logId);
});
for (const button of reviewRatings.querySelectorAll("button")) {
  button.addEventListener("click", () => rateReview(Number(button.dataset.rating), null));
}
practiceStop.addEventListener("click", lookBack);
practiceLeave.addEventListener("click", leavePractice);
practiceHelp.addEventListener("click", showEnglish);
initSetting();

for (const lang of VOICE_LANGS) {
  for (const part of VOICE_PARTS) {
    // Moving a slider plays nothing: the sample button does.
    voiceInput(lang, part).addEventListener("input", () => {
      showVoice(lang, part);
      storeVoice(voiceValues());
    });
  }
  byId("voice-" + lang + "-sample").addEventListener("click", () => playSample(lang));
}
voiceReset.addEventListener("click", () => {
  setVoice(voiceInfo.defaults);
  storeVoice(null);
});
mixInput.addEventListener("input", () => {
  showMix();
  storeMix(mixInput.value);
});

pronunciationBlock.addEventListener("toggle", () => {
  if (pronunciationBlock.open) refreshFlags();
});

player.addEventListener("play", playbackStarted);
player.addEventListener("timeupdate", tick);
player.addEventListener("pause", playbackStopped);
player.addEventListener("ended", playbackStopped);

window.addEventListener("pagehide", () => {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
});

init();

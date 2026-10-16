"""Send one turn to a running server, print the reply and save its audio.

    python scripts/smoke.py
    python scripts/smoke.py --text "Can you explain how to conjugate goen?"

Start the server first (run.ps1). Without --text the turn is a recording, by default the LOD
example <data_dir>/lod_audio/moien.m4a. The spoken reply goes to <data_dir>/last_reply.wav.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402

# run.ps1 takes the port from the LOCI_PORT environment variable in the same way.
DEFAULT_URL = f"http://127.0.0.1:{os.environ.get('LOCI_PORT') or 8770}"
# A turn waits for the tutor, and the tutor may be asked twice.
TIMEOUT_S = 300
MIMES = {
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
}


def call(client: httpx.Client, method: str, path: str, **kwargs) -> dict:
    """The JSON object of one request; exits with the server's error and fix when it fails."""
    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        raise SystemExit(
            f"{method} {client.base_url}{path} failed: {type(exc).__name__}: {exc}\n"
            "fix: start the server with run.ps1, or pass its address with --url"
        ) from exc
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.is_success and isinstance(body, dict):
        return body
    if isinstance(body, dict) and "error" in body:
        fix = f"\nfix: {body['fix']}" if body.get("fix") else ""
        raise SystemExit(f"{method} {path} answered HTTP {response.status_code}\nerror: {body['error']}{fix}")
    raise SystemExit(f"{method} {path} answered HTTP {response.status_code}: {response.text[:500]}")


def smoke(client: httpx.Client, audio: Path, text: str | None, out_path: Path) -> None:
    health = call(client, "GET", "/api/health")
    print(f"health: {health['status']}")
    for kind, engine in health["engines"].items():
        state = "ready" if engine["ready"] else "NOT READY"
        print(f"  {kind}: {engine['name']} {state} {engine['detail']}".rstrip())
    lod = health["lod"]
    print(f"  lod store: {'ready' if lod['ready'] else 'NOT READY'} {lod['detail']}".rstrip())

    form = {"session_id": call(client, "POST", "/api/session")["session_id"]}
    if text:
        print(f"sending the text: {text}")
        turn = call(client, "POST", "/api/turn", data={**form, "text": text})
    else:
        if not audio.is_file():
            raise SystemExit(
                f"no recording {audio}\nfix: python scripts/download_models.py --lod-audio, or pass --audio"
            )
        print(f"sending the recording: {audio}")
        mime = MIMES.get(audio.suffix.lower(), "application/octet-stream")
        files = {"audio": (audio.name, audio.read_bytes(), mime)}
        turn = call(client, "POST", "/api/turn", data=form, files=files)
    if not turn["heard"]:
        raise SystemExit(f"the server heard no speech in {audio} (timings: {turn['timings']})")

    print(f"transcript: {turn['transcript']}")
    print(f"mode: {turn['mode']}")
    for segment in turn["segments"]:
        print(f"[{segment['lang']}] {segment['text']}")
    for claim in turn["claims"]:
        checked = "verified in LOD" if claim["verified"] else "not verified in LOD"
        print(f"claim: {claim['form']} ({claim['lemma']}) - {checked}")
    if not turn["claims"]:
        print("claims: none")
    if turn["shape_warning"]:
        print(f"format note: {turn['shape_warning']}")
    print("timings: " + ", ".join(f"{name} {value}" for name, value in turn["timings"].items()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(turn["audio_wav_base64"]))
    print(f"reply audio: {out_path} ({turn['segment_offsets'][-1]['end_s']:.1f} s)")


def read_token(data_dir: Path) -> str:
    """The access token: LOCI_ACCESS_TOKEN, else the file the server keeps in the data folder."""
    token = os.environ.get("LOCI_ACCESS_TOKEN")
    if token:
        return token
    path = data_dir / "access_token"
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    if not token:
        raise SystemExit(f"no access token {path}\nfix: start the server once with run.ps1, or set LOCI_ACCESS_TOKEN")
    return token


def main(argv: list[str] | None = None) -> None:
    data_dir = Settings.from_env().data_dir
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL, help="address of the running server (default: %(default)s)")
    parser.add_argument(
        "--audio",
        type=Path,
        default=data_dir / "lod_audio" / "moien.m4a",
        help="recording to send (default: %(default)s)",
    )
    parser.add_argument("--text", help="send this message as typed text instead of a recording")
    args = parser.parse_args(argv)
    # The console code page may lack a character of the reply.
    sys.stdout.reconfigure(errors="backslashreplace")
    headers = {"X-Loci-Token": read_token(data_dir)}
    with httpx.Client(base_url=args.url, timeout=TIMEOUT_S, headers=headers) as client:
        smoke(client, args.audio, args.text, data_dir / "last_reply.wav")


if __name__ == "__main__":
    main()

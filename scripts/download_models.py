"""Download the model files and test recordings the app and its `models` tests use.

    python scripts/download_models.py --stt --tts --lod-audio

Files go to Settings.models_dir and Settings.data_dir (LOCI_MODELS_DIR / LOCI_DATA_DIR).
Each flag skips what is already on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402

# lod.lu answers 403 to some default client user agents (python-httpx among them).
USER_AGENT = "loci-language-app/0.1 (model and test-audio downloader)"

# Example sentences from the Lëtzebuerger Online Dictionnaire (lod.lu, CC0), as returned by
# https://lod.lu/api/lb/entry/<entry> under microStructures/.../examples.
LOD_EXAMPLES = [
    {
        "id": "benotzen",
        "entry": "BENOTZEN1",
        "text": "dat Wuert gëtt bal ni benotzt",
        "url": "https://lod.lu/uploads/examples/AAC/ba/ba475ecade153bef72bb6f29f623fb48.m4a",
    },
    {
        "id": "goen",
        "entry": "GOEN1",
        "text": "géi méi séier, soss verpass de den Zuch!",
        "url": "https://lod.lu/uploads/examples/AAC/81/8190ef4a1f594b8ddf2f698ec1f55543.m4a",
    },
    {
        "id": "moien",
        "entry": "MOIEN1",
        "text": "gudde Moien, Madamm, wat kann ech fir Iech maachen?",
        "url": "https://lod.lu/uploads/examples/AAC/ee/ee04cddd4072d1d7a63aef7f1c667f6f.m4a",
    },
]


def download_stt(settings: Settings) -> None:
    target = settings.models_dir / "whisper-lb"
    if (target / "model.bin").is_file():
        print(f"stt: {target} is present, skipped")
        return
    from huggingface_hub import snapshot_download

    print(f"stt: downloading {settings.stt_model} to {target}")
    snapshot_download(settings.stt_model, local_dir=target)
    print("stt: done")


def download_lod_audio(settings: Settings) -> None:
    target = settings.data_dir / "lod_audio"
    target.mkdir(parents=True, exist_ok=True)
    texts = {}
    for example in LOD_EXAMPLES:
        name = example["id"] + Path(example["url"]).suffix
        texts[name] = example["text"]
        path = target / name
        if path.is_file():
            print(f"lod-audio: {path} is present, skipped")
            continue
        request = urllib.request.Request(example["url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
        print(f"lod-audio: wrote {path}")
    (target / "texts.json").write_text(
        json.dumps(texts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def download_tts(settings: Settings) -> None:
    target = settings.models_dir / "piper"
    target.mkdir(parents=True, exist_ok=True)
    for voice in (settings.tts_voice_lb, settings.tts_voice_en):
        files = [target / f"{voice}.onnx", target / f"{voice}.onnx.json"]
        if all(path.is_file() and path.stat().st_size for path in files):
            print(f"tts: {voice} is present in {target}, skipped")
            continue
        from piper.download_voices import download_voice

        print(f"tts: downloading {voice} to {target}")
        # Fetches only the files of the voice that are missing or empty.
        download_voice(voice, target)
    print("tts: done")


# flag -> (download function, help text); a new kind of download is one more entry.
DOWNLOADS = {
    "stt": (download_stt, "the Luxembourgish Whisper model, into <models_dir>/whisper-lb"),
    "tts": (download_tts, "the two Piper voices, into <models_dir>/piper"),
    "lod-audio": (download_lod_audio, "three LOD example recordings, into <data_dir>/lod_audio"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag, (_, help_text) in DOWNLOADS.items():
        parser.add_argument(f"--{flag}", action="store_true", help=help_text)
    args = parser.parse_args(argv)
    chosen = [flag for flag in DOWNLOADS if getattr(args, flag.replace("-", "_"))]
    if not chosen:
        parser.error("choose at least one of: " + ", ".join(f"--{flag}" for flag in DOWNLOADS))
    settings = Settings.from_env()
    for flag in chosen:
        DOWNLOADS[flag][0](settings)


if __name__ == "__main__":
    main()

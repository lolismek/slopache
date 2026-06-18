#!/usr/bin/env python3
"""Generate a Romanian voice clip with the ElevenLabs API.

Drop-in alternative to gen_voice.py (XTTS): same --text/--out surface, so the wav
feeds the same downstream — Wan 2.2 S2V for talking shots (copy into ComfyUI/input/,
then gen_video.py --audio ...) or voiceover at the ffmpeg assembly stage.

Unlike the XTTS/Fish/MMS paths this needs NO GPU and NO weights — it's a pure
stdlib network call, so it runs anywhere (the box, inside make_reel.py, or the Mac).

NOTE: ElevenLabs is a paid 3rd-party API and contradicts PIPELINE.md's "open-source
only / non-commercial" constraint. This is an experiment to evaluate Romanian voice
quality vs the self-hosted options; revisit before any monetization.

Key is read from .env (ELEVENLABS_API_KEY=...), which is gitignored.

Usage:
    # find an older-female voice to use as the bunica
    python scripts/gen_voice_eleven.py --list-voices

    # synthesize (pcm_44100 wrapped as wav by default)
    python scripts/gen_voice_eleven.py --voice-id <ID> \
        --text "Bună ziua, dragii mei! Astăzi vă arăt ce mănânc." \
        --out outputs/voice/eleven_test.wav
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import wave

API = "https://api.elevenlabs.io/v1"


def load_key():
    """Read ELEVENLABS_API_KEY from the environment or the repo-root .env."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if key:
        return key.strip()
    here = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(os.path.dirname(here), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ELEVENLABS_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("[eleven] ELEVENLABS_API_KEY not found in env or .env")


def list_voices(key):
    req = urllib.request.Request(f"{API}/voices", headers={"xi-api-key": key})
    data = json.loads(urllib.request.urlopen(req).read())
    for v in data.get("voices", []):
        labels = v.get("labels", {}) or {}
        tag = " ".join(f"{k}={labels[k]}" for k in ("gender", "age", "accent") if labels.get(k))
        print(f"{v['voice_id']}  {v.get('name','?'):<22} {tag}")


def synth(key, args):
    # pcm_* returns headerless 16-bit LE mono PCM at the named rate; we wrap it as
    # a real wav. (mp3_* would save raw mp3 bytes — but S2V/ffmpeg prefer wav.)
    rate = int(args.format.split("_")[1]) if args.format.startswith("pcm_") else None
    body = json.dumps({
        "text": args.text,
        "model_id": args.model,
        "voice_settings": {"stability": args.stability, "similarity_boost": args.similarity,
                           "style": args.style, "use_speaker_boost": True},
    }).encode()
    url = f"{API}/text-to-speech/{args.voice_id}?output_format={args.format}"
    req = urllib.request.Request(url, data=body, headers={
        "xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/*"})
    try:
        audio = urllib.request.urlopen(req).read()
    except urllib.error.HTTPError as e:
        sys.exit(f"[eleven] HTTP {e.code}: {e.read().decode(errors='replace')[:400]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if rate:  # wrap PCM in a WAV container
        with wave.open(args.out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(audio)
    else:  # mp3 etc. — write bytes as-is
        with open(args.out, "wb") as f:
            f.write(audio)
    print(f"[eleven] wrote {args.out} ({len(audio)} bytes, {args.model}, voice {args.voice_id})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--out", default="outputs/voice/eleven.wav")
    ap.add_argument("--voice-id", default=None)
    # eleven_multilingual_v2: stable, supports Romanian. eleven_v3: more expressive.
    ap.add_argument("--model", default="eleven_multilingual_v2")
    # mp3 works on all tiers; pcm_* requires ElevenLabs Pro. mp3 is fine downstream
    # (ComfyUI LoadAudio + ffmpeg read it; S2V resamples internally). Transcode to
    # wav with ffmpeg only where a clean wav is needed.
    ap.add_argument("--format", default="mp3_44100_128", help="mp3_44100_128 | pcm_44100 (Pro) | pcm_24000 (Pro)")
    ap.add_argument("--stability", type=float, default=0.5)
    ap.add_argument("--similarity", type=float, default=0.75)
    ap.add_argument("--style", type=float, default=0.0)
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args()

    key = load_key()
    if args.list_voices:
        list_voices(key)
        return
    if not args.text or not args.voice_id:
        sys.exit("[eleven] --text and --voice-id are required (use --list-voices to find one)")
    synth(key, args)


if __name__ == "__main__":
    main()

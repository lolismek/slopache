#!/usr/bin/env python3
"""Generate a sound effect / foley clip from a text prompt with the ElevenLabs
sound-generation API.

Same shape and key handling as gen_voice_eleven.py (pure stdlib HTTP, no GPU/
weights, runs anywhere). The reel orchestrator calls this per shot for diegetic
audio the video model can't make — sizzling slănină, pouring coffee, a rooster,
chickens in the yard — so b-roll isn't dead silent.

`--duration` is optional; ElevenLabs caps it at 22s and auto-picks a length if
omitted. We usually pass the shot's clip duration so the effect fills the shot.

Key is read from .env (ELEVENLABS_API_KEY=...), which is gitignored.

Usage:
    python scripts/gen_sfx_eleven.py --prompt "sizzling bacon frying in a pan, close up" \
        --duration 3.0 --out episodes/ep02/sfx/shot2.mp3
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

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
    sys.exit("[sfx] ELEVENLABS_API_KEY not found in env or .env")


def synth(key, args):
    payload = {"text": args.prompt, "prompt_influence": args.influence}
    if args.duration:
        # API range is 0.5–22s; clamp so a long shot doesn't get rejected.
        payload["duration_seconds"] = max(0.5, min(22.0, args.duration))
    body = json.dumps(payload).encode()
    url = f"{API}/sound-generation?output_format={args.format}"
    req = urllib.request.Request(url, data=body, headers={
        "xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
    try:
        audio = urllib.request.urlopen(req).read()
    except urllib.error.HTTPError as e:
        sys.exit(f"[sfx] HTTP {e.code}: {e.read().decode(errors='replace')[:400]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(audio)
    print(f"[sfx] wrote {args.out} ({len(audio)} bytes) <- {args.prompt!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="text description of the sound")
    ap.add_argument("--out", default="outputs/sfx/sfx.mp3")
    ap.add_argument("--duration", type=float, default=None, help="seconds (0.5–22)")
    # 0 = follow the prompt loosely (more variety); 1 = stick tightly to the prompt.
    ap.add_argument("--influence", type=float, default=0.4)
    ap.add_argument("--format", default="mp3_44100_128")
    args = ap.parse_args()
    synth(load_key(), args)


if __name__ == "__main__":
    main()

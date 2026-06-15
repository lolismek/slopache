#!/usr/bin/env python3
"""Generate a Romanian voice clip with XTTS-v2 (run in the TTS venv).

For the S2V bake-off this uses base XTTS-v2's built-in multilingual speakers with
language="ro". For production we'll swap in the eduardm Romanian finetune + its
named bunică voices (Marioara / Lăcrămioara) — same TTS.api surface.

Usage (in /ephemeral/tts-venv):
    python scripts/gen_voice.py --text "Bună ziua, dragii mei!" \
        --speaker "Ana Florence" --out outputs/voice/line1.wav
"""
import argparse
import os

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # accept CPML non-commercial TOS headlessly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker", default="Ana Florence", help="built-in XTTS speaker")
    ap.add_argument("--language", default="ro")
    ap.add_argument("--out", default="voice.wav")
    ap.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    from TTS.api import TTS
    tts = TTS(args.model).to(args.device)
    tts.tts_to_file(text=args.text, speaker=args.speaker,
                    language=args.language, file_path=args.out)
    print(f"[voice] wrote {args.out}")


if __name__ == "__main__":
    main()

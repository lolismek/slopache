#!/usr/bin/env bash
# Runs ON the box (fish venv). Fish Audio S2 Pro voice cloning, 3 stages:
#   1) reference wav -> VQ tokens (fake.npy)
#   2) text + VQ tokens -> semantic tokens (codes_N.npy)
#   3) semantic tokens -> waveform (fake.wav)
# Language is auto-detected from the text (no lang flag needed).
#
# Usage: gen_voice_fish.sh <ref.wav> "<text>" <out.wav> ["<optional ref transcript>"]
set -euo pipefail
FISH_DIR="${FISH_DIR:-/ephemeral/fish-speech}"
CKPT="${CKPT:-checkpoints/s2-pro}"
# shellcheck disable=SC1091
source /ephemeral/fish-venv/bin/activate
cd "$FISH_DIR"

REF="$1"; TEXT="$2"; OUT="$3"; PROMPT_TEXT="${4:-}"
rm -f fake.npy fake.wav output/codes_*.npy codes_*.npy 2>/dev/null || true

echo "[fish] (1/3) reference -> VQ tokens"
python fish_speech/models/dac/inference.py -i "$REF" --checkpoint-path "$CKPT/codec.pth"

echo "[fish] (2/3) text -> semantic tokens"
if [ -n "$PROMPT_TEXT" ]; then
  python fish_speech/models/text2semantic/inference.py --text "$TEXT" \
    --prompt-tokens fake.npy --prompt-text "$PROMPT_TEXT" --checkpoint-path "$CKPT"
else
  python fish_speech/models/text2semantic/inference.py --text "$TEXT" \
    --prompt-tokens fake.npy --checkpoint-path "$CKPT"
fi

CODES="$(ls -t output/codes_*.npy codes_*.npy 2>/dev/null | head -1)"
[ -n "$CODES" ] || { echo "[fish] ERROR: no codes file produced"; exit 1; }

echo "[fish] (3/3) semantic tokens ($CODES) -> waveform"
python fish_speech/models/dac/inference.py -i "$CODES" --checkpoint-path "$CKPT/codec.pth"

mkdir -p "$(dirname "$OUT")"
cp fake.wav "$OUT"
echo "[fish] wrote $OUT"

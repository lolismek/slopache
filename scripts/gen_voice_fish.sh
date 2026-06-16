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
# Force UTF-8 so Romanian diacritics survive argv decoding (box has LANG unset).
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8
# Sampling: default temperature 1.0 is too hot (breathy/dramatic, erratic pacing).
TEMP="${TEMP:-0.7}"; TOPP="${TOPP:-0.8}"
# shellcheck disable=SC1091
source /ephemeral/fish-venv/bin/activate
cd "$FISH_DIR"

REF="$1"; TEXT="$2"; OUT="$3"; PROMPT_TEXT="${4:-}"
rm -f fake.npy fake.wav output/codes_*.npy codes_*.npy 2>/dev/null || true

# Verify the diacritics actually reached Python (not mangled by locale).
python -c "import sys; t=sys.argv[1]; print('[fish] text repr:', repr(t))" "$TEXT"

echo "[fish] (1/3) reference -> VQ tokens"
python fish_speech/models/dac/inference.py -i "$REF" --checkpoint-path "$CKPT/codec.pth"

echo "[fish] (2/3) text -> semantic tokens"
if [ -n "$PROMPT_TEXT" ]; then
  python fish_speech/models/text2semantic/inference.py --text "$TEXT" \
    --prompt-tokens fake.npy --prompt-text "$PROMPT_TEXT" \
    --temperature "$TEMP" --top-p "$TOPP" --checkpoint-path "$CKPT"
else
  python fish_speech/models/text2semantic/inference.py --text "$TEXT" \
    --prompt-tokens fake.npy \
    --temperature "$TEMP" --top-p "$TOPP" --checkpoint-path "$CKPT"
fi

CODES="$(ls -t output/codes_*.npy 2>/dev/null | head -1 || true)"
[ -n "$CODES" ] || { echo "[fish] ERROR: no codes file produced"; exit 1; }

echo "[fish] (3/3) semantic tokens ($CODES) -> waveform"
python fish_speech/models/dac/inference.py -i "$CODES" --checkpoint-path "$CKPT/codec.pth"

mkdir -p "$(dirname "$OUT")"
cp fake.wav "$OUT"
echo "[fish] wrote $OUT"

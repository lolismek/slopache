#!/usr/bin/env bash
# Runs ON the box. Creates a SEPARATE venv for TTS (coqui-tts) so its deps don't
# collide with ComfyUI's. coqui-tts is the maintained fork of Coqui TTS that
# works with modern torch and ships XTTS-v2 (multilingual, incl. Romanian).
set -euo pipefail
TTS_VENV="${TTS_VENV:-/ephemeral/tts-venv}"

[ -d "$TTS_VENV" ] || python3 -m venv "$TTS_VENV"
# shellcheck disable=SC1091
source "$TTS_VENV/bin/activate"
pip install -q --upgrade pip wheel
echo "[tts] installing coqui-tts (pulls torch + XTTS deps)"
pip install -q coqui-tts
python -c "import TTS; print('[tts] coqui-tts', TTS.__version__)"

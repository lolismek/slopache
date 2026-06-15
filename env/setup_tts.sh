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
echo "[tts] installing torch + torchaudio (CUDA 12.4)"
pip install -q torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
echo "[tts] installing coqui-tts"
pip install -q coqui-tts
# coqui-tts 0.27 declares transformers>=4.57 but its tortoise layer breaks on
# transformers 5.x — pin to the 4.57.x window.
pip install -q "transformers>=4.57,<5.0"
python -c "import torch, TTS; print('[tts] coqui-tts', TTS.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

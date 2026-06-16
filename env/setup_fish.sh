#!/usr/bin/env bash
# Runs ON the box. Installs Fish Audio S2 Pro (2026, top-tier audio fidelity) in
# its OWN venv (torch 2.8 — different from ComfyUI/XTTS venvs). Downloads the
# ~10GB s2-pro weights. License: Fish Audio Research License (non-commercial OK).
# WARNING: Romanian is NOT in s2-pro's trained language list (model card lists
# ~38 langs incl. it/es/fr but NOT ro). It accepts ă/â/î but renders them with
# Romance-neighbor phonetics — wrong vowels. Use XTTS-ro for correct Romanian.
set -euo pipefail
FISH_DIR="${FISH_DIR:-/ephemeral/fish-speech}"
FISH_VENV="${FISH_VENV:-/ephemeral/fish-venv}"

echo "[fish] system audio deps"
sudo apt-get install -y -q portaudio19-dev libsox-dev ffmpeg || echo "[fish][warn] apt issues; continuing"

[ -d "$FISH_DIR/.git" ] || git clone --depth 1 https://github.com/fishaudio/fish-speech "$FISH_DIR"
[ -d "$FISH_VENV" ] || python3 -m venv "$FISH_VENV"
# shellcheck disable=SC1091
source "$FISH_VENV/bin/activate"
pip install -q --upgrade pip wheel

echo "[fish] installing fish-speech (pulls torch 2.8 + deps)"
cd "$FISH_DIR"
pip install -e .

echo "[fish] downloading s2-pro weights (~10GB)"
pip install -q "huggingface_hub[cli]" 2>/dev/null || true
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro

python -c "import torch;print('[fish] torch',torch.__version__,'| cuda',torch.cuda.is_available())"
echo "[fish] done."

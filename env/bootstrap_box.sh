#!/usr/bin/env bash
# Runs ON the box. Idempotent: installs ComfyUI + a Python venv + torch on the
# /ephemeral volume (the root disk is only ~97GB). Safe to re-run.
set -euo pipefail
COMFY_DIR="${COMFY_DIR:-/ephemeral/ComfyUI}"

echo "[bootstrap] apt deps"
sudo apt-get update -y -q && sudo apt-get install -y -q \
  git python3-venv python3-pip ffmpeg libgl1 libglib2.0-0 \
  || echo "[bootstrap][warn] apt step had issues; continuing"

echo "[bootstrap] clone ComfyUI -> $COMFY_DIR"
if [ ! -d "$COMFY_DIR/.git" ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
else
  git -C "$COMFY_DIR" pull --ff-only || true
fi

cd "$COMFY_DIR"
[ -d venv ] || python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install -q --upgrade pip wheel

echo "[bootstrap] torch (CUDA 12.4 wheels)"
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "[bootstrap] ComfyUI requirements + hf hub"
pip install -q -r requirements.txt
pip install -q "huggingface_hub[hf_transfer]"

echo "[bootstrap] verify torch/CUDA:"
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
echo "[bootstrap] done."

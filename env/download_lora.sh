#!/usr/bin/env bash
# Runs ON the box. Pulls the trained bunica FLUX.2 character-LoRA from the PRIVATE
# HF repo (alexjerpelea/bunica-flux2-lora) and symlinks it into ComfyUI's loras dir
# so workflows/flux2_lora_txt2img_api.json can load it. Mirror of upload_lora_hf.py.
# Private repo => needs HF_TOKEN (read from env or repo-root .env, which is synced).
set -euo pipefail
COMFY_DIR="${COMFY_DIR:-/ephemeral/ComfyUI}"
DL="${DL:-/ephemeral/hf}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$COMFY_DIR/venv/bin/activate"

# HF_TOKEN: env first, else repo-root .env (same lookup as gen_voice_eleven.py's key).
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HERE/../.env" ]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' "$HERE/../.env" | head -1 | cut -d= -f2- | tr -d '"'"'"'' )"
fi
[ -n "${HF_TOKEN:-}" ] || { echo "[dl-lora] HF_TOKEN not found in env or .env" >&2; exit 1; }
export HF_TOKEN

if pip install -q hf_transfer 2>/dev/null; then export HF_HUB_ENABLE_HF_TRANSFER=1; fi
if command -v hf >/dev/null 2>&1; then HF=hf; else HF=huggingface-cli; fi

REPO="${LORA_REPO:-alexjerpelea/bunica-flux2-lora}"
LORA="${LORA_FILE:-bunica_flux2_v1_step1000.safetensors}"

echo "[dl-lora] $REPO/$LORA -> $DL/lora"
"$HF" download "$REPO" "$LORA" --local-dir "$DL/lora" --token "$HF_TOKEN"

mkdir -p "$COMFY_DIR/models/loras"
ln -sf "$DL/lora/$LORA" "$COMFY_DIR/models/loras/"
echo "[dl-lora] linked:"
ls -lhL "$COMFY_DIR/models/loras/$LORA"
echo "[dl-lora] done."

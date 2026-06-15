#!/usr/bin/env bash
# Runs ON the box. Downloads Wan 2.2 S2V (audio-driven talking-head) — a single
# 14B model (no MoE) + the wav2vec audio encoder. Reuses the umt5 text encoder
# and Wan VAE already fetched by download_wan.sh.
set -euo pipefail
COMFY_DIR="${COMFY_DIR:-/ephemeral/ComfyUI}"
DL="${DL:-/ephemeral/hf_wan}"
# shellcheck disable=SC1091
source "$COMFY_DIR/venv/bin/activate"
if pip install -q hf_transfer 2>/dev/null; then export HF_HUB_ENABLE_HF_TRANSFER=1; fi
if command -v hf >/dev/null 2>&1; then HF=hf; else HF=huggingface-cli; fi

REPO=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
S2V="split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors"
AENC="split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors"

echo "[dl-s2v] $REPO -> $DL"
"$HF" download "$REPO" "$S2V" "$AENC" --local-dir "$DL"

mkdir -p "$COMFY_DIR"/models/{diffusion_models,audio_encoders}
ln -sf "$DL/$S2V"  "$COMFY_DIR/models/diffusion_models/"
ln -sf "$DL/$AENC" "$COMFY_DIR/models/audio_encoders/"
echo "[dl-s2v] linked:"
ls -lhL "$COMFY_DIR"/models/diffusion_models/wan2.2_s2v_* "$COMFY_DIR"/models/audio_encoders/

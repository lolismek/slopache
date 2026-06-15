#!/usr/bin/env bash
# Runs ON the box. Downloads Wan 2.2 14B I2V (two-expert MoE: high+low noise),
# the umt5 text encoder, and the Wan VAE from the Comfy-Org repackaged mirror,
# then symlinks them into ComfyUI's model dirs.
set -euo pipefail
COMFY_DIR="${COMFY_DIR:-/ephemeral/ComfyUI}"
DL="${DL:-/ephemeral/hf_wan}"
# shellcheck disable=SC1091
source "$COMFY_DIR/venv/bin/activate"

if pip install -q hf_transfer 2>/dev/null; then export HF_HUB_ENABLE_HF_TRANSFER=1; fi
if command -v hf >/dev/null 2>&1; then HF=hf; else HF=huggingface-cli; fi

REPO=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
HI="split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
LO="split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
TE="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE="split_files/vae/wan_2.1_vae.safetensors"

echo "[dl-wan] $REPO -> $DL (via $HF)"
"$HF" download "$REPO" "$HI" "$LO" "$TE" "$VAE" --local-dir "$DL"

mkdir -p "$COMFY_DIR"/models/{diffusion_models,text_encoders,vae}
for f in "$HI" "$LO"; do ln -sf "$DL/$f" "$COMFY_DIR/models/diffusion_models/"; done
ln -sf "$DL/$TE"  "$COMFY_DIR/models/text_encoders/"
ln -sf "$DL/$VAE" "$COMFY_DIR/models/vae/"

echo "[dl-wan] linked:"
ls -lhL "$COMFY_DIR"/models/diffusion_models/wan2.2_* "$COMFY_DIR"/models/text_encoders/umt5_* "$COMFY_DIR"/models/vae/wan_2.1_vae.safetensors

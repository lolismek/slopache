#!/usr/bin/env bash
# Runs ON the box. Downloads ComfyUI-ready Wan 2.2 I2V (14B MoE) split files from
# the (ungated) Comfy-Org repackage and symlinks them into ComfyUI's model dirs.
# Wan 2.2 I2V is a Mixture-of-Experts: BOTH high- and low-noise experts required.
# Files land on /ephemeral/hf (one copy) and are linked, not duplicated.
set -euo pipefail
COMFY_DIR="${COMFY_DIR:-/ephemeral/ComfyUI}"
DL="${DL:-/ephemeral/hf}"
# shellcheck disable=SC1091
source "$COMFY_DIR/venv/bin/activate"

if pip install -q hf_transfer 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
else
  echo "[dl][warn] hf_transfer unavailable; using standard download"
fi
if command -v hf >/dev/null 2>&1; then HF=hf; else HF=huggingface-cli; fi

REPO=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
HIGH="split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
LOW="split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
TE="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE="split_files/vae/wan_2.1_vae.safetensors"   # I2V 14B uses the 2.1 VAE
LORA_HI="split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
LORA_LO="split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

echo "[dl] $REPO -> $DL (this is ~30GB)"
"$HF" download "$REPO" "$HIGH" "$LOW" "$TE" "$VAE" "$LORA_HI" "$LORA_LO" --local-dir "$DL"

mkdir -p "$COMFY_DIR"/models/{diffusion_models,text_encoders,vae,loras}
ln -sf "$DL/$HIGH" "$COMFY_DIR/models/diffusion_models/"
ln -sf "$DL/$LOW"  "$COMFY_DIR/models/diffusion_models/"
ln -sf "$DL/$TE"   "$COMFY_DIR/models/text_encoders/"
ln -sf "$DL/$VAE"  "$COMFY_DIR/models/vae/"
ln -sf "$DL/$LORA_HI" "$COMFY_DIR/models/loras/"
ln -sf "$DL/$LORA_LO" "$COMFY_DIR/models/loras/"

echo "[dl] linked into ComfyUI/models:"
ls -lhL "$COMFY_DIR"/models/diffusion_models/wan2.2_i2v_* \
        "$COMFY_DIR"/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
        "$COMFY_DIR"/models/vae/wan_2.1_vae.safetensors \
        "$COMFY_DIR"/models/loras/wan2.2_i2v_lightx2v_4steps_* 2>/dev/null
echo "[dl] done."

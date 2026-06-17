#!/usr/bin/env bash
# Runs ON the box. Downloads ComfyUI-ready Qwen-Image-Edit-2511 split files from
# the (ungated) Comfy-Org mirrors and symlinks them into ComfyUI's model dirs.
# Used to generate the LoRA training dataset: edit the founder image into ~30
# on-model variations (pose/angle/expression/scene/light) while keeping the face.
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

# Diffusion model: fp8mixed 2511 (~20GB) — ample on an 80GB A100. Override DIFF= for bf16.
EDIT_REPO=Comfy-Org/Qwen-Image-Edit_ComfyUI
DIFF="${DIFF:-split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors}"
# Text-encoder (Qwen2.5-VL 7B) + VAE live in the base Qwen-Image repo.
BASE_REPO=Comfy-Org/Qwen-Image_ComfyUI
TE="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE="split_files/vae/qwen_image_vae.safetensors"

echo "[dl] $EDIT_REPO -> $DL"
"$HF" download "$EDIT_REPO" "$DIFF" --local-dir "$DL"
echo "[dl] $BASE_REPO -> $DL"
"$HF" download "$BASE_REPO" "$TE" "$VAE" --local-dir "$DL"

mkdir -p "$COMFY_DIR"/models/{diffusion_models,text_encoders,vae}
ln -sf "$DL/$DIFF" "$COMFY_DIR/models/diffusion_models/"
ln -sf "$DL/$TE"   "$COMFY_DIR/models/text_encoders/"
ln -sf "$DL/$VAE"  "$COMFY_DIR/models/vae/"

echo "[dl] linked into ComfyUI/models:"
ls -lhL "$COMFY_DIR"/models/diffusion_models/qwen_image_edit_* \
        "$COMFY_DIR"/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
        "$COMFY_DIR"/models/vae/qwen_image_vae.safetensors
echo "[dl] done."

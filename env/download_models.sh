#!/usr/bin/env bash
# Runs ON the box. Downloads the ComfyUI-ready fp8 FLUX.2-dev split files from
# the (ungated) Comfy-Org mirror and symlinks them into ComfyUI's model dirs.
# Files land on /ephemeral/hf (one copy) and are linked, not duplicated.
set -euo pipefail
COMFY_DIR="${COMFY_DIR:-/ephemeral/ComfyUI}"
DL="${DL:-/ephemeral/hf}"
# shellcheck disable=SC1091
source "$COMFY_DIR/venv/bin/activate"

# hf_transfer gives multi-connection downloads; install it, fall back if it fails.
if pip install -q hf_transfer 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
else
  echo "[dl][warn] hf_transfer unavailable; using standard download"
fi

# CLI was renamed huggingface-cli -> hf in huggingface_hub 1.x.
if command -v hf >/dev/null 2>&1; then HF=hf; else HF=huggingface-cli; fi

REPO=Comfy-Org/flux2-dev
DIFF="split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"
VAE="split_files/vae/flux2-vae.safetensors"
# Text-encoder: fp8 (18GB) is ample on an 80GB A100; override via TE=... for bf16.
TE="${TE:-split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors}"

echo "[dl] $REPO -> $DL (via $HF)"
"$HF" download "$REPO" "$DIFF" "$VAE" "$TE" --local-dir "$DL"

mkdir -p "$COMFY_DIR"/models/{diffusion_models,text_encoders,vae}
ln -sf "$DL/$DIFF" "$COMFY_DIR/models/diffusion_models/"
ln -sf "$DL/$VAE"  "$COMFY_DIR/models/vae/"
ln -sf "$DL/$TE"   "$COMFY_DIR/models/text_encoders/"

echo "[dl] linked into ComfyUI/models:"
ls -lhL "$COMFY_DIR"/models/diffusion_models/ "$COMFY_DIR"/models/text_encoders/ "$COMFY_DIR"/models/vae/

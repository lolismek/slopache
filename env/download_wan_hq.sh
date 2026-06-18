#!/usr/bin/env bash
# Runs ON the box. Downloads the HIGH-QUALITY (fp16/bf16) Wan 2.2 weights and the
# fp16 umt5 text encoder, then symlinks them next to the fp8 ones in ComfyUI.
#
# Why: the A100 (compute 8.0) has no hardware fp8, so fp8_scaled buys nothing on
# speed here and only costs fidelity. Running the real fp16/bf16 weights is the
# single biggest "generation quality" lever (sharper detail, more coherent motion,
# fewer artifacts) — at the cost of ~2.5x the VRAM/disk, which the 80GB box has.
#
# Files coexist with the fp8 variants; the workflows choose which to load by name.
# ~84GB download (3 x 14B @ fp16/bf16 + ~11GB text encoder).
set -euo pipefail
COMFY_DIR="${COMFY_DIR:-/ephemeral/ComfyUI}"
DL="${DL:-/ephemeral/hf}"          # i2v + text encoder land here (matches download_wan_i2v.sh)
DL_S2V="${DL_S2V:-/ephemeral/hf_wan}"  # s2v lands here (matches download_s2v.sh)
# shellcheck disable=SC1091
source "$COMFY_DIR/venv/bin/activate"

if pip install -q hf_transfer 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
else
  echo "[dl-hq][warn] hf_transfer unavailable; using standard download"
fi
if command -v hf >/dev/null 2>&1; then HF=hf; else HF=huggingface-cli; fi

REPO=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
HIGH="split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
LOW="split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
S2V="split_files/diffusion_models/wan2.2_s2v_14B_bf16.safetensors"
TE="split_files/text_encoders/umt5_xxl_fp16.safetensors"

echo "[dl-hq] i2v fp16 + umt5 fp16 -> $DL"
"$HF" download "$REPO" "$HIGH" "$LOW" "$TE" --local-dir "$DL"
echo "[dl-hq] s2v bf16 -> $DL_S2V"
"$HF" download "$REPO" "$S2V" --local-dir "$DL_S2V"

mkdir -p "$COMFY_DIR"/models/{diffusion_models,text_encoders}
ln -sf "$DL/$HIGH"    "$COMFY_DIR/models/diffusion_models/"
ln -sf "$DL/$LOW"     "$COMFY_DIR/models/diffusion_models/"
ln -sf "$DL_S2V/$S2V" "$COMFY_DIR/models/diffusion_models/"
ln -sf "$DL/$TE"      "$COMFY_DIR/models/text_encoders/"

echo "[dl-hq] linked into ComfyUI/models:"
ls -lhL "$COMFY_DIR"/models/diffusion_models/wan2.2_i2v_*_fp16.safetensors \
        "$COMFY_DIR"/models/diffusion_models/wan2.2_s2v_14B_bf16.safetensors \
        "$COMFY_DIR"/models/text_encoders/umt5_xxl_fp16.safetensors 2>/dev/null
echo "[dl-hq] done."

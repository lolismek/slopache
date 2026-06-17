#!/usr/bin/env python3
"""Upload the bunica FLUX.2 character-LoRA checkpoints to a PRIVATE HF repo.
Runs ON the box (checkpoints + huggingface_hub live there). Token read from
HF_HOME/token (HF_HOME=/ephemeral/hf)."""
import os
from huggingface_hub import HfApi

REPO = "alexjerpelea/bunica-flux2-lora"
CKPT_DIR = "/ephemeral/slopache/output/lora/bunica_flux2_v1"
# (local checkpoint file, name in repo)
UPLOADS = [
    ("bunica_flux2_v1_000001000.safetensors", "bunica_flux2_v1_step1000.safetensors"),
    ("bunica_flux2_v1_000000750.safetensors",  "bunica_flux2_v1_step750.safetensors"),
    ("bunica_flux2_v1_000000500.safetensors",  "bunica_flux2_v1_step500.safetensors"),
]

CARD = """---
base_model: black-forest-labs/FLUX.2-dev
tags:
  - flux2
  - lora
  - character-lora
  - text-to-image
library_name: diffusers
inference: false
---

# Bunica Influenceriță — FLUX.2 Character LoRA

Character LoRA for a consistent Pixar-style Romanian grandmother, trained on
FLUX.2-dev from a curated 28-image synthetic dataset (one founder image varied
across decorrelated backgrounds / framings / expressions / camera angles).

**Trigger word:** `ohwxbunica`  (always pair with the class/style in prose, e.g.
`ohwxbunica, a 3D Pixar-style elderly Romanian grandmother ...`)

## Checkpoints

| File | Step | Notes |
|------|------|-------|
| `bunica_flux2_v1_step1000.safetensors` | 1000 | **Chosen / default.** Strongest, most polished frontals. |
| `bunica_flux2_v1_step750.safetensors`  | 750  | Backup. Slightly less stylization. |
| `bunica_flux2_v1_step500.safetensors`  | 500  | Conservative backup, mildest caricature. |

## Training

- Base: `black-forest-labs/FLUX.2-dev` (32B), quantized for a single A100 80GB
  (quanto fp8 weights + bf16 compute; no hardware fp8 on A100).
- Trainer: ostris/ai-toolkit, `arch: flux2`.
- LoRA rank 16 / alpha 16, lr 1e-4, adamw8bit, batch 1, 1024px, flowmatch.
- Captions: natural-language prose (FLUX.2 Mistral text encoder); trigger first,
  face left undescribed (binds to trigger), costume fully described (promptable).

## Evaluation notes

Validated on out-of-training prompts (unseen scenes, outfit swap, accessories).
Identity locked in by ~step 500 and held through an outfit change. The adapter
weight-norm grew linearly with no plateau, so training was stopped at 1000;
later steps only amplified caricature (notably the nose in big expressions)
without improving identity.

> Derivative of FLUX.2-dev — inherits the FLUX.2-dev license terms (non-commercial).
"""

def main():
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    print(f"[hf] repo ready (private): {REPO}")
    # model card
    from huggingface_hub import upload_file
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(CARD); card_path = f.name
    api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                    repo_id=REPO, repo_type="model")
    print("[hf] uploaded README.md")
    for local, name in UPLOADS:
        p = os.path.join(CKPT_DIR, local)
        if not os.path.exists(p):
            print(f"[hf] SKIP missing {local}"); continue
        api.upload_file(path_or_fileobj=p, path_in_repo=name,
                        repo_id=REPO, repo_type="model")
        print(f"[hf] uploaded {name} ({os.path.getsize(p)/1e9:.2f} GB)")
    print(f"[hf] done -> https://huggingface.co/{REPO}")

if __name__ == "__main__":
    main()

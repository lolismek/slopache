# slopache

Production pipeline for an Instagram account built around a **consistent
AI-generated character** — *Bunica Influenceriță*, a Romanian grandmother who has
accidentally become a lifestyle influencer. See [`PIPELINE.md`](./PIPELINE.md) for
the full architecture and the end goal.

## Build order (incremental)

We are **not** building the whole pipeline at once. Current focus:

1. **▶ Image generation** — stand up FLUX.2-dev on the GPU box and produce the
   "founder" grandma image. *(in progress)*
2. Qwen-Image-Edit — turn the founder image into a consistent dataset.
3. LoRA training — lock the identity (`ai-toolkit`, FLUX.2 LoRA).
4. …then animation, voice, captions, assembly (see `PIPELINE.md`).

## Infrastructure

All generation runs on a Brev A100 80GB instance (`visible-rose-reptile`). Heavy
data (ComfyUI, model weights) lives on the box's `/ephemeral` volume; this repo is
the source of truth for code/config and is rsync'd to the box.

```
infra/box.env          connection config (IP resolved dynamically)
infra/remote.sh        ssh / sync / pull / push wrapper
env/bootstrap_box.sh   installs ComfyUI + venv + torch on the box   (run on box)
env/download_models.sh downloads FLUX.2-dev fp8 weights             (run on box)
scripts/gen_image.py   headless ComfyUI client (prompt -> image)
workflows/             ComfyUI API-format workflow templates
character/prompts/     identity + style prompt fragments
```

## Quickstart (image generation)

```bash
# 1. push code to the box
./infra/remote.sh sync

# 2. one-time: install ComfyUI + download weights (on the box)
./infra/remote.sh ssh 'cd /ephemeral/slopache && COMFY_DIR=/ephemeral/ComfyUI bash env/bootstrap_box.sh'
./infra/remote.sh ssh 'cd /ephemeral/slopache && COMFY_DIR=/ephemeral/ComfyUI bash env/download_models.sh'

# 3. start ComfyUI server (on the box, backgrounded)
./infra/remote.sh ssh 'cd /ephemeral/ComfyUI && nohup venv/bin/python main.py --listen 127.0.0.1 --port 8188 >/tmp/comfy.log 2>&1 &'

# 4. generate the founder image (on the box) and pull it back
./infra/remote.sh ssh 'cd /ephemeral/slopache && /ephemeral/ComfyUI/venv/bin/python scripts/gen_image.py --prompt "..." --out grandma --batch 4'
./infra/remote.sh pull '/ephemeral/slopache/outputs/*' ./outputs/
```

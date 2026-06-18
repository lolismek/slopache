# Build Log & Insights — Bunica Influenceriță pipeline

Running record of what's built, what we learned, and the gotchas that cost real
iteration. Newest section first. See `PIPELINE.md` for the target architecture.

---

## Pipeline status (as of 2026-06-17)

| Stage | Status | Notes |
|---|---|---|
| Dataset builder (Qwen-Image-Edit-2511) | ✅ done | `scripts/gen_dataset.py`, workflows in `workflows/`. 28-img v2 set. |
| Character LoRA (FLUX.2-dev) | ✅ **trained & chosen** | step-1000 keeper on HF (private) `alexjerpelea/bunica-flux2-lora`. Trigger `ohwxbunica`. |
| Stills (FLUX.2 + LoRA) | ⏳ next | LoRA done; haven't generated production stills yet. Want **native 9:16**. |
| Video — b-roll (Wan 2.2 I2V) | ✅ **validated** | `scripts/run_wan_i2v.py`. ~85–100s per 5s clip (4-step). |
| Video — talking (Wan 2.2 S2V) | ❌ not started | Needs a still **+ audio**. Audio = voice stage (deferred). |
| Voice (RO XTTS-v2) | ⏸ deferred | Earlier track; see memory `fish-s2-cloning-needs-prompt-text`. |
| Captions (WhisperX) / Assembly (ffmpeg) | ❌ not started | Deterministic tail. |

**Box:** brev `stale-blush-orca` (A100 80GB PCIe, driver 580.126.09). `infra/remote.sh {ssh|sync|pull|push|host}`.
`/ephemeral` is scratch — **wiped on deprovision**. Source of truth = this git repo
(code) + HF (LoRA weights). Model weights re-downloadable via `env/download_*.sh`.

---

## Character LoRA — recipe & results

**Recipe (validated):** FLUX.2-dev, ostris/ai-toolkit `arch: flux2`, rank 16 / alpha 16,
lr 1e-4, adamw8bit, batch 1, 1024px, flowmatch. Config: `training/bunica_flux2_v1.yaml`.
Captions: **natural-language prose** (FLUX.2 uses a Mistral-24B text encoder, NOT
tag-soup), trigger first, **face left undescribed** (binds to trigger), **costume fully
described** (stays promptable). `scripts/gen_captions.py`.

**Results:** identity locked by ~step 500, held through an unseen outfit change.
Validated on **out-of-training** prompts (bus, sunglasses selfie, red-puffer supermarket,
field laugh). Chose **step 1000**; 750/500 kept as backups. Caricature (nose) amplifies
with more steps, worst in big expressions — so we stopped at 1000, not the full 2000.

**Convergence metric (model-free, `scripts/lora_weight_norm.py`):** effective adapter
norm ‖scale·B·A‖ per checkpoint + delta vs prev, via 16×16 trace identities (no big
matrices → no OOM). Observed:

```
   step   ||W||    ||dW||   rel dW
    250    6.41      —        —
    500   11.71    8.58     0.733
    750   16.91   10.09     0.596
   1000   21.61   10.17     0.471
```

‖W‖ grew **linearly** (no plateau); rel dW fell only because the denominator grew.
Confirms: training loss is useless for character LoRAs, and there was no convergence
plateau — visual samples were the real stop signal.

---

## Key insights / gotchas (durable)

**Diffusion-LoRA loss is not a convergence signal.** Denoising MSE is sampled at random
timesteps → extremely noisy, flat at ~0.42 the whole run while identity went from absent
→ locked. Judge by samples. Model-free quantitative proxies that need NO extra model:
(1) held-out denoising loss with *fixed* noise/timesteps (overfitting gap), (2) LoRA
weight-norm curve (we used this), (3) fixed-seed inter-checkpoint output drift. Identity
embedding (ArcFace) is the gold standard but needs another model.

**A100 has no hardware FP8.** A100 = compute capability 8.0; native FP8 needs ≥8.9
(H100/4090). So quantize with **int8/bf16 or quanto fp8 (software, bf16 compute)** — NOT
hardware float8. The float8 defaults in most guides assume newer GPUs.

**torch/driver mismatch on the box.** ai-toolkit pip-installs the *latest* torch
(2.12.0+cu130), which needs driver ≥580; box driver is 570.195 (supports CUDA 12.8 max).
Fix: pin **torch 2.11.0+cu128 + torchvision 0.26.0 + torchaudio 2.11.0**, **numpy 1.26.4**,
**setuptools 69.5.1** (≥70 removed `pkg_resources.packaging` that bundled CLIP imports).

**CPU OOM on the analysis box, not the trainer.** Loading checkpoints + forming B@A in
*system RAM* alongside the running trainer (which holds the offloaded 24B encoder) gets
OOM-killed (exit 137). Use trace identities to avoid materializing big matrices.

**Nested-quote python over `remote.sh ssh` breaks.** Write a `.py`, `push` it, run it.

---

## Architecture clarifications (settled this session)

- **Multi-character is a *stills* problem, not a video problem.** Compose both characters
  into ONE still (per-character LoRA + **regional/masked conditioning**, or sequential
  inpaint), then animate that single still. Recurring 2nd character = its own saved LoRA.
- **Wan I2V takes ONE start frame** (+ optional FLF2V *last* frame = temporal endpoint,
  + optional single clip-vision embedding). It's a scene *animator*, not a multi-reference
  compositor. The start frame already contains everything in shot.
- **I2V ≠ talking.** Mouth motion under I2V is hallucinated, not lip-synced. Real dialogue
  = **S2V** (still + audio). Two people talking → **shot/reverse-shot** single-char S2V.
- **Video models are multi-modal in conditioning** (T2V / I2V / FLF2V / reference (VACE) /
  control / audio) over a shared latent-diffusion DiT + 3D-VAE core. "Animate one still"
  is just I2V. Our **still-first** choice is a *consistency* strategy, not a universal law.

---

## Wan 2.2 I2V — concrete setup

Files (all `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`, `env/download_wan_i2v.sh`): two 14B
experts (high+low noise, fp8_scaled, MoE — **both required**), `umt5_xxl` text encoder,
**`wan_2.1_vae`** (I2V uses the 2.1 VAE), + LightX2V 4-step LoRAs. Graph: LoadImage →
WanImageToVideo(start_image) + CLIPTextEncode×2(umt5,`wan`) + UNETLoader×2 →
ModelSamplingSD3(shift 5) → (4-step LoRA) → KSamplerAdvanced×2 (split 0→2 / 2→4, cfg 1,
euler/simple) → VAEDecode(2.1) → CreateVideo(16fps) → SaveVideo(mp4/h264). One gen =
81 frames @ 16fps ≈ 5s. **Timing on A100:** ~100s cold, ~85s warm (4-step, 768²);
20-step quality ≈ 3–4 min; 720p/9:16 longer. Longer clips = chain last→next or FLF2V.

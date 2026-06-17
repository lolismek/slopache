# "Bunica Influenceriță" — AI Reel Production Pipeline

## Context

We're building a repeatable pipeline to produce short-form vertical reels for an
Instagram account featuring a **consistent AI-generated character** — a Romanian
grandmother ("bunică") who has accidentally become a lifestyle influencer (GRWM in
ie & catrință, unboxing gadgets, "ce mănânc într-o zi" = all slănină). The model
is the Zupi / "Porumbeii din Cluj" format: recurring AI character, continuous story
across episodes.

**Hard requirements / constraints (settled over prior discussion):**
- **Open-source models only**, self-hosted. No paid 3rd-party generation APIs.
- **Character consistency is the #1 priority** — the face must be identical across
  every reel. This is the account's whole moat.
- **Non-commercial use assumed** (unlocks FLUX.2-dev and CPML-licensed XTTS weights).
- **Single GPU box:** one **A100 80 GB** at brev instance `zany-brown-owl`
  (resolved dynamically by `infra/remote.sh`). Everything runs on this one warm box
  — no two-machine split. **Current build status & insights: see `docs/BUILD_LOG.md`.**
- **Orchestration:** ComfyUI-first (manual node graphs), with scripted glue only for
  the deterministic tail (voice, captions, ffmpeg assembly).
- **Review:** approve-each-shot — human reviews/regenerates stills and clips before
  assembly. Right for early episodes while we dial in the character.
- **Code/config in this git repo**, synced to the box; heavy weights/assets live on
  the box (git-ignored), reproducible via a download manifest.

**Intended outcome:** a reproducible per-reel loop — *story bible → script → stills →
voice → animate → caption → assemble* — that reliably yields a publishable 9:16 MP4
of the same recognizable grandma.

---

## Architecture overview

Core principle: **lock identity at the still-image layer (a trained LoRA), then
animate the stills.** Video models drift; a trained character model does not.

```
                 ┌─────────────────────────────────────────────────┐
                 │  visible-rose-reptile  (A100 80GB, one warm box)  │
                 └─────────────────────────────────────────────────┘

  story bible ─► SCRIPT ─► STILLS ──────► ANIMATE ──────► CAPTION ─► ASSEMBLE
   (.md/.json)   (Qwen3)   (FLUX.2        (Wan 2.2 fam.)  (WhisperX)  (ffmpeg)
                            + bunică LoRA)  ├ talking → S2V (+RO audio)
                                            └ b-roll  → I2V (+prompt)
                              ▲                                ▲
                              │ approve-each-shot gate         │ optional
                              └── re-roll until on-model       └ LatentSync lip touch-up
                                                          VOICE: RO XTTS-v2 → wav
```

A reel = **4–6 short shots stitched together**. Talking shots carry dialogue
(audio-driven), b-roll shots are silent under voiceover/music.

---

## Final model selection (research-backed, mid-2026)

| Layer | Choice | Notes |
|---|---|---|
| Orchestration | **ComfyUI** | One install hosts every model below as nodes. |
| Stills / identity | **FLUX.2-dev + trained character LoRA** | 80 GB box unlocks FLUX.2 LoRA training; best face quality/consistency. |
| Dataset builder | **Qwen-Image-Edit-2511** | Generate the consistent 20–50-img LoRA dataset from one founder image (varied pose/light, same face). |
| LoRA training | **Ostris `ai-toolkit`** (FLUX.2 LoRA, ~1–2 h on A100) | One-time. LoKr optional for small character sets. |
| Video — talking | **Wan 2.2 S2V** (image + RO audio → lip-synced, face+body) | Audio-driven ⇒ language-agnostic ⇒ Romanian works. Supersedes InfiniteTalk. |
| Video — b-roll | **Wan 2.2 I2V** (image + prompt → motion) | Same family/nodes as S2V — one ecosystem, two modes. |
| Lip touch-up | **LatentSync** *(optional post-pass)* | Only if S2V mouths look soft; not a 2nd generator. |
| Voice (RO) | **eduardm Romanian XTTS-v2 finetune** (6.3% WER) | Ships ready female voices (Marioara / Lăcrămioara) — likely the bunică voice with no cloning. CPML (non-commercial). |
| Script / continuity | **Qwen3** (or Llama 3.3 70B) | Reads story bible, emits dialogue + per-shot image/motion prompts. |
| Captions | **WhisperX** | Word-level timestamps → karaoke-style burned subs. |
| Assembly | **ffmpeg** | Concat shots, mix music, burn subs, normalize to 9:16 1080×1920. |

> **Decision recorded:** standardize on the **Wan 2.2 family** (S2V + I2V), drop the
> separate InfiniteTalk lineage. Base Wan 2.2 I2V cannot lip-sync to a supplied audio
> track (no audio conditioning); S2V is the first-party audio-driven member.

---

## Repository layout (this repo, synced to box)

```
slopache/
├── README.md
├── .gitignore                 # ignore weights/, assets/, outputs/, .venv/
├── env/
│   ├── bootstrap.sh           # one-time box setup: ComfyUI + custom nodes + venv
│   ├── download_weights.sh    # model manifest → pulls all weights to weights/
│   └── requirements.txt       # XTTS, whisperx, ffmpeg-python, huggingface_hub, etc.
├── comfyui/
│   └── workflows/
│       ├── 01_dataset_gen.json     # Qwen-Image-Edit: founder img → dataset variations
│       ├── 02_still_flux2_lora.json# FLUX.2 + bunică LoRA → shot keyframes
│       ├── 03_talking_wan_s2v.json # Wan 2.2 S2V: still + wav → talking clip
│       └── 04_broll_wan_i2v.json   # Wan 2.2 I2V: still + prompt → b-roll clip
├── training/
│   ├── flux2_bunica_lora.yaml      # ai-toolkit training config
│   └── dataset/                    # (git-ignored) curated 20–50 imgs + captions
├── character/
│   ├── story_bible.md              # canon: name, personality, look, running gags
│   ├── continuity_log.md           # what happened each episode (plot memory)
│   └── prompts/
│       ├── identity.txt            # reusable LoRA trigger + wardrobe fragment
│       └── style.txt               # lighting/lens/look fragment for consistency
├── scripts/
│   ├── write_script.py             # Qwen3: bible + idea → script.json (shotlist)
│   ├── gen_voice.py                # RO XTTS-v2 → per-shot wavs
│   ├── gen_captions.py             # WhisperX → captions.ass
│   └── assemble.py                 # ffmpeg: shots + audio + music + subs → reel.mp4
├── episodes/
│   └── ep01_ce_mananc/             # (git-ignored heavy files) per-episode workdir
│       ├── script.json             # shotlist: dialogue + image/motion prompts
│       ├── stills/  voice/  clips/ captions/  final/
└── assets/
    └── music/                      # (git-ignored) trending/folk audio beds
```

`script.json` schema (per reel):
```json
{
  "episode": "ep01_ce_mananc",
  "music": "assets/music/folk_loop.mp3",
  "shots": [
    {"id": 1, "type": "talking", "line": "Astăzi vă arăt ce mănânc...",
     "image_prompt": "bunicaX woman, front-facing, kitchen, ie", "voice": "Marioara"},
    {"id": 2, "type": "broll", "line": null,
     "image_prompt": "slănină on cutting board, hands", "motion_prompt": "slicing, steam"}
  ]
}
```

---

## Build phases

### Phase 0 — Box bootstrap (one-time, on `visible-rose-reptile`)
1. `env/bootstrap.sh`: clone ComfyUI, create `.venv`, install ComfyUI + custom nodes
   (Wan 2.2 video wrapper, FLUX.2 nodes, Qwen-Image-Edit nodes, LatentSync node),
   install `env/requirements.txt` (XTTS / whisperx / ffmpeg / huggingface_hub).
2. `env/download_weights.sh`: pull FLUX.2-dev, Wan 2.2 S2V + I2V, Qwen-Image-Edit-2511,
   RO XTTS-v2 finetune, WhisperX model, VAE/text-encoders → `weights/` (git-ignored).
3. Smoke test: launch ComfyUI, confirm all 4 workflow JSONs load with no missing nodes.

### Phase 1 — Create the character (one-time, the critical investment)
1. In ComfyUI, generate one **founder image** of the grandma (face, build, ie+catrință).
2. `01_dataset_gen.json` (Qwen-Image-Edit-2511): spin ~30 on-model variations
   (angles, lighting, expressions, indoor/outdoor) from the founder face.
3. Curate to 20–50 best, caption with trigger token (e.g. `bunicaX woman`), drop in
   `training/dataset/`.
4. Train FLUX.2 LoRA via `ai-toolkit` + `flux2_bunica_lora.yaml` (~1–2 h).
5. **Acceptance gate:** generate her in 5 unseen scenes; she must be unmistakably the
   same person. Re-train/curate until she is. *Do not proceed until this passes.*

### Phase 2 — Voice setup (one-time)
1. Audition Marioara / Lăcrămioara from the RO XTTS finetune; pick the bunică voice.
2. If neither fits, clone from a short reference clip (XTTS supports zero-shot).
3. Lock the choice in `character/story_bible.md`.

### Phase 3 — Per-reel production loop (repeat per episode, approve-each-shot)
1. **Script:** `write_script.py` reads `story_bible.md` + `continuity_log.md` + episode
   idea → `script.json` (dialogue + per-shot prompts). Human edits as needed.
2. **Stills:** `02_still_flux2_lora.json` → one keyframe per shot (9:16, e.g. 768×1344).
   **Approve/re-roll each still** before continuing.
3. **Voice:** `gen_voice.py` → wav for each talking shot.
4. **Animate** (per shot): talking → `03_talking_wan_s2v.json` (still+wav);
   b-roll → `04_broll_wan_i2v.json` (still+prompt). Optional LatentSync touch-up.
   **Approve/re-roll each clip.**
5. **Captions:** `gen_captions.py` (WhisperX) → `captions.ass`.
6. **Assemble:** `assemble.py` → `final/reel.mp4` (concat + music duck + burn subs +
   9:16 1080×1920 + loudness normalize).
7. **Update `continuity_log.md`** with what happened so the next script stays canon.

### Phase 4 — Publish (manual to start)
Download `reel.mp4`, review, upload to Instagram by hand. (Automated posting is an
explicit non-goal for now.)

---

## Verification

- **Phase 0:** ComfyUI starts on the box; all 4 workflow JSONs load with zero missing
  nodes; `python -c "import TTS, whisperx"` succeeds; `ffmpeg -version` works.
- **Phase 1 (the make-or-break test):** render the grandma in 5 distinct unseen
  prompts → a human confirms identical face/identity across all 5. This is the gate
  everything else depends on.
- **Phase 3 dry run (end-to-end on one short reel):**
  1. `write_script.py` produces valid `script.json` matching the schema.
  2. Each still renders on-model; each clip lip-syncs (talking) / moves plausibly (b-roll).
  3. `assemble.py` outputs a playable 9:16 1080×1920 MP4 with burned captions, voiceover
     audible over ducked music, correct shot order and total length (~20–35 s).
  4. Manually play the MP4 end-to-end — confirm it reads as one coherent grandma reel.
- **Throughput sanity:** time one full reel on the A100 (~25–45 min expected, dominated
  by Wan generation) to confirm the box handles it without OOM (offload idle weights).

---

## Open considerations / risks

- **LoRA quality is everything.** If identity drifts shot-to-shot, the account looks
  like AI slop. Budget real iteration time on Phase 1; treat its gate as non-negotiable.
- **VRAM choreography.** FLUX.2 + Wan 2.2 + XTTS won't all stay resident at once on
  80 GB during heavy steps — rely on ComfyUI model offloading / sequential stage
  execution (we run stages in order anyway, so this is natural).
- **Romanian TTS edge cases.** Diacritics (ș/ț comma-below vs cedilla), foreign brand
  names in "unboxing" bits. Keep a pronunciation-fix list; spot-check generated audio.
- **Wan 2.2 S2V Romanian:** audio-driven so should be language-agnostic — verify on a
  real Romanian clip early (it's listed as a follow-up check, do it in Phase 0/1).
- **Licensing:** FLUX.2-dev + XTTS CPML are non-commercial. If the account ever
  monetizes, the identity model must move to a commercially-licensed base (Qwen-Image /
  SDXL) and the voice re-sourced — a known future fork, out of scope now.
- **Continuity discipline:** the story bible + continuity log are what make this a
  "story across reels" and not random clips. Keep them updated every episode.

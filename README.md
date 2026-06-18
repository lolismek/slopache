# slopache

Production pipeline for an Instagram/TikTok account built around a **consistent
AI-generated character** — *Bunica Influenceriță*, a Romanian village grandmother
who has accidentally become a lifestyle influencer. The comedy is the clash:
modern girly-influencer tropes executed with rural-village reality.

One command turns a hand-written `script.json` into a finished vertical reel —
voice, talking-head lip-sync, b-roll, sound effects, captions and all. See
[`PIPELINE.md`](./PIPELINE.md) for the full architecture and [`docs/BUILD_LOG.md`](./docs/BUILD_LOG.md)
for the build history.

## Status

The full reel pipeline works end to end. Produced so far:

- **`episodes/ep01_ce_mananc`** — first prototype ("ce mănânc într-o zi").
- **`episodes/ep02_influencer`** — "Morning routine de soft life (la țară)", the
  current quality bar: fp16/bf16 generation, layered audio, optional 60 fps.

Phase 1 (identity) is locked: the FLUX.2 character LoRA is trained and on HF
(`alexjerpelea/bunica-flux2-lora`, trigger **`ohwxbunica`**). The character canon
lives in [`character/story_bible.md`](./character/story_bible.md); the prompt method
that makes episodes look good is in [`character/prompt_recipe.md`](./character/prompt_recipe.md).

## How a reel is made

`scripts/make_reel.py` reads an episode `script.json` and, per shot, chains the
single-purpose clients (everything runs on **one** machine, the GPU box):

```
talking shot:  ElevenLabs voice ─► FLUX.2 + LoRA still ─► Wan 2.2 S2V (lip-synced)
b-roll shot:   ElevenLabs voice (voiceover) ─► FLUX.2 + LoRA still ─► Wan 2.2 I2V (LightX2V, fast)
per shot:      optional ElevenLabs SFX / foley
───────────────────────────────────────────────────────────────────────────────
assemble.py:   trim edge frames ─► layer audio (dialogue / voiceover / SFX / music)
               ─► caption cards ─► concat ─► (optional 60 fps interpolation)
               ─► loudnorm + burn subs ─► final/reel.mp4
```

Generation runs at **480p fp16/bf16** (the A100 has no hardware fp8, so fp8 only
cost fidelity); b-roll uses the LightX2V 4-step LoRAs so it stays fast. See the
box memory / BUILD_LOG for the why.

## Repo layout

```
infra/box.env            box connection (instance, ports; IP resolved dynamically)
infra/remote.sh          ssh / sync / pull / push wrapper  (run from the Mac)
infra/comfy.sh           start/stop/wait the ComfyUI server (run on the box)

env/bootstrap_box.sh     installs ComfyUI + venv + torch              (run on box)
env/download_*.sh        fetch weights: FLUX.2, Wan I2V/S2V, fp16/bf16 (HQ), LoRA
env/setup_*.sh           optional TTS / ai-toolkit setups

scripts/make_reel.py     one-command orchestrator (script.json -> reel.mp4)
scripts/gen_image.py     headless ComfyUI text->image client (FLUX.2 + LoRA)
scripts/gen_video.py     headless ComfyUI image->video client (Wan S2V / I2V)
scripts/gen_voice_eleven.py   ElevenLabs TTS (Romanian, pure stdlib)
scripts/gen_sfx_eleven.py     ElevenLabs sound-generation (foley from a text prompt)
scripts/gen_subs.py      karaoke .ass caption cards (even-split / optional whisperx)
scripts/assemble.py      ffmpeg assembly: trim, audio layers, captions, interpolation
scripts/api_to_ui.py     convert API workflows -> UI graph (for inspection)

workflows/               ComfyUI API-format workflow templates
  flux2_lora_txt2img_api.json   stills with the bunica LoRA
  wan22_s2v_api.json            talking head (audio-driven, fp16/bf16)
  wan22_i2v_api.json            b-roll, full quality (fp16, 30 steps)
  wan22_i2v_fast_api.json       b-roll, LightX2V 4-step (fast)

character/               story_bible.md, prompt_recipe.md, prompts/, lora_dataset/
episodes/<ep>/script.json the per-episode spec (heavy artifacts are gitignored)
docs/BUILD_LOG.md        chronological build notes + hard-won gotchas
```

## Quickstart — generate a reel

Only two actions happen on the Mac: `sync` (push code) and `pull` (fetch the MP4).
Everything else runs on the box. Secrets (`ELEVENLABS_API_KEY`, `HF_TOKEN`) live in
a gitignored repo-root `.env`, which `sync` copies to the box.

```bash
# 0. (one-time) bootstrap the box + download weights
./infra/remote.sh sync
./infra/remote.sh ssh 'cd /ephemeral/slopache && COMFY_DIR=/ephemeral/ComfyUI bash env/bootstrap_box.sh'
./infra/remote.sh ssh 'cd /ephemeral/slopache && COMFY_DIR=/ephemeral/ComfyUI bash env/download_models.sh'   # FLUX.2
./infra/remote.sh ssh 'cd /ephemeral/slopache && COMFY_DIR=/ephemeral/ComfyUI bash env/download_wan_i2v.sh'  # Wan I2V + LightX2V
./infra/remote.sh ssh 'cd /ephemeral/slopache && COMFY_DIR=/ephemeral/ComfyUI bash env/download_s2v.sh'      # Wan S2V
./infra/remote.sh ssh 'cd /ephemeral/slopache && COMFY_DIR=/ephemeral/ComfyUI bash env/download_wan_hq.sh'   # fp16/bf16 (quality)
./infra/remote.sh ssh 'cd /ephemeral/slopache && COMFY_DIR=/ephemeral/ComfyUI bash env/download_lora.sh'     # bunica LoRA

# 1. push code, start ComfyUI
./infra/remote.sh sync
./infra/remote.sh ssh 'cd /ephemeral/slopache && COMFY_DIR=/ephemeral/ComfyUI infra/comfy.sh start && COMFY_DIR=/ephemeral/ComfyUI infra/comfy.sh wait'

# 2. build the reel (one command, fully hands-off with --yes)
./infra/remote.sh ssh 'cd /ephemeral/slopache && python scripts/make_reel.py episodes/ep02_influencer/script.json --yes'

# 3. pull the finished reel back to the Mac
./infra/remote.sh pull /ephemeral/slopache/episodes/ep02_influencer/final/reel.mp4 ./outputs/
```

### Batching episodes (run many overnight)

`scripts/batch_reels.py` runs `make_reel.py --yes` over a queue of episodes in
sequence on the one box. A failing episode is logged (`<ep>/build.log`) and the
batch continues; `--skip-existing` skips episodes that already have a
`final/reel.mp4`, so an interrupted run resumes cleanly. Launch it detached so it
survives an ssh disconnect:

```bash
./infra/remote.sh ssh 'cd /ephemeral/slopache && \
  COMFY_DIR=/ephemeral/ComfyUI infra/comfy.sh start && COMFY_DIR=/ephemeral/ComfyUI infra/comfy.sh wait && \
  nohup python scripts/batch_reels.py --all --skip-existing > batch.log 2>&1 &'

./infra/remote.sh ssh 'tail -n 40 /ephemeral/slopache/batch.log'      # check progress
./infra/remote.sh pull '/ephemeral/slopache/episodes/*/final/reel.mp4' ./outputs/   # pull all finished
```

A full episode is ~45–60 min at 480p, so ~10 fit in an overnight run.

## Writing an episode (`script.json`)

Top-level keys set the look, voice and render config; `shots` is an ordered list.
Each shot is `type: "talking"` (lip-synced dialogue) or `"broll"` (silent video +
optional voiceover). Audio layers are all optional per shot.

```jsonc
{
  "eleven_voice_id": "…", "eleven_model": "eleven_multilingual_v2", "language": "ro",
  "video_width": 480, "video_height": 832, "fps": 16,
  "output_fps": 60,                 // optional: interpolate the final video (off by default)
  "interp_backend": "rife",         // "rife" (fast, GPU; needs env/setup_rife.sh) or "minterpolate" (slow, CPU)
  "music": null, "music_volume": 0.15,
  "trim_head": 2, "trim_tail": 2,   // edge frames trimmed off each clip
  "i2v_workflow": "workflows/wan22_i2v_fast_api.json",   // per-episode workflow override
  "style_prefix": "3D Pixar-style …, Romanian village aesthetic, …",  // every shot
  "character_prefix": "ohwxbunica, … white ie blouse and red floral headscarf",  // grandma shots
  "shots": [
    { "id": 1, "type": "talking", "line": "Bună, fetelor! …",
      "image_prompt": "holding a phone on a ring-light …",
      "motion_prompt": "she talks to the camera, subtle coherent motion" },
    { "id": 2, "type": "broll", "voiceover": "Primul pas: …",
      "image_prompt": "a rooster crowing on a fence at dawn …",
      "motion_prompt": "the rooster flaps and crows, steady camera",
      "sfx": "a rooster crowing loudly at dawn" }
  ]
}
```

`make_reel.py` prepends `style_prefix` (+ `character_prefix` on grandma shots) to
each `image_prompt`, so the look and wardrobe stay consistent across cuts. See
`character/prompt_recipe.md` for the full method (prop continuity, motion wording,
the comedy formula).

## Infrastructure

Generation runs on a Brev **A100 80GB** instance (`stale-blush-orca`). Heavy data
(ComfyUI, model weights, episode artifacts) lives on the box's `/ephemeral` volume,
which is **wiped on deprovision** — this repo (code/config) plus HuggingFace
(weights/LoRA) are the source of truth. Per-episode `stills/voice/sfx/clips/captions/final`
are gitignored; only `script.json` is tracked.

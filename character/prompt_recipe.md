# Prompt recipe — what made ep02 work

The reel that landed (`episodes/ep02_influencer/`) used a layered prompt structure
plus a comedy formula. Reuse this for new episodes; it's the difference between the
rough ep01 and the much-improved ep02.

## 1. Layered image prompts (continuity across cuts)

Every still is built by `make_reel.py` as:

    style_prefix  +  (character_prefix if the shot shows the grandma)  +  per-shot image_prompt

- **`style_prefix`** (episode-level, applied to EVERY shot) — the world + render look, so
  cuts share one aesthetic. What worked:
  > 3D Pixar-style animation, Romanian village aesthetic, warm golden cinematic lighting,
  > soft global illumination, cohesive warm film color grade, highly detailed, vertical 9:16 composition
- **`character_prefix`** (episode-level, applied only to shots with the grandma) — identity +
  fixed wardrobe, so she never changes outfit between cuts. What worked:
  > ohwxbunica, a cheerful elderly Romanian grandmother with rosy cheeks and kind eyes,
  > wearing a white embroidered Romanian ie blouse and a red floral headscarf
  - Talking shots get it automatically; on a b-roll that should show her, set `"character": true`.
  - Pure-scenery b-roll (rooster, well, hens) omits it so the character isn't forced in.
- **Per-shot `image_prompt`** — ONLY the scene-specific subject/action/composition. Don't repeat
  style or wardrobe here; the prefixes own that. Keep `ohwxbunica` out of per-shot prompts too —
  it's already in `character_prefix`.

## 2. Prop continuity (the ep01 "different food" bug, fixed)

When a prop appears in a talking shot AND its b-roll, describe it **identically** in both, and
say "the same …". ep02 example — the sour-cream jar:
- talking (shot 3): "holding a small ceramic jar of thick white sour cream …"
- b-roll  (shot 4): "extreme close-up of **the same** small ceramic jar of thick white sour cream …"

## 3. Motion prompts that reduce incoherent gestures

Describe a single, plausible, natural motion and explicitly ask for coherence/stability. Avoid
piling on multiple actions. What worked:
> she dabs a little cream on her cheek with a fingertip and smiles at the camera,
> playful natural head movement, subtle **coherent** motion

Useful tail words: "subtle natural motion", "coherent", "steady camera", "smooth", "stable framing".
(Negative prompt — blur/warp/extra limbs/flicker — is baked into the Wan workflows.)

## 4. Comedy formula — influencer × Romanian village

The joke is the clash: a modern girly-influencer trope executed with rural village reality.
Lines are Romanian with influencer English loanwords (the loanword IS the punchline), kept short
and punchy so they read well as caption cards.

| Influencer trope        | Village reality (ep02)                                |
|-------------------------|-------------------------------------------------------|
| morning / soft-life routine | wake at 5 to milk the cow ("vaca nu așteaptă like-uri") |
| skincare / "noul retinol"   | smântână (sour cream) from vaca Florica            |
| fit check / sustainable     | hand-sewn ie + "batic vintage din '67"             |
| Paris catwalk               | hens strutting down the ulița                       |
| hot girl walk / hidratare   | walk to the fântână, "apă fără microplastice"       |
| like & share outro          | "dați la găini niște grăunțe. Pupici!"             |

Structure that worked: talking **hook** → alternate talking/b-roll pairs (each b-roll illustrates
the line before it) → talking **outro punchline**. Keep total ~25–35s (ep02 ran 44s — trim a pair).

## 5. Config the prompts were tuned on

480p fp16 (Wan S2V talking + Wan I2V b-roll via LightX2V 4-step), FLUX.2 + bunica LoRA stills,
ElevenLabs voice + sound-generation SFX. See `docs/BUILD_LOG.md` and the box memory for the why.

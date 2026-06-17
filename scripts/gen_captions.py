#!/usr/bin/env python3
"""Build the FLUX.2 character-LoRA training folder from the v2 dataset.

FLUX.2 uses a Mistral-24B VLM text encoder, so captions are NATURAL-LANGUAGE
PROSE (not Danbooru tag soup). Identity-binding discipline:
  - TRIGGER token first on every caption (the identity handle).
  - Class noun only ("elderly Romanian grandmother") -> face stays UN-described
    so it binds to the trigger.
  - Costume is FULLY described (user's choice) -> stays promptable, won't bind.
  - Variable context (scene/framing/expression/lighting/accessory/viewpoint) is
    described so it stays promptable and decorrelated from identity.

Output: character/lora_dataset/ with <name>.png + <name>.txt sidecars, ready to
sync to the box and point ai-toolkit's folder_path at.
"""
import os
import shutil

SRC = "outputs/v2"
DST = "character/lora_dataset"

# Single string, easy to rebrand later (sed the .txt files). Rare-ish coined
# token + we always pair it with the class noun in prose.
TRIGGER = "ohwxbunica"
CLASS = "a 3D Pixar-style elderly Romanian grandmother"
COSTUME = ("a white embroidered Romanian ie blouse, a red floral headscarf, "
           "and a dark catrinta skirt")

def cap(costume, ctx):
    return f"{TRIGGER}, {CLASS} wearing {costume}, {ctx}"

# label -> (source filename, context prose). Costume is COSTUME unless overridden.
SCENES = {
    "kitchen_proud":   "standing in a cozy rustic kitchen with warm hanging lights, a waist-up shot, beaming with a proud smile in warm indoor light.",
    "garden_laugh":    "in a sunny flower garden during golden hour, a full-body shot, laughing warmly with her mouth open.",
    "market_surprise": "at a busy outdoor village market stall in bright daylight, a waist-up shot, with a comically surprised expression.",
    "porch_wave":      "on a wooden village porch in soft overcast daylight, a full-body shot, waving at the camera with a friendly smile.",
    "snow_content":    "on a snowy village street in cool blue daylight, a waist-up shot, with a content closed-mouth smile.",
    "livingroom_think":"in a homely living room lit by a warm lamp, a close-up shot, with a thoughtful expression looking off camera.",
    "field_gentle":    "in an autumn field under hazy afternoon sun, a full-body shot, with a gentle smile.",
    "courtyard_neutral":"in an old stone church courtyard in soft daylight, a waist-up shot, with a relaxed neutral expression.",
    "balcony_wink":    "on a balcony with a town behind her in warm evening light, a waist-up shot, giving a playful wink and grin.",
    "studio_proud":    "against a plain warm grey studio backdrop in soft studio light, a close-up shot, with a proud smile.",
    "lake_look":       "at a lakeside at dawn in bright soft light, a full-body shot, one hand shading her eyes as she looks into the distance.",
    "vineyard_grapes": "in sunny vineyard rows, a waist-up shot, holding a bunch of grapes and smiling.",
    "bakery_smile":    "in a warm village bakery interior, a close-up shot, smiling gently at the camera.",
    "fireplace_cozy":  "in a room with a lit fireplace casting a firelight glow, a waist-up shot, with a cozy content expression.",
    "terrace_laugh":   "on a rooftop terrace in late afternoon light, a close-up shot, laughing.",
    "station_basket":  "on a rustic train platform in daylight, a full-body shot, holding a wicker basket with a calm smile.",
}

# Accessory shots: costume overridden to include the accessory (promptable).
ACCESSORIES = {
    "acc_sunglasses": ("a white embroidered Romanian ie blouse and a red floral headscarf, plus oversized white sunglasses",
                       "in a sunny village yard, a waist-up shot, arms crossed in a confident cool pose."),
    "acc_glasses":    ("a white embroidered Romanian ie blouse and a red floral headscarf, plus small reading glasses perched on her nose",
                       "in a cozy kitchen, a close-up shot, smiling gently."),
    "acc_shawl":      ("a white embroidered Romanian ie blouse and a red floral headscarf, with a heavy knitted wool shawl wrapped over her shoulders",
                       "in a snowy doorway, a waist-up shot, with a warm smile."),
}

# Camera-angle shots (multi-angle LoRA). Describe viewpoint+framing only; the
# background is the founder scene, so we don't assert one (keeps it from baking).
ANGLES = {
    "front_eye":  "seen from a front view at eye level, a medium shot, with a neutral expression.",
    "frq_eye":    "seen from a three-quarter front-right view at eye level, a medium shot.",
    "flq_eye":    "seen from a three-quarter front-left view at eye level, a medium shot.",
    "right_eye":  "seen from a right-side profile view at eye level, a medium shot.",
    "left_eye":   "seen from a left-side profile view at eye level, a medium shot.",
    "front_low":  "seen from a front view at a low angle looking up, a medium shot.",
    "front_high": "seen from a front view at a high angle looking down, a medium shot.",
    "frq_cu":     "seen from a three-quarter front-right view at eye level, a close-up shot.",
}

def find_src(label):
    """Match dataset_<label>_<NNNNN>_.png in SRC."""
    for f in sorted(os.listdir(SRC)):
        if f.endswith(".png") and (f.startswith(f"dataset_{label}_")
                                   or f == f"dataset_{label}.png"):
            return f
    return None

def main():
    os.makedirs(DST, exist_ok=True)
    rows = []
    for label, ctx in SCENES.items():
        rows.append((label, COSTUME, ctx))
    for label, (costume, ctx) in ACCESSORIES.items():
        rows.append((label, costume, ctx))
    for label, ctx in ANGLES.items():
        rows.append((label, COSTUME, ctx))
    # founder anchor
    rows.append(("00_founder", COSTUME,
                 "seen from a front view, a close-up portrait, with a gentle smile."))

    n = 0
    missing = []
    for label, costume, ctx in rows:
        src = find_src(label)
        if not src:
            missing.append(label); continue
        stem = f"{label}"
        shutil.copy(os.path.join(SRC, src), os.path.join(DST, stem + ".png"))
        with open(os.path.join(DST, stem + ".txt"), "w") as fh:
            fh.write(cap(costume, ctx) + "\n")
        n += 1
    print(f"[captions] wrote {n} image+caption pairs -> {DST}")
    print(f"[captions] trigger token = '{TRIGGER}'")
    if missing:
        print(f"[captions] WARNING missing sources for: {missing}")

if __name__ == "__main__":
    main()

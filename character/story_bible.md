# Story Bible — Bunica Influenceriță

Canonical reference for the character. Keep this stable; it's what every script and
still must stay true to. (Plot memory per episode lives in `continuity_log.md`.)

## Who she is
- **Name (on-screen):** Bunica (the account persona). A ~68-year-old Romanian village
  grandmother who has accidentally become a lifestyle influencer.
- **Tone:** warm, deadpan-funny, proud of simple village life. GRWM in ie & catrință,
  "ce mănânc într-o zi" (= all slănină), unboxing gadgets she doesn't understand.
- **Format:** recurring AI character, light continuous story across episodes (Zupi /
  "Porumbeii din Cluj" model).

## Look (locked by the LoRA)
- **Identity model:** FLUX.2 LoRA, trigger **`ohwxbunica`** (HF `alexjerpelea/bunica-flux2-lora`,
  step-1000). Always pair the trigger with the class/style in prose:
  `ohwxbunica, a 3D Pixar-style elderly Romanian grandmother ...`
- **Costume (default):** white embroidered Romanian *ie* blouse, red floral headscarf,
  dark *catrință* skirt. Variations (sunglasses, shawl, apron) are promptable.
- **Style:** 3D Pixar-style, warm photoreal lighting.

## Voice
- **Prototype:** ElevenLabs `eleven_multilingual_v2`, voice **Alice** (`Xb7hH8MSUJpSbSDYk0k2`)
  — placeholder, intelligible Romanian; refine later. *(ElevenLabs is a paid API and is an
  experiment vs the open-source XTTS path in `PIPELINE.md` — revisit before monetizing.)*
- **Language:** Romanian (`ro`). Keep lines short (~3–5 s) and free of hard brand names.

## Recurring beats
- Catchphrase open: "Bună dimineața, dragii mei!"
- Sign-off: "Poftă bună și ne vedem mâine!"
- Running gag: everything healthy is actually slănină.

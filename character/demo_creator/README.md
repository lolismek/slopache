# demo_creator — placeholder reference sheet

This folder is the target of `episodes/_template_noLoRA/script.json`'s
`character_refs`. It ships **empty on purpose** — mint your own canonical
reference sheet here before running the template:

```bash
# front view (seed-search a few, keep the best one)
python scripts/gen_image.py --workflow workflows/flux2_txt2img_api.json \
    --prompt "<style_prefix>, a cheerful young woman …, neutral front portrait" \
    --out front --batch 4 --outdir character/demo_creator/refs

# expression view, conditioned on the front so the set agrees (front angle = clean)
python scripts/gen_image.py --workflow workflows/flux2_txt2img_api.json \
    --ref character/demo_creator/refs/front.png \
    --prompt "<style_prefix>, the same woman, warm smile, front view" \
    --out smile --outdir character/demo_creator/refs
```

NOTE: asking `--ref` for a turned *angle* ("three-quarter view") tends to make
FLUX.2 compose the reference plus a second copy side-by-side — mint angles as
their own txt2img, and keep `--ref` for expression changes on the front angle.

Keep 2–4 clean, neutral, well-lit views (front / expression / a turned angle).
Commit them — they are the no-LoRA identity artifact (the `*.png` gitignore has a
carve-out for `character/*/refs/`). Then `python scripts/make_reel.py
episodes/_template_noLoRA/script.json --yes`.

Use the **same** fixed set for every shot; never chain off the previous still.
See the README section "Identity without a trained LoRA".

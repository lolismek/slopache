#!/usr/bin/env python3
"""Build the LoRA training dataset by editing ONE founder image into many on-model
variations with Qwen-Image-Edit-2511 (run on the box, ComfyUI serving on :8188).

Why edit instead of re-prompt: re-prompting FLUX yields a different face each time;
Qwen-Image-Edit takes the actual founder image as input and changes pose/angle/
expression/scene/light while holding the identity + wardrobe fixed — exactly what a
character LoRA needs.

Reads an edits file ("label | instruction" per line, '#' comments ignored), runs
each instruction over --image, varies the seed per item, and saves dataset_<label>*.
Curate the outputs to 20-50 keepers, caption with the trigger token, then train.

The founder image must already be in ComfyUI/input/ (push it with infra/remote.sh).

Usage (on the box):
    python scripts/gen_dataset.py --image pixar_portrait_b_00001_.png \
        --edits character/prompts/dataset_edits.txt --outdir outputs/dataset
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import uuid


def load_edits(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            label, instruction = line.split("|", 1)
            items.append((label.strip(), instruction.strip()))
    return items


def patch(wf, image, prompt, seed, out):
    for node in wf.values():
        ct = node.get("class_type", "")
        title = (node.get("_meta", {}).get("title") or "")
        ins = node.setdefault("inputs", {})
        if title == "POSITIVE_PROMPT":
            for key in ("prompt", "text"):  # Qwen edit node uses 'prompt'; CLIP encoders use 'text'
                if key in ins:
                    ins[key] = prompt
        if ct == "LoadImage":
            ins["image"] = image
        for k in ("seed", "noise_seed"):
            if k in ins:
                ins[k] = seed
        if ct == "SaveImage":
            ins["filename_prefix"] = out
    return wf


def submit(server, wf):
    body = json.dumps({"prompt": wf, "client_id": uuid.uuid4().hex}).encode()
    req = urllib.request.Request(f"http://{server}/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["prompt_id"]


def wait(server, pid, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = json.loads(urllib.request.urlopen(f"http://{server}/history/{pid}").read())
        if pid in hist:
            return hist[pid]
        time.sleep(2)
    raise TimeoutError(f"prompt {pid} did not finish within {timeout}s")


def download(server, hist, outdir):
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for out in hist.get("outputs", {}).values():
        for img in out.get("images", []):
            q = urllib.parse.urlencode({"filename": img["filename"],
                                        "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "output")})
            data = urllib.request.urlopen(f"http://{server}/view?{q}").read()
            path = os.path.join(outdir, img["filename"])
            with open(path, "wb") as f:
                f.write(data)
            saved.append(path)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="founder filename in ComfyUI/input/")
    ap.add_argument("--edits", default="character/prompts/dataset_edits.txt")
    ap.add_argument("--workflow", default="workflows/qwen_edit_api.json")
    ap.add_argument("--outdir", default="outputs/dataset")
    ap.add_argument("--seed", type=int, default=1000, help="base seed; +i per item")
    ap.add_argument("--server", default="127.0.0.1:8188")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    with open(args.workflow) as f:
        template = f.read()
    edits = load_edits(args.edits)
    print(f"[dataset] {len(edits)} edits over {args.image} -> {args.outdir}")
    for i, (label, instruction) in enumerate(edits):
        wf = patch(json.loads(template), args.image, instruction,
                   args.seed + i, f"dataset_{label}")
        pid = submit(args.server, wf)
        hist = wait(args.server, pid, args.timeout)
        saved = download(args.server, hist, args.outdir)
        print(f"  [{i+1}/{len(edits)}] {label}: {', '.join(os.path.basename(p) for p in saved)}")
    print(f"[dataset] done -> {args.outdir} (curate to 20-50, caption with trigger token)")


if __name__ == "__main__":
    main()

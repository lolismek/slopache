#!/usr/bin/env python3
"""Headless ComfyUI text-to-image client.

Loads an API-format workflow template, patches prompt / seed / size / output
prefix, submits it to a running ComfyUI server, waits, and downloads the images.

Built to scale: drive many prompts/seeds from one process; outputs are named
deterministically so batches stay organized. The workflow template is data, so
swapping FLUX.2 for another model later is a template change, not a code change.

The template tags the nodes it patches via their `_meta.title`:
    POSITIVE_PROMPT   CLIPTextEncode whose text becomes --prompt
    NEGATIVE_PROMPT   CLIPTextEncode whose text becomes --negative
Any node exposing seed/noise_seed, width/height/batch_size, steps, or a
SaveImage filename_prefix is patched automatically.

Usage (run on the box, where ComfyUI serves on 127.0.0.1:8188):
    python scripts/gen_image.py --prompt "$(cat character/prompts/founder.txt)" \
        --out grandma_founder --batch 4 --seed 0
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import uuid


def load_workflow(path):
    with open(path) as f:
        return json.load(f)


def patch(wf, args):
    for node in wf.values():
        ct = node.get("class_type", "")
        title = (node.get("_meta", {}).get("title") or "")
        ins = node.setdefault("inputs", {})
        if title == "POSITIVE_PROMPT":
            ins["text"] = args.prompt
        if title == "NEGATIVE_PROMPT" and args.negative is not None:
            ins["text"] = args.negative
        for k in ("seed", "noise_seed"):
            if k in ins and args.seed is not None:
                ins[k] = args.seed
        if {"width", "height"} <= ins.keys():
            ins["width"], ins["height"] = args.width, args.height
            if "batch_size" in ins:
                ins["batch_size"] = args.batch
        if "steps" in ins and args.steps is not None:
            ins["steps"] = args.steps
        if ct == "SaveImage":
            ins["filename_prefix"] = args.out
    return wf


def submit(server, wf, client_id):
    body = json.dumps({"prompt": wf, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"http://{server}/prompt", data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req).read())["prompt_id"]


def wait(server, pid, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = json.loads(urllib.request.urlopen(f"http://{server}/history/{pid}").read())
        if pid in hist:
            entry = hist[pid]
            # Presence in history != success; a run can be there because it errored
            # or was interrupted. Fail loudly so callers don't treat it as done.
            st = entry.get("status", {})
            if st and st.get("status_str") == "error":
                raise RuntimeError(f"prompt {pid} errored in ComfyUI: "
                                   f"{st.get('messages', '?')}")
            if st and st.get("completed") is False:
                raise RuntimeError(f"prompt {pid} did not complete (interrupted)")
            return entry
        time.sleep(2)
    raise TimeoutError(f"prompt {pid} did not finish within {timeout}s")


def download(server, hist, outdir):
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for out in hist.get("outputs", {}).values():
        for img in out.get("images", []):
            q = urllib.parse.urlencode({
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            })
            data = urllib.request.urlopen(f"http://{server}/view?{q}").read()
            path = os.path.join(outdir, img["filename"])
            with open(path, "wb") as f:
                f.write(data)
            saved.append(path)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=None)
    ap.add_argument("--workflow", default="workflows/flux2_txt2img_api.json")
    ap.add_argument("--out", default="img")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1344)  # 9:16
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--server", default="127.0.0.1:8188")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    wf = patch(load_workflow(args.workflow), args)
    client_id = uuid.uuid4().hex
    pid = submit(args.server, wf, client_id)
    print(f"[gen] submitted prompt_id={pid} ({args.batch}x {args.width}x{args.height} seed={args.seed})")
    hist = wait(args.server, pid, args.timeout)
    saved = download(args.server, hist, args.outdir)
    print(f"[gen] saved {len(saved)} image(s):")
    for p in saved:
        print("   ", p)


if __name__ == "__main__":
    main()

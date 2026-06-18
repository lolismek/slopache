#!/usr/bin/env python3
"""Headless ComfyUI image-to-video client (Wan 2.2 I2V).

Patches an API-format I2V workflow with a start image, motion prompt, seed, size,
clip length, and output prefix, submits to a running ComfyUI server, and waits
for completion. The produced video lands in ComfyUI/output/<prefix>* — pull it
from the box with infra/remote.sh.

The start image must already be in ComfyUI/input/ (the runner copies it there).

Patch targets (by node class / _meta.title):
    POSITIVE_PROMPT  CLIPTextEncode whose text becomes --prompt
    LoadImage        inputs.image  := --image
    WanImageToVideo  width/height/length := --width/--height/--length
    *                noise_seed    := --seed
    SaveVideo        filename_prefix := --out

Usage (on the box):
    python scripts/gen_video.py --workflow workflows/wan22_i2v_api.json \
        --image grandma_photoreal.png --prompt "she smiles and waves" \
        --out i2v_photoreal --length 49
"""
import argparse
import json
import time
import urllib.request
import uuid


def submit(server, wf, client_id):
    body = json.dumps({"prompt": wf, "client_id": client_id}).encode()
    req = urllib.request.Request(f"http://{server}/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["prompt_id"]


def wait(server, pid, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = json.loads(urllib.request.urlopen(f"http://{server}/history/{pid}").read())
        if pid in hist:
            entry = hist[pid]
            # ComfyUI writes the history entry whether the run SUCCEEDED, errored,
            # or was interrupted — so presence != success. Surface failures loudly
            # instead of returning an empty result that downstream reads as "done".
            st = entry.get("status", {})
            if st and st.get("status_str") == "error":
                msgs = st.get("messages", [])
                detail = next((m for m in msgs if isinstance(m, list)
                               and m and "error" in str(m[0]).lower()), msgs[-1:] or "?")
                raise RuntimeError(f"prompt {pid} errored in ComfyUI: {detail}")
            if st and st.get("completed") is False:
                raise RuntimeError(f"prompt {pid} did not complete (interrupted)")
            return entry
        time.sleep(3)
    raise TimeoutError(f"prompt {pid} did not finish within {timeout}s")


def patch(wf, args):
    for node in wf.values():
        ct = node.get("class_type", "")
        title = (node.get("_meta", {}).get("title") or "")
        ins = node.setdefault("inputs", {})
        if title == "POSITIVE_PROMPT":
            ins["text"] = args.prompt
        if title == "NEGATIVE_PROMPT" and args.negative is not None:
            ins["text"] = args.negative
        if ct == "LoadImage":
            ins["image"] = args.image
        if ct == "LoadAudio" and args.audio:
            ins["audio"] = args.audio
        if ct in ("WanImageToVideo", "WanSoundImageToVideo"):
            ins["width"], ins["height"], ins["length"] = args.width, args.height, args.length
        for k in ("seed", "noise_seed"):   # KSampler uses seed; KSamplerAdvanced uses noise_seed
            if k in ins:
                ins[k] = args.seed
        if ct == "SaveVideo":
            ins["filename_prefix"] = args.out
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default="workflows/wan22_i2v_api.json")
    ap.add_argument("--image", required=True, help="filename in ComfyUI/input/")
    ap.add_argument("--audio", default=None, help="audio filename in ComfyUI/input/ (S2V)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=None)
    ap.add_argument("--out", default="i2v")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--length", type=int, default=49)
    ap.add_argument("--server", default="127.0.0.1:8188")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    with open(args.workflow) as f:
        wf = patch(json.load(f), args)
    pid = submit(args.server, wf, uuid.uuid4().hex)
    print(f"[i2v] submitted {pid}: {args.image} -> {args.out} "
          f"({args.width}x{args.height}, {args.length} frames, seed {args.seed})")
    hist = wait(args.server, pid, args.timeout)
    outs = hist.get("outputs", {})
    print("[i2v] done. output nodes:")
    for nid, out in outs.items():
        for key, items in out.items():
            for it in items:
                if isinstance(it, dict) and "filename" in it:
                    print(f"    node {nid} [{key}]: {it.get('subfolder','')}/{it['filename']}")
    print("[i2v] video saved under ComfyUI/output/<prefix>* on the box.")


if __name__ == "__main__":
    main()

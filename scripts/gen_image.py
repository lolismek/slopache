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


# ComfyUI node that appends a VAE-encoded reference image to the positive
# conditioning so FLUX.2 anchors identity/look on it (shared with Flux Kontext).
# If a given ComfyUI build names it differently, change this one constant —
# discover the real name via GET /object_info on the box.
REF_NODE = "ReferenceLatent"


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


def find_node(wf, class_type):
    """First node id with the given class_type, else None."""
    for nid, node in wf.items():
        if node.get("class_type") == class_type:
            return nid
    return None


def upload_image(server, path):
    """Upload a local image into ComfyUI's input folder so a LoadImage node can
    reference it by name. Returns the server-side filename (subfolder-qualified)."""
    fname = os.path.basename(path)
    with open(path, "rb") as f:
        blob = f.read()
    boundary = "----comfy" + uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{fname}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n", blob, b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
        f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"http://{server}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    r = json.loads(urllib.request.urlopen(req).read())
    return f"{r['subfolder']}/{r['name']}" if r.get("subfolder") else r["name"]


def inject_references(wf, ref_names):
    """Splice VAE-encoded reference images into the positive conditioning chain.
    Identity comes from the references, not a trained LoRA — the no-LoRA path.
    ref_names are server-side input filenames (from upload_image), applied in
    order. No-op when ref_names is empty, so every existing call is unaffected."""
    if not ref_names:
        return wf
    vae = find_node(wf, "VAELoader")
    guider = find_node(wf, "FluxGuidance")
    if not vae or not guider:
        raise SystemExit("[gen] --ref needs a FLUX.2 template with VAELoader + FluxGuidance")
    cond = wf[guider]["inputs"]["conditioning"]  # current positive conditioning [id, idx]
    for i, name in enumerate(ref_names, 1):
        lid, eid, rid = f"ref_load{i}", f"ref_enc{i}", f"ref_lat{i}"
        wf[lid] = {"class_type": "LoadImage", "inputs": {"image": name},
                   "_meta": {"title": f"REF_IMAGE_{i}"}}
        wf[eid] = {"class_type": "VAEEncode", "inputs": {"pixels": [lid, 0], "vae": [vae, 0]},
                   "_meta": {"title": f"REF_ENCODE_{i}"}}
        wf[rid] = {"class_type": REF_NODE, "inputs": {"conditioning": cond, "latent": [eid, 0]},
                   "_meta": {"title": f"REF_LATENT_{i}"}}
        cond = [rid, 0]
    wf[guider]["inputs"]["conditioning"] = cond
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
    ap.add_argument("--ref", action="append", default=[], metavar="IMG",
                    help="reference image for identity (no-LoRA FLUX.2 path); repeatable")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--server", default="127.0.0.1:8188")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    wf = patch(load_workflow(args.workflow), args)
    if args.ref:
        ref_names = [upload_image(args.server, r) for r in args.ref]
        wf = inject_references(wf, ref_names)
        print(f"[gen] conditioning on {len(ref_names)} reference image(s): {', '.join(ref_names)}")
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

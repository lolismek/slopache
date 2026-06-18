#!/usr/bin/env python3
"""One-command reel orchestrator. Runs ON the box; ComfyUI must be up (infra/comfy.sh).

Reads an episode script.json and, per shot, chains the existing single-purpose
clients (no reimplementation):

  talking shot:  gen_voice_eleven.py -> mp3  (ElevenLabs; pure HTTP, runs anywhere)
                 gen_image.py (flux2_lora_txt2img) -> on-model still
                 gen_video.py (wan22_s2v, still+audio, length sized to the voice) -> clip
  b-roll shot:   gen_image.py -> still
                 gen_video.py (wan22_i2v, still+motion prompt) -> silent clip

then assemble.py stitches clips + captions + (optional) music into final/reel.mp4.

Everything is on one machine: ElevenLabs is just an API call, so voice happens here
too. Trigger from the Mac with:
    ./infra/remote.sh ssh 'cd $REMOTE_ROOT && python scripts/make_reel.py \
        episodes/ep01_ce_mananc/script.json --yes'
then ./infra/remote.sh pull .../final/reel.mp4 ./outputs/
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
COMFY_DIR = os.environ.get("COMFY_DIR", "/ephemeral/ComfyUI")
PY = sys.executable

STILL_WF = "workflows/flux2_lora_txt2img_api.json"
S2V_WF = "workflows/wan22_s2v_api.json"
I2V_WF = "workflows/wan22_i2v_api.json"


def run(cmd):
    print("[reel]", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO)


def newest(pattern):
    hits = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    return hits[-1] if hits else None


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", path], capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def snap_len(seconds, fps, pad=8, lo=49, hi=161):
    """Wan length must be ~4k+1 frames. Size to the audio + a little tail."""
    n = int(round(seconds * fps)) + pad
    n = max(lo, min(n, hi))
    k = round((n - 1) / 4)
    return 4 * k + 1


def comfy_up(server):
    try:
        with urllib.request.urlopen(f"http://{server}/", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--server", default="127.0.0.1:8188")
    ap.add_argument("--yes", action="store_true", help="no per-shot approval prompts")
    ap.add_argument("--skip-existing", action="store_true", help="reuse clips already in clips/")
    ap.add_argument("--still-steps", type=int, default=None)
    ap.add_argument("--no-assemble", action="store_true")
    ap.add_argument("--align", action="store_true", help="pass through to assemble (whisperx)")
    args = ap.parse_args()

    if not comfy_up(args.server):
        sys.exit(f"[reel] ComfyUI not reachable at {args.server} — start it: "
                 f"COMFY_DIR={COMFY_DIR} infra/comfy.sh start && infra/comfy.sh wait")

    with open(os.path.join(REPO, args.script)) as f:
        spec = json.load(f)
    ep = os.path.dirname(os.path.abspath(os.path.join(REPO, args.script)))
    for sub in ("stills", "voice", "clips", "captions", "final"):
        os.makedirs(os.path.join(ep, sub), exist_ok=True)
    cin = os.path.join(COMFY_DIR, "input"); os.makedirs(cin, exist_ok=True)
    cout = os.path.join(COMFY_DIR, "output")

    sw, sh = spec.get("still_width", 768), spec.get("still_height", 1344)
    vw, vh = spec.get("video_width", 480), spec.get("video_height", 832)
    fps = spec.get("fps", 16)
    voice_id = spec["eleven_voice_id"]
    voice_model = spec.get("eleven_model", "eleven_multilingual_v2")

    for s in spec["shots"]:
        sid = s["id"]
        clip_dst = os.path.join(ep, "clips", f"shot{sid}.mp4")
        if args.skip_existing and os.path.exists(clip_dst):
            print(f"[reel] shot {sid}: clip exists, skipping")
            continue
        talking = s.get("type") == "talking"
        print(f"\n[reel] === shot {sid} ({s.get('type')}) ===")

        # 1. voice (talking only) — ElevenLabs, then copy into ComfyUI/input
        s2v_len = None
        if talking:
            vmp3 = os.path.join(ep, "voice", f"shot{sid}.mp3")
            run([PY, "scripts/gen_voice_eleven.py", "--voice-id", voice_id,
                 "--model", voice_model, "--text", s["line"], "--out", vmp3])
            # ComfyUI LoadAudio is happiest with wav; transcode into the input dir.
            wav_in = os.path.join(cin, f"shot{sid}.wav")
            subprocess.run(["ffmpeg", "-y", "-i", vmp3, "-ar", "44100", "-ac", "1", wav_in],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            s2v_len = snap_len(probe_duration(vmp3), fps)
            print(f"[reel] voice {probe_duration(vmp3):.1f}s -> S2V length {s2v_len} frames")

        # 2. still — FLUX.2 + bunica LoRA, then copy into ComfyUI/input
        still_cmd = [PY, "scripts/gen_image.py", "--workflow", STILL_WF,
                     "--prompt", s["image_prompt"], "--out", f"shot{sid}",
                     "--width", sw, "--height", sh, "--seed", sid,
                     "--outdir", os.path.join(ep, "stills")]
        if args.still_steps:
            still_cmd += ["--steps", args.still_steps]
        run(still_cmd)
        still = newest(os.path.join(ep, "stills", f"shot{sid}_*.png"))
        if not still:
            sys.exit(f"[reel] no still produced for shot {sid}")
        shutil.copy(still, os.path.join(cin, f"shot{sid}.png"))

        # 3. animate — S2V (talking) or I2V (b-roll)
        while True:
            if talking:
                run([PY, "scripts/gen_video.py", "--workflow", S2V_WF,
                     "--image", f"shot{sid}.png", "--audio", f"shot{sid}.wav",
                     "--prompt", s["motion_prompt"], "--out", f"shot{sid}",
                     "--width", vw, "--height", vh, "--length", s2v_len, "--seed", sid])
            else:
                run([PY, "scripts/gen_video.py", "--workflow", I2V_WF,
                     "--image", f"shot{sid}.png", "--prompt", s["motion_prompt"],
                     "--out", f"shot{sid}", "--width", vw, "--height", vh,
                     "--length", s.get("frames", 49), "--seed", sid])
            produced = newest(os.path.join(cout, "**", f"shot{sid}*.mp4")) or \
                newest(os.path.join(cout, f"shot{sid}*.mp4"))
            if not produced:
                sys.exit(f"[reel] no clip produced for shot {sid} (check ComfyUI logs)")
            shutil.copy(produced, clip_dst)
            print(f"[reel] shot {sid} -> {clip_dst}")
            if args.yes:
                break
            ans = input(f"[reel] accept shot {sid}? [Enter=yes / r=regenerate] ").strip().lower()
            if ans != "r":
                break

    if args.no_assemble:
        print("[reel] clips done; skipping assembly (--no-assemble)")
        return
    asm = [PY, "scripts/assemble.py", "--script", args.script]
    if args.align:
        asm.append("--align")
    run(asm)
    print(f"\n[reel] REEL READY -> {os.path.join(ep, 'final', 'reel.mp4')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assemble per-shot clips into a finished 9:16 reel with ffmpeg.

Reads an episode `script.json` and the per-shot clips the orchestrator produced in
`<ep>/clips/shot<id>.mp4` (talking S2V clips carry their voice audio baked in;
b-roll I2V clips are silent). Stages, each writing an inspectable intermediate to
`<ep>/.tmp/`:

  A. normalize every clip to the target WxH (cover+crop), fps, 44.1k stereo audio
     (b-roll gets a silent track so concat lines up)
  B. concat in shot order            -> body.mp4
  C. karaoke captions from the lines -> captions/captions.ass   (scripts/gen_subs.py)
  D. duck-mix optional music + loudnorm + burn subs -> final/reel.mp4

Usage (on the box, where ffmpeg lives):
    python scripts/assemble.py --script episodes/ep01_ce_mananc/script.json
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def run(cmd):
    print("[ff]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", path], capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                          "-show_entries", "stream=index", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    return bool(json.loads(out.stdout).get("streams"))


def normalize(src, dst, w, h, fps):
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
          f"crop={w}:{h},fps={fps},setsar=1,format=yuv420p")
    cmd = ["ffmpeg", "-y", "-i", src]
    if not has_audio(src):
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-ar", "44100", "-ac", "2", "-shortest", dst]
    run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--target", default="1080x1920")
    ap.add_argument("--font-size", type=int, default=84)
    ap.add_argument("--align", action="store_true", help="force-align captions with whisperx")
    ap.add_argument("--align-device", default="cuda")
    ap.add_argument("--music", default=None, help="override music path (else script.music)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.script) as f:
        spec = json.load(f)
    ep = os.path.dirname(os.path.abspath(args.script))
    w, h = (int(x) for x in args.target.lower().split("x"))
    fps = spec.get("fps", 16)
    shots = spec["shots"]

    clips_dir = os.path.join(ep, "clips")
    tmp = os.path.join(ep, ".tmp"); os.makedirs(tmp, exist_ok=True)
    cap_dir = os.path.join(ep, "captions"); os.makedirs(cap_dir, exist_ok=True)
    out = args.out or os.path.join(ep, "final", "reel.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # A. normalize, and build the caption timeline from real clip durations.
    norm_files, segments, t = [], [], 0.0
    for s in shots:
        clip = os.path.join(clips_dir, f"shot{s['id']}.mp4")
        if not os.path.exists(clip):
            sys.exit(f"[assemble] missing clip: {clip}")
        dst = os.path.join(tmp, f"norm{s['id']}.mp4")
        normalize(clip, dst, w, h, fps)
        dur = probe_duration(dst)
        if s.get("type") == "talking" and (s.get("line") or "").strip():
            segments.append({"text": s["line"].strip(), "start": round(t, 3),
                             "end": round(t + dur, 3)})
        norm_files.append(dst)
        t += dur
    print(f"[assemble] total duration ~{t:.1f}s, {len(segments)} caption segment(s)")

    # B. concat (re-encode via concat filter so minor param drift can't break it).
    body = os.path.join(tmp, "body.mp4")
    inputs = []
    for nf in norm_files:
        inputs += ["-i", nf]
    streams = "".join(f"[{i}:v][{i}:a]" for i in range(len(norm_files)))
    fc = f"{streams}concat=n={len(norm_files)}:v=1:a=1[v][a]"
    run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", body])

    # C. captions
    ass = os.path.join(cap_dir, "captions.ass")
    if segments:
        seg_json = os.path.join(tmp, "segments.json")
        with open(seg_json, "w") as f:
            json.dump(segments, f, ensure_ascii=False)
        cmd = [sys.executable, os.path.join(HERE, "gen_subs.py"), "--segments", seg_json,
               "--out", ass, "--play-res", f"{w}x{h}", "--font-size", str(args.font_size)]
        if args.align:
            # whisperx aligns against the concatenated audio track.
            body_wav = os.path.join(tmp, "body.wav")
            run(["ffmpeg", "-y", "-i", body, "-vn", "-ac", "1", "-ar", "16000", body_wav])
            cmd += ["--audio", body_wav, "--language", spec.get("language", "ro"),
                    "--device", args.align_device]
        subprocess.run(cmd, check=True)
    else:
        ass = None

    # D. music duck-mix + loudnorm + burn subs -> final
    music = args.music or spec.get("music")
    cmd = ["ffmpeg", "-y"]
    if music:
        music_path = music if os.path.isabs(music) else os.path.join(REPO, music)
        if not os.path.exists(music_path):
            sys.exit(f"[assemble] music not found: {music_path}")
        cmd += ["-i", body, "-stream_loop", "-1", "-i", music_path]
        afilter = ("[1:a]volume=0.18[m];[0:a][m]amix=inputs=2:duration=first:"
                   "dropout_transition=0[mx];[mx]loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    else:
        cmd += ["-i", body]
        afilter = "[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[a]"
    vfilter = f"[0:v]ass=filename='{ass}'[v]" if ass else "[0:v]copy[v]"
    cmd += ["-filter_complex", f"{vfilter};{afilter}", "-map", "[v]", "-map", "[a]",
            "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
            "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-shortest", out]
    run(cmd)
    print(f"[assemble] done -> {out}")


if __name__ == "__main__":
    main()

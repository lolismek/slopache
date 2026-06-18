#!/usr/bin/env python3
"""Assemble per-shot clips into a finished 9:16 reel with ffmpeg.

Reads an episode `script.json` and the per-shot clips the orchestrator produced in
`<ep>/clips/shot<id>.mp4`, plus any per-shot audio assets it generated:
`<ep>/voice/shot<id>.mp3` (dialogue for talking shots / voiceover for b-roll) and
`<ep>/sfx/shot<id>.mp3` (foley). Stages, each writing an inspectable intermediate
to `<ep>/.tmp/`:

  A. per shot: TRIM the distorted head/tail frames (a Wan boundary artifact),
     scale/crop to target WxH, then build a layered audio track:
         base   = baked dialogue (talking) OR voiceover (b-roll) OR silence
         + sfx  = mixed under the base at a lower volume (if present)
     -> norm<id>.mp4
  B. concat in shot order            -> body.mp4
  C. caption cards from the spoken lines -> captions/captions.ass (gen_subs.py)
  D. duck-mix optional music bed + loudnorm + burn subs -> final/reel.mp4

Audio layers are all optional per shot, so each episode uses only what it needs.

Usage (on the box, where ffmpeg lives):
    python scripts/assemble.py --script episodes/ep02_influencer/script.json
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def run(cmd):
    print("[ff]", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", path], capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                          "-show_entries", "stream=index", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    return bool(json.loads(out.stdout).get("streams"))


def find_asset(ep, kind, sid):
    """First existing <ep>/<kind>/shot<sid>.<ext> (mp3/wav/m4a), or None."""
    for ext in ("mp3", "wav", "m4a", "aac"):
        p = os.path.join(ep, kind, f"shot{sid}.{ext}")
        if os.path.exists(p):
            return p
    return None


def normalize_shot(clip, dst, w, h, fps, keep, head, talking,
                   voiceover, sfx, sfx_vol):
    """Trim head/tail, scale/crop the video, and assemble the shot's audio layers
    into one normalized clip of length `keep` seconds."""
    tmp = os.path.dirname(dst)
    vsil = os.path.join(tmp, f"v_{os.path.basename(dst)}")
    shot_a = os.path.join(tmp, f"a_{os.path.basename(dst)}.wav")

    # --- video: input-seek past the distorted head, keep `keep`s, drop audio ---
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
          f"crop={w}:{h},fps={fps},setsar=1,format=yuv420p")
    run(["ffmpeg", "-y", "-ss", f"{head:.3f}", "-i", clip, "-t", f"{keep:.3f}",
         "-an", "-vf", vf, "-r", str(fps), "-c:v", "libx264", "-preset", "medium",
         "-crf", "18", vsil])

    # --- audio: base (dialogue/voiceover/silence) with sfx mixed under ---
    inputs, idx = [], 0
    base_clip_audio = talking and has_audio(clip)
    if base_clip_audio:
        inputs += ["-ss", f"{head:.3f}", "-i", clip, "-t", f"{keep:.3f}"]  # trimmed dialogue
    elif voiceover:
        inputs += ["-i", voiceover]
    else:
        inputs += ["-f", "lavfi", "-t", f"{keep:.3f}",
                   "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    base_idx = idx
    idx += 1

    parts = []
    if sfx:
        inputs += ["-i", sfx]
        sfx_idx = idx
        parts.append(f"[{base_idx}:a]apad[base]")
        parts.append(f"[{sfx_idx}:a]volume={sfx_vol},apad[sfx]")
        parts.append(f"[base][sfx]amix=inputs=2:duration=longest:dropout_transition=0,"
                     f"atrim=0:{keep:.3f},asetpts=N/SR/TB[a]")
    else:
        parts.append(f"[{base_idx}:a]apad,atrim=0:{keep:.3f},asetpts=N/SR/TB[a]")
    run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(parts),
         "-map", "[a]", "-ar", "44100", "-ac", "2", shot_a])

    # --- mux trimmed video + layered audio ---
    run(["ffmpeg", "-y", "-i", vsil, "-i", shot_a, "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-shortest", dst])
    for f in (vsil, shot_a):
        try:
            os.remove(f)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--target", default="1080x1920")
    ap.add_argument("--font-size", type=int, default=84)
    ap.add_argument("--align", action="store_true", help="force-align captions with whisperx")
    ap.add_argument("--align-device", default="cuda")
    ap.add_argument("--music", default=None, help="override music path (else script.music)")
    ap.add_argument("--output-fps", type=int, default=None,
                    help="interpolate the final video up to this fps (motion-compensated "
                         "ffmpeg minterpolate); default = script.output_fps or no interpolation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.script) as f:
        spec = json.load(f)
    ep = os.path.dirname(os.path.abspath(args.script))
    w, h = (int(x) for x in args.target.lower().split("x"))
    fps = spec.get("fps", 16)
    shots = spec["shots"]
    # Frames trimmed off each clip's head/tail to drop Wan's boundary distortion.
    g_head = spec.get("trim_head", 2)
    g_tail = spec.get("trim_tail", 2)

    clips_dir = os.path.join(ep, "clips")
    tmp = os.path.join(ep, ".tmp"); os.makedirs(tmp, exist_ok=True)
    cap_dir = os.path.join(ep, "captions"); os.makedirs(cap_dir, exist_ok=True)
    out = args.out or os.path.join(ep, "final", "reel.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # A. normalize + trim + layer audio; build caption timeline from real durations.
    norm_files, segments, t = [], [], 0.0
    for s in shots:
        sid = s["id"]
        clip = os.path.join(clips_dir, f"shot{sid}.mp4")
        if not os.path.exists(clip):
            sys.exit(f"[assemble] missing clip: {clip}")
        talking = s.get("type") == "talking"
        src_dur = probe_duration(clip)
        head = s.get("trim_head", g_head) / fps  # seconds
        tail = s.get("trim_tail", g_tail) / fps
        keep = max(0.3, src_dur - head - tail)

        voiceover = None if talking else find_asset(ep, "voice", sid)
        # talking shots can fall back to the dialogue mp3 if the clip lost its audio
        if talking and not has_audio(clip):
            voiceover = find_asset(ep, "voice", sid)
        sfx = find_asset(ep, "sfx", sid)
        sfx_vol = 0.5
        if isinstance(s.get("sfx"), dict):
            sfx_vol = s["sfx"].get("volume", 0.5)

        dst = os.path.join(tmp, f"norm{sid}.mp4")
        normalize_shot(clip, dst, w, h, fps, keep, head, talking,
                       voiceover, sfx, sfx_vol)
        dur = probe_duration(dst)

        # caption text: dialogue (talking) or voiceover narration (b-roll)
        cap_text = (s.get("line") if talking else s.get("voiceover")) or ""
        if cap_text.strip():
            segments.append({"text": cap_text.strip(), "start": round(t, 3),
                             "end": round(t + dur, 3)})
        norm_files.append(dst)
        t += dur
    print(f"[assemble] total ~{t:.1f}s, {len(segments)} caption segment(s)")

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
            body_wav = os.path.join(tmp, "body.wav")
            run(["ffmpeg", "-y", "-i", body, "-vn", "-ac", "1", "-ar", "16000", body_wav])
            cmd += ["--audio", body_wav, "--language", spec.get("language", "ro"),
                    "--device", args.align_device]
        subprocess.run([str(c) for c in cmd], check=True)
    else:
        ass = None

    # D. (optional fps interpolation) + music duck-mix + loudnorm + burn subs -> final
    out_fps = args.output_fps or spec.get("output_fps") or fps
    music = args.music or spec.get("music")
    music_vol = spec.get("music_volume", 0.15)
    cmd = ["ffmpeg", "-y"]
    if music:
        music_path = music if os.path.isabs(music) else os.path.join(REPO, music)
        if not os.path.exists(music_path):
            sys.exit(f"[assemble] music not found: {music_path}")
        cmd += ["-i", body, "-stream_loop", "-1", "-i", music_path]
        afilter = (f"[1:a]volume={music_vol}[m];[0:a][m]amix=inputs=2:duration=first:"
                   "dropout_transition=0[mx];[mx]loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    else:
        cmd += ["-i", body]
        afilter = "[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[a]"
    # Video chain: interpolate motion-compensated frames FIRST (Wan is 16fps native and
    # reads choppy), THEN draw captions so text is rendered crisply per output frame and
    # never morphed. Scene-change detection keeps hard cuts from ghosting.
    vchain = []
    if out_fps > fps:
        vchain.append(f"minterpolate=fps={out_fps}:mi_mode=mci:mc_mode=aobmc:"
                      "me_mode=bidir:vsbmc=1")
        print(f"[assemble] interpolating {fps} -> {out_fps} fps (minterpolate mci)")
    if ass:
        vchain.append(f"ass=filename='{ass}'")
    vfilter = "[0:v]" + (",".join(vchain) if vchain else "copy") + "[v]"
    cmd += ["-filter_complex", f"{vfilter};{afilter}", "-map", "[v]", "-map", "[a]",
            "-r", str(out_fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
            "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-shortest", out]
    run(cmd)
    print(f"[assemble] done -> {out}")


if __name__ == "__main__":
    main()

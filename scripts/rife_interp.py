#!/usr/bin/env python3
"""GPU frame interpolation with RIFE (Practical-RIFE) — the fast backend for
assemble.py's `output_fps` step (alternative to ffmpeg `minterpolate`, which is
single-threaded CPU and slow).

Reads a video by piping raw frames from ffmpeg (no cv2 dependency), interpolates
to the target fps on the GPU at arbitrary timesteps (so any source->target ratio
works, not just 2x/4x), and re-encodes — copying the source audio through.

Needs Practical-RIFE checked out with its weights (env/setup_rife.sh). Point at it
with --rife-dir or $RIFE_DIR (default /ephemeral/Practical-RIFE).

Usage:
    python scripts/rife_interp.py --video body.mp4 --out body60.mp4 --fps 60
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F


def vinfo(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,r_frame_rate",
                          "-show_entries", "format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    j = json.loads(out.stdout)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), float(j["format"]["duration"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, required=True, help="target output fps")
    ap.add_argument("--rife-dir", default=os.environ.get("RIFE_DIR", "/ephemeral/Practical-RIFE"))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.rife_dir)
    from train_log.RIFE_HDv3 import Model  # noqa: E402

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model()
    model.load_model(os.path.join(args.rife_dir, "train_log"), -1)
    model.eval()

    W, H, src_fps, dur = vinfo(args.video)
    tmp = max(128, int(128 / args.scale))
    ph = ((H - 1) // tmp + 1) * tmp
    pw = ((W - 1) // tmp + 1) * tmp
    pad = (0, pw - W, 0, ph - H)
    fsize = W * H * 3
    print(f"[rife] {W}x{H} {src_fps:.3f}->{args.fps} fps, ~{dur:.1f}s, scale {args.scale}, "
          f"fp16={args.fp16}")

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", args.video,
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE, bufsize=10**8)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(args.fps), "-i", "-",
                            "-i", args.video, "-map", "0:v:0", "-map", "1:a:0?",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                            "-preset", "medium", "-c:a", "aac", "-b:a", "192k",
                            "-shortest", args.out], stdin=subprocess.PIPE)

    def read():
        buf = dec.stdout.read(fsize)
        return None if len(buf) < fsize else np.frombuffer(buf, np.uint8).reshape(H, W, 3)

    def to_t(fr):
        t = torch.from_numpy(fr.copy()).to(dev).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if args.fp16:
            t = t.half()
        return F.pad(t, pad)

    def to_bytes(t):
        a = (t[0, :, :H, :W].float().clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous()
        return a.cpu().numpy().tobytes()

    if args.fp16:
        model.flownet.half()

    src_dt, out_dt = 1.0 / src_fps, 1.0 / args.fps
    n = 0
    with torch.no_grad():
        i, a, b = 0, read(), read()
        if a is None:
            sys.exit("[rife] no frames")
        k = 0
        while True:
            t = k * out_dt
            # b is None => a is the last source frame at time i*src_dt
            if b is not None and t >= (i + 1) * src_dt - 1e-9:
                i += 1
                a, b = b, read()
                continue
            if b is None:
                if t <= i * src_dt + 1e-9:
                    enc.stdin.write(a.tobytes())
                    n += 1
                    k += 1
                    continue
                break
            frac = (t - i * src_dt) / src_dt
            if frac <= 1e-4:
                enc.stdin.write(a.tobytes())
            elif frac >= 1 - 1e-4:
                enc.stdin.write(b.tobytes())
            else:
                enc.stdin.write(to_bytes(model.inference(to_t(a), to_t(b), frac, args.scale)))
            n += 1
            k += 1

    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(f"[rife] wrote {n} frames -> {args.out}")


if __name__ == "__main__":
    main()

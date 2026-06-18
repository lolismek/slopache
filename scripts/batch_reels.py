#!/usr/bin/env python3
"""Queue many episodes back-to-back, unattended (e.g. overnight).

Runs make_reel.py --yes on each episode in sequence on the single GPU box.
A failing episode does NOT stop the batch — it's logged and the run moves on,
so one bad script can't waste the whole night. Each episode's full output goes
to <ep>/build.log; a one-line status per episode goes to this process's stdout,
and a summary table prints at the end.

Run it detached so it survives an ssh disconnect (this is the "queue overnight"
part — the box keeps working after you close the laptop):

    ./infra/remote.sh ssh 'cd /ephemeral/slopache && \
        COMFY_DIR=/ephemeral/ComfyUI infra/comfy.sh start && infra/comfy.sh wait && \
        nohup python scripts/batch_reels.py --all --skip-existing > batch.log 2>&1 &'

then check on it any time with:
    ./infra/remote.sh ssh 'tail -n 40 /ephemeral/slopache/batch.log'

and pull everything finished:
    ./infra/remote.sh pull '/ephemeral/slopache/episodes/*/final/reel.mp4' ./outputs/
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def resolve_scripts(items, use_all):
    """Turn CLI args (ep dirs, ep globs, or script.json paths) into script.json paths."""
    paths = []
    if use_all:
        items = list(items) + ["episodes/*"]
    for it in items:
        for hit in sorted(glob.glob(os.path.join(REPO, it))) or [os.path.join(REPO, it)]:
            if os.path.isdir(hit):
                sj = os.path.join(hit, "script.json")
                if os.path.exists(sj):
                    paths.append(sj)
            elif hit.endswith(".json") and os.path.exists(hit):
                paths.append(hit)
    # de-dupe, keep order
    seen, out = set(), []
    for p in paths:
        rp = os.path.relpath(p, REPO)
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def fmt(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser(description="Run many episodes unattended, in sequence.")
    ap.add_argument("episodes", nargs="*", help="episode dirs, globs, or script.json paths")
    ap.add_argument("--all", action="store_true", help="queue every episodes/*/script.json")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip episodes whose final/reel.mp4 already exists, and reuse clips inside an episode")
    ap.add_argument("--server", default="127.0.0.1:8188")
    ap.add_argument("--align", action="store_true", help="pass through to assemble (whisperx)")
    args = ap.parse_args()

    scripts = resolve_scripts(args.episodes, args.all)
    if not scripts:
        sys.exit("[batch] no episodes resolved — pass dirs/globs or --all")

    print(f"[batch] queue of {len(scripts)} episode(s):")
    for sj in scripts:
        print(f"          {sj}")
    print(f"[batch] starting {time.strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)

    results = []
    batch_t0 = time.time()
    for i, sj in enumerate(scripts, 1):
        ep_dir = os.path.join(REPO, os.path.dirname(sj))
        final = os.path.join(ep_dir, "final", "reel.mp4")
        name = os.path.basename(os.path.dirname(sj))

        if args.skip_existing and os.path.exists(final):
            print(f"[batch] ({i}/{len(scripts)}) {name}: reel.mp4 exists — SKIP", flush=True)
            results.append((name, "skipped", 0.0))
            continue

        os.makedirs(os.path.join(ep_dir, "final"), exist_ok=True)
        logpath = os.path.join(ep_dir, "build.log")
        cmd = [sys.executable, "scripts/make_reel.py", sj, "--yes", "--server", args.server]
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.align:
            cmd.append("--align")

        print(f"[batch] ({i}/{len(scripts)}) {name}: START  (log -> {os.path.relpath(logpath, REPO)})",
              flush=True)
        t0 = time.time()
        with open(logpath, "w") as log:
            log.write(f"# {name} build {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write("# " + " ".join(cmd) + "\n\n")
            log.flush()
            rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0

        if rc == 0 and os.path.exists(final):
            print(f"[batch] ({i}/{len(scripts)}) {name}: OK    in {fmt(dt)} -> {os.path.relpath(final, REPO)}",
                  flush=True)
            results.append((name, "ok", dt))
        else:
            print(f"[batch] ({i}/{len(scripts)}) {name}: FAIL  (rc={rc}) after {fmt(dt)} — "
                  f"see {os.path.relpath(logpath, REPO)}", flush=True)
            results.append((name, f"FAIL(rc={rc})", dt))

    total = time.time() - batch_t0
    ok = sum(1 for _, st, _ in results if st == "ok")
    print(f"\n[batch] DONE {time.strftime('%Y-%m-%d %H:%M:%S')} — "
          f"{ok}/{len(results)} reels in {fmt(total)}")
    print("[batch] summary:")
    for name, st, dt in results:
        print(f"          {name:28s} {st:14s} {fmt(dt)}")
    # non-zero exit if anything failed, so a wrapping cron/script can notice
    if any(st.startswith("FAIL") for _, st, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

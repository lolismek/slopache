#!/usr/bin/env python3
"""Model-free LoRA convergence metric (method 2).

For each saved checkpoint, compute the EFFECTIVE adapter magnitude per layer,
||scale * B @ A||_F (scale = alpha/rank), aggregate to a global norm, and report
the delta between consecutive checkpoints. When ||dW|| (and its relative size)
trends toward zero, the adapter has stopped changing -> converged/plateaued.
No external model required; reads only the LoRA .safetensors.
"""
import glob, os, re, sys
import torch
from safetensors.torch import load_file

CKPT_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/ephemeral/slopache/output/lora/bunica_flux2_v1"
SCALE = 1.0  # alpha/rank = 16/16

def step_of(p):
    m = re.search(r"_(\d+)\.safetensors$", p)
    return int(m.group(1)) if m else -1

def pairs(sd):
    """Return {layer: (A,B) float32} raw factors. Memory-cheap: never forms B@A.
    A is (r x in), B is (out x r)."""
    out = {}
    for k in sd:
        if k.endswith(".lora_A.weight"):
            base = k[:-len(".lora_A.weight")]
            B = sd.get(base + ".lora_B.weight")
            if B is None:
                continue
            out[base] = (sd[k].float(), B.float())
    return out

def _fro2(A, B):
    """||B@A||_F^2 = sum( (B^T B) ⊙ (A A^T) ), both r x r (tiny)."""
    return float(((B.T @ B) * (A @ A.T)).sum())

def gnorm(p):
    return float(SCALE * (sum(_fro2(A, B) for A, B in p.values())) ** 0.5)

def delta_norm(cur, prev):
    """||scale*(B_t A_t - B_p A_p)||_F via r x r trace identities only."""
    s = 0.0
    for k, (At, Bt) in cur.items():
        nx = _fro2(At, Bt)
        if k in prev:
            Ap, Bp = prev[k]
            ny = _fro2(Ap, Bp)
            cross = float(torch.trace((Bp.T @ Bt) @ (At @ Ap.T)))  # <X,Y>_F
            s += nx + ny - 2 * cross
        else:
            s += nx
    return float(SCALE * max(s, 0.0) ** 0.5)

ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.safetensors")), key=step_of)
print(f"{'step':>7} {'||W|| (adapter)':>16} {'||dW|| vs prev':>15} {'rel dW':>8}")
prev = None
for p in ckpts:
    eff = pairs(load_file(p))
    nrm = gnorm(eff)
    if prev is None:
        print(f"{step_of(p):>7} {nrm:16.3f} {'—':>15} {'—':>8}")
    else:
        d = delta_norm(eff, prev)
        rel = d / nrm if nrm else 0.0
        print(f"{step_of(p):>7} {nrm:16.3f} {d:15.3f} {rel:8.3f}")
    prev = eff

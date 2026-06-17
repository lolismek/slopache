#!/usr/bin/env bash
# Bootstrap ostris/ai-toolkit on the brev A100 box and fetch the gated FLUX.2
# weights. Run ON the box (via infra/remote.sh ssh) AFTER `huggingface-cli login`
# with a token that has accepted BOTH licenses:
#   - https://huggingface.co/black-forest-labs/FLUX.2-dev   (gated)
#   - https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503 (gated)
set -euo pipefail

AITK=/ephemeral/ai-toolkit
HF_HOME=/ephemeral/hf
export HF_HOME

# 1) ai-toolkit
if [ ! -d "$AITK" ]; then
  git clone https://github.com/ostris/ai-toolkit "$AITK"
fi
cd "$AITK"
git submodule update --init --recursive

# 2) python deps (reuse a venv on /ephemeral so it survives root-fs limits)
if [ ! -d "$AITK/venv" ]; then
  python3 -m venv "$AITK/venv"
fi
# shellcheck disable=SC1091
source "$AITK/venv/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 3) sanity: confirm the gated repos are reachable with the logged-in token
python3 - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for r in ["black-forest-labs/FLUX.2-dev",
          "mistralai/Mistral-Small-3.1-24B-Instruct-2503"]:
    try:
        api.model_info(r)
        print(f"OK  access: {r}")
    except Exception as e:
        print(f"!!  NO access: {r} -> {type(e).__name__}: {e}")
        print("    Accept the license on the HF page and re-login. Aborting.")
        raise SystemExit(1)
PY

echo "[setup] ai-toolkit ready at $AITK"
echo "[setup] weights download lazily on first run; ~64GB (FLUX.2) + ~48GB (Mistral)."
echo "[setup] launch: source $AITK/venv/bin/activate && \\"
echo "        python $AITK/run.py /ephemeral/slopache/training/bunica_flux2_v1.yaml"

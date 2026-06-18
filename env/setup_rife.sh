#!/usr/bin/env bash
# Runs ON the box. Sets up Practical-RIFE for GPU frame interpolation, the fast
# backend for assemble.py (interp_backend "rife") vs the slow CPU minterpolate.
# Clones the repo and fetches the 4.26 model weights from Google Drive via gdown.
set -euo pipefail
RIFE_DIR="${RIFE_DIR:-/ephemeral/Practical-RIFE}"
COMFY_DIR="${COMFY_DIR:-/ephemeral/ComfyUI}"
# shellcheck disable=SC1091
source "$COMFY_DIR/venv/bin/activate"
pip install -q gdown

[ -d "$RIFE_DIR" ] || git clone --depth 1 https://github.com/hzwer/Practical-RIFE "$RIFE_DIR"
cd "$RIFE_DIR"
if [ ! -f train_log/flownet.pkl ]; then
  echo "[rife] downloading 4.26 weights..."
  gdown 1gViYvvQrtETBgU1w8axZSsr7YUuw31uy -O rife426.zip
  unzip -o -q rife426.zip && rm -rf __MACOSX rife426.zip
fi
python -c "import sys; sys.path.insert(0, '$RIFE_DIR'); \
from train_log.RIFE_HDv3 import Model; m=Model(); m.load_model('$RIFE_DIR/train_log', -1); \
print('[rife] model loads OK')"
echo "[rife] ready at $RIFE_DIR"

#!/usr/bin/env bash
# Setup the ResWAM training and inference Python environment.
# Usage:
#   bash scripts/env/setup_reswam.sh [--venv ./envs/reswam] [--download-weights] [--train-kernels]
#
# --train-kernels additionally installs flash-linear-attention 0.4.2 + causal-conv1d (the
# Gated-DeltaNet kernels of the official training runs; requires a CUDA toolchain).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${RESWAM_VENV_DIR:-${REPO_ROOT}/envs/reswam}"
DOWNLOAD_WEIGHTS=0
TRAIN_KERNELS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --download-weights)
      DOWNLOAD_WEIGHTS=1
      shift
      ;;
    --train-kernels)
      TRAIN_KERNELS=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup_reswam] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "[setup_reswam] Creating Python 3.10 virtual environment at ${VENV_DIR}"
uv venv --python 3.10 "${VENV_DIR}"

echo "[setup_reswam] Installing dependencies from requirements.txt and editable package"
uv pip install --python "${VENV_DIR}/bin/python" -r "${REPO_ROOT}/requirements.txt"
uv pip install --python "${VENV_DIR}/bin/python" --no-deps -e "${REPO_ROOT}"
if [[ "${TRAIN_KERNELS}" -eq 1 ]]; then
  echo "[setup_reswam] Installing Gated-DeltaNet training kernels (requirements-train.txt)"
  uv pip install --python "${VENV_DIR}/bin/python" --no-build-isolation -r "${REPO_ROOT}/requirements-train.txt"
fi

if [[ "${DOWNLOAD_WEIGHTS}" -eq 1 ]]; then
  CKPT_ROOT="${RESWAM_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints}"
  mkdir -p "${CKPT_ROOT}"
  echo "[setup_reswam] Downloading RynnBrain1.1-2B and DINOv3-ViT-B/16 weights into ${CKPT_ROOT}"
  "${VENV_DIR}/bin/python" - "${CKPT_ROOT}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

ckpt_root = Path(sys.argv[1])
snapshot_download(
    repo_id="Alibaba-DAMO-Academy/RynnBrain1.1-2B",
    revision="36069aa841fc58f500d307316cbfdf379777ce8a",
    local_dir=ckpt_root / "RynnBrain1.1-2B",
)
snapshot_download(
    repo_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
    revision="5931719e67bbdb9737e363e781fb0c67687896bc",
    local_dir=ckpt_root / "dinov3-vitb16-pretrain-lvd1689m",
)
PY
fi

"${VENV_DIR}/bin/python" - <<'PY'
import json, sys, torch, transformers, av, numpy, pyarrow, reswam
print(json.dumps({
    "reswam": reswam.__version__,
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "av": av.__version__,
    "numpy": numpy.__version__,
    "pyarrow": pyarrow.__version__,
}, indent=2))
PY
echo "[setup_reswam] Environment ready at ${VENV_DIR}"

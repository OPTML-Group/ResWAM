#!/usr/bin/env bash
# Setup the LIBERO-Plus simulator and evaluation environment.
# Official LIBERO-Plus commit: 4976dc30028e805ff8094b55501d532c48fec182
# Usage:
#   bash scripts/env/setup_libero_plus.sh [--dir ./third_party/LIBERO-plus] [--venv ./envs/libero_plus]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LP_DIR="${LIBERO_PLUS_ROOT:-${REPO_ROOT}/third_party/LIBERO-plus}"
VENV_DIR="${LIBERO_PLUS_VENV_DIR:-${REPO_ROOT}/envs/libero_plus}"
LP_COMMIT="4976dc30028e805ff8094b55501d532c48fec182"
LP_ASSETS_SHA256="96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      LP_DIR="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "${LP_DIR}")"
if [[ ! -d "${LP_DIR}/.git" ]]; then
  echo "[setup_libero_plus] Cloning sylvestf/LIBERO-plus into ${LP_DIR}"
  git clone https://github.com/sylvestf/LIBERO-plus.git "${LP_DIR}"
fi
git -C "${LP_DIR}" fetch --all --tags
git -C "${LP_DIR}" checkout "${LP_COMMIT}"
touch "${LP_DIR}/libero/__init__.py"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# Simulator stack of the official ResWAM LIBERO-Plus evaluation (81.1%): Python 3.11,
# MuJoCo 3.3.2, robosuite 1.4.1, bddl 3.6.0, gym 0.26.2, Wand 0.7.0 (needs ImageMagick).
if ! "${MAGICK_CONFIG:-MagickWand-config}" --version >/dev/null 2>&1 && \
   ! ldconfig -p 2>/dev/null | grep -q libMagickWand; then
  echo "[setup_libero_plus] WARNING: ImageMagick (libMagickWand) not found. LIBERO-Plus' sensor-noise" >&2
  echo "  perturbations use Wand; install it (e.g. apt install libmagickwand-dev) or set MAGICK_HOME." >&2
fi

echo "[setup_libero_plus] Creating Python 3.11 virtual environment at ${VENV_DIR}"
uv venv --python 3.11 "${VENV_DIR}"
uv pip install --python "${VENV_DIR}/bin/python" \
  "mujoco==3.3.2" "robosuite==1.4.1" "bddl==3.6.0" "easydict==1.13" "cloudpickle" \
  "gym==0.26.2" "imageio[ffmpeg]" "opencv-python-headless" "Wand==0.7.0" "tyro"
# The policy runs in-process in the evaluation worker: install the ResWAM runtime as well.
uv pip install --python "${VENV_DIR}/bin/python" -r "${REPO_ROOT}/requirements.txt"
uv pip install --python "${VENV_DIR}/bin/python" --no-deps -e "${LP_DIR}"
uv pip install --python "${VENV_DIR}/bin/python" --no-deps -e "${REPO_ROOT}"

echo "[setup_libero_plus] Downloading LIBERO-Plus assets from Hugging Face (Sylvest/LIBERO-plus)..."
"${VENV_DIR}/bin/python" - "${LP_DIR}" "${LP_ASSETS_SHA256}" <<'PY'
import sys, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

import hashlib

lp_dir = Path(sys.argv[1])
expected = sys.argv[2]
assets_dir = lp_dir / "libero" / "libero" / "assets"
if not (assets_dir / "textures").is_dir():
    zip_path = hf_hub_download(
        repo_id="Sylvest/LIBERO-plus",
        filename="assets.zip",
        repo_type="dataset",
    )
    digest = hashlib.sha256()
    with open(zip_path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit(
            f"LIBERO-Plus assets.zip sha256 {digest.hexdigest()} != official {expected}"
        )
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(lp_dir / "libero" / "libero")
print(f"[setup_libero_plus] Assets verified under {assets_dir}")
PY

echo "[setup_libero_plus] LIBERO-Plus environment ready at ${VENV_DIR}"

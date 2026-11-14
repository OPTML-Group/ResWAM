#!/usr/bin/env bash
# Setup the RoboTwin 2.0 simulator and evaluation environment.
# Official RoboTwin 2.0 upstream commit: bf44be51cf5717a5595ce59447f2cf5263d2aa95
# Usage:
#   bash scripts/env/setup_robotwin2.sh [--dir ./third_party/RoboTwin] [--venv ./envs/robotwin2]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RT_DIR="${ROBOTWIN_ROOT:-${REPO_ROOT}/third_party/RoboTwin}"
VENV_DIR="${ROBOTWIN_VENV_DIR:-${REPO_ROOT}/envs/robotwin2}"
RT_COMMIT="bf44be51cf5717a5595ce59447f2cf5263d2aa95"
CUROBO_COMMIT="cca894de9ec74e77a0a4071319c81958991a9108"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      RT_DIR="$2"
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

mkdir -p "$(dirname "${RT_DIR}")"
if [[ ! -d "${RT_DIR}/.git" ]]; then
  echo "[setup_robotwin2] Cloning RoboTwin-Platform/RoboTwin into ${RT_DIR}"
  git clone https://github.com/RoboTwin-Platform/RoboTwin.git "${RT_DIR}"
fi
git -C "${RT_DIR}" fetch --all --tags
git -C "${RT_DIR}" checkout "${RT_COMMIT}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "[setup_robotwin2] Creating Python 3.10 virtual environment at ${VENV_DIR}"
uv venv --python 3.10 "${VENV_DIR}"
# Simulator stack of the official ResWAM RoboTwin 2.0 evaluation (93.8%).
uv pip install --python "${VENV_DIR}/bin/python" -r "${RT_DIR}/script/requirements.txt"
uv pip install --python "${VENV_DIR}/bin/python" \
  "sapien==3.0.0b1" "mplib==0.2.1" "trimesh==4.4.3" "open3d==0.18.0" \
  "transforms3d==0.4.2" "gymnasium==0.29.1" "toppra==0.6.10" "warp-lang==1.8.0" \
  "numpy==1.26.4" "scipy==1.15.3" "imageio[ffmpeg]" "opencv-python-headless"
# The policy runs in-process in the evaluation worker: install the ResWAM runtime as well.
uv pip install --python "${VENV_DIR}/bin/python" -r "${REPO_ROOT}/requirements.txt"
uv pip install --python "${VENV_DIR}/bin/python" --no-deps -e "${REPO_ROOT}"

# CuRobo (expert planner used for seed acceptance), pinned to the commit of the official run.
# Note: RoboTwin's script/_install.sh also patches mplib/planner.py and sapien's urdf_loader.py;
# the official ResWAM evaluation ran WITHOUT those patches, so they are intentionally skipped
# (the mplib patch changes which candidate seeds the expert planner accepts).
CUROBO_DIR="${RT_DIR}/envs/curobo"
if [[ ! -d "${CUROBO_DIR}/.git" ]]; then
  git clone https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
fi
git -C "${CUROBO_DIR}" checkout "${CUROBO_COMMIT}"
uv pip install --python "${VENV_DIR}/bin/python" --no-build-isolation -e "${CUROBO_DIR}"

echo "[setup_robotwin2] RoboTwin 2.0 environment ready at ${VENV_DIR}"
echo "[setup_robotwin2] Download the RoboTwin 2.0 assets before evaluating:"
echo "  cd ${RT_DIR} && bash script/_download_assets.sh"

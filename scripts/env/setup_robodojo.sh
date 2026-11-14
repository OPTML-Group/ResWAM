#!/usr/bin/env bash
# Setup the RoboDojo benchmark (Isaac Sim 5.1 simulator) and the ResWAM policy environment.
#
# The official ResWAM RoboDojo evaluation (SR 9.40 / Score 13.41) used two environments:
#   * simulator client: RoboDojo's conda env from `scripts/install.sh -i`
#       Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3 (isaaclab 0.54.3), PyTorch 2.7.0/cu128, CuRobo
#   * policy server:    the ResWAM environment (Python 3.10, PyTorch 2.6.0/cu124, transformers 5.2.0)
#                       plus msgpack-numpy and h5py for XPolicyLab's websocket server
#
# Host requirements (Isaac Sim 5.1): Ubuntu 22.04 x86-64, an NVIDIA RTX GPU with RT cores
# (A100/H100 are NOT supported), driver 570/580, CUDA 12.8, a working Vulkan ICD, >= 16 GB VRAM.
# conda (or mamba) must be on PATH for the simulator environment.
#
# Pinned sources: RoboDojo 726e9aabfaa642203722eb126f5eaf0f37f3e1ad with submodules
# XPolicyLab bb9a0b5f, IsaacLab afca7b09, curobo d17b54ce.
#
# Usage:
#   bash scripts/env/setup_robodojo.sh [--dir ./third_party/RoboDojo] [--policy-venv ./envs/reswam]
#                                      [--skip-sim] [--skip-assets]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RD_DIR="${ROBODOJO_ROOT:-${REPO_ROOT}/third_party/RoboDojo}"
POLICY_VENV="${RESWAM_VENV_DIR:-${REPO_ROOT}/envs/reswam}"
RD_COMMIT="726e9aabfaa642203722eb126f5eaf0f37f3e1ad"
declare -A SUBMODULE_COMMITS=(
  [XPolicyLab]="bb9a0b5f5136a74503b679af830bfd0a3a837d5c"
  [third_party/IsaacLab]="afca7b09d60d8beb9c1cb28b43066499940b969b"
  [third_party/curobo]="d17b54ce32cba095c0b000c4c58777075d11de0e"
)
SKIP_SIM=0
SKIP_ASSETS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      RD_DIR="$2"
      shift 2
      ;;
    --policy-venv)
      POLICY_VENV="$2"
      shift 2
      ;;
    --skip-sim)
      SKIP_SIM=1
      shift
      ;;
    --skip-assets)
      SKIP_ASSETS=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "${RD_DIR}")"
if [[ ! -d "${RD_DIR}/.git" ]]; then
  echo "[setup_robodojo] Cloning RoboDojo-Benchmark/RoboDojo into ${RD_DIR}"
  git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git "${RD_DIR}"
fi
git -C "${RD_DIR}" fetch --all --tags
git -C "${RD_DIR}" checkout "${RD_COMMIT}"
git -C "${RD_DIR}" submodule update --init --recursive
for path in "${!SUBMODULE_COMMITS[@]}"; do
  actual="$(git -C "${RD_DIR}/${path}" rev-parse HEAD)"
  if [[ "${actual}" != "${SUBMODULE_COMMITS[${path}]}" ]]; then
    echo "[setup_robodojo] ERROR: submodule ${path} is at ${actual}, expected ${SUBMODULE_COMMITS[${path}]}" >&2
    exit 1
  fi
done

if [[ "${SKIP_SIM}" -eq 0 ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[setup_robodojo] ERROR: conda is required for RoboDojo's Isaac Sim environment" >&2
    exit 1
  fi
  echo "[setup_robodojo] Installing the official simulator environment (conda env 'RoboDojo')"
  (cd "${RD_DIR}" && OMNI_KIT_ACCEPT_EULA=YES bash scripts/install.sh -i)
fi

if [[ "${SKIP_ASSETS}" -eq 0 ]]; then
  echo "[setup_robodojo] Downloading RoboDojo assets and updating embodiment paths"
  (cd "${RD_DIR}" && bash scripts/init_assets.sh)
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate RoboDojo
    (cd "${RD_DIR}" && python utils/update_embodiment_config_path.py)
    conda deactivate
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -x "${POLICY_VENV}/bin/python" ]]; then
  echo "[setup_robodojo] Creating the ResWAM policy environment at ${POLICY_VENV}"
  uv venv --python 3.10 "${POLICY_VENV}"
fi
uv pip install --python "${POLICY_VENV}/bin/python" -r "${REPO_ROOT}/requirements.txt"
uv pip install --python "${POLICY_VENV}/bin/python" "msgpack-numpy==0.4.8" "h5py"
uv pip install --python "${POLICY_VENV}/bin/python" --no-deps -e "${REPO_ROOT}"

echo "[setup_robodojo] RoboDojo ready at ${RD_DIR}"
echo "  simulator env : conda env 'RoboDojo'  (pass --sim-env RoboDojo to eval_robodojo.py)"
echo "  policy env    : ${POLICY_VENV}          (pass --policy-env ${POLICY_VENV})"

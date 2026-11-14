#!/usr/bin/env bash
# Launch distributed FSDP training for ResWAM on RoboDojo (H=32, lambda_w=0.1, GBS=256).
# Two stages, as in the paper run:
#   stage 1: fresh, warmup 1500 + cosine to 200k            (configs/robodojo.yaml)
#   stage 2: 200k -> 400k at constant floor LR, from stage 1 (configs/robodojo_stage2.yaml)
# Supports both online DINOv3 residual extraction (default) and pre-extracted cache (--use-dino-cache).
# Usage:
#   bash scripts/train/train_robodojo.sh                 # stage 1
#   RESWAM_STAGE=2 bash scripts/train/train_robodojo.sh  # stage 2 (after stage 1 finished)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${RESWAM_PYTHON:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29503}"
if [[ "${RESWAM_STAGE:-1}" == "2" ]]; then
  CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/robodojo_stage2.yaml}"
else
  CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/robodojo.yaml}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m reswam.engine.trainer \
  --config "${CONFIG_PATH}" \
  "$@"

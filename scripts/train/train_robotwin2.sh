#!/usr/bin/env bash
# Launch distributed FSDP training for ResWAM on RoboTwin 2.0 (H=32, lambda_w=0.1, GBS=256, 100k steps).
# Supports both online DINOv3 residual extraction (default) and pre-extracted cache (--use-dino-cache).
# Usage:
#   bash scripts/train/train_robotwin2.sh [extra trainer flags, e.g. --use-dino-cache --steps 100000]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${RESWAM_PYTHON:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29502}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/robotwin2.yaml}"

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

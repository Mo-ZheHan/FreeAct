#!/bin/bash
# Stage 1: physically grounded latent action model.
#   bash scripts/train_lam.sh [extra LightningCLI args, e.g. --data.root /path/to/data]
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then   # GPUs in CUDA_VISIBLE_DEVICES if set, otherwise all GPUs of the node
    IFS=, read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
    NPROC=${NPROC:-${#gpus[@]}}
else
    NPROC=${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}
fi
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

torchrun --standalone --nproc_per_node="$NPROC" -m freeact.lam.train fit \
    --config configs/lam.yaml \
    --trainer.devices="$NPROC" \
    "$@"

#!/bin/bash
# Stage 2 of the tokenizer baselines: VLA pretraining with the UniVLA or LAPA tokenizer.
#   LAM_CKPT=checkpoints/univla/lam-stage-2.ckpt RUN_NAME=stage2_univla bash scripts/train_stage2_baseline.sh
#   LAM_CKPT=checkpoints/lapa/laq_openx.pt       RUN_NAME=stage2_lapa   bash scripts/train_stage2_baseline.sh
set -euo pipefail
: "${LAM_CKPT:?set LAM_CKPT to the UniVLA lam-stage-2.ckpt or the LAPA laq_openx.pt tokenizer}"
export RUN_NAME=${RUN_NAME:-stage2_baseline}
export DATASETS=${DATASETS:-"droid/dual_cam:1.0:True,droid/single_cam:1.0:True,droid/dual:1.0:False,droid/single:1.0:False,libero/libero_spatial:5.0:True,libero/libero_object:5.0:True,libero/libero_goal:5.0:True,libero/libero_10:5.0:True,libero/libero_90:5.0:True"}
export LR=${LR:-5e-5} NUM_EPOCHS=${NUM_EPOCHS:-1}
exec bash "$(dirname "$0")/train_stage2.sh" \
    --lr_scheduler_type constant_with_warmup --warmup_ratio 0 --warmup_steps 3000 \
    --max_grad_norm 5 --weight_decay 0.01 --lerobot_delta_time 0.6 \
    "$@"

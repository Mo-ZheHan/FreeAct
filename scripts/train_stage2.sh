#!/bin/bash
# Stage 2: latent action-guided VLA pretraining on DROID + LIBERO.
set -euo pipefail
source "$(dirname "$0")/_vla_common.sh"

RUN_NAME=${RUN_NAME:-stage2}
DATASETS=${DATASETS:-"droid/dual_cam:1.0:True,droid/single_cam:1.0:True,droid/dual:1.0:False,droid/single:1.0:False,libero/libero_spatial:1.0:True,libero/libero_object:1.0:True,libero/libero_goal:1.0:True,libero/libero_10:1.0:True,libero/libero_90:1.0:True"}

run_vla \
    --vla_training_phase pretrain \
    --model_name_or_path "${BASE_MODEL:-Qwen/Qwen3-VL-4B-Instruct}" \
    --dataset_use "$DATASETS" \
    --output_dir "$OUTPUT_DIR/$RUN_NAME" \
    --num_train_epochs "${NUM_EPOCHS:-2}" \
    --learning_rate "${LR:-1e-4}" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_grad_norm 2 \
    --save_strategy steps --save_steps 5000 --save_total_limit 3 \
    "$@"

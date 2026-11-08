#!/bin/bash
# Stage 3: deployment adaptation from generated target-domain videos.
#
# Source stream:    libero_spatial demonstrations (latent tokens + continuous actions)
# Generated stream: Wan 2.6 generated videos, pseudo-labeled by the LAM (tokens only)
# The two streams are combined with gradient projection (GRAD_PROJ=False for the "w/o GradProj" ablation).
set -euo pipefail
source "$(dirname "$0")/_vla_common.sh"

RUN_NAME=${RUN_NAME:-stage3}
STAGE2_CKPT=${STAGE2_CKPT:-./checkpoints/FreeAct/stage2}
DATASETS=${DATASETS:-"libero/libero_spatial:1.0:True"}
VIDEO_DATASETS=${VIDEO_DATASETS:-"generated/libero_spatial_wan2.6:0.1,generated/libero_target_wan2.6:0.1"}   # weight 0.1 = lambda_v
if [[ ! -f "$STAGE2_CKPT/adapter_config.json" ]]; then
    echo "STAGE2_CKPT=$STAGE2_CKPT is not a stage-2 checkpoint (no adapter_config.json); download it with: hf download Mo-ZheHan/FreeAct --include 'stage2/*' --local-dir checkpoints/FreeAct" >&2
    exit 1
fi

run_vla \
    --vla_training_phase finetune \
    --model_name_or_path "$STAGE2_CKPT" \
    --dataset_use "$DATASETS" \
    ${VIDEO_DATASETS:+--video_dataset_use "$VIDEO_DATASETS"} \
    --action_norm_dataset libero/libero_spatial \
    --action_dim 7 \
    --action_decoder_hidden_dim 512 \
    --action_decoder_lr 3.5e-4 \
    --action_decoder_weight_decay 1e-3 \
    --grad_proj "${GRAD_PROJ:-True}" \
    --output_dir "$OUTPUT_DIR/$RUN_NAME" \
    --num_train_epochs "${NUM_EPOCHS:-75}" \
    --learning_rate "${LR:-1e-4}" \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --warmup_steps 200 \
    --max_grad_norm 2 \
    --save_strategy epoch --save_total_limit 2 \
    "$@"

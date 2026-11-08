# Shared launcher for stages 2 and 3 (sourced by the stage scripts).
# Variables set as ${VAR:-default} can be overridden from the environment.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# GPUs: those in CUDA_VISIBLE_DEVICES if set, otherwise all GPUs of the node
default_nproc() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local gpus
        IFS=, read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
        echo "${#gpus[@]}"
    else
        nvidia-smi -L 2>/dev/null | wc -l
    fi
}
NPROC=${NPROC:-$(default_nproc)}
if (( NPROC < 2 )); then
    echo "NPROC=$NPROC: stages 2-3 need at least two GPUs on one node (DeepSpeed ZeRO-2 shards the optimizer states across GPUs to fit in memory)" >&2
    exit 1
fi
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
(( GLOBAL_BATCH_SIZE % (PER_DEVICE_BATCH * NPROC) == 0 )) || {
    echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by PER_DEVICE_BATCH*NPROC=$((PER_DEVICE_BATCH * NPROC))" >&2
    exit 1
}
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / PER_DEVICE_BATCH / NPROC))

DATA_ROOT=${DATA_ROOT:-./data}
LAM_CKPT=${LAM_CKPT:-./checkpoints/FreeAct/lam/lam.ckpt}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs}
REPORT_TO=${REPORT_TO:-none}          # set to "wandb" to log
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# DeepSpeed ZeRO-2 here uses no compiled ops, so a CUDA toolkit (CUDA_HOME) is not required
export DS_IGNORE_CUDA_DETECTION=${DS_IGNORE_CUDA_DETECTION:-1}

# Fail before torchrun when an input is missing, instead of deep inside the trainer.
check_inputs() {
    local spec missing=()
    if [[ ! -f "$LAM_CKPT" ]]; then
        echo "LAM_CKPT=$LAM_CKPT not found; download it with: hf download Mo-ZheHan/FreeAct --include 'lam/*' --local-dir checkpoints/FreeAct" >&2
        exit 1
    fi
    for spec in ${DATASETS//,/ } ${VIDEO_DATASETS:+${VIDEO_DATASETS//,/ }}; do
        [[ -f "$DATA_ROOT/${spec%%:*}/meta/info.json" ]] || missing+=("$DATA_ROOT/${spec%%:*}")
    done
    if (( ${#missing[@]} )); then
        echo "datasets not found under DATA_ROOT=$DATA_ROOT: ${missing[*]}; download them as described in the Data section of README.md" >&2
        exit 1
    fi
}

run_vla() {
    check_inputs
    echo "[$RUN_NAME] ${NPROC} GPUs | batch ${GLOBAL_BATCH_SIZE} = ${PER_DEVICE_BATCH} x ${NPROC} x ${GRAD_ACCUM}"
    torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" -m freeact.vla.train.train \
        --deepspeed configs/zero2.json \
        --lam_checkpoint "$LAM_CKPT" \
        --lerobot_root "$DATA_ROOT" \
        --lerobot_resolution 224 \
        --lerobot_image_aug True \
        --lerobot_delta_time 1.0 \
        --lora_r 64 --lora_alpha 128 --lora_dropout 0 \
        --bf16 \
        --per_device_train_batch_size "$PER_DEVICE_BATCH" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --gradient_checkpointing True \
        --dataloader_num_workers "${NUM_WORKERS:-8}" \
        --eval_strategy no \
        --logging_steps 10 \
        --report_to "$REPORT_TO" \
        --run_name "$RUN_NAME" \
        --seed 42 \
        "$@"
}

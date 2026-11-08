#!/bin/bash
# Self-evolution: stage 3 again from the stage-2 checkpoint, adding MLLM-selected rollouts of the
# adapted (stage-3) policy.
#   bash scripts/train_self_evolve.sh                      # tasks A-E
#   SUITE=libero_camera bash scripts/train_self_evolve.sh  # camera-shift environment (Task B)
#   SUITE=libero_bottle bash scripts/train_self_evolve.sh  # ketchup-bottle environment (Task C)
set -euo pipefail
SUITE=${SUITE:-libero_target}
if [[ "$SUITE" == libero_target ]]; then
    ROLLOUTS=${ROLLOUTS:-"self_evolution/libero_target_A:5.0:True,self_evolution/libero_target_B:5.0:True,self_evolution/libero_target_C:5.0:True,self_evolution/libero_target_D:5.0:True,self_evolution/libero_target_E:5.0:True"}
    export VIDEO_DATASETS=${VIDEO_DATASETS:-"generated/libero_target_wan2.6:0.1"}
    export RUN_NAME=${RUN_NAME:-self_evolve}
elif [[ "$SUITE" == libero_camera || "$SUITE" == libero_bottle ]]; then
    ROLLOUTS=${ROLLOUTS:-"self_evolution/${SUITE}:2.0:True"}
    export VIDEO_DATASETS=${VIDEO_DATASETS:-"generated/libero_target_wan2.6:0.1,generated/${SUITE}_wan2.6:0.1"}
    export RUN_NAME=${RUN_NAME:-self_evolve_${SUITE#libero_}}
else
    echo "Unsupported self-evolution suite: $SUITE (libero_target, libero_camera or libero_bottle)" >&2
    exit 2
fi
DATASETS="libero/libero_spatial:1.0:True,${ROLLOUTS}" \
    exec bash "$(dirname "$0")/train_stage3.sh" "$@"

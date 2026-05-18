#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export LOG_ROOT=${LOG_ROOT:-logs/h200_fastexplore}
export PIPELINE_TS=${PIPELINE_TS:-$(date +"%Y%m%d_%H%M%S")_server4}
export ROUTE=p3_lora_adapter
export TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}
export NUM_PROCESSES=${NUM_PROCESSES:-8}

bash examples/calvin/train_files/run_h200_fastexplore_route.sh

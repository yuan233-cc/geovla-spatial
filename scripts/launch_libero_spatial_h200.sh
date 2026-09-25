#!/usr/bin/env bash
set -euo pipefail

# Host-side foreground launcher. Keep this process alive (normally under
# nohup) so Enroot is not torn down while the trainer is running.

: "${SLURM_JOB_ID:?This script must run inside a Slurm allocation}"
: "${RUN_ID:?Set a unique run ID}"
: "${WANDB_ENTITY:?Set the verified W&B entity slug}"
[[ "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]
NUM_GPUS="${NUM_GPUS:-1}"
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer" >&2
  exit 2
fi
# A fresh SSH session into the Slurm job may not inherit CUDA_VISIBLE_DEVICES.
# The device cgroup exposes exactly the allocated GPUs, re-indexed from zero.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
fi

RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-/tmp/yuan/geovla_runtime/job-${SLURM_JOB_ID}/runtime.env}"
test -f "${RUNTIME_ENV_FILE}"
# shellcheck disable=SC1090
source "${RUNTIME_ENV_FILE}"

WANDB_PROJECT="${WANDB_PROJECT:-geovla_libero_spatial}"
WANDB_NETRC="${WANDB_NETRC:-/tmp/yuan/credentials/job-${SLURM_JOB_ID}/wandb.netrc}"
MAX_STEPS="${MAX_STEPS:-}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
SAVE_ON_TERMINATE="${SAVE_ON_TERMINATE:-True}"

# The paired H200 NVL devices on Aachen are connected through PCIe (PIX), and
# the default NCCL P2P/IB path was observed to hang at the first collective.
# Shared-memory collectives are slower but reliable for this single-node job.
if (( NUM_GPUS > 1 )); then
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
fi

test -s "${WANDB_NETRC}"
test "$(stat -c %a "${WANDB_NETRC}")" = "600"
test ! -e "${RUN_ROOT_DIR}/${RUN_ID}"
test -w "${RUN_ROOT_DIR}"

WANDB_ROOT="/tmp/yuan/wandb/job-${SLURM_JOB_ID}/${RUN_ID}"
HF_HOME="/tmp/yuan/huggingface/job-${SLURM_JOB_ID}"
mkdir -p \
  "${WANDB_ROOT}/data" \
  "${WANDB_ROOT}/cache" \
  "${WANDB_ROOT}/config" \
  "${HF_HOME}"

export ENROOT_DATA_PATH="/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/data"
export ENROOT_CACHE_PATH="/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/cache"
export ENROOT_RUNTIME_PATH="/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/runtime"
export ENROOT_TEMP_PATH="/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/tmp"

enroot_args=(
  start --root --rw
  --mount /mnt:/mnt
  --mount /tmp:/tmp
  --mount "${GEOVLA_STORAGE_MOUNT}"
  --mount "${GEOVLA_VENV_DIR}:/opt/conda/envs/geovla"
)

container_env=(
  "PATH=/opt/conda/envs/geovla/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  "PYTHONPATH=${CODE_ROOT}"
  "PYTHON_BIN=/opt/conda/envs/geovla/bin/python"
  "DATA_ROOT_DIR=${DATA_ROOT_DIR}"
  "PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT}"
  "LLAMA2_7B_PATH=${LLAMA2_7B_PATH}"
  "RUN_ROOT_DIR=${RUN_ROOT_DIR}"
  "RUN_ID=${RUN_ID}"
  "WANDB_ENTITY=${WANDB_ENTITY}"
  "WANDB_PROJECT=${WANDB_PROJECT}"
  "WANDB_MODE=online"
  "WANDB_DIR=${WANDB_ROOT}"
  "WANDB_DATA_DIR=${WANDB_ROOT}/data"
  "WANDB_CACHE_DIR=${WANDB_ROOT}/cache"
  "WANDB_CONFIG_DIR=${WANDB_ROOT}/config"
  "WANDB_FINISH_WAIT_SECONDS=${WANDB_FINISH_WAIT_SECONDS:-10}"
  "NETRC=${WANDB_NETRC}"
  "HF_HOME=${HF_HOME}"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  "NUM_GPUS=${NUM_GPUS}"
  "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
  "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
  "EPOCHS=${EPOCHS}"
  "SAVE_INTERVAL=${SAVE_INTERVAL}"
  "SAVE_ON_TERMINATE=${SAVE_ON_TERMINATE}"
  "TF_CPP_MIN_LOG_LEVEL=2"
)
if [[ -n "${NCCL_P2P_DISABLE:-}" ]]; then
  container_env+=("NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}")
fi
if [[ -n "${NCCL_IB_DISABLE:-}" ]]; then
  container_env+=("NCCL_IB_DISABLE=${NCCL_IB_DISABLE}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  container_env+=("MAX_STEPS=${MAX_STEPS}")
fi

env_args=()
for item in "${container_env[@]}"; do
  env_args+=(--env "${item}")
done

exec enroot "${enroot_args[@]}" "${env_args[@]}" \
  "${CONTAINER_IMAGE}" \
  bash "${CODE_ROOT}/scripts/train_libero_spatial_h200.sh"

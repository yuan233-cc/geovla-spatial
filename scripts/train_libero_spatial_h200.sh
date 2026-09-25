#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

# Paper-style GeoVLA 3D-MoE fine-tuning on the LIBERO-Spatial-only RLDS
# release. The original recipe uses 8 GPUs with global batch 256. This wrapper
# supports either one or multiple GPUs on a single node.

: "${DATA_ROOT_DIR:?Set DATA_ROOT_DIR to the extracted/mounted RLDS root}"
: "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to the OpenVLA .pt file}"
: "${RUN_ROOT_DIR:?Set RUN_ROOT_DIR to a persistent output parent}"
: "${RUN_ID:?Set a unique run name}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY to the verified W&B entity slug}"
: "${LLAMA2_7B_PATH:?Set LLAMA2_7B_PATH to a staged meta-llama/Llama-2-7b-hf directory}"

# All architecture/tokenizer files and pretrained weights are local for this
# reproduction.  train.py still resolves its legacy hf_token setting through
# an environment-variable name, so provide a harmless placeholder when no Hub
# credential is needed.
export HF_TOKEN="${HF_TOKEN:-unused-local-assets}"

NUM_GPUS="${NUM_GPUS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-8}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
SAVE_ON_TERMINATE="${SAVE_ON_TERMINATE:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-geovla_libero_spatial}"
DATASET_TRANSITIONS="${DATASET_TRANSITIONS:-62153}"
MAX_STEPS="${MAX_STEPS:-$(( (DATASET_TRANSITIONS + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE * EPOCHS ))}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_VERSION_DIR="${DATA_ROOT_DIR}/libero_spatial_state_pc_no_noop/1.1.0"
test -f "${DATASET_VERSION_DIR}/dataset_info.json"
test -f "${DATASET_VERSION_DIR}/features.json"
test -s "${PRETRAINED_CHECKPOINT}"
test "$(basename "$(dirname "${PRETRAINED_CHECKPOINT}")")" = "checkpoints"
PRETRAINED_RUN_DIR="$(dirname "$(dirname "${PRETRAINED_CHECKPOINT}")")"
test -f "${PRETRAINED_RUN_DIR}/config.json"
test -f "${PRETRAINED_RUN_DIR}/dataset_statistics.json"
test -f "${LLAMA2_7B_PATH}/config.json"
test -f "${LLAMA2_7B_PATH}/tokenizer.json"
test -d "${RUN_ROOT_DIR}"
test -w "${RUN_ROOT_DIR}"
test ! -e "${RUN_ROOT_DIR}/${RUN_ID}"

if (( GLOBAL_BATCH_SIZE % (NUM_GPUS * PER_DEVICE_BATCH_SIZE) != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS * PER_DEVICE_BATCH_SIZE" >&2
  exit 2
fi

args=(
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
  --vla.type prism-dinosiglip-224px+oxe+diffusion
  --vla.data_mix libero_spatial_state_pc_no_noop
  --vla.expected_world_size "${NUM_GPUS}"
  --vla.global_batch_size "${GLOBAL_BATCH_SIZE}"
  --vla.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --vla.learning_rate 2e-5
  --vla.epochs "${EPOCHS}"
  --vla.shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}"
  --data_root_dir "${DATA_ROOT_DIR}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
  --image_aug False
  --wandb_project "${WANDB_PROJECT}"
  --wandb_entity "${WANDB_ENTITY}"
  --save_interval "${SAVE_INTERVAL}"
  --save_on_terminate "${SAVE_ON_TERMINATE}"
  --repeated_diffusion_steps 8
  --future_action_window_size 15
  --action_model_type DiT-B
  --is_resume False
  --load_depth True
  --depth_type dit_condition_self
  --dit_moe True
  --proprio_type shift_ee
  --load_wrist False
  --wrist_first False
)

args+=(--vla.max_steps "${MAX_STEPS}")

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes 1 \
  --nproc-per-node "${NUM_GPUS}" \
  scripts/train.py "${args[@]}"

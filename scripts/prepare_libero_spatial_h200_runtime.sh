#!/usr/bin/env bash
set -euo pipefail

# Run on the allocated H200 host after the already-validated artifacts have
# been copied into their final Aachen storage locations. This script performs
# only deterministic verification, publishes the prebuilt Python environment,
# mounts RLDS, and runs one real GeoVLA loader batch. It does not train.

: "${SLURM_JOB_ID:?This script must run inside a Slurm allocation}"
[[ "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]

: "${EXPECTED_GEOVLA_COMMIT:?Set the exact reviewed geovla-spatial commit SHA}"
POINTACT_ROOT="${POINTACT_ROOT:-/mnt/home/weihangli/pointact_project}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${POINTACT_ROOT}/containers/pointact.sqsh}"
STAGE_ROOT="${STAGE_ROOT:-/tmp/yuan/geovla_stage/job-${SLURM_JOB_ID}}"

host="$(hostname -s)"
if [[ "${host}" == "aachen" ]]; then
  STORAGE_ROOT="${GEOVLA_STORAGE_ROOT:-/local/weihangli}"
  STORAGE_MOUNT="/local:/local"
else
  STORAGE_ROOT="${GEOVLA_STORAGE_ROOT:-/nfs/aachen/weihangli}"
  STORAGE_MOUNT="/nfs:/nfs"
  storage_source="$(findmnt -T "${STORAGE_ROOT}" -n -o SOURCE)"
  [[ "${storage_source}" == 131.159.11.84:/local* ]]
fi

DATASET_DIR="${DATASET_DIR:-${STORAGE_ROOT}/datasets/libero-spatial-clean-geovla-rlds-v1}"
DATA_ARCHIVE="${DATA_ARCHIVE:-${DATASET_DIR}/libero_spatial_clean_geovla_rlds_v1.zip}"
MODEL_ROOT="${MODEL_ROOT:-${STORAGE_ROOT}/models/openvla-7b-prismatic}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${MODEL_ROOT}/checkpoints/step-295000-epoch-40-loss=0.2200.pt}"
CODE_ROOT="${CODE_ROOT:-${STORAGE_ROOT}/code/geovla-spatial}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${STORAGE_ROOT}/geovla_runtime/geovla-py310-torch2.7-cu126}"
LLAMA2_7B_PATH="${LLAMA2_7B_PATH:-${STORAGE_ROOT}/models/llama2-config-tokenizer}"
VENV_ARCHIVE="${VENV_ARCHIVE:-${STAGE_ROOT}/geovla-venv-py310-torch2.7-cu126.tar.zst}"
LLAMA_STAGE_DIR="${LLAMA_STAGE_DIR:-${STAGE_ROOT}/llama2-config-tokenizer}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/tmp/yuan/geovla_dataset/job-${SLURM_JOB_ID}/libero-spatial-rlds}"
RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-/tmp/yuan/geovla_runtime/job-${SLURM_JOB_ID}/runtime.env}"

test "$(stat -c %U "${STORAGE_ROOT}")" = "weihangli"
test -w "${STORAGE_ROOT}"
test -s "${CONTAINER_IMAGE}"
test "$(git -C "${CODE_ROOT}" rev-parse HEAD)" = "${EXPECTED_GEOVLA_COMMIT}"
test -z "$(git -C "${CODE_ROOT}" status --porcelain)"

echo "2c057763e4803e961e63740ffa877ae91a43dbfb01e09ae9aaebe7038163fc94  ${DATA_ARCHIVE}" | sha256sum -c -
echo "7d91ac278fa6a5ef6ab6d7daa18f67309ced793d3820ca1b3251df723fc47853  ${MODEL_ROOT}/config.json" | sha256sum -c -
echo "45ea0007022ebbe1223bdfec05d49a27246b8e37df0d30e859f79e8d163a6ed5  ${MODEL_ROOT}/dataset_statistics.json" | sha256sum -c -
echo "2c2497cd9e0ecced65e54b0771172f22e3ed64d0c0af339e094349715d3b3602  ${PRETRAINED_CHECKPOINT}" | sha256sum -c -

if [[ ! -e "${RUNTIME_ROOT}" ]]; then
  runtime_partial="${RUNTIME_ROOT}.partial.${SLURM_JOB_ID}"
  test ! -e "${runtime_partial}"
  echo "bf40537e2f66d7d9d7393cb5a7b49421cc4b2d5c08b97620a03efaf28fdfda6e  ${VENV_ARCHIVE}" | sha256sum -c -
  mkdir -p "${runtime_partial}"
  tar --zstd -xf "${VENV_ARCHIVE}" -C "${runtime_partial}"
  test -L "${runtime_partial}/opt/conda/envs/geovla/bin/python" || \
    test -x "${runtime_partial}/opt/conda/envs/geovla/bin/python"
  mv -T "${runtime_partial}" "${RUNTIME_ROOT}"
fi
VENV_DIR="${RUNTIME_ROOT}/opt/conda/envs/geovla"
test -L "${VENV_DIR}/bin/python" || test -x "${VENV_DIR}/bin/python"

if [[ ! -e "${LLAMA2_7B_PATH}" ]]; then
  llama_partial="${LLAMA2_7B_PATH}.partial.${SLURM_JOB_ID}"
  test ! -e "${llama_partial}"
  mkdir -p "${llama_partial}"
  cp -a "${LLAMA_STAGE_DIR}/." "${llama_partial}/"
  (cd "${llama_partial}" && sha256sum -c SHA256SUMS)
  mv -T "${llama_partial}" "${LLAMA2_7B_PATH}"
fi
(cd "${LLAMA2_7B_PATH}" && sha256sum -c SHA256SUMS)

mkdir -p \
  "${DATA_ROOT_DIR}" \
  "${STORAGE_ROOT}/checkpoints" \
  "$(dirname "${RUNTIME_ENV_FILE}")" \
  "/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/data" \
  "/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/cache" \
  "/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/runtime" \
  "/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/tmp"
chmod 700 \
  "/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/runtime" \
  "/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/tmp"

if ! mountpoint -q "${DATA_ROOT_DIR}"; then
  python3 /mnt/datasets/tools/mount_dataset.py "${DATA_ARCHIVE}" "${DATA_ROOT_DIR}"
fi
mountpoint -q "${DATA_ROOT_DIR}"
test -f "${DATA_ROOT_DIR}/libero_spatial_state_pc_no_noop/1.1.0/dataset_info.json"
test -f "${DATA_ROOT_DIR}/libero_spatial_state_pc_no_noop/1.1.0/features.json"

export ENROOT_DATA_PATH="/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/data"
export ENROOT_CACHE_PATH="/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/cache"
export ENROOT_RUNTIME_PATH="/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/runtime"
export ENROOT_TEMP_PATH="/tmp/yuan/geovla_enroot/job-${SLURM_JOB_ID}/tmp"

enroot start \
  --root \
  --rw \
  --mount /mnt:/mnt \
  --mount /tmp:/tmp \
  --mount "${STORAGE_MOUNT}" \
  --mount "${VENV_DIR}:/opt/conda/envs/geovla" \
  "${CONTAINER_IMAGE}" \
  /usr/bin/env \
    "PATH=/opt/conda/envs/geovla/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "PYTHONPATH=${CODE_ROOT}" \
    TF_CPP_MIN_LOG_LEVEL=2 \
    /opt/conda/envs/geovla/bin/python \
    "${CODE_ROOT}/rlds_dataset_builder/libero_spatial_state_pc_no_noop/smoke_geovla_loader.py" \
    --data-root "${DATA_ROOT_DIR}"

{
  printf 'export GEOVLA_STORAGE_ROOT=%q\n' "${STORAGE_ROOT}"
  printf 'export GEOVLA_STORAGE_MOUNT=%q\n' "${STORAGE_MOUNT}"
  printf 'export CODE_ROOT=%q\n' "${CODE_ROOT}"
  printf 'export DATA_ROOT_DIR=%q\n' "${DATA_ROOT_DIR}"
  printf 'export PRETRAINED_CHECKPOINT=%q\n' "${PRETRAINED_CHECKPOINT}"
  printf 'export LLAMA2_7B_PATH=%q\n' "${LLAMA2_7B_PATH}"
  printf 'export GEOVLA_VENV_DIR=%q\n' "${VENV_DIR}"
  printf 'export RUN_ROOT_DIR=%q\n' "${STORAGE_ROOT}/checkpoints"
  printf 'export CONTAINER_IMAGE=%q\n' "${CONTAINER_IMAGE}"
} > "${RUNTIME_ENV_FILE}"
chmod 600 "${RUNTIME_ENV_FILE}"

echo "GEOVLA_RUNTIME_READY ${RUNTIME_ENV_FILE}"

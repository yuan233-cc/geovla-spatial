#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
SOURCE="${SOURCE:-/media/hyunjun/wli_data/libero_spatial_clean_geovla_npz_v1}"
DATA_ROOT="${DATA_ROOT:-/media/hyunjun/wli_data/libero_spatial_clean_geovla_rlds_v1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BUILDER_DIR="${ROOT}/geovla.code/rlds_dataset_builder/libero_spatial_clean_state_pc_no_noop"

export PYTHONNOUSERSITE=1
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

TFDS_PARALLEL_ARGS=()
if (( NUM_PROCESSES > 1 )); then
  TFDS_PARALLEL_ARGS=(--num-processes "${NUM_PROCESSES}")
fi

cd "${ROOT}"
if [[ ! -f "${SOURCE}/validation_report.json" ]]; then
  /home/hyunjun/anaconda3/envs/cavla3d/bin/python \
    "${ROOT}/robot-PointAct/experiments/libero_polar/validate_libero_spatial_clean_npz.py" \
    --source "${SOURCE}"
fi

/home/hyunjun/anaconda3/envs/cavla3d/bin/tfds build \
  "${BUILDER_DIR}/libero_spatial_clean_state_pc_no_noop_dataset_builder.py" \
  --manual_dir "${SOURCE}" \
  --data_dir "${DATA_ROOT}" \
  "${TFDS_PARALLEL_ARGS[@]}" \
  --overwrite

/home/hyunjun/anaconda3/envs/cavla3d/bin/python \
  "${BUILDER_DIR}/validate_rlds.py" \
  --source "${SOURCE}" \
  --data-root "${DATA_ROOT}"

/home/hyunjun/anaconda3/envs/cavla3d/bin/python \
  "${BUILDER_DIR}/smoke_geovla_loader.py" \
  --data-root "${DATA_ROOT}" \
  --future-action-window-size 15

cp "${BUILDER_DIR}/DATASET_CARD.md" "${DATA_ROOT}/README.md"
cp "${SOURCE}/validation_report.json" "${DATA_ROOT}/source_validation_report.json"

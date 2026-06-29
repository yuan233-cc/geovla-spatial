
export HF_HOME="/data/model/"
export TORCH_HOME="/data/model/"
export CUDA_VISIBLE_DEVICES=0

# 默认都不开启，可通过运行脚本时传入参数来开启
PROPRIO_TYPE_FLAG=""
LOAD_DEPTH_FLAG=""
LOAD_PCD_FLAG=""
DEPTH_TYPE=""
DIT_MOE=""
LOAD_WRIST_FLAG=""
SAVED_MODEL_PATH=""
DEBUG=""

# 解析传入的参数
while [[ $# -gt 0 ]]; do
  key="$1"
  value="$2"
  case $key in
    --saved_model_path) SAVED_MODEL_PATH="--saved_model_path $value"; shift 2;;
    --proprio_type) PROPRIO_TYPE_FLAG="--proprio_type $value"; shift 2;; # shift_ee, vlm_condition@8, goal
    --load_depth) LOAD_DEPTH_FLAG="--load_depth"; shift;; # from depth to pcd
    --depth_type) DEPTH_TYPE="--depth_type $value"; shift 2;;
    --dit_moe) DIT_MOE="--dit_moe"; shift;; 
    --load_pcd) LOAD_PCD_FLAG="--load_pcd"; shift;;
    --load_wrist) LOAD_WRIST_FLAG="--load_wrist"; shift;;
    --debug) DEBUG="--debug"; shift;;
  esac
done


####################################################################### maniskill
# --action_chunking --action_chunking_window 16  
# --depth_type dit_condition_self 
# --action_ensemble --action_ensemble_horizon 7
python scripts/maniskill_deploy.py --use_bf16 \
      --action_ensemble --action_ensemble_horizon 7 \
      --cfg_scale 1.5 \
      $SAVED_MODEL_PATH \
      $DEPTH_TYPE \
      $PROPRIO_TYPE_FLAG $LOAD_DEPTH_FLAG $LOAD_PCD_FLAG $LOAD_WRIST_FLAG $DIT_MOE \
      $DEBUG \
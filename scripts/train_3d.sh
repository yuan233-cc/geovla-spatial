export WANDB_PROJECT=geovla

# 模型缓存位置，/mnt/sunl-benchmark/models/  /data/model/
export HF_HOME="/data/model/"
export TORCH_HOME="/data/model/"
: "${HF_TOKEN:?Please set HF_TOKEN in your environment}"

DEBUG=False
NUMS=(8 8 256 32) 

if [[ " $@ " =~ " --debug " ]]; then
    NUMS=(1 1 32 1) # (8 8 256 32) (1 1 4 4)
    DEBUG=True
    SAVE_PATH=debug
    export WANDB_MODE=disabled
    export CUDA_VISIBLE_DEVICES=0
fi

##################################################################################################################
# initialized from OpenVLA checkpoint
# /mnt/sunl-benchmark/models/hub/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt
# /data/model/hub/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt

###### /mnt/sunl-benchmark/datasets/widowx_4/tfds/delta_auto_05cm_10ang_drop ###################################
# x4_3d_base_task, x14_shift_task, x4_3d_black_task, x4_3d_task, x4_put_basketball
# x4_matryoshka_doll, x4_higher_task
## widowx ###################################################################################################
SAVE_PATH=20250718_3dvla_x4_shift_task
OUTPUT_DIR=/data/outputs/geovla/widowx

torchrun --standalone --nnodes 1 --nproc-per-node ${NUMS[0]} scripts/train.py \
  --pretrained_checkpoint /data/model/hub/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix "x14_shift_task" \
  --vla.expected_world_size ${NUMS[1]} \
  --vla.global_batch_size ${NUMS[2]} \
  --vla.per_device_batch_size ${NUMS[3]} \
  --vla.learning_rate 2e-5 \
  --vla.epochs 1 \
  --vla.shuffle_buffer_size 5000 \
  --data_root_dir /mnt/sunl-benchmark/datasets/widowx_all/tfds/delta_auto_05cm_10ang_drop_camera \
  --run_root_dir  $OUTPUT_DIR \
  --run_id $SAVE_PATH \
  --image_aug False \
  --wandb_project geovla_widowx \
  --save_interval 2000 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume False \
  --load_depth True \
  --depth_type dit_condition_self \
  --proprio_type extrinsic \
  --ssh_debug $DEBUG \
  --load_wrist True \
  --wrist_first False \


##################################################################################################################
# pick_cube_pcd_goal_state, pick_single_egad_pcd_goal_state, pick_single_ycb_pcd_goal_state
# stack_cube_pcd_goal_state, pick_clutter_ycb_pcd_goal_state
# agg_maniskill_pcd_goal_state, agg_maniskill_with_state_wrist_pc_tcp, agg_maniskill_with_state_wrist_depth_tcp
# SAVE_PATH=20250702_maniskill_3dvla_new_ori_10000
# OUTPUT_DIR=/data/outputs/geovla/maniskill

# torchrun --standalone --nnodes 1 --nproc-per-node ${NUMS[0]} scripts/train.py \
#   --pretrained_checkpoint /data/model/hub/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
#   --vla.type prism-dinosiglip-224px+oxe+diffusion \
#   --vla.data_mix "agg_maniskill_pcd_goal_state" \
#   --vla.expected_world_size ${NUMS[1]} \
#   --vla.global_batch_size ${NUMS[2]} \
#   --vla.per_device_batch_size ${NUMS[3]} \
#   --vla.learning_rate 2e-5 \
#   --vla.epochs 1 \
#   --vla.shuffle_buffer_size 10000 \
#   --data_root_dir /mnt/sunl-benchmark/datasets/maniskill/tfds \
#   --run_root_dir  $OUTPUT_DIR \
#   --run_id $SAVE_PATH \
#   --image_aug False \
#   --wandb_project geovla_maniskill \
#   --save_interval 2000 \
#   --repeated_diffusion_steps 8 \
#   --future_action_window_size 15 \
#   --action_model_type DiT-B \
#   --is_resume False \
#   --load_depth True \
#   --depth_type dit_condition_self \
#   --proprio_type goal+shift_ee+vlm_condition@8 \
#   --ssh_debug $DEBUG \
#   --load_wrist False \
#   --wrist_first False \


##################################################################################################################
# libero_all_with_state_wrist_pc, libero_all_state_pc_no_noop
# SAVE_PATH=20250708_libero_all_3dvla_from24_noop_100000
# OUTPUT_DIR=/data/outputs/geovla/libero

# torchrun --standalone --nnodes 1 --nproc-per-node ${NUMS[0]} scripts/train.py \
#   --pretrained_checkpoint /mnt/sunl-benchmark/outputs/geovla/libero/20250705_libero_all_3dvla_10000/checkpoints/step-024000-epoch-06-loss=0.0282.pt \
#   --vla.type prism-dinosiglip-224px+oxe+diffusion \
#   --vla.data_mix "libero_all_state_pc_no_noop" \
#   --vla.expected_world_size ${NUMS[1]} \
#   --vla.global_batch_size ${NUMS[2]} \
#   --vla.per_device_batch_size ${NUMS[3]} \
#   --vla.learning_rate 2e-5 \
#   --vla.epochs 1 \
#   --vla.shuffle_buffer_size 10000 \
#   --data_root_dir /mnt/sunl-benchmark/datasets/libero/tfds \
#   --run_root_dir  $OUTPUT_DIR \
#   --run_id $SAVE_PATH \
#   --image_aug False \
#   --wandb_project geovla_libero \
#   --save_interval 2000 \
#   --repeated_diffusion_steps 8 \
#   --future_action_window_size 15 \
#   --action_model_type DiT-B \
#   --is_resume False \
#   --load_depth True \
#   --depth_type dit_condition_self \
#   --proprio_type shift_ee \
#   --ssh_debug $DEBUG \
#   --load_wrist False \
#   --wrist_first False \
"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

# fmt: off
OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === Bridge V2 Dataset ===
    "bridge": [
        # ("bridge_oxe", 1.0),                                    # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],


    # === [Moderate-Scale] Bridge++ Mixtures ===
    "bridge_rt_1": [
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website

        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
    ],

    # === RT-X Mixtures ===
    "rtx": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),
    ],

    "rtx_franka": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),

        ("taco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("viola", 1.0),
        ("toto", 1.0),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 3.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("maniskill_dataset_converted_externally_to_rlds", 0.1),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 5.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("berkeley_rpt_converted_externally_to_rlds", 1.0),
        ("kaist_nonprehensile_converted_externally_to_rlds", 3.0),
        ("stanford_robocook_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("cmu_play_fusion", 1.0),
    ],

    # === Open-X Magic Soup ===
    "oxe_magic_soup": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        # ("nyu_door_opening_surprising_effectiveness", 1.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        # ("bc_z", 0.2),                                        # Note --> raw data is broken!
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        # ("uiuc_d3field", 1.0),                                # Note --> raw data is broken!
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
    ],

    # === Open-X Magic Soup++ ===
    "oxe_magic_soup_plus": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        ("fmb_dataset", 1.0),
        ("dobbe", 0.2),
        ("droid", 0.06),
    ],

    "oxe_magic_soup_plus_minus": [
        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        # ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        ("fmb_dataset", 1.0),
        ("dobbe", 0.2),
        # ("droid", 0.06),
    ],

    # === T-DROID Dataset ===
    "tdroid_carrot_in_bowl": [
        ("tdroid_carrot_in_bowl", 1.0),
    ],
    "tdroid_pour_corn_in_pot": [
        ("tdroid_pour_corn_in_pot", 1.0),
    ],
    "tdroid_flip_pot_upright": [
        ("tdroid_flip_pot_upright", 1.0),
    ],
    "tdroid_move_object_onto_plate": [
        ("tdroid_move_object_onto_plate", 1.0),
    ],
    "tdroid_knock_object_over": [
        ("tdroid_knock_object_over", 1.0),
    ],
    "tdroid_cover_object_with_towel": [
        ("tdroid_cover_object_with_towel", 1.0),
    ],

    # === DROID Finetuning Datasets ===
    "droid_wipe": [
        ("droid_wipe", 1.0),
    ],

    # === Custom Finetuning Datasets ===
    "custom_finetuning": [
        ("custom_finetuning", 1.0),
    ],

    # === Maniskill Finetuning Datasets ===
    #######################################################################################
    "agg_maniskill": [ # 带有state信息的
        ("pick_cube", 1.0), 
        ("stack_cube", 1.0),
        ("pick_single_ycb", 1.0),
        ("pick_single_egad", 1.0),
        ("pick_clutter_ycb", 1.0),
    ], 
    "pick_cube": [("pick_cube", 1.0)],
    "stack_cube": [("stack_cube", 1.0)],
    "pick_single_ycb": [("pick_single_ycb", 1.0)],
    "pick_single_egad": [("pick_single_egad", 1.0)],
    "pick_clutter_ycb": [("pick_clutter_ycb", 1.0)],
    #######################################################################################
    "agg_maniskill_with_state_wrist_depth_tcp": [ # 带有state信息, wrist, depth, tcp
        ("pick_cube_with_state_wrist_depth_tcp", 1.0), 
        ("stack_cube_with_state_wrist_depth_tcp", 1.0),
        ("pick_single_ycb_with_state_wrist_depth_tcp", 1.0),
        ("pick_single_egad_with_state_wrist_depth_tcp", 1.0),
        ("pick_clutter_ycb_with_state_wrist_depth_tcp", 1.0),
    ], 
    "pick_cube_with_state_wrist_depth_tcp": [
        ("pick_cube_with_state_wrist_depth_tcp", 1.0)
    ],
    "stack_cube_with_state_wrist_depth_tcp": [
        ("stack_cube_with_state_wrist_depth_tcp", 1.0)
    ],
    "pick_single_ycb_with_state_wrist_depth_tcp": [
        ("pick_single_ycb_with_state_wrist_depth_tcp", 1.0)
    ],
    "pick_single_egad_with_state_wrist_depth_tcp": [
        ("pick_single_egad_with_state_wrist_depth_tcp", 1.0)
    ],
    "pick_clutter_ycb_with_state_wrist_depth_tcp": [
        ("pick_clutter_ycb_with_state_wrist_depth_tcp", 1.0)
    ],
    
    "agg_maniskill_with_state_wrist_pc_tcp": [ # 带有state信息, wrist, pc, tcp
        ("pick_cube_with_state_wrist_pc_tcp", 1.0), 
        ("stack_cube_with_state_wrist_pc_tcp", 1.0),
        ("pick_single_ycb_with_state_wrist_pc_tcp", 1.0),
        ("pick_single_egad_with_state_wrist_pc_tcp", 1.0),
        ("pick_clutter_ycb_with_state_wrist_pc_tcp", 1.0),
    ], 
    "pick_cube_with_state_wrist_pc_tcp": [
        ("pick_cube_with_state_wrist_pc_tcp", 1.0)
    ],
    "stack_cube_with_state_wrist_pc_tcp": [
        ("stack_cube_with_state_wrist_pc_tcp", 1.0)
    ],
    "pick_single_ycb_with_state_wrist_pc_tcp": [
        ("pick_single_ycb_with_state_wrist_pc_tcp", 1.0)
    ],
    "pick_single_egad_with_state_wrist_pc_tcp": [
        ("pick_single_egad_with_state_wrist_pc_tcp", 1.0)
    ],
    "pick_clutter_ycb_with_state_wrist_pc_tcp": [
        ("pick_clutter_ycb_with_state_wrist_pc_tcp", 1.0)
    ],
    #######################################################################################
    "maniskill_with_state_wrist_depth_tcp": [ # 带有state信息, wrist, depth, tcp 整体
        ("maniskill_with_state_wrist_depth_tcp", 1.0)
    ], 
    "stack_cube_with_state": [
        ("stack_cube_with_state", 1.0)
    ], 

    "agg_maniskill_pcd_goal_state": [ # 带有state信息, wrist, pc, tcp
        ("pick_cube_pcd_goal_state", 1.0), 
        ("stack_cube_pcd_goal_state", 1.0),
        ("pick_single_ycb_pcd_goal_state", 1.0),
        ("pick_single_egad_pcd_goal_state", 1.0),
        ("pick_clutter_ycb_pcd_goal_state", 1.0),
    ], 
    "pick_cube_pcd_goal_state": [("pick_cube_pcd_goal_state", 1.0)],

    #######################################################################################
    "libero_10_no_noops":[("libero_10_no_noops", 1.0)],
    "libero_10_with_state_wrist_pc":[("libero_10_with_state_wrist_pc", 1.0)],
    "libero_90_with_state_wrist_depth":[("libero_90_with_state_wrist_depth", 1.0)],
    "libero_90_with_state_wrist_pc":[("libero_90_with_state_wrist_pc", 1.0)],
    "libero_90_with_state_wrist_pc_x256":[("libero_90_with_state_wrist_pc_x256", 1.0)],
    "libero_all_with_state_wrist_pc":[
        ("libero_10_with_state_wrist_pc", 1.0), 
        ("libero_90_with_state_wrist_pc", 1.0),
        ("libero_goal_with_state_wrist_pc", 1.0), 
        ("libero_spatial_with_state_wrist_pc", 1.0),
        ("libero_object_with_state_wrist_pc", 1.0)
    ],
    "libero_all_state_pc_no_noop":[
        ("libero_10_state_pc_no_noop", 1.0), 
        ("libero_90_state_pc_no_noop", 1.0), 
        ("libero_goal_state_pc_no_noop", 1.0),
        ("libero_object_state_pc_no_noop", 1.0), 
        ("libero_spatial_state_pc_no_noop", 1.0),
    ],
    "libero_100_state_pc_no_noop":[
        ("libero_10_state_pc_no_noop", 1.0), 
        ("libero_90_state_pc_no_noop", 1.0),
    ],
    "libero_fun_state_pc_no_noop":[
        ("libero_goal_state_pc_no_noop", 1.0),
        ("libero_object_state_pc_no_noop", 1.0), 
        ("libero_spatial_state_pc_no_noop", 1.0),
    ],
    "libero_goal_state_pc_no_noop":[
        ("libero_goal_state_pc_no_noop", 1.0),
    ],
    "libero_object_state_pc_no_noop":[
        ("libero_object_state_pc_no_noop", 1.0),
    ],
    "libero_spatial_state_pc_no_noop":[
        ("libero_spatial_state_pc_no_noop", 1.0),
    ],
    #######################################################################################
    "x4_3d_base_task":[
        ("x4_stack_block_0514", 1.0), # 叠块
        ("x4_stack_cup_0516", 1.0), ("x4_stack_cup_other_0704", 1.0), # 叠杯子
        ("x4_stack_circle_0512", 1.0), # 
        ("x4_hang_cup_0606", 1.0), ("x4_hang_cup_other0701", 1.0), # 
        ("x4_insert_shape_other0703", 1.0)
    ],

    "x4_3d_task":[
        ("x4_hang_cup_0606", 1.0),
        ("x4_hang_cup_other0701", 1.0),
        ("x4_put_basketball5_0609", 1.0),
        ("x4_put_basketball5_other0617", 1.0),
        ("x4_matryoshka_doll_other0617", 1.0),
        ("x4_pick_carrot_0605", 1.0),
    ],

    "x4_higher_task":[
        ("x4_pick_glue_0604", 1.0),
        ("x4_stack_blue_cup_0603", 1.0),
        ("x4_pick_carrot_0605", 1.0),
    ],

    "x4_matryoshka_doll": [("x4_matryoshka_doll_other0617", 1.0)], 
    "x4_put_basketball": [("x4_put_basketball5_0609", 1.0), ("x4_put_basketball5_other0617", 1.0)], 

    "x4_3d_black_task":[
        ("x4_pick_out_scissor_other_0701", 1.0), 
        ("x4_pick_hairclip_other0702", 1.0), # black
        ("x4_stack_block_in_black_background_other0703", 1.0),
    ],

    "x14_shift_task":[
        ("x4_stack_block_0514", 1.0), 
        ("x4_stack_cup_0516", 1.0), 
        ("x4_stack_circle_0512", 1.0), # x4老任务
        ("x4_pick_carrot_0605", 1.0), 
        ("x4_hang_cup_0606", 1.0), 
        ("x4_put_basketball_0609", 1.0), # 新数据
        ("x1_insert_ring_0509", 1.0), 
        ("x1_push_button_0507", 1.0), 
        ("x1_put_carrot_0507", 1.0),
        ("x1_put_glue_0506", 1.0), 
        ("x1_slide_block_0509", 1.0), 
        ("x1_stack_cube_0509", 1.0), 
        ("x1_stack_cup_0508", 1.0), # x1
        ("x4_hang_cup_other0701", 1.0), 
        ("x4_matryoshka_doll_other0617", 1.0), 
        ("x4_put_basketball5_other0617", 1.0),
        ("x4_put_basketball5_0609", 1.0),
        ("x4_pick_out_scissor_other_0701", 1.0),
        ("x4_pick_hairclip_other0702", 1.0), # black
        ("x4_stack_block_in_black_background_other0703", 1.0),
        ("x4_insert_shape_other0703", 1.0),
        ("x4_stack_cup_other_0704", 1.0),
    ]
}
# fmt: on

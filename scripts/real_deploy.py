from copy import deepcopy
import pickle
import time
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union
import os
import argparse
import json
import math
from flask import Flask, request, jsonify
import tempfile
import torch
from vla import load_vla
from vla.adaptive_ensemble import AdaptiveEnsembler

app = Flask(__name__)
ENV_ID = None


class GeoVLAService:
    def __init__(
        self,
        saved_model_path: str,
        unnorm_key: str = None,
        image_size: list[int] = [224, 224],
        action_model_type: str = "DiT-B",  # choose from ['DiT-Small', 'DiT-Base', 'DiT-Large'] to match the model weight
        future_action_window_size: int = 15,
        cfg_scale: float = 1.5,
        num_ddim_steps: int = 10, 
        use_ddim: bool = True,
        use_bf16: bool = True,
        action_dim: int = 7,
        action_ensemble: bool = True,
        adaptive_ensemble_alpha: float = 0.1,
        action_ensemble_horizon: int = 2,
        action_chunking: bool = False,
        action_chunking_window: Optional[int] = None,
        args=None,
        **kwargs,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert not (action_chunking and action_ensemble), "Now 'action_chunking' and 'action_ensemble' cannot both be True."  

        self.unnorm_key = unnorm_key

        print(f"*** unnorm_key: {unnorm_key} ***")
        self.vla = load_vla(
          saved_model_path,
          load_for_training=False, 
          action_model_type=action_model_type,
          future_action_window_size=future_action_window_size,
          action_dim=action_dim, 
          **kwargs
        )
        if use_bf16:
            self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        self.vla = self.vla.to("cuda").eval()
        self.cfg_scale = cfg_scale

        self.image_size = image_size
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.action_chunking = action_chunking
        self.action_chunking_window = action_chunking_window
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None

        self.args = args
        self.reset()

    def reset(self) -> None:
        self.vla.reset_bank()
        if self.action_ensemble:
            self.action_ensembler.reset()

    def step(
        self, image: Image.Image, 
        task_description: Optional[str] = None, 
        **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: Image.image
            task_description: Optional[str], task description
        Output:
            action: list[float], the ensembled 7-DoFs action of End-effector and gripper

        """

        # [IMPORTANT!]: Please process the input images here in exactly the same way as the images
        # were processed during finetuning to ensure alignment between inference and training.
        # Make sure, as much as possible, that the gripper is visible in the processed images.
        resized_image = resize_image(image, size=self.image_size)
        resized_image.save("real.jpg")
        if kwargs.get("wrist_img") is not None:
            kwargs['wrist_img'] = resize_image(kwargs.pop("wrist_img"), size=self.image_size)
            kwargs['wrist_img'].save("maniskill_wrist.jpg")

        unnormed_actions, normalized_actions = self.vla.predict_action(
            image=resized_image, 
            instruction=task_description, 
            unnorm_key=self.unnorm_key, 
            do_sample=False, 
            cfg_scale=self.cfg_scale, 
            use_ddim=self.use_ddim, 
            num_ddim_steps=self.num_ddim_steps,
            **kwargs
            )

        if self.action_ensemble:
            unnormed_actions = self.action_ensembler.ensemble_action(unnormed_actions)
            # Translate the value of the gripper's open/close state to 0 or 1.
            # Please adjust this line according to the control mode of different grippers.
            unnormed_actions[6] = unnormed_actions[6] > 0.5
            action = [unnormed_actions.tolist()]
        elif self.action_chunking:
            # [IMPORTANT!]: Please modify the code here to output multiple actions at once.
            # The code below only outputs the first action in the chunking.
            # The chunking window size can be adjusted by modifying the 'action_chunking_window' parameter.
            if self.action_chunking_window is not None:
                chunked_actions = []
                for i in range(0, self.action_chunking_window):
                    chunked_actions.append(unnormed_actions[i].tolist())
                action = chunked_actions
            else:
                raise ValueError("Please specify the 'action_chunking_window' when using action chunking.")
        else:
            # Output the first action in the chunking. Can be modified to output multiple actions at once.
            unnormed_actions = unnormed_actions[0]
            action = unnormed_actions.tolist()

        return action


# [IMPORTANT!]: Please modify the image processing code here to ensure that the input images  
# are handled in exactly the same way as during the finetuning phase.
# Make sure, as much as possible, that the gripper is visible in the processed images.
def resize_image(image: Image, size=(224, 224), shift_to_left=0):
    w, h = image.size
    assert h <= w, "Height should be less than width"
    left_margin = (w - h) // 2 - shift_to_left
    left_margin = min(max(left_margin, 0), w - h)
    image = image.crop((left_margin, 0, left_margin + h, h))

    image = image.resize(size, resample=Image.LANCZOS)
    
    return image


parser = argparse.ArgumentParser()
parser.add_argument("--saved_model_path", type=str, required=True)
parser.add_argument("--unnorm_key", type=str, default=None)
parser.add_argument("--image_size", type=list[int], default=[224, 224])
parser.add_argument("--action_model_type", type=str, default="DiT-B")
parser.add_argument("--future_action_window_size", type=int, default=15)
parser.add_argument("--cfg_scale", type=float, default=1.5)
parser.add_argument("--port", type=int, default=1234)
parser.add_argument("--use_bf16", action="store_true")
parser.add_argument("--action_dim", type=int, default=7)
parser.add_argument("--action_ensemble", action="store_true")
parser.add_argument("--action_ensemble_horizon", type=int, default=2)
parser.add_argument("--adaptive_ensemble_alpha", type=float, default=0.1)
parser.add_argument("--action_chunking", action="store_true")
parser.add_argument("--action_chunking_window", type=int, default=None)

parser.add_argument("--proprio_type", type=str, default="none")
parser.add_argument("--load_depth", action="store_true")
parser.add_argument("--depth_type", type=str, default="none")
parser.add_argument("--dit_moe", action="store_true")
parser.add_argument("--load_pcd", action="store_true")
parser.add_argument("--load_wrist", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()
########################################
from pprint import pprint
pprint(vars(args))
########################################
# start debug
if args.debug:
    import debugpy
    try:
        port = 9501
        debugpy.listen(("0.0.0.0", port))
        print(f"Waiting for debugger attach: {port}")
        debugpy.wait_for_client()
        torch.set_default_device("cuda")
    except Exception as e:
        print(f"error when attach: {e}")
        exit(0)
else:
    pass

INTRINSICS = np.array([
    [604.762512207031, 0, 328.612335205078],
    [0, 604.416931152344, 245.81575012207],
    [0, 0, 1]
])
EXTRINSICS60 = np.array([
    [-0.856091,  0.227101, -0.464255,  0.660566],
    [ 0.511669,  0.498983, -0.699435,  0.624032],
    [ 0.072813, -0.836326, -0.543376,  0.524639],
    [ 0.0,       0.0,       0.0,       1.0]
])
EXTRINSICS0 = np.array([
    [-0.019236171415071723,  0.39703106017287504, -0.9176035674338311,  0.8072784446125323],
    [ 0.9997645746975246,    0.016852857646320052, -0.013666615127061188, -0.039304540745199706],
    [ 0.010038171604846213, -0.9176504336876632,  -0.3972617734739603,   0.4973309533042442],
    [ 0.0,                   0.0,                  0.0,                  1.0]
])
EXTRINSICS30 = np.array([
    [-0.429252,  0.46677,  -0.77322,   0.767541],
    [ 0.897265,  0.318242, -0.306003,  0.249611],
    [ 0.103238, -0.825135, -0.555422,  0.514936],
    [ 0.0,       0.0,       0.0,       1.0]
])
extrinsics = np.eye(4)

def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
    return uniform_pixel_coords

def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(
        transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector,
                      (h, w, -1))

def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate(
        [pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate(
        [world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo


def pointcloud_from_depth_and_camera_params(
        depth: np.ndarray, intrinsics: np.ndarray,
        extrinsics: np.ndarray=np.eye(4)) -> np.ndarray:
    """Converts depth (in meters) to point cloud in word frame.
    :return: A numpy array of size (width, height, 3)
    """
    upc = _create_uniform_pixel_coords_image(depth.shape)
    pc = upc * np.expand_dims(depth, -1)
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(_pixel_to_world_coords(
        pc, cam_proj_mat_inv), 0)
    world_coords = world_coords_homo[..., :-1][0]
    return world_coords

inferencer = GeoVLAService(
    saved_model_path=args.saved_model_path,
    unnorm_key=args.unnorm_key,
    image_size=args.image_size,
    action_model_type=args.action_model_type,
    future_action_window_size=args.future_action_window_size,
    cfg_scale=args.cfg_scale,
    use_bf16=args.use_bf16,
    action_dim=args.action_dim,
    action_ensemble=args.action_ensemble,
    adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
    action_ensemble_horizon=args.action_ensemble_horizon,
    action_chunking=args.action_chunking,
    action_chunking_window=args.action_chunking_window,
    args=deepcopy(args),
    depth_type=args.depth_type,
    proprio_type=args.proprio_type,
    dit_moe=args.dit_moe,
)

@app.route('/reset', methods=['GET', 'POST'])
def reset():
    inferencer.reset()
    return jsonify({"response": "OK"})

@app.route('/camera/<camera_id>', methods=['GET', 'POST'])
def set_camera(camera_id):
    global extrinsics
    if int(camera_id) == 60:
        extrinsics = EXTRINSICS60
    elif int(camera_id) == 30:
        extrinsics = EXTRINSICS30
    elif int(camera_id) == 0:
        extrinsics = EXTRINSICS0
    else:
        extrinsics = np.eye(4)
    return jsonify({"response": "OK"})

@app.route('/reload/<path:model_path>', methods=['GET', 'POST'])
def reload(model_path):
    global inferencer
    model_path = os.path.join("/", model_path)
    print(f"Reloading model from {model_path}")
    pprint(vars(args))
    inferencer = GeoVLAService(
        saved_model_path=model_path,
        unnorm_key=args.unnorm_key,
        image_size=args.image_size,
        action_model_type=args.action_model_type,
        future_action_window_size=args.future_action_window_size,
        cfg_scale=args.cfg_scale,
        use_bf16=args.use_bf16,
        action_dim=args.action_dim,
        action_ensemble=args.action_ensemble,
        adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
        action_ensemble_horizon=args.action_ensemble_horizon,
        action_chunking=args.action_chunking,
        action_chunking_window=args.action_chunking_window,
        args=deepcopy(args),
        depth_type=args.depth_type,
        proprio_type=args.proprio_type,
    )
    return jsonify({"response": "OK"})

@app.route('/type/<env_id>', methods=['GET', 'POST'])
def handle_request(env_id):
    global ENV_ID, inferencer
    ENV_ID = env_id
    inferencer.unnorm_key = env_id
    return jsonify({"response": f"setting inferencer.unnorm_key = {env_id}"})

# maniskill中的接口是用的process_frame,这里需要同步, 传入的是image和text
@app.route('/process_frame', methods=['POST'])
def inference():
    image = request.files.get("image", None)
    wrist_img = request.files.get("wrist", None)
    depth = request.files.get('depth', None)
    query = request.form.get('text', '')
    
    if "states" in request.files and request.files["states"].filename != "":
        robot_states = pickle.load(request.files['states'].stream)
        state = np.array(robot_states['position']+robot_states['orientation'], dtype=np.float32)
    else:
        state = None
    
    if "goal" in request.files and request.files["goal"].filename != "":
        goal_bytes = request.files["goal"].read()
        goal = np.frombuffer(goal_bytes, dtype=np.float32)
    else:
        goal = None
    
    if "base_pc" in request.files and request.files["base_pc"].filename != "":
        base_pc = request.files["base_pc"].read()
        base_pc = np.frombuffer(base_pc, dtype=np.float16)
    else:
        base_pc = None

    if "wrist_pc" in request.files and request.files["wrist_pc"].filename != "":
        wrist_pc = request.files["wrist_pc"].read()
        wrist_pc = np.frombuffer(wrist_pc, dtype=np.float32)
    else:
        wrist_pc = None
    
    if "depth" in request.files and request.files['depth'].filename != "":
        depth = request.files["depth"].read()
        depth = np.frombuffer(depth, dtype=np.uint16).reshape(480, 640) * 1.0 / 1000 # depth is in uint16 format
    else:
        depth = None

    input_kwargs = {"task_description": query}
    if "vlm_condition" in args.proprio_type:
        assert state is not None, "state is None, please check the input"
        input_kwargs["states"] = torch.tensor(state)[None].cuda().float()
        
    if "goal" in args.proprio_type:
        assert goal is not None, "goal is None, please check the input"
        input_kwargs["goal"] = torch.tensor(goal)[None].cuda().float() if goal.sum() != 0 else None

    if args.load_wrist and wrist_img is not None:
        input_kwargs["wrist_img"] = Image.open(wrist_img)
    
    if args.load_depth and depth is not None:
        base_pc = pointcloud_from_depth_and_camera_params(depth, INTRINSICS, extrinsics=extrinsics) # [480, 640, 3]
        # 中心裁剪
        base_pc = base_pc[:, 80:640-80] # [480, 480, 3]
        base_pc = cv2.resize(base_pc, (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        input_kwargs["pcd"] = {"pc": base_pc[None] - (state[:3] if "shift_ee" in args.proprio_type else 0)}
        pass
    
    # base_pc 输入尺寸是224x224
    if args.load_pcd:
        base_pc = base_pc.reshape(256, 256, 3)
        base_pc = cv2.resize(base_pc, (224, 224), interpolation=cv2.INTER_NEAREST)
        input_kwargs["pcd"] = {"pc": base_pc[None] - (state[:3] if "shift_ee" in args.proprio_type else 0)}  # [N(B), H, W, 3]

    # print info
    print(f"lang: {query}")
    print(f"states: {', '.join([f'{x:.4f}' for x in state.tolist()]) if state is not None else 'None'}")
    print(f"goal: {', '.join([f'{x:.4f}' for x in goal.tolist()]) if goal is not None else 'None'}")

    for k, v in input_kwargs.items():
        if hasattr(v, 'shape'):
            print(f"{k}: type={type(v).__name__}, shape={tuple(v.shape)}")
        elif isinstance(v, dict):
            print(f"{k}: type={type(v).__name__}, keys={list(v.keys())}")
        else:
            print(f"{k}: type={type(v).__name__}")

    answer = inferencer.step(Image.open(image), **input_kwargs)
    print(f"action: {', '.join([f'{x:.4f}' for x in answer[0]]) if isinstance(answer, list) else 'None'}")
    
    
    # Convert action array to string based on different modes
    if inferencer.action_ensemble:
        # For action ensemble mode, directly convert the action list
        action_str = ' '.join([str(x) for x in answer])
    elif inferencer.action_chunking:
        # For action chunking mode, convert the chunked actions
        action_str = ';'.join([' '.join([str(x) for x in chunk]) for chunk in answer])
    else:
        # For single action mode
        action_str = ' '.join([str(x) for x in answer])

    return jsonify({'response': action_str})

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", debug=False, port=args.port)

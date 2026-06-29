from copy import deepcopy
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
        # resized_image.save("maniskill.jpg")
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
    dit_moe=args.dit_moe
)

@app.route('/reset', methods=['GET', 'POST'])
def reset():
    inferencer.reset()
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
        dit_moe=args.dit_moe
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
        state_bytes = request.files["states"].read()
        state = np.frombuffer(state_bytes, dtype=np.float32)
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

    input_kwargs = {"task_description": query}
    if "vlm_condition" in args.proprio_type:
        assert state is not None, "state is None, please check the input"
        if True:
            assert inferencer.unnorm_key is not None, "inferencer.unnorm_key should not be none"
            v = inferencer.vla.norm_stats[inferencer.unnorm_key]
            property_dim = int(args.proprio_type.split("@")[-1])
            low = np.array(v['proprio']["q01"])[:property_dim]
            high = np.array(v['proprio']["q99"])[:property_dim]
            state = 2 * (state - low) / (high - low + 1e-8) - 1
            np.clip(state, -1, 1, out=state)
        input_kwargs["states"] = torch.tensor(state)[None].cuda().float()
        
    if "goal" in args.proprio_type:
        assert goal is not None, "goal is None, please check the input"
        if True:
            assert inferencer.unnorm_key is not None, "inferencer.unnorm_key should not be none"
            v = inferencer.vla.norm_stats[inferencer.unnorm_key]
            low = np.array(v['proprio']["q01"])[-3:]
            high = np.array(v['proprio']["q99"])[-3:]
            goal = 2 * (goal - low) / (high - low + 1e-8) - 1
            np.clip(goal, -1, 1, out=goal)
            goals = torch.tensor(goal)[None].cuda().float() if goal.sum() != -3 else None # -1 means None
        else:
            goals = torch.tensor(goal)[None].cuda().float() if goal.sum() != 0 else None # 0 means None
        input_kwargs["goal"] = goals

    if args.load_wrist and wrist_img is not None:
        input_kwargs["wrist_img"] = Image.open(wrist_img)
    
    # 深度图像 输入尺寸是224x224
    if args.load_depth and depth is not None: # TODO: from depth to pcd
        depth = Image.open(depth).resize(224, 224, resample=Image.LANCZOS)
        input_kwargs["depth"] = depth

    # base_pc 输入尺寸是224x224
    if args.load_pcd:
        size = int((base_pc.shape[0] / 3) ** 0.5)
        assert size * size * 3 == base_pc.shape[0], f"the shape of pcd {base_pc.shape[0]} should be [{size}^2 * 3]"
        base_pc = base_pc.reshape(size, size, 3)
        base_pc = cv2.resize(base_pc, (224, 224), interpolation=cv2.INTER_NEAREST)
        pc = base_pc[None] - (state[:3] if "shift_ee" in args.proprio_type else 0)  # [N(B), H, W, 3]
        input_kwargs['pcd'] = {"pc": pc.astype(np.float32)}

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
    # clean files
    return jsonify({"response": answer})

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", debug=False, port=args.port)

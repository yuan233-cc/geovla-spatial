"""
geovlavla.py

"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaTokenizerFast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

from action_model.action_model import ActionModel
from action_model.models import DiT

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class GeoVLA(nn.Module):
    def __init__(
        self,
        vlm: PrismaticVLM,
        action_model_type: str = 'DiT-B',
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.dit_moe = kwargs.get("dit_moe", False)

        self.action_model = ActionModel(model_type = action_model_type, 
                                            token_size = token_size, 
                                            in_channels = action_dim, 
                                            future_action_window_size = future_action_window_size, 
                                            past_action_window_size = past_action_window_size,
                                            **kwargs)
        
        self.proprio_type = kwargs.get("proprio_type", "")
        
        embedding_dim = vlm.llm_backbone.llm.model.embed_tokens.embedding_dim
        ### add for processing the states
        if "vlm_condition@" in self.proprio_type:
            property_dim = int(self.proprio_type.split("@")[-1])
            self.states_encoder = nn.Sequential(
                nn.Linear(property_dim, embedding_dim), nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            overwatch.warning(f"No using states_encoder!!!")
        ### add for processing the states

        ### add for processing the depth
        self.depth_type = kwargs.get("depth_type", "")
        if "film_condition" in self.depth_type :
            from vla import FilmDepthEncoder
            self.depth_encoder = FilmDepthEncoder()
            self.depth_preprocess = self.depth_encoder.depth_preprocess
        elif "dit_condition" in self.depth_type:
            from vla import DiTDepthEncoder
            self.depth_encoder = DiTDepthEncoder()
            self.depth_preprocess = self.depth_encoder.depth_preprocess
        else:
            overwatch.warning(f"No using depth_encoder!!!")
        ### add for processing the depth

        from modules.fusion_module import CrossFusionModule
        # self.vlm_fusion = CrossFusionModule(embedding_dim=embedding_dim, num_attn_heads=8, num_layers=4, use_adaln=True)
        self.pcd_cond_fusion = CrossFusionModule(embedding_dim=embedding_dim, num_attn_heads=8, num_layers=4, use_adaln=True)
        ###

        ### add for goal postion
        if "goal" in self.proprio_type:
            self.goal_encoder = nn.Sequential(
                nn.Linear(3, embedding_dim), nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            overwatch.warning(f"No using goal_encoder!!!")
        ### add for goal postion

        if self.dit_moe:
            overwatch.warning(f"using moe !!!")
            assert "dit_condition_self" in self.depth_type, "moe module must using with dit_condition_self"

        self.vlm = vlm
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys = ['action_model', 'ema_diffusion']
        else:
            self.all_module_keys = ['action_model']
        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)
        self.all_module_keys += ['states_encoder',  "depth_encoder", 
                                 'goal_encoder', "pcd_cond_fusion", "vlm_fusion"]

        # Diffusion head is always trainable
        self._trainable_module_keys = ['action_model']

        # states_encoder is also trainable
        self._trainable_module_keys += ['states_encoder']
        # depth_encoder is also trainable
        self._trainable_module_keys += ['depth_encoder']
        # goal encoder
        self._trainable_module_keys += ['goal_encoder', "pcd_cond_fusion", "vlm_fusion"]
        self.norm_stats = norm_stats

    @property
    def trainable_module_keys(self) -> List[str]:
        keys = []
        for module_keys in self.vlm.trainable_module_keys:
            keys.append("vlm." + module_keys)
        keys += self._trainable_module_keys
        return keys
    
    @property
    def llm_backbone(self) -> LLMBackbone:
        return self.vlm.llm_backbone
    
    @property
    def vision_backbone(self) -> VisionBackbone:
        return self.vlm.vision_backbone
    
    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)
    
    def reset_bank(self):
        pass

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        depth: Optional[torch.FloatTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        goal: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        action_masks = None,
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        
        others={}   # others except pos used for rope will be present as input embeddings

        # process the states
        if "vlm_condition" in self.proprio_type:
            assert states is not None, "states should not be None when proprio_type is vlm_condition"
            states = self.states_encoder(states)
            others["states"] = states
        if "goal" in self.proprio_type:
            assert goal is not None, "goal should not be None when proprio_type is goal"
            goal = self.goal_encoder(goal) if torch.randint(0, 3, (1,)).item() < 2 else None # 一定概率不加
            if goal is not None:
                others["goal"] = goal

        V = len(pixel_values)

        if self.dit_moe:
            rand_val = torch.randint(0, 3, ()).item()
            if rand_val == 0: # rgb zero
                pixel_values = {k:{k1:torch.zeros_like(v1) for k1, v1, in v.items()} for k, v in pixel_values.items()}
                self.action_model.net.enable_moe_type("pcd")
            elif rand_val == 1: # pcd zero
                depth = {k:torch.zeros_like(v) for k, v in depth.items()}
                self.action_model.net.enable_moe_type("rgb")
            else: # no zero
                self.action_model.net.enable_moe_type("all")

        pcd_feature, pcd_ee_feature, pcd_pos, pcd_ee_pos = None, None, None, None
        if "dit_condition" in self.depth_type or "vlm_condition" in self.depth_type:
            depth = self.depth_encoder(depth) # 需要对其图片的分辨率
            others["query_pos"] = depth['pos'][0] # [N, B, 3]
            others["value_pos"] = depth['pos'][1] if len(depth['pos']) > 1 else None    # [N, B, 3]
            if "vlm_condition" in self.depth_type:
                others["pcd_vlm_token"] = depth['ee_feature'].mean(0).permute(1, 0, 2)      # [B, 1, C]

            if "dit_condition" in self.depth_type:
                pcd_feature = depth['x_adapt']                                # [M, 256, B, C]
                pcd_pos = depth['pos']                                        # [M, 256, B, 3]
                pcd_ee_feature = depth['ee_feature']                          # [M, 1, B, C]
                pcd_ee_pos = depth['ee_pos']                                  # [M, 1, B, 3]

        output: CausalLMOutputWithPast = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            others=others
        )

        # extract the last hidden state and the learnable EOS token feature
        last_hidden = output.hidden_states[-1]
        
        # 保证剩下的长度是文本的长度
        num_patch = self.vlm.img_patches + len(others)
        
        last_hidden = last_hidden[:, num_patch :]

        # extract the VLM feature
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))
        vlm_features = last_hidden.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]

        actions_history = actions[:,0:self.past_action_window_size,:]
        actions_future = actions[:, -(self.future_action_window_size+1):, :]
        
        # # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        actions_history_repeated = actions_history.repeat(repeated_diffusion_steps, 1, 1)
        vlm_features_repeated = vlm_features.repeat(repeated_diffusion_steps, 1, 1) # [repeated_diffusion_steps*B, 1, D]

        pcd_cross_condition = None
        if "dit_condition" in self.depth_type: # [V, N, B, C]

            assert pcd_feature is not None, "The pcd_feature should not be None when self.depth_type is dit_condition"
            assert pcd_ee_feature is not None, "The pcd_ee_feature should not be None when self.depth_type is dit_condition"
            assert self.vlm.lang_embeddings is not None, "The lang_embedings should not be None in the PrismaticVLM"
            assert self.vlm.img_embeddings is not None, "The img_embedings should not be None in the PrismaticVLM"
            
            
            # vlm_embeddings = torch.cat([o[:, num_patch:].gather(1, expanded_indices.unsqueeze(1)) 
            #                             for o in output.hidden_states], dim=-2).permute(1, 0, 2) # [B, N, C] -> [N, B, C]
            
            pcd_queries = torch.cat([pcd_ee_feature, pcd_feature], dim=1).flatten(0, 1) # [V * (1 + 256), B, C]
            pcd_query_pos = torch.cat([pcd_ee_pos, pcd_pos], dim=1).flatten(0, 1)       # [V * (1 + 256), B, 3]

            # image and language fusion
            # pcd_queries = self.vlm_fusion(queries=pcd_queries, values=vlm_embeddings)
            
            pcd_feature = self.pcd_cond_fusion(queries=pcd_queries, values=pcd_queries, 
                                                query_pos=pcd_query_pos, value_pos=pcd_query_pos
                                                ).permute(1, 0, 2).unflatten(sizes=(V, -1), dim=1) 
            # [V * (1 + 256), B, C] -> [B, V * (1 + 256), C] -> [B, V, (1 + 256), C]

            if "dit_condition_self" in self.depth_type:
                ee_embeddings = pcd_feature[..., 0, :].mean(1, keepdim=True)                    # [B, 1, C]
                pcd_self_feature = ee_embeddings.repeat(repeated_diffusion_steps, 1, 1)         # [B * R, 1, C]
                vlm_features_repeated = torch.cat([vlm_features_repeated, pcd_self_feature], dim=-2) # 只要pcd的条件
                # vlm_features_repeated = pcd_self_feature
            if "dit_condition_cross" in self.depth_type:
                pcd_embeddings = pcd_feature[..., 1:, :].flatten(1, 2) # [B, V * 256, C]
                pcd_cross_condition = pcd_embeddings.repeat(repeated_diffusion_steps, 1, 1) # [B * R, 1, C]
            

        # Action model forward and compute loss
        loss_noise = self.action_model.loss(actions_repeated, vlm_features_repeated, 
                                            cross_condition=pcd_cross_condition)

        return loss_noise, output

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        action_model_type: str = 'DiT-B',
        use_ema: bool = False,
        norm_stats = None,
        **kwargs,
    ) -> GeoVLA:

        # Load VLM backbone, borrowed from PrismaticVLM
        vlm = PrismaticVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cuda")["model"]
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict.keys():
            # if loading depth and conditioned by Film, we need replace the Block
            if "film_condition" in kwargs.get("depth_type", ""):
                from vla import replace_vision_backbone
                replace_vision_backbone(vlm.vision_backbone)
            try:
                vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"], strict=True)
            except RuntimeError as e:
                overwatch.warning(f"vlm.vision_backbone is modified. Some weights will be newly initialized: {e}")
                vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"], strict=False)

        if "vision_fusion" in model_state_dict.keys():
            try:
                vlm.vision_fusion.load_state_dict(model_state_dict["vision_fusion"], strict=True)
            except RuntimeError as e:
                overwatch.warning(f"vlm.vision_fusion is modified. Some weights will be newly initialized: {e}")
                vlm.vision_fusion.load_state_dict(model_state_dict["vision_fusion"], strict=False)
        else:
            overwatch.warning("No vision_fusion found in the pretrained checkpoint. Initializing a new one.")

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # Initialize GeoVLA
        geovla = GeoVLA(vlm,
                        token_size = vlm.llm_backbone.llm.lm_head.in_features,
                        action_dim = action_dim,
                        future_action_window_size = future_action_window_size,
                        past_action_window_size = past_action_window_size,
                        action_model_type = action_model_type,
                        use_ema = use_ema,
                        norm_stats = norm_stats,
                        **kwargs
                        )

        # Load ActionModel from Checkpoint
        if "action_model" in model_state_dict:
            try:
                geovla.action_model.load_state_dict(model_state_dict["action_model"], strict=True)
            except RuntimeError as e:
                overwatch.warning(f"geovla.action_model is modified. Some weights will be newly initialized: {e}")
                geovla.action_model.load_state_dict(model_state_dict["action_model"], strict=False) # 可能加载了额外的模块
            if "ema_diffusion" in model_state_dict and use_ema:
                geovla.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"])
            elif use_ema:
                geovla.ema_diffusion.load_state_dict(model_state_dict["action_model"])
        else:
            overwatch.warning("No ActionModel found in the pretrained checkpoint. Initializing a new one.")
        
        # Load states_encoder from Checkpoint
        if "states_encoder" in model_state_dict and hasattr(geovla, "states_encoder"):
            try:
                geovla.states_encoder.load_state_dict(model_state_dict["states_encoder"], strict=True)
            except RuntimeError as e:
                overwatch.warning(f"geovla.states_encoder is modified. Some weights will be newly initialized: {e}")
                geovla.states_encoder.load_state_dict(model_state_dict["states_encoder"], strict=False)
        else:
            overwatch.warning("No StatesEncoder found in the pretrained checkpoint. Initializing a new one.")

        # Load goal_encoder from Checkpoint
        if "goal_encoder" in model_state_dict  and hasattr(geovla, "goal_encoder"):
            try:
                geovla.goal_encoder.load_state_dict(model_state_dict["goal_encoder"], strict=True)
            except RuntimeError as e:
                overwatch.warning(f"geovla.goal_encoder is modified. Some weights will be newly initialized: {e}")
                geovla.goal_encoder.load_state_dict(model_state_dict["goal_encoder"], strict=False)
        else:
            overwatch.warning("No GoalEncoder found in the pretrained checkpoint or in the model")

        # Load depth_encoder from Checkpoint
        if "depth_encoder" in model_state_dict and hasattr(geovla, "depth_encoder"):
            try:
                geovla.depth_encoder.load_state_dict(model_state_dict["depth_encoder"], strict=True)
            except RuntimeError as e:
                overwatch.warning(f"geovla.depth_encoder is modified. Some weights will be newly initialized: {e}")
                geovla.depth_encoder.load_state_dict(model_state_dict["depth_encoder"], strict=False)
        else:
            overwatch.warning("No DepthEncoder found in the pretrained checkpoint or in the model")

        if "pcd_cond_fusion" in model_state_dict and hasattr(geovla, "pcd_cond_fusion"):
            try:
                geovla.pcd_cond_fusion.load_state_dict(model_state_dict["pcd_cond_fusion"], strict=True)
            except RuntimeError as e:
                overwatch.warning(f"geovla.pcd_cond_fusion is modified. Some weights will be newly initialized: {e}")
                geovla.pcd_cond_fusion.load_state_dict(model_state_dict["pcd_cond_fusion"], strict=False)
        else:
            overwatch.warning("No pcd_cond_fusion found in the pretrained checkpoint or in the model")

        # if "vlm_fusion" in model_state_dict:
        #     try:
        #         geovla.vlm_fusion.load_state_dict(model_state_dict["vlm_fusion"], strict=True)
        #     except RuntimeError as e:
        #         overwatch.warning(f"geovla.vlm_fusion is modified. Some weights will be newly initialized: {e}")
        #         geovla.vlm_fusion.load_state_dict(model_state_dict["vlm_fusion"], strict=False)
        # else:
        #     overwatch.warning("No vlm_fusion found in the pretrained checkpoint. Initializing a new one.")
            
        return geovla

    @torch.inference_mode()
    def predict_action(
        self, image: Image, 
        instruction: str, 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """

        others = {}

        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       insert it to match the inputs seen at training time. The empty token is at index 29871.
            #       We also need to add the special VLM summary token at index 2 (i.e. the EOS token).
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871, 2]).long(), dim=0).to(self.vlm.device)), dim=1
            )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = {"img": image_transform(image)}
        wrist_img = kwargs.pop("wrist_img", None)
        if wrist_img is not None:
            pixel_values["img_wrist"] = image_transform(wrist_img)

        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vlm.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: {k1:v1[None, ...].to(self.vlm.device) for k1, v1 in v.items()} for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        V = len(pixel_values)

        # using states
        states = kwargs.pop("states", None)
        if "vlm_condition" in self.proprio_type:
            assert states is not None, "states should not be None when proprio_type is vlm_condition"
            others["states"] = self.states_encoder(states)[:, None, :]

        goal = kwargs.pop("goal", None)
        if "goal" in self.proprio_type and goal is not None: # goal may be none in the task of stackcube
            others["goal"] = self.goal_encoder(goal)[:, None, :]
        
        # using pcd
        pcd = kwargs.pop("pcd", None) # pcd 是一个dict，包括pc和pc_wrist， as rope encoding
        pcd_feature, pcd_pos = None, None
        if "dit_condition" in self.depth_type or "vlm_condition" in self.depth_type:
            pcd = {k: torch.from_numpy(v).permute(0, 3, 1, 2).to(self.vlm.device) for k, v in pcd.items()}
            depth = self.depth_encoder(pcd) # 需要对其图片的分辨率
            others["query_pos"] = depth['pos'][0] # [N, B, 3]
            others["value_pos"] = depth['pos'][1] if len(depth['pos']) > 1 else None # [N, B, 3]
            if "vlm_condition" in self.depth_type:
                others["pcd_vlm_token"] = depth['ee_feature'].mean(0).permute(1, 0, 2) # [B, 1, C]
            if "dit_condition" in self.depth_type:
                pcd_feature = depth['x_adapt']                                # [M, 256, B, C]
                pcd_pos = depth['pos']                                        # [M, 256, B, 3]
                pcd_ee_feature = depth['ee_feature']                          # [M, 1, B, C]
                pcd_ee_pos = depth['ee_pos']                                  # [M, 1, B, 3]

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype

        # Generate VLM feature through vlm
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                others=others,
                **kwargs
            )
            # fmt: on

        # Extract VLM feature
        vlm_features = output.hidden_states[0][-1][:,-1,:]
        assert (vlm_features.shape[0], vlm_features.shape[1]) == (1,4096), "Batch size must be 1 for action prediction"
        using_cfg = cfg_scale > 1.0

        model_dtype = next(self.action_model.net.parameters()).dtype
        B = vlm_features.shape[0]

        vlm_features = vlm_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]
        pcd_cross_condition = None
        if "dit_condition" in self.depth_type: # [M * 256 + M, B, C]
            
            assert pcd_feature is not None, "The pcd_feature should not be None when self.depth_type is dit_condition"
            assert pcd_ee_feature is not None, "The pcd_ee_feature should not be None when self.depth_type is dit_condition"
            assert self.vlm.lang_embeddings is not None, "The lang_embedings should not be None in the PrismaticVLM"
            assert self.vlm.img_embeddings is not None, "The img_embedings should not be None in the PrismaticVLM"
            
            # vlm_embeddings = torch.stack([o[:, -1, :] for o in output.hidden_states[0]], 
            #                              dim=-2).to(model_dtype).permute(1, 0, 2) # [B, N, C] -> [N, B, C]
            
            pcd_queries = torch.cat([pcd_ee_feature, pcd_feature], dim=1).flatten(0, 1) # [V * (1 + 256), B, C]
            pcd_query_pos = torch.cat([pcd_ee_pos, pcd_pos], dim=1).flatten(0, 1)       # [V * (1 + 256), B, 3]

            # image and language fusion
            # pcd_feature = self.vlm_fusion(queries=pcd_queries, values=vlm_embeddings)
            
            pcd_feature = self.pcd_cond_fusion(queries=pcd_queries, values=pcd_queries, 
                                                query_pos=pcd_query_pos, value_pos=pcd_query_pos
                                                ).permute(1, 0, 2).unflatten(sizes=(V, -1), dim=1) 
            # [V * (1 + 256), B, C] -> [B, V * (1 + 256), C] -> [B, V, (1 + 256), C]
            
            if "dit_condition_self" in self.depth_type:
                pcd_self_feature = pcd_feature[..., 0, :].mean(1, keepdim=True)                     # [B, 1, C]
                vlm_features = torch.cat([vlm_features, pcd_self_feature], dim=-2)      # 只要pcd的条件
                # vlm_features = pcd_self_feature
            if "dit_condition_cross" in self.depth_type:
                pcd_cross_condition = pcd_feature[..., 1:, :].flatten(1, 2) # [B, V * 256, C]
        
        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels,
                            device=vlm_features.device).to(model_dtype)  #[B, T, D]
    
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition, cross_uncondition = self.action_model.net.z_embedder.uncondition, self.action_model.net.z_embedder.cross_uncondition
            uncondition, cross_uncondition = uncondition.unsqueeze(0), cross_uncondition.unsqueeze(0)  #[1, D], [256, D]
            uncondition = uncondition.expand(B, vlm_features.shape[-2], -1) #[B, 1, D]
            cross_uncondition = cross_uncondition.expand(B, cross_uncondition.shape[-2], -1) #[B, 256, D]
            z = torch.cat([vlm_features, uncondition], 0)

            if pcd_cross_condition is not None:
                cross_uncondition = cross_uncondition.repeat(B, pcd_cross_condition.shape[-2] // cross_uncondition.shape[-2], 1)
                pcd_cross_condition = torch.cat([pcd_cross_condition, cross_uncondition], 0)

            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale, cross_condition=pcd_cross_condition)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=vlm_features, cross_condition=pcd_cross_condition)
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=vlm_features.device,
                                                                eta=0.0
                                                                )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=vlm_features.device
                                                                    )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions        
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0, 0, 1) # 原来阈值是0.5，但是我在训练的时候传入的是-1，1 阈值应取0
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions

    @torch.inference_mode()
    def predict_action_batch(
        self, image: List[Image], 
        instruction: List[str], 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference in batch; maps input image and task instruction to continuous action.
        This function is used for batch inference in the simulators.
        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer
        
        input_ids = []
        pixel_values = []

        # Build VLA Prompt
        B = len(image)

        if isinstance(tokenizer, LlamaTokenizerFast):
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        for id in range(B):
            prompt_builder = self.vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction[id].lower()}?")
            prompt_text = prompt_builder.get_prompt()
            # Prepare Inputs
            single_input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device).squeeze(0)
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       insert it to match the inputs seen at training time. The empty token is at index 29871.
            #       We also need to add the special VLM summary token at index 2 (i.e. the EOS token).
            single_input_ids = torch.cat(
                (single_input_ids, torch.Tensor([29871, 2]).long().to(self.vlm.device)), dim=0
            ) # [seq]

            input_ids.append(single_input_ids)
            # Preprocess Image
            pixel_values.append(image_transform(image[id]))

        # Padding
        padding_side = "right"
        # For now, we only support Tokenizers with `padding_side = "right"`
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert padding_side == "right", f"Invalid Tokenizer `{padding_side = }`"

        model_max_length = tokenizer.model_max_length
        pad_token_id = tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        # Truncate (if necessary)
        input_ids = input_ids[:, : model_max_length]
        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(pad_token_id)

        # Preprocess Image
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(self.vlm.device)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]).to(self.vlm.device) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                attention_mask = attention_mask,
                **kwargs
            )
            # fmt: on

        # Extract VLM feature
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif hasattr(self.vlm.vision_backbone, 'siglip_featurizer') and self.vlm.vision_backbone.siglip_featurizer is not None:
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        last_hidden = output.hidden_states[0][-1]
        last_hidden = last_hidden[:, num_patch :]

        cumulative_sum = attention_mask.cumsum(dim=1)  
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))  
        vlm_features = last_hidden.gather(1, expanded_indices.unsqueeze(1)).squeeze(1) #[B, D]

        assert (vlm_features.shape[0], vlm_features.shape[1]) == (B, 4096), "Batch size must be B for action prediction"
        using_cfg = cfg_scale > 1.0


        model_dtype = next(self.action_model.net.parameters()).dtype

        B = vlm_features.shape[0]
        
        vlm_features = vlm_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=vlm_features.device).to(model_dtype)  #[B, T, D]
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, 1, -1) #[B, 1, D]
            z = torch.cat([vlm_features, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=vlm_features)
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,#False, try to set True 
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=vlm_features.device,
                                                                eta=0.0)
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,#False, try to set True 
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=vlm_features.device)
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples.cpu().numpy()

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, :, 6] = np.where(normalized_actions[:, :, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions, normalized_actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

import copy
from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from typing import Any, Callable, Dict, Sequence, Tuple, Union

def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper

def replace_vision_backbone(vision_backbone, condition_dim=1024):
    self = vision_backbone
    
    replace_vision_transformer(self.dino_featurizer, condition_dim=condition_dim)
    replace_vision_transformer(self.siglip_featurizer, condition_dim=condition_dim)

    def forward(pixel_values: Dict[str, torch.Tensor], condition=None) -> torch.Tensor:
        """Runs the transformed image/pixel tensors through each vision backbone, returning concatenated patches."""
        if condition is not None:
            dino_patches = self.dino_featurizer(pixel_values["dino"], condition=condition)
            siglip_patches = self.siglip_featurizer(pixel_values["siglip"], condition=condition)
        else:
            dino_patches = self.dino_featurizer(pixel_values["dino"])
            siglip_patches = self.siglip_featurizer(pixel_values["siglip"])

        return torch.cat([dino_patches, siglip_patches], dim=2)
    self.forward = forward


def replace_block(block: Block, condition_dim=1024):
    self = block
    # condition_mlp 输出 gamma 和 beta 参数，两个LayerNorm，每个都有 gamma beta
    self.condition_mlp = nn.Sequential(
        nn.Conv2d(condition_dim, 2 * self.norm1.normalized_shape[0], 1),  # 4倍输出
        nn.GELU(),
        nn.Conv2d(2 * self.norm1.normalized_shape[0], 2 * self.norm1.normalized_shape[0], 1)
    )
    ## 初始化
    with torch.no_grad():
        nn.init.zeros_(self.condition_mlp[-1].weight)
        nn.init.zeros_(self.condition_mlp[-1].bias)

    def forward(x, condition=None):
        # condition: [B, condition_dim]
        if condition is None:
            gamma1, beta1 = 1, 0
        else:
            gamma1, beta1 = torch.zeros_like(x), torch.zeros_like(x)
            cond_out = self.condition_mlp(condition)  # 输出 [B, 2 * dim]
            cgamma1, cbeta1 = cond_out.flatten(-2, -1).transpose(-1, -2).chunk(2, dim=-1)  # 分成2份
            num_img_patch = cgamma1.shape[1]
            
            gamma1 = torch.cat([gamma1[:, :-num_img_patch], cgamma1], dim=1)
            beta1 = torch.cat([beta1[:, :-num_img_patch], cbeta1], dim=1)

        x_norm = self.norm1(x)
        x = x + self.drop_path1(self.ls1(self.attn(x_norm)))
        x = (1 + gamma1) * x + beta1  # FiLM调制
        # 调制 norm2
        x_norm = self.norm2(x)
        x = x + self.drop_path2(self.ls2(self.mlp(x_norm)))

        return x
    
    self.forward = forward


def replace_vision_transformer(featurizer: nn.Module, condition_dim=1024):
    self = featurizer
    # 替换原来的block
    for i in range(len(self.blocks)):
        replace_block(self.blocks[i], condition_dim=condition_dim)
    
    def _intermediate_layers(
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            **kwargs
    ):
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, **kwargs)
            if i in take_indices:
                outputs.append(x)

        return outputs
    
    def get_intermediate_layers(
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            **kwargs
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n, **kwargs)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    self._intermediate_layers = _intermediate_layers
    self.get_intermediate_layers = get_intermediate_layers
    self.forward = unpack_tuple(
        partial(self.get_intermediate_layers, n={len(self.blocks) - 2})
    )
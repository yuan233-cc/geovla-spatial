import torch.nn as nn

from .position_encodings import RotaryPositionEncoding3D
from .layers import FFWRelativeSelfCrossAttentionModule, FFWRelativeCrossAttentionModule

class FusionModule(nn.Module):

    def __init__(self, embedding_dim: int=4096, num_attn_heads: int=8,
        num_self_attn_layers: int=12, num_cross_attn_layers: int=6,
        use_adaln: bool=False):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.num_attn_heads=num_attn_heads
        self.num_self_attn_layers=num_self_attn_layers
        self.num_cross_attn_layers=num_cross_attn_layers
        self.use_adaln=use_adaln
        self.transformer_layers = FFWRelativeSelfCrossAttentionModule(
            embedding_dim, num_attn_heads, num_self_attn_layers, num_cross_attn_layers, use_adaln=use_adaln
        )
        self.rope_encoding3d = RotaryPositionEncoding3D(embedding_dim)
    def forward(self, queries, values, query_pos=None, value_pos=None, condition=None):
        # the shape of input should be like [N, B, D], and pos should be [N, B, D, 3]
        query_pos = self.rope_encoding3d(query_pos)[..., :self.embedding_dim, :] if query_pos is not None else None
        query_pos = query_pos.permute(1, 0, 2, 3) if query_pos is not None else None
        value_pos = self.rope_encoding3d(value_pos)[..., :self.embedding_dim, :] if value_pos is not None else None
        value_pos = value_pos.permute(1, 0, 2, 3) if value_pos is not None else None
        output = self.transformer_layers(
            query=queries, context=values, 
            query_pos=query_pos, context_pos=value_pos, 
            diff_ts=condition
        )[-1]
        return output
    

class CrossFusionModule(nn.Module):

    def __init__(self, embedding_dim: int=4096, num_attn_heads: int=8, num_layers: int=2,
        use_adaln: bool=False):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.num_attn_heads=num_attn_heads
        self.num_layers=num_layers
        self.use_adaln=use_adaln
        self.transformer_layers = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers, use_adaln=True
        )
        self.rope_encoding3d = RotaryPositionEncoding3D(embedding_dim)
    def forward(self, queries, values, query_pos=None, value_pos=None, condition=None):
        # the shape of input should be like [N, B, D], and pos should be [N, B, 3], and condition should be [B, C]
        query_pos = self.rope_encoding3d(query_pos)[..., :self.embedding_dim, :] if query_pos is not None else None
        query_pos = query_pos.permute(1, 0, 2, 3) if query_pos is not None else None
        value_pos = self.rope_encoding3d(value_pos)[..., :self.embedding_dim, :] if value_pos is not None else None
        value_pos = value_pos.permute(1, 0, 2, 3) if value_pos is not None else None
        output = self.transformer_layers(
            query=queries, value=values, 
            query_pos=query_pos, value_pos=value_pos, 
            diff_ts=condition
        )[-1]
        return output
    
class VisionFusionModule(nn.Module):

    def __init__(self, input_embedding_dim:int=4096, embedding_dim: int=4096, num_attn_heads: int=8, num_layers: int=2, use_adaln: bool=False):
        super().__init__()
        self.vision_fusion = CrossFusionModule(embedding_dim=embedding_dim, num_attn_heads=num_attn_heads, num_layers=num_layers, use_adaln=use_adaln)
        self.projector_in = nn.Linear(input_embedding_dim, embedding_dim)
        self.projector_out = nn.Linear(embedding_dim, embedding_dim)
        nn.init.zeros_(self.projector_out.weight)
        nn.init.zeros_(self.projector_out.bias)
    def forward(self, main_embeddings, other_embeddings, main_pos=None, other_pos=None):
        # main_embeddings & other_embeddings: [B, N, D]
        fused_embeddings = self.projector_out(self.vision_fusion(
                queries=main_embeddings.permute(1, 0, 2), 
                values=self.projector_in(other_embeddings).permute(1, 0, 2),
                query_pos=main_pos,
                value_pos=other_pos,
            ).permute(1, 0, 2)) + main_embeddings
        return fused_embeddings
    
class PCDFusionModule(nn.Module):

    def __init__(self, embedding_dim=4096, num_attn_heads: int=8, 
                 lang_fusion_layer: int=2, vlm_fusion_layer: int=0, pcd_fusion_layer: int=2, 
                 img_fusion_layer: int=2, ee_fusion_layer: int=2):
        super().__init__()

        self.load_lang_fusion = lang_fusion_layer > 0
        if self.load_lang_fusion:
            self.lang_fusion = CrossFusionModule(embedding_dim=embedding_dim, num_attn_heads=num_attn_heads, num_layers=lang_fusion_layer)

        self.load_vlm_fusion = vlm_fusion_layer > 0
        if self.load_vlm_fusion:
            self.vlm_fusion = CrossFusionModule(embedding_dim=embedding_dim, num_attn_heads=num_attn_heads, num_layers=vlm_fusion_layer)
    
        self.load_pcd_fusion = pcd_fusion_layer > 0
        if self.load_pcd_fusion:
            self.pcd_fusion = CrossFusionModule(embedding_dim=embedding_dim, num_attn_heads=num_attn_heads, num_layers=pcd_fusion_layer)

        self.load_img_fusion = img_fusion_layer > 0
        if self.load_img_fusion:
            self.img_fusion = CrossFusionModule(embedding_dim=embedding_dim, num_attn_heads=num_attn_heads, num_layers=img_fusion_layer)

        self.load_ee_fusion = ee_fusion_layer > 0
        if self.load_ee_fusion:
            self.ee_fusion = CrossFusionModule(embedding_dim=embedding_dim, num_attn_heads=num_attn_heads, num_layers=ee_fusion_layer)

        self.pcd_self_fusion = CrossFusionModule(embedding_dim=embedding_dim, num_attn_heads=num_attn_heads, num_layers=4)

    def forward(self, pcd_embeddings, lang_embeddings=None, vlm_embeddings=None, img_embeddings=None,
                 pcd_pos=None, img_pos=None):

        V, N, B, C = pcd_embeddings.shape
        pcd_embeddings = pcd_embeddings.permute(1, 0, 2, 3).flatten(1, 2) # N, VB, C
        pcd_pos = pcd_pos.permute(1, 0, 2, 3).flatten(1, 2) if pcd_pos is not None else None # N, VB, 3

        assert V <= 2, "only support base, wrist or both now"

        if self.load_lang_fusion and lang_embeddings is not None:
            lang_embeddings = lang_embeddings[None].repeat(V, 1, 1, 1).flatten(0, 1).permute(1, 0, 2)
            pcd_embeddings = self.lang_fusion(queries=pcd_embeddings, values=lang_embeddings)
        
        if self.load_vlm_fusion and vlm_embeddings is not None:
            vlm_embeddings = vlm_embeddings[None].repeat(V, 1, 1, 1).flatten(0, 1).permute(1, 0, 2)
            pcd_embeddings = self.vlm_fusion(queries=pcd_embeddings, values=vlm_embeddings)

        if self.load_pcd_fusion:
            assert V == 2, "pcd fusion must contain 2 more view"

            pcd_embeddings = pcd_embeddings.reshape(N, V, B, C)
            pcd_pos = pcd_pos.reshape(N, V, B, 3)
            
            pcd_embeddings = self.pcd_fusion(queries=pcd_embeddings[:, 0], values=pcd_embeddings[:, 1], query_pos=pcd_pos[:, 0], value_pos=pcd_pos[:, 1])
            pcd_pos = pcd_pos[:, 0] if pcd_pos is not None else None # N, B, C
        
        # pcd_embeddings and pcd_pos == N, B, C

        if self.load_img_fusion and img_embeddings is not None:
            pcd_embeddings = self.img_fusion(queries=pcd_embeddings, values=img_embeddings.permute(1, 0, 2), query_pos=pcd_pos, 
                                             value_pos=img_pos if img_pos is not None else None)

        pcd_embeddings = self.pcd_self_fusion(queries=pcd_embeddings, values=pcd_embeddings, query_pos=pcd_pos, value_pos=pcd_pos) # [N, B, C]
        
        if self.load_ee_fusion:
            ee_patch = (pcd_pos ** 2).sum(dim=-1).argmin(dim=0)
            ee_pos = pcd_pos.gather(dim=0, index=ee_patch[None, ..., None].expand(-1, -1, pcd_pos.size(2)))
            ee_embeddings = pcd_embeddings.gather(dim=0, index=ee_patch[None, ..., None].expand(-1, -1, pcd_embeddings.size(2)))
            ee_embeddings = self.ee_fusion(queries=ee_embeddings, values=pcd_embeddings, query_pos=ee_pos, value_pos=pcd_pos) # pcd_embeddings is [1, B, C]
            ee_embeddings = ee_embeddings.permute(1, 0, 2) # [N, 1, C]

        return pcd_embeddings, ee_embeddings # [N, B, C] and [B, 1, C]
        

        
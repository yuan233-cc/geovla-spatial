import torch
import torch.nn as nn

class TokenLearner(nn.Module):
    def __init__(self, dim, num_output_tokens=128):
        super().__init__()
        self.num_output_tokens = num_output_tokens

        self.attn_maps = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_output_tokens),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, N, D]  (patch tokens only)
        returns: [B, M, D]
        """
        attn = self.attn_maps(x)         # [B, N, M]
        attn = attn.permute(0, 2, 1)     # [B, M, N]
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        out = torch.matmul(attn, x)      # [B, M, D]
        return out

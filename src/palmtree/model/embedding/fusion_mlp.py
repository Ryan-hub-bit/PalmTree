from typing import Optional, Literal
import torch
import torch.nn as nn

class MLPFuse(nn.Module):
    """
    Fuse [tokpos, x_src, x_des] via a 2-layer MLP, with optional residual to tokpos.
    Input:  tokpos[B, L, D], x_src[B, L, D], x_des[B, L, D]
    Output: fused[B, L, D]
    """
    def __init__(self, d_model: int, p_drop: float = 0.1, expansion: float = 2.0, use_residual: bool = True):
        super().__init__()
        in_dim = d_model * 3
        hidden = int(d_model * expansion)

        self.norm_in = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop2 = nn.Dropout(p_drop)

        self.use_residual = use_residual
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, tokpos: torch.Tensor, x_src: torch.Tensor, x_des: torch.Tensor) -> torch.Tensor:
        x = torch.cat([tokpos, x_src, x_des], dim=-1)          # [B, L, 3D]
        x = self.norm_in(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)                                      # [B, L, D]
        if self.use_residual:
            x = x + tokpos
        return self.norm_out(x)

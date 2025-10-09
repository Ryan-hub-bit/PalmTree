import torch.nn as nn
import torch


class GatedAdd(nn.Module):
    """
    Additive fusion with learned gates.
    x = tok + g_src*W_src(src) + g_des*W_des(des)
    """
    def __init__(self, d_model: int, p_drop: float = 0.1):
        super().__init__()
        self.w_src = nn.Linear(d_model, d_model)
        self.w_des = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, 2)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)
        self.alpha_src = nn.Parameter(torch.tensor(1.0))
        self.alpha_des = nn.Parameter(torch.tensor(1.0))

    def forward(self, tok, src, des):
        g = torch.sigmoid(self.gate(tok))
        g_src, g_des = g[..., 0:1], g[..., 1:2]
        out = tok + g_src * (self.alpha_src * self.w_src(src)) + g_des * (self.alpha_des * self.w_des(des))
        return self.drop(self.norm(out))

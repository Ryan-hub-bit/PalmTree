import math
import torch
import torch.nn as nn
from typing import Optional, Literal


class AddressEncoding(nn.Module):
    """
    Encodes source/destination addresses with sincos or rope mode.
    addr: [B] or [B, L]
    returns: [B, L, dim]
    """
    def __init__(self, dim: int, mode: Literal["sincos", "rope"] = "sincos", base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.half = dim // 2
        self.mode = mode

        freq = torch.arange(self.half, dtype=torch.float32)
        inv_freq = base ** (-freq / self.half)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, addr: torch.Tensor, L: Optional[int] = None) -> torch.Tensor:
        if addr.dim() == 1:
            if L is None:
                raise ValueError("When addr is [B], pass L to broadcast across tokens.")
            addr = addr.unsqueeze(1).expand(-1, L)
        elif addr.dim() != 2:
            raise ValueError("addr must be [B] or [B, L]")

        B, L = addr.shape
        angles = addr.unsqueeze(-1).float() * self.inv_freq  # [B, L, half]
        sin, cos = torch.sin(angles), torch.cos(angles)

        if self.mode == "sincos":
            out = torch.empty(B, L, self.dim, device=addr.device, dtype=angles.dtype)
            out[..., 0::2] = sin
            out[..., 1::2] = cos
            return out
        elif self.mode == "rope":
            return torch.cat([cos, sin], dim=-1)
        else:
            raise ValueError(f"Unsupported mode {self.mode}")

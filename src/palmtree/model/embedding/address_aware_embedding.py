import torch
import torch.nn as nn
from typing import Optional, Literal
from .bert import BERTEmbedding
from .address_encoding import AddressEncoding
from .fusion_concat import ConcatProject
from .fusion_gated import GatedAdd
from .fusion_mlp import GatedAdd


class AddressAwareEmbedding(nn.Module):
    """
    Wrapper around BERTEmbedding that adds source/destination address channels.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        addr_mode: Literal["sincos", "rope"] = "sincos",
        fuse_mode: Literal["concat", "gated", "mlp"] = "concat",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base = BERTEmbedding(vocab_size=vocab_size, embed_size=embed_size, dropout=dropout)
        self.addr_enc = AddressEncoding(dim=embed_size, mode=addr_mode)

        if fuse_mode == "concat":
            self.fuse = ConcatProject(in_dim=embed_size * 3, out_dim=embed_size, p_drop=dropout)
        elif fuse_mode == "gated":
            self.fuse = GatedAdd(d_model=embed_size, p_drop=dropout)
        elif fuse_mode == "mlp":
            self.fuse = MLPFuse(d_model=embed_size, p_drop=dropout, expansion=2.0, use_residual=True)
        else:
            raise ValueError(f"Unknown fuse_mode: {fuse_mode}")

        self.fuse_mode = fuse_mode

    def forward(
        self,
        sequence: torch.Tensor,
        segment_label: Optional[torch.Tensor] = None,
        addr_src: Optional[torch.Tensor] = None,
        addr_des: Optional[torch.Tensor] = None,
    ):
        tokpos = self.base(sequence, segment_label)  # [B, L, D]
        if addr_src is None or addr_des is None:
            return tokpos
        B, L, D = tokpos.shape
        x_src = self.addr_enc(addr_src, L=L)  # [B, L, D]
        x_des = self.addr_enc(addr_des, L=L)  # [B, L, D]
        return self.fuse(tokpos, x_src, x_des)
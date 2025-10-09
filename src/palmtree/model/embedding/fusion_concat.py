import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatProject(nn.Module):
    def __init__(self, in_dim, out_dim, p_drop=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p_drop)

    def forward(self, *xs):
        # 1️⃣ Concatenate all input tensors along the last dimension
        x = torch.cat(xs, dim=-1)

        # 2️⃣ Normalize each token’s features
        x = F.layer_norm(x, x.shape[-1:])

        # 3️⃣ Project back to model dimension
        x = self.proj(x)

        # 4️⃣ Dropout for regularization
        return self.drop(x)


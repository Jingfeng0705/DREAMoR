import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseTokenizer(nn.Module):
    def __init__(self, part_dims, pose_token_dim):
        super().__init__()
        self.part_proj = nn.ModuleList([
            nn.Linear(d, pose_token_dim) for d in part_dims
        ])

    def forward(self, x):
        """
        x: [B, D] full pose vector
        returns: [B, N, pose_token_dim] tokenized pose by parts
        """
        tokens = []
        start = 0
        for proj in self.part_proj:
            end = start + proj.in_features
            tokens.append(proj(x[:, start:end]))
            start = end
        return torch.stack(tokens, dim=1)  # [B, N, pose_token_dim]

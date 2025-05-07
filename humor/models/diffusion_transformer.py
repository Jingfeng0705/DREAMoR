import torch
import torch.nn as nn
import torch.nn.functional as F
from humor.models.diffusion.pose_tokenizer import PoseTokenizer

class DiffusionTransformer(nn.Module):
    def __init__(self, latent_dim, pose_token_dim, pose_dim_list, d_model=256, nhead=4, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        self.pose_tokenizer = PoseTokenizer(pose_dim_list, pose_token_dim)
        
        # Projections
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.pose_proj = nn.Linear(pose_token_dim, d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        num_pose_tokens = len(pose_dim_list)  # Number of pose tokens (parts

        # Learnable null token for CFG
        self.null_token = nn.Parameter(torch.zeros(num_pose_tokens, d_model))

        # Positional encoding (optional)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_pose_tokens, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, latent_dim)

    def forward(self, z_noisy, x_prev, t, cond_drop_mask=None):
        """
        z_noisy: [B, latent_dim]
        x_prev: [B, pose_dim]
        t: [B] or scalar int
        cond_drop_mask: [B] boolean mask for CFG (True if condition dropped)
        """
        B = z_noisy.shape[0]
        device = z_noisy.device
        
        # tokenization
        x_prev_tokens = self.pose_tokenizer(x_prev)

        # Time embedding
        if isinstance(t, int):
            t = torch.full((B,), t, dtype=torch.float32, device=device)
        half_dim = self.latent_proj.out_features // 2
        freq = torch.exp(-torch.arange(half_dim, device=device).float() * (torch.log(torch.tensor(10000.0)) / half_dim))
        t_emb = torch.cat([torch.sin(t[:, None] * freq), torch.cos(t[:, None] * freq)], dim=-1)
        t_emb = self.time_embed(t_emb)  # [B, d_model]

        # Project latent token
        latent_token = self.latent_proj(z_noisy) + t_emb  # [B, d_model]

        # Project pose tokens
        if cond_drop_mask is not None:
            # If condition is dropped, replace with null token
            x_proj = self.pose_proj(x_prev_tokens)  # [B, N, d_model]
            null_tokens = self.null_token.unsqueeze(0).expand(B, -1, -1)  # [B, N, d_model]
            x_proj = torch.where(cond_drop_mask[:, None, None], null_tokens, x_proj)  # [B, N, d_model]
            x_proj = x_proj + t_emb[:, None, :]
        else:
            x_proj = self.pose_proj(x_prev_tokens) + t_emb[:, None, :]

        # Concatenate sequence
        token_seq = torch.cat([latent_token[:, None, :], x_proj], dim=1)  # [B, 1 + N, d_model]
        token_seq = token_seq + self.pos_embed[:, :token_seq.size(1), :]

        # Transformer encoding
        encoded = self.transformer(token_seq)  # [B, 1 + N, d_model]
        latent_out = encoded[:, 0, :]  # [B, d_model] only output the latent token

        return self.output_proj(latent_out)  # [B, latent_dim]


    def forward_with_cfg(self, x, t, cond, cfg_scale):
        """
        Perform classifier-free guidance inference.

        Args:
            model: TransformerLatentDenoiser
            z_noisy: [B, latent_dim]
            x_prev: [B, pose_dim]
            t: [B] or scalar int
            guidance_scale: float multiplier for guidance strength

        Returns:
            eps_cfg: [B, latent_dim] guided noise prediction
        """
        B = x.size(0)
        device = x.device

        # Conditioned prediction
        eps_cond = self.forward(x, cond, t, cond_drop_mask=torch.zeros(B, dtype=torch.bool, device=device))

        # Unconditioned prediction
        eps_uncond = self.forward(x, cond, t, cond_drop_mask=torch.ones(B, dtype=torch.bool, device=device))

        # Classifier-Free Guidance combination
        eps_cfg = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        return eps_cfg
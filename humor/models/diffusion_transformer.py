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


    def forward_with_cfg(self, z_noisy, t, x_prev, cfg_scale):
        """
        Perform classifier-free guidance inference.

        Args:
            z_noisy: [B, latent_dim]
            x_prev: [B, pose_dim]
            t: [B] or scalar int
            guidance_scale: float multiplier for guidance strength

        Returns:
            eps_cfg: [B, latent_dim] guided noise prediction
        """
        B = z_noisy.size(0)
        device = z_noisy.device

        # Conditioned prediction
        eps_cond = self.forward(z_noisy, x_prev, t, cond_drop_mask=torch.zeros(B, dtype=torch.bool, device=device))

        # Unconditioned prediction
        eps_uncond = self.forward(z_noisy, x_prev, t, cond_drop_mask=torch.ones(B, dtype=torch.bool, device=device))

        # Classifier-Free Guidance combination
        eps_cfg = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        return eps_cfg
    
    # def ddim_inference(self, z_clean, x_prev, t_index, T, cfg_scale, num_steps=1):
    #     """
    #     Perform multi-step DDIM-style denoising starting from timestep t_index.

    #     Args:
    #         z_clean: [B, latent_dim] clean latent to be noised and denoised
    #         x_prev: Conditioning input
    #         t_index: Starting timestep index (0 <= t_index < T)
    #         T: Total number of diffusion timesteps
    #         cfg_scale: Guidance scale
    #         num_steps: Number of denoising steps to perform (from t_index towards 0)

    #     Returns:
    #         z_t: Final noised latent at step t_index - num_steps
    #         z_0_hat: Final estimated clean latent
    #     """
    #     B, latent_dim = z_clean.shape
    #     device = z_clean.device

    #     # Diffusion schedule
    #     betas = torch.linspace(1e-4, 0.02, T, device=device)
    #     alphas = 1. - betas
    #     alpha_bars = torch.cumprod(alphas, dim=0)

    #     # Noise clean latent to current t_index
    #     alpha_bar_t = alpha_bars[t_index].view(1, 1).expand(B, 1)
    #     sqrt_alpha_bar = alpha_bar_t.sqrt()
    #     sqrt_one_minus_alpha_bar = (1 - alpha_bar_t).sqrt()
    #     eps = torch.randn_like(z_clean)
    #     z_t = sqrt_alpha_bar * z_clean + sqrt_one_minus_alpha_bar * eps

    #     # Iteratively denoise
    #     for i in range(num_steps):
    #         t = t_index - i
    #         if t <= 0:
    #             break

    #         t_tensor = torch.full((B,), t, dtype=torch.float32, device=device)
    #         alpha_bar_t = alpha_bars[t].view(B, 1)
    #         sqrt_alpha_bar_t = alpha_bar_t.sqrt()
    #         sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()

    #         eps_pred = self.forward_with_cfg(z_t, t_tensor, x_prev, cfg_scale=cfg_scale)
    #         z_0_hat = (z_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

    #         # Predict z_t-1 using DDIM update (no noise since deterministic)
    #         if t > 1:
    #             alpha_bar_prev = alpha_bars[t - 1].view(B, 1)
    #         else:
    #             alpha_bar_prev = torch.ones_like(alpha_bar_t)

    #         z_t = (
    #             z_0_hat * alpha_bar_prev.sqrt() +
    #             (1 - alpha_bar_prev).sqrt() * eps_pred
    #         )

    #     return z_t, z_0_hat

    def ddim_inference(self, z_clean, x_prev, t_index, T, cfg_scale, num_steps=1):
        """
        Perform DDIM-style denoising from timestep t_index to t=0 in num_steps.

        Args:
            z_clean: [B, latent_dim] clean latent
            x_prev: Conditioning input
            t_index: Starting timestep (e.g., 999)
            T: Total diffusion steps (e.g., 1000)
            cfg_scale: Guidance scale
            num_steps: Number of denoising steps

        Returns:
            z_t: Final denoised latent at t=0
            z_0_hat: Final predicted clean latent
        """
        B, latent_dim = z_clean.shape
        device = z_clean.device

        # Define DDIM diffusion schedule
        betas = torch.linspace(1e-4, 0.02, T, device=device)
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)  # [T]

        # Step schedule: e.g., [t_index, ..., 0] with num_steps steps
        t_steps = torch.linspace(t_index, 0, steps=num_steps + 1, dtype=torch.long, device=device)  # includes t=0
        t_steps = t_steps.round().long()  # Make sure t_steps are ints

        # Initial noisy latent from z_clean
        t_start = t_steps[0]
        alpha_bar_start = alpha_bars[t_start].view(1, 1)
        eps = torch.randn_like(z_clean)
        z_t = alpha_bar_start.sqrt() * z_clean + (1 - alpha_bar_start).sqrt() * eps

        # DDIM denoising loop
        for i in range(num_steps):
            t = t_steps[i]
            t_prev = t_steps[i + 1]

            t_tensor = torch.full((B,), t, dtype=torch.float32, device=device)
            alpha_bar_t = alpha_bars[t].view(1, 1)
            alpha_bar_prev = alpha_bars[t_prev].view(1, 1)

            sqrt_alpha_bar_t = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()

            # Predict noise
            eps_pred = self.forward_with_cfg(z_t, t_tensor, x_prev, cfg_scale=cfg_scale)

            # Estimate z_0
            z_0_hat = (z_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

            # Deterministic DDIM update
            z_t = alpha_bar_prev.sqrt() * z_0_hat + (1 - alpha_bar_prev).sqrt() * eps_pred

        return z_t, z_0_hat
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def sinusoidal_embedding(timesteps, dim):
    """
    Generate sinusoidal embeddings for timesteps (like positional encoding).
    `timesteps` is a tensor of shape [batch] with integer values.
    Returns a tensor of shape [batch, dim] with sinusoidal embeddings.
    """
    half_dim = dim // 2
    # Prepare frequency scales
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim)
    # Outer product: batch timesteps with freq scales
    args = timesteps.unsqueeze(1).float() * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        # if odd dimension, pad one zero to make even
        embedding = F.pad(embedding, (0, 1))
    return embedding



# Residual block used in the U-Net
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, dropout=0.1):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        # Layers for the first convolution
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        # Layers for the second convolution
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        # Linear layer to project the time+cond embedding to scale channels
        self.emb_proj = nn.Linear(embed_dim, out_channels)
        # If channels change, use a 1x1 conv for the residual connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x, emb):
        # x: (B, in_channels, L), emb: (B, embed_dim) broadcasted via linear to (B, out_channels)
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time and condition embedding as a bias, in the middle of two conv layers, as DDPM
        h = h + self.emb_proj(F.silu(emb))[:, :, None]  # shape emb_proj(...) -> (B, out_channels), unsqueeze to (B,out_ch,1) for broadcasting
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)  # residual connection



# Downsampling block: two ResBlocks then downsample (stride-2 conv)
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super(DownBlock, self).__init__()
        self.res1 = ResBlock(in_channels, in_channels, embed_dim)
        self.res2 = ResBlock(in_channels, in_channels, embed_dim)
        # Downsample with stride 2 conv instead of pooling, which is used in DDPM
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)


    def forward(self, x, emb):
        x = self.res1(x, emb)
        x = self.res2(x, emb)
        skip = x  # save for skip connection
        # x: B, 64, 1
        x = self.downsample(x)
        return x, skip



# Upsampling block: upsample (transpose conv) then two ResBlocks with skip connection
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super(UpBlock, self).__init__()
        # Transposed conv to upsample (stride 2)
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.res1 = ResBlock(out_channels * 2, out_channels, embed_dim)  # after concat skip, channels = out*2
        self.res2 = ResBlock(out_channels, out_channels, embed_dim)


    def forward(self, x, skip, emb):
        x = self.upconv(x)
        # If needed, adjust for mismatch of one sample in length due to odd sizes (output_padding) – omitted for simplicity
        # Concatenate skip connection from down path
        x = torch.cat([x, skip], dim=1)  # concat on channel dimension
        x = self.res1(x, emb)
        x = self.res2(x, emb)
        return x



# U-Net model definition
class UNetConditional(nn.Module):
    def __init__(self, in_channels, cond_dim, base_channels, channel_mults, embed_dim):
        super(UNetConditional, self).__init__()
        self.in_channels = in_channels

        # Time embedding dimensionality for sinusoidal features
        self.time_dim = base_channels
        # Embedding MLPs for time and condition
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Initial projection from input channels to base_channels
        self.init_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        # Determine channel sizes at each U-Net level
        channels = [base_channels * m for m in channel_mults]

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(DownBlock(channels[i], channels[i+1], embed_dim))
            
        # Bottom (latent) layers - one or two ResBlocks at the lowest resolution
        self.mid_block1 = ResBlock(channels[-1], channels[-1], embed_dim)
        self.mid_block2 = ResBlock(channels[-1], channels[-1], embed_dim)

        # Upsampling blocks (in reverse order)
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            # Note: skip_channels = channels[i-1]
            self.up_blocks.append(UpBlock(channels[i], channels[i-1], embed_dim))

        # Final output layer: project to original latent dimension (predict noise)
        self.final_conv = nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)


    def forward(self, x, cond, t):
        """
        x: [B, in_channels,] noisy latent at time t
        cond: [B, cond_dim] condition vector
        t: [B] time step indices
        """
        # Embed time step t and condition
        # Time-step embedding (sinusoidal, then MLP)
        t_emb = sinusoidal_embedding(t, self.time_dim)               # [B, time_dim]
        t_emb = self.time_embed(t_emb)                               # [B, embed_dim]
        # Condition embedding (MLP)
        c_emb = self.cond_embed(cond)                                # [B, embed_dim]
        # Combined embedding
        emb = t_emb + c_emb                                          # [B, embed_dim]

        # U-Net forward
        # x: [B, D]
        x = x.unsqueeze(1)
        h = self.init_conv(x) # h: [B, base_channels, D]
        # Downsample 
        skips = []
        for block in self.down_blocks:
            h, skip = block(h, emb)
            skips.append(skip)
        # Bottom layers
        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)
        # Upsample 
        for block in self.up_blocks:
            skip = skips.pop()       # pop out last skip
            h = block(h, skip, emb)
        # Final conv to predict noise
        
        h = self.final_conv(h) # h: [B, in_channels, D]
        return h.squeeze()


    def forward_with_cfg(self, x, t, cond, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, cond, t)
        # Apply classifier-free guidance
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
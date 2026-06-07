import torch
from torch import nn
from torch.nn import functional as F
import torch.fft
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2).reshape(input_shape)
        return self.out_proj(output)

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x = x.transpose(-1, -2)
        x = self.attention(x).transpose(-1, -2)
        return x + residue

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_layer = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.conv_1(F.silu(self.groupnorm_1(x)))
        x = self.conv_2(F.silu(self.groupnorm_2(x)))
        return x + self.residual_layer(residue)

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv1d(12, 128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512), 
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            
            nn.Conv1d(512, 8, kernel_size=3, padding=1),
            nn.Conv1d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor=None):
        x = x.transpose(1, 2)
        for module in self:
            if getattr(module, 'stride', None) == (2,):
                x = F.pad(x, (0, 1))
            x = module(x)
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        stdev = log_var.mul(0.5).exp()
        if noise is None:
            noise = torch.randn_like(stdev)
        return mean + stdev * noise, mean, log_var

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv1d(4, 4, kernel_size=1, padding=0),
            nn.Conv1d(4, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv1d(128, 12, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor):
        for module in self:
            x = module(x)
        return x.transpose(1, 2)

def spectral_loss(x, x_hat):
    x_f, x_hat_f = x.float(), x_hat.float()
    fft_x = torch.fft.rfft(x_f, dim=1)
    fft_x_hat = torch.fft.rfft(x_hat_f, dim=1)
    return (fft_x.abs() - fft_x_hat.abs()).pow(2).sum() / x.size(0)

def loss_function(recons, x, mu, log_var, kld_weight=1.0):
    batch_size = x.size(0)
    
    recons_loss = F.mse_loss(recons, x, reduction='sum') / batch_size
    
    spec_loss = spectral_loss(x, recons)
    
    kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=[1, 2]).mean()
    
    loss = recons_loss + 0.1 * spec_loss + kld_weight * kld_loss
    
    return {'loss': loss, 'mse': recons_loss.detach(), 'spectral': spec_loss.detach(), 'KLD': kld_loss.detach()}

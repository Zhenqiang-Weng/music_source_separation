import numpy as np
from functools import partial
from typing import Optional, Callable
import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from ..registry import register_diffusion_model
import math
from .waveunet import ResidualGatedBlock, WaveNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time steps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B] time steps in [0, 1]
        returns: [B, dim] embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvBlockRes(nn.Module):
    """Two 3x3 conv layers with BatchNorm+ReLU and residual/shortcut, with optional time embedding."""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: Optional[int] = None, momentum: float = 0.01):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True),
        )
        
        # Time embedding projection
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.time_mlp = None
            
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.use_shortcut = True
        else:
            self.use_shortcut = False

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.conv(x)
        
        # Add time embedding
        if self.time_mlp is not None and time_emb is not None:
            time_feat = self.time_mlp(time_emb)  # [B, out_channels]
            time_feat = time_feat[:, :, None, None]  # [B, out_channels, 1, 1]
            out = out + time_feat
        
        if self.use_shortcut:
            return out + self.shortcut(x)
        else:
            return out + x


class ResEncoderBlock(nn.Module):
    """Encoder block with time embedding support."""
    def __init__(self, in_channels: int, out_channels: int, pool_kernel: Optional[Tuple[int, int]], 
                 n_blocks: int = 1, time_emb_dim: Optional[int] = None, momentum: float = 0.01):
        super().__init__()
        layers = []
        layers.append(ConvBlockRes(in_channels, out_channels, time_emb_dim=time_emb_dim, momentum=momentum))
        for _ in range(n_blocks - 1):
            layers.append(ConvBlockRes(out_channels, out_channels, time_emb_dim=time_emb_dim, momentum=momentum))
        self.blocks = nn.ModuleList(layers)
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel) if pool_kernel is not None else None

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None):
        for block in self.blocks:
            x = block(x, time_emb)
        x_pre = x  # feature before pooling (skip)
        if self.pool is not None:
            x_down = self.pool(x_pre)
            return x_pre, x_down
        else:
            return x_pre


class ResDecoderBlock(nn.Module):
    """Decoder block with time embedding support."""
    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int], 
                 n_blocks: int = 1, time_emb_dim: Optional[int] = None, momentum: float = 0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                               stride=stride, padding=1, output_padding=out_padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True),
        )
        convs = []
        convs.append(ConvBlockRes(out_channels * 2, out_channels, time_emb_dim=time_emb_dim, momentum=momentum))
        for _ in range(n_blocks - 1):
            convs.append(ConvBlockRes(out_channels, out_channels, time_emb_dim=time_emb_dim, momentum=momentum))
        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = _fit_to(skip, x.shape[-2:])
        x = torch.cat([x, skip], dim=1)
        for conv in self.convs:
            x = conv(x, time_emb)
        return x


def _fit_to(t: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Center-crop or pad tensor t to target height/width (H,W)."""
    _, _, h, w = t.shape
    th, tw = target_hw
    # crop if larger
    if h >= th and w >= tw:
        sh = (h - th) // 2
        sw = (w - tw) // 2
        return t[..., sh:sh + th, sw:sw + tw]
    # pad if smaller
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    # pad: (left, right, top, bottom)
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return F.pad(t, pad)


class Encoder(nn.Module):
    """Stack of ResEncoderBlock with time embedding support."""
    def __init__(self, in_channels: int, n_mels: int, n_layers: int, pool_kernel: Tuple[int, int], 
                 n_blocks: int, base_channels: int = 16, time_emb_dim: Optional[int] = None, momentum: float = 0.01):
        super().__init__()
        self.bn_in = nn.BatchNorm2d(in_channels, momentum=momentum)
        layers = []
        out_ch = base_channels
        cur_ch = in_channels
        cur_mels = n_mels
        self.latent_info = []
        for i in range(n_layers):
            layers.append(ResEncoderBlock(cur_ch, out_ch, pool_kernel=pool_kernel, n_blocks=n_blocks, 
                                         time_emb_dim=time_emb_dim, momentum=momentum))
            self.latent_info.append((out_ch, cur_mels))
            cur_ch = out_ch
            out_ch *= 2
            cur_mels = cur_mels // 2
        self.layers = nn.ModuleList(layers)
        self.out_channel = cur_ch
        self.out_size = cur_mels

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None):
        x = self.bn_in(x)
        skips = []
        for l in self.layers:
            res = l(x, time_emb)
            if isinstance(res, tuple):
                pre, x = res
                skips.append(pre)
            else:
                pre = res
                x = pre
                skips.append(pre)
        return x, skips


class Intermediate(nn.Module):
    """Bottleneck stack with time embedding support."""
    def __init__(self, in_channels: int, out_channels: int, n_inters: int, n_blocks: int, 
                 time_emb_dim: Optional[int] = None, momentum: float = 0.01):
        super().__init__()
        layers = []
        layers.append(ResEncoderBlock(in_channels, out_channels, pool_kernel=None, n_blocks=n_blocks, 
                                     time_emb_dim=time_emb_dim, momentum=momentum))
        for _ in range(n_inters - 1):
            layers.append(ResEncoderBlock(out_channels, out_channels, pool_kernel=None, n_blocks=n_blocks, 
                                         time_emb_dim=time_emb_dim, momentum=momentum))
        self.blocks = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, time_emb)
        return x


class Decoder(nn.Module):
    """Decoder with time embedding support."""
    def __init__(self, in_channels: int, n_layers: int, stride: Tuple[int, int], n_blocks: int, 
                 time_emb_dim: Optional[int] = None, momentum: float = 0.01):
        super().__init__()
        layers = []
        cur_ch = in_channels
        for _ in range(n_layers):
            out_ch = cur_ch // 2
            layers.append(ResDecoderBlock(cur_ch, out_ch, stride=stride, n_blocks=n_blocks, 
                                         time_emb_dim=time_emb_dim, momentum=momentum))
            cur_ch = out_ch
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor], time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            skip = skips[-1 - i]
            x = layer(x, skip, time_emb)
        return x


class TimbreFilter(nn.Module):
    """Apply a ConvBlockRes to each skip map with time embedding."""
    def __init__(self, latent_channels: List[Tuple[int, int]], time_emb_dim: Optional[int] = None, momentum: float = 0.01):
        super().__init__()
        layers = []
        for ch_info in latent_channels:
            ch = ch_info[0]
            layers.append(ConvBlockRes(ch, ch, time_emb_dim=time_emb_dim, momentum=momentum))
        self.layers = nn.ModuleList(layers)

    def forward(self, skip_maps: List[torch.Tensor], time_emb: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        out = []
        for i, l in enumerate(self.layers):
            out.append(l(skip_maps[i], time_emb))
        return out

# input_size=[1025, 690], in_channels=2, out_channels=2
@register_diffusion_model("unet2cs")
class UNet(nn.Module):
    """
    UNet with time step embedding for diffusion models.
    Input: [B, 2, 1025, 690]
    Output: [B, 2, 1025, 690]
    """
    def __init__(self,
                 in_channels: int = 2,
                 n_fft: int = 1025,
                 n_layers: int = 5,
                 inter_layers: int = 4,
                 n_blocks: int = 1,
                 base_channels: int = 16,
                 pool_kernel: Tuple[int, int] = (1, 2),
                 stride: Tuple[int, int] = (1, 2),
                 use_timbre: bool = True,
                 time_emb_dim: int = 128):
        super().__init__()
        
        # Time embedding
        self.time_emb_dim = time_emb_dim
        self.time_pos_emb = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        self.encoder = Encoder(in_channels, n_fft, n_layers, pool_kernel, n_blocks, base_channels, 
                              time_emb_dim=time_emb_dim)
        bottom_in = self.encoder.out_channel
        bottom_out = bottom_in * 2
        self.intermediate = Intermediate(bottom_in, bottom_out, inter_layers, n_blocks, 
                                        time_emb_dim=time_emb_dim)
        self.use_timbre = use_timbre
        if self.use_timbre:
            self.tf = TimbreFilter(self.encoder.latent_info, time_emb_dim=time_emb_dim)
        else:
            self.tf = None
        self.decoder = Decoder(bottom_out, n_layers, stride, n_blocks, time_emb_dim=time_emb_dim)
        out_ch = base_channels
        self.out_proj = nn.Conv2d(out_ch, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 2, 1025, T] (complex spectrogram: 2 channels for real/imag)
        t: [B] time steps in [0, 1]
        returns: [B, 2, 1025, T] (same shape as input)
        """
        input_shape = x.shape  # [B, 2, F, T]
        
        # Create time embedding
        time_emb = self.time_pos_emb(t)  # [B, time_emb_dim]
        time_emb = self.time_mlp(time_emb)  # [B, time_emb_dim]
        
        x_enc, skips = self.encoder(x, time_emb)
        x_mid = self.intermediate(x_enc, time_emb)
        if self.tf is not None:
            skips = self.tf(skips, time_emb)
        x_dec = self.decoder(x_mid, skips, time_emb)
        out = self.out_proj(x_dec)  # [B, 2, F', T']
        
        # Ensure output matches input spatial dimensions
        if out.shape[-2:] != input_shape[-2:]:
            out = _fit_to(out, input_shape[-2:])
        
        return out





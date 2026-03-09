import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_norm_conv2d(in_ch, out_ch, kernel_size, stride=(1, 1), padding=(0, 0), groups=1):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups)
    return nn.utils.weight_norm(conv)


def spectral_norm_conv1d(in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1):
    conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups)
    return nn.utils.spectral_norm(conv)


class DiscriminatorP(nn.Module):
    """
    Period-based discriminator (MPD). Operates on waveform shaped [B, 1, T].
    For a given period p, reshape to [B, 1, T//p, p] and apply 2D convs.
    Returns final score and intermediate feature maps for FM.
    """

    def __init__(self, period: int, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.period = period
        c = base_channels
        layers = [
            weight_norm_conv2d(in_channels, c, (5, 1), (3, 1), (2, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm_conv2d(c, c, (3, 1), (2, 1), (1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm_conv2d(c, c * 2, (3, 1), (2, 1), (1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm_conv2d(c * 2, c * 4, (3, 1), (2, 1), (1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm_conv2d(c * 4, c * 8, (3, 1), (1, 1), (1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.layers = nn.ModuleList(layers)
        self.final = weight_norm_conv2d(c * 8, 1, (3, 1), (1, 1), (1, 0))

    def forward(self, x: torch.Tensor):
        # x: [B, 1, T]
        B, C, T = x.shape
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len), mode='reflect')
            T = T + pad_len
        x = x.view(B, C, T // self.period, self.period)  # [B, 1, T//p, p]

        fmaps = []
        h = x
        for layer in self.layers:
            h = layer(h)
            fmaps.append(h)
        out = self.final(h)
        fmaps.append(out)
        out = out.flatten(1)  # [B, -1]
        return out, fmaps


class DiscriminatorS(nn.Module):
    """
    Scale-based discriminator (MSD). Operates on waveform [B, 1, T].
    Optionally applies average pooling to create lower-resolution inputs.
    Uses 1D conv stack with spectral norm.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.layers = nn.ModuleList([
            spectral_norm_conv1d(in_channels, c, 15, 1, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_conv1d(c, c, 41, 2, padding=20, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_conv1d(c, c * 2, 41, 2, padding=20, groups=16),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_conv1d(c * 2, c * 4, 41, 4, padding=20, groups=16),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_conv1d(c * 4, c * 8, 41, 4, padding=20, groups=16),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_conv1d(c * 8, c * 16, 41, 1, padding=20, groups=16),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_conv1d(c * 16, c * 16, 5, 1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.final = spectral_norm_conv1d(c * 16, 1, 3, 1, padding=1)

    def forward(self, x: torch.Tensor):
        # x: [B, 1, T]
        fmaps = []
        h = x
        for layer in self.layers:
            h = layer(h)
            fmaps.append(h)
        out = self.final(h)
        fmaps.append(out)
        out = out.flatten(1)  # [B, -1]
        return out, fmaps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, x: torch.Tensor):
        ys, fmaps = [], []
        for d in self.discriminators:
            y, fm = d(x)
            ys.append(y)
            fmaps.append(fm)
        y_sum = sum(ys)
        # Flatten feature maps list-of-lists
        flat_fm = [f for sub in fmaps for f in sub]
        return y_sum, flat_fm


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, poolings=(1, 2, 4)):
        super().__init__()
        self.poolings = poolings
        self.discriminators = nn.ModuleList([DiscriminatorS() for _ in poolings])
        self.avgpools = nn.ModuleList([
            nn.Identity() if p == 1 else nn.AvgPool1d(kernel_size=p * 2, stride=p, padding=p // 2, count_include_pad=False)
            for p in poolings
        ])

    def forward(self, x: torch.Tensor):
        ys, fmaps = [], []
        for pool, disc in zip(self.avgpools, self.discriminators):
            x_in = x if isinstance(pool, nn.Identity) else pool(x)
            y, fm = disc(x_in)
            ys.append(y)
            fmaps.append(fm)
        y_sum = sum(ys)
        flat_fm = [f for sub in fmaps for f in sub]
        return y_sum, flat_fm


class HiFiGANDiscriminator(nn.Module):
    """
    Aggregates MPD and MSD; input is waveform [B, 1, T] or [B, T].
    Returns (y, None, h_list) to be compatible with mel-based discriminator wrapper
    where the second return value is a placeholder for start_frames.
    """

    def __init__(self, periods=(2, 3, 5, 7, 11), poolings=(1, 2, 4)):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods=periods)
        self.msd = MultiScaleDiscriminator(poolings=poolings)

    def forward(self, x: torch.Tensor, x_len=None, start_frames_wins=None):
        # Accept [B, T] or [B, 1, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(1) == 1, "HiFiGANDiscriminator expects [B, 1, T] or [B, T]"
        y_mpd, h_mpd = self.mpd(x)
        y_msd, h_msd = self.msd(x)
        y = y_mpd + y_msd
        h_all = h_mpd + h_msd
        return y, None, h_all

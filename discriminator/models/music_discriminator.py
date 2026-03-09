"""
Music-friendly GAN Discriminators: MSD + MPD (+ optional multi-res spectrogram branch)
多分辨率、多周期、多尺度判别器，专为音乐分离设计

- Pure PyTorch, no external deps (optional torchaudio for CQT/Mel if you enable it).
- Designed for wideband music (44.1/48 kHz). Larger receptive fields & stable feature maps.
- Returns per-subdiscriminator logits and intermediate feature maps for Feature Matching.

Usage (minimal):

    D = MusicDiscriminator(
            sample_rate=44100,
            mpd_periods=(2,3,5,7,11,13),
            msd_pool_scales=(1,2,4,8),
            enable_spec=True,
            mr_stft_cfg=[
                dict(n_fft=1024, hop=256, win=1024),
                dict(n_fft=2048, hop=512, win=2048),
                dict(n_fft=4096, hop=1024, win=4096),
            ],
        )
    out = D(x_real, x_fake)  # x_*: (B, 1, T)
    lossD = hinge_d_loss(out)
    lossG = hinge_g_loss(out) + 10.0 * feature_matching_loss(out)
"""
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================
# Utility blocks
# ========================================
class ConvBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, groups=1):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, k, s, p, groups=groups))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, k: Tuple[int, int], s: Tuple[int, int], p: Tuple[int, int], groups=1):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


# ========================================
# Multi-Scale Discriminator (MSD) - waveform (1D)
# ========================================
class SubMSD(nn.Module):
    """One scale of the Multi-Scale Discriminator operating on (B,1,T)."""

    def __init__(self, channels: List[int] = None):
        super().__init__()
        if channels is None:
            channels = [16, 64, 256, 512, 1024, 1024]
        c = channels
        layers = [ConvBlock1d(1, c[0], k=15, s=1, p=7)]
        layers += [ConvBlock1d(c[0], c[1], k=41, s=4, p=20, groups=4)]
        layers += [ConvBlock1d(c[1], c[2], k=41, s=4, p=20, groups=16)]
        layers += [ConvBlock1d(c[2], c[3], k=41, s=4, p=20, groups=16)]
        layers += [ConvBlock1d(c[3], c[4], k=41, s=4, p=20, groups=16)]
        layers += [ConvBlock1d(c[4], c[5], k=5, s=1, p=2)]
        self.body = nn.ModuleList(layers)
        self.proj = nn.utils.weight_norm(nn.Conv1d(c[5], 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmaps = []
        h = x
        for layer in self.body:
            h = layer(h)
            fmaps.append(h)
        logit = self.proj(h)
        fmaps.append(logit)
        return logit, fmaps


class MSD(nn.Module):
    def __init__(self, pool_scales: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.pool_scales = pool_scales
        self.scales = nn.ModuleList([SubMSD() for _ in pool_scales])
        self.avgpools = nn.ModuleList([
            nn.Identity() if s == 1 else nn.AvgPool1d(kernel_size=s, stride=s, padding=0)
            for s in pool_scales
        ])

    def forward(self, x_real: torch.Tensor, x_fake: torch.Tensor):
        outputs = []
        for pool, sub in zip(self.avgpools, self.scales):
            xr = pool(x_real)
            xf = pool(x_fake)
            r_logit, r_f = sub(xr)
            f_logit, f_f = sub(xf)
            outputs.append({
                'type': 'msd',
                'scale': pool.__class__.__name__,
                'real_logit': r_logit,
                'fake_logit': f_logit,
                'real_fmaps': r_f,
                'fake_fmaps': f_f,
            })
        return outputs


# ========================================
# Multi-Period Discriminator (MPD)
# ========================================
class SubMPD(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        ch = [32, 128, 512, 1024, 1024]
        self.body = nn.ModuleList([
            ConvBlock2d(1, ch[0], k=(5, 1), s=(3, 1), p=(2, 0)),
            ConvBlock2d(ch[0], ch[1], k=(5, 1), s=(3, 1), p=(2, 0)),
            ConvBlock2d(ch[1], ch[2], k=(5, 1), s=(3, 1), p=(2, 0)),
            ConvBlock2d(ch[2], ch[3], k=(5, 1), s=(3, 1), p=(2, 0)),
            ConvBlock2d(ch[3], ch[4], k=(5, 1), s=(1, 1), p=(2, 0)),
        ])
        self.proj = nn.utils.weight_norm(nn.Conv2d(ch[4], 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, C, T = x.shape
        p = self.period
        if T % p != 0:
            pad_len = p - (T % p)
            x = F.pad(x, (0, pad_len), mode='reflect')
            T = T + pad_len
        x = x.view(B, C, T // p, p)  # (B, 1, frames, period)
        fmaps = []
        h = x
        for layer in self.body:
            h = layer(h)
            fmaps.append(h)
        logit = self.proj(h)
        fmaps.append(logit)
        return logit, fmaps


class MPD(nn.Module):
    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.subs = nn.ModuleList([SubMPD(p) for p in periods])
        self.periods = periods

    def forward(self, x_real: torch.Tensor, x_fake: torch.Tensor):
        outputs = []
        for p, sub in zip(self.periods, self.subs):
            r_logit, r_f = sub(x_real)
            f_logit, f_f = sub(x_fake)
            outputs.append({
                'type': 'mpd',
                'period': p,
                'real_logit': r_logit,
                'fake_logit': f_logit,
                'real_fmaps': r_f,
                'fake_fmaps': f_f,
            })
        return outputs


# ========================================
# Optional: Multi-Resolution Spectrogram Discriminator (2D)
# ========================================
class SpecExtractor(nn.Module):
    """Multi-resolution magnitude spectrograms via torch.stft (no external deps)."""

    def __init__(self, cfg: List[Dict]):
        super().__init__()
        self.cfg = cfg
        # Precreate Hann windows
        self.windows = nn.ParameterList([])
        for c in cfg:
            win = torch.hann_window(c['win'], periodic=False)
            self.windows.append(nn.Parameter(win, requires_grad=False))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B,1,T)
        x = x.squeeze(1)  # (B, T)
        specs = []
        for (c, win) in zip(self.cfg, self.windows):
            S = torch.stft(
                x, n_fft=c['n_fft'], hop_length=c['hop'], win_length=c['win'],
                window=win.to(x.device), center=True, return_complex=True, pad_mode='reflect'
            )
            mag = (S.real ** 2 + S.imag ** 2).clamp_min(1e-12).sqrt()  # (B, F, frames)
            specs.append(mag.unsqueeze(1))  # (B,1,F,frames)
        return specs


class SubSpecD(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 8]
        self.body = nn.ModuleList([
            ConvBlock2d(in_ch, ch[0], k=(3, 3), s=(1, 1), p=(1, 1)),
            ConvBlock2d(ch[0], ch[1], k=(5, 5), s=(2, 2), p=(2, 2)),
            ConvBlock2d(ch[1], ch[2], k=(5, 5), s=(2, 2), p=(2, 2)),
            ConvBlock2d(ch[2], ch[3], k=(5, 5), s=(2, 2), p=(2, 2)),
            ConvBlock2d(ch[3], ch[4], k=(3, 3), s=(1, 1), p=(1, 1)),
        ])
        self.proj = nn.utils.weight_norm(nn.Conv2d(ch[4], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

    def forward(self, x2d: torch.Tensor):
        fmaps = []
        h = x2d
        for layer in self.body:
            h = layer(h)
            fmaps.append(h)
        logit = self.proj(h)
        fmaps.append(logit)
        return logit, fmaps


class MRSpecD(nn.Module):
    def __init__(self, mr_cfg: List[Dict]):
        super().__init__()
        self.extractor = SpecExtractor(mr_cfg)
        self.subs = nn.ModuleList([SubSpecD() for _ in mr_cfg])

    def forward(self, x_real: torch.Tensor, x_fake: torch.Tensor):
        outs = []
        specs_r = self.extractor(x_real)
        specs_f = self.extractor(x_fake)
        for i, (sr, sf, sub) in enumerate(zip(specs_r, specs_f, self.subs)):
            r_logit, r_f = sub(sr)
            f_logit, f_f = sub(sf)
            outs.append({
                'type': 'spec',
                'res': i,
                'real_logit': r_logit,
                'fake_logit': f_logit,
                'real_fmaps': r_f,
                'fake_fmaps': f_f,
            })
        return outs


# ========================================
# Wrapper: Music Discriminator (MSD + MPD [+ Spec])
# ========================================
class MusicDiscriminator(nn.Module):
    """
    Aggregates MPD + MSD + optional multi-resolution spectrogram discriminators.
    Input: x_real, x_fake both [B, 1, T] or [B, T] (auto-unsqueeze if needed).
    Output: list of dicts with per-sub-discriminator real/fake logits and feature maps.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        mpd_periods: Tuple[int, ...] = (2, 3, 5, 7, 11, 13),
        msd_pool_scales: Tuple[int, ...] = (1, 2, 4, 8),
        enable_spec: bool = True,
        mr_stft_cfg: Optional[List[Dict]] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.mpd = MPD(periods=mpd_periods)
        self.msd = MSD(pool_scales=msd_pool_scales)
        self.spec = None
        if enable_spec:
            if mr_stft_cfg is None:
                mr_stft_cfg = [
                    dict(n_fft=1024, hop=256, win=1024),
                    dict(n_fft=2048, hop=512, win=2048),
                    dict(n_fft=4096, hop=1024, win=4096),
                ]
            self.spec = MRSpecD(mr_cfg=mr_stft_cfg)

    def forward(self, x_real: torch.Tensor, x_fake: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            x_real: [B, 1, T] or [B, T]
            x_fake: [B, 1, T] or [B, T]
        Returns:
            List of dicts with keys: type, real_logit, fake_logit, real_fmaps, fake_fmaps
        """
        if x_real.dim() == 2:
            x_real = x_real.unsqueeze(1)
        if x_fake.dim() == 2:
            x_fake = x_fake.unsqueeze(1)

        outputs = []
        outputs += self.mpd(x_real, x_fake)
        outputs += self.msd(x_real, x_fake)
        if self.spec is not None:
            outputs += self.spec(x_real, x_fake)
        return outputs
    
    def single_score(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        scores = []

        # MPD
        for item in self.mpd(x, x):  # 用 x 与自身比较，只取 real_logit
            logit = item["real_logit"]
            # 全局平均到一个数
            score = logit.mean(dim=list(range(1, logit.dim())))
            scores.append(score)

        # MSD
        for item in self.msd(x, x):
            logit = item["real_logit"]
            score = logit.mean(dim=list(range(1, logit.dim())))
            scores.append(score)

        # SPEC
        if self.spec is not None:
            for item in self.spec(x, x):
                logit = item["real_logit"]
                score = logit.mean(dim=list(range(1, logit.dim())))
                scores.append(score)

        # 融合所有子判别器的得分，得到最终 scalar
        return torch.stack(scores, dim=1).mean(dim=1)



# ========================================
# Loss functions: hinge GAN + feature matching
# ========================================

def _flatten_logits(d: List[Dict[str, torch.Tensor]]):
    r, f = [], []
    for item in d:
        r.append(item['real_logit'])
        f.append(item['fake_logit'])
    return r, f


def hinge_d_loss(d_out: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Hinge discriminator loss: maximize (1 - D(real)) and (1 + D(fake))."""
    r_list, f_list = _flatten_logits(d_out)
    loss = 0.0
    for r, f in zip(r_list, f_list):
        loss += torch.relu(1.0 - r).mean() + torch.relu(1.0 + f).mean()
    return loss / len(r_list)


def hinge_g_loss(d_out: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Hinge generator loss: maximize D(fake)."""
    _, f_list = _flatten_logits(d_out)
    loss = 0.0
    for f in f_list:
        loss += (-f).mean()
    return loss / len(f_list)


def rank_loss(d_out: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Rank loss (preference pair loss): -log(sigmoid(real - fake)).
    Equivalent to softplus(fake - real).
    """
    loss = 0.0
    count = 0
    for item in d_out:
        real_logit = item['real_logit']
        fake_logit = item['fake_logit']
        # -log(sigmoid(r - f)) = log(1 + exp(f - r)) = softplus(f - r)
        loss += F.softplus(fake_logit - real_logit).mean()
        count += 1
    return loss / max(count, 1)


def feature_matching_loss(d_out: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """L1 feature matching loss between real and fake feature maps (except final logit)."""
    loss = 0.0
    count = 0
    for item in d_out:
        rf = item['real_fmaps']
        ff = item['fake_fmaps']
        # Exclude final logit from FM (optional); both work
        for r, f in zip(rf[:-1], ff[:-1]):
            loss += F.l1_loss(f, r.detach())
            count += 1
    return loss / max(count, 1)




# ========================================
# Quick self-test (shapes only)
# ========================================
if __name__ == '__main__':
    B, T = 2, 44100 * 2  # 2 sec at 44.1k
    x_real = torch.randn(B, 1, T)
    x_fake = torch.randn(B, 1, T)

    D = MusicDiscriminator(
        sample_rate=44100,
        mpd_periods=(2, 3, 5, 7, 11, 13),
        msd_pool_scales=(1, 2, 4, 8),
        enable_spec=True,
        mr_stft_cfg=[
            dict(n_fft=1024, hop=256, win=1024),
            dict(n_fft=2048, hop=512, win=2048),
            dict(n_fft=4096, hop=1024, win=4096),
        ],
    )

    outs = D(x_real, x_fake)
    print(f"#sub-discriminators: {len(outs)}")
    for i, o in enumerate(outs[:3]):
        print(i, o['type'], o['real_logit'].shape, len(o['real_fmaps']))

    d_loss = hinge_d_loss(outs)
    g_loss = hinge_g_loss(outs) + 10.0 * feature_matching_loss(outs)
    print('lossD:', float(d_loss), 'lossG:', float(g_loss))

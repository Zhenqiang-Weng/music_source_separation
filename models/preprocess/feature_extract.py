from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
import torch
import torch.nn.functional as F

@dataclass
class STFTConfig:
    # === main STFT (match MelBandRoformer.stft_kwargs + window) ===
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    normalized: bool = False
    center: bool = True
    window_fn: Callable = torch.hann_window  

    # === iSTFT output length control ===
    istft_length: Optional[int] = None  

    # === audio meta ===
    stereo: bool = True
    num_stems: int = 3

    # === multi-resolution STFT loss (match MelBandRoformer.*) ===
    use_multi_stft_loss: bool = True
    multi_weight: float = 1.0
    multi_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256)
    multi_hop_length: int = 147
    multi_normalized: bool = False
    multi_center: bool = True
    multi_window_fn: Callable = torch.hann_window  


class STFTProcessorBatch:
    """STFT/iSTFT processor - batch, stereo-aware, with multi-res STFT loss."""

    def __init__(self, config: STFTConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        # main STFT window cache
        self.window = self.config.window_fn(self.config.win_length, device=device)

        # Cache for multi-resolution windows
        self._multi_stft_windows: Dict[int, torch.Tensor] = {}

    @classmethod
    def from_melband_roformer(cls, model, device: str = "cuda", istft_length: Optional[int] = None):
        """
        Build processor config from a MelBandRoformer-like instance.
        Only uses STFT / multi-STFT / stereo / stems attributes.
        """
        # 1) main stft kwargs
        stft_kwargs = getattr(model, "stft_kwargs", {})
        n_fft = int(stft_kwargs.get("n_fft", 2048))
        hop_length = int(stft_kwargs.get("hop_length", 512))
        win_length = int(stft_kwargs.get("win_length", n_fft))
        normalized = bool(stft_kwargs.get("normalized", False))

        # 2) multi stft
        multi_weight = float(getattr(model, "multi_stft_resolution_loss_weight", 1.0))
        multi_window_sizes = tuple(getattr(model, "multi_stft_resolutions_window_sizes", (4096, 2048, 1024, 512, 256)))
        multi_kwargs = getattr(model, "multi_stft_kwargs", {})
        multi_hop = int(multi_kwargs.get("hop_length", 147))
        multi_norm = bool(multi_kwargs.get("normalized", False))
        multi_window_fn = getattr(model, "multi_stft_window_fn", torch.hann_window)

        # 3) stereo / stems
        stereo = bool(getattr(model, "stereo", True))
        num_stems = int(getattr(model, "num_stems", 1))

        cfg = STFTConfig(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            normalized=normalized,
            center=True,                
            window_fn=torch.hann_window, 

            istft_length=istft_length,  

            stereo=stereo,
            num_stems=num_stems,

            use_multi_stft_loss=True,
            multi_weight=multi_weight,
            multi_window_sizes=multi_window_sizes,
            multi_hop_length=multi_hop,
            multi_normalized=multi_norm,
            multi_center=True,
            multi_window_fn=multi_window_fn,
        )
        return cls(cfg, device=device)

    def to(self, device: str):
        self.device = device
        self.window = self.window.to(device)
        self._multi_stft_windows = {k: v.to(device) for k, v in self._multi_stft_windows.items()}
        return self

    def _get_multi_stft_window(self, window_size: int):
        if window_size not in self._multi_stft_windows:
            self._multi_stft_windows[window_size] = self.config.multi_window_fn(window_size, device=self.device)
        return self._multi_stft_windows[window_size]

    def stft_batch(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: (batch, 2, samples) if stereo else (batch, 1, samples)
        return: (batch, FREQ_BINS*channels, TIME_FRAMES, 2)
        """
        batch_size, channels, samples = audio.shape

        # (batch, ch, samples) -> (batch*ch, samples)
        audio_flat = audio.reshape(batch_size * channels, samples)

        stft_complex = torch.stft(
            audio_flat,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            center=self.config.center,
            normalized=self.config.normalized,
            return_complex=True
        )
        stft_real = torch.view_as_real(stft_complex)  # (batch*ch, freq, time, 2)

        freq_bins = stft_real.shape[1]
        time_frames = stft_real.shape[2]

        # (batch*ch, freq, time, 2) -> (batch, ch, freq, time, 2)
        stft_real = stft_real.view(batch_size, channels, freq_bins, time_frames, 2)

        # -> (batch, freq, ch, time, 2) -> (batch, freq*ch, time, 2)
        stft_real = stft_real.permute(0, 2, 1, 3, 4).reshape(batch_size, freq_bins * channels, time_frames, 2)

        return stft_real

    def istft_batch(self, masked_stft: torch.Tensor) -> torch.Tensor:
        """
        masked_stft: (batch, num_stems, FREQ_BINS*channels, TIME_FRAMES, 2)
        return:     (batch, num_stems, channels, samples)
        """
        batch_size, num_stems, freq_ch, time_frames, _ = masked_stft.shape

        # 推断 channels
        channels = 2 if self.config.stereo else 1
        freq_bins = freq_ch // channels

        # (b, n, f*ch, t, 2) -> (b, n, f, ch, t, 2)
        x = masked_stft.view(batch_size, num_stems, freq_bins, channels, time_frames, 2)

        # -> (b*n*ch, f, t, 2)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch_size * num_stems * channels, freq_bins, time_frames, 2)
        x = torch.view_as_complex(x.contiguous())

        audio = torch.istft(
            x,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            center=self.config.center,
            normalized=self.config.normalized,
            return_complex=False,
            length=self.config.istft_length
        )

        return audio.view(batch_size, num_stems, channels, -1)

    def compute_loss(self, recon_audio: torch.Tensor, target_audio: torch.Tensor,
                     use_multi_stft: Optional[bool] = None) -> torch.Tensor:
        """
        recon_audio / target_audio: (batch, num_stems, channels, samples)
        """
        if use_multi_stft is None:
            use_multi_stft = self.config.use_multi_stft_loss

        target_audio = target_audio[..., :recon_audio.shape[-1]]
        loss = F.l1_loss(recon_audio, target_audio)

        if not use_multi_stft:
            return loss

        multi_loss = 0.0

        recon_flat = recon_audio.reshape(-1, recon_audio.shape[-1])
        target_flat = target_audio.reshape(-1, target_audio.shape[-1])

        for window_size in self.config.multi_window_sizes:
            window = self._get_multi_stft_window(window_size)

            res_stft_kwargs = dict(
                n_fft=max(window_size, self.config.n_fft),
                hop_length=self.config.multi_hop_length,
                win_length=window_size,
                window=window,
                center=self.config.multi_center,
                normalized=self.config.multi_normalized,
                return_complex=True,
            )

            recon_Y = torch.stft(recon_flat, **res_stft_kwargs)
            target_Y = torch.stft(target_flat, **res_stft_kwargs)

            multi_loss = multi_loss + F.l1_loss(recon_Y, target_Y)

        return loss + multi_loss * self.config.multi_weight
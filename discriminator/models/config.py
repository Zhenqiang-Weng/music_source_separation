"""
DiscriminatorConfig dataclass - central config used by wrappers.

Keep wrapper-level configuration compact and place discriminator-specific model
details under `model_kwargs`.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


def _default_mr_stft_cfg() -> List[Dict[str, int]]:
    return [
        {"n_fft": 1024, "hop": 256, "win": 1024},
        {"n_fft": 2048, "hop": 512, "win": 2048},
        {"n_fft": 4096, "hop": 1024, "win": 4096},
    ]


@dataclass
class DiscriminatorConfig:
    """Discriminator configuration (compact version)."""

    input_type: str = "mel"

    loss_type: str = "rank"
    rank_margin: float = 0.0

    n_train_start: int = 0
    gan_weight: float = 0.5
    fml_weight: float = 1.0
    gan_loss_lambda: Optional[float] = None
    enable_hdn_loss: bool = True

    lr: float = 1e-4
    betas: Optional[List[float]] = None
    eps: float = 1e-9
    weight_decay: float = 0.0

    n_warmup: int = 2000
    init_scale: float = 0.25
    grad_clip_thresh: float = 1.0
    grad_acc_step: int = 1

    model_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.betas is None:
            self.betas = [0.9, 0.98]

        if self.model_kwargs is None:
            self.model_kwargs = {}

        if self.gan_loss_lambda is not None:
            try:
                val = float(self.gan_loss_lambda)
            except Exception:
                val = 0.1
            self.gan_weight = val
            self.fml_weight = val

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscriminatorConfig":
        """Create config from dict and absorb legacy top-level fields."""
        payload = dict(data)
        model_kwargs = dict(payload.pop("model_kwargs", {}) or {})

        legacy_keys = {
            "time_lengths",
            "freq_lengths",
            "kernel_size",
            "hidden_size",
            "n_mel_channels",
            "mpd_periods",
            "msd_pool_scales",
            "ms_poolings",
            "enable_music_spec",
            "mr_stft_cfg",
            "sample_rate",
        }

        for key in list(payload.keys()):
            if key in legacy_keys:
                model_kwargs[key] = payload.pop(key)

        payload["model_kwargs"] = model_kwargs
        return cls(**payload)

    def get_mel_hparams(self) -> SimpleNamespace:
        kwargs = self.model_kwargs or {}
        return SimpleNamespace(
            time_lengths=kwargs.get("time_lengths", [64, 64, 64]),
            freq_lengths=kwargs.get("freq_lengths", [64, 64, 64]),
            kernel_size=kwargs.get("kernel_size", 3),
            hidden_size=kwargs.get("hidden_size", 128),
            n_mel_channels=kwargs.get("n_mel_channels", 128),
        )

    def min_mel_frames(self) -> int:
        return max(self.get_mel_hparams().time_lengths)

    def get_wave_kwargs(self) -> Dict[str, Any]:
        kwargs = self.model_kwargs or {}
        periods = kwargs.get("mpd_periods", [2, 3, 5, 7, 11])
        poolings = kwargs.get("msd_pool_scales", kwargs.get("ms_poolings", [1, 2, 4]))
        return {
            "periods": tuple(periods),
            "poolings": tuple(poolings),
        }

    def get_music_kwargs(self) -> Dict[str, Any]:
        kwargs = self.model_kwargs or {}
        periods = kwargs.get("mpd_periods", [2, 3, 5, 7, 11, 13])
        pool_scales = kwargs.get("msd_pool_scales", kwargs.get("ms_poolings", [1, 2, 4, 8]))
        return {
            "sample_rate": kwargs.get("sample_rate", 44100),
            "mpd_periods": tuple(periods),
            "msd_pool_scales": tuple(pool_scales),
            "enable_spec": kwargs.get("enable_music_spec", False),
            "mr_stft_cfg": kwargs.get("mr_stft_cfg", _default_mr_stft_cfg()),
        }

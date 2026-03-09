"""
BS-Roformer Exportable Core Model (without STFT/iSTFT)
For ONNX/TensorRT export

Input: STFT real representation (batch, freq_bins*2, time_frames, 2)
Output: Mask (batch, 3, freq_bins*2, time_frames, 2)
"""

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from typing import Optional, Tuple
from beartype import beartype

from .bs_roformer import (
    RMSNorm, BandSplit, MaskEstimator, ScaleTransformer,
    DEFAULT_FREQS_PER_BANDS, exists
)
from .conditioner import BandEmbedder
from rotary_embedding_torch import RotaryEmbedding


class BDCSGBSRoformerExportable(Module):
    """
    Exportable BS-Roformer Core without STFT/iSTFT
    STFT preprocessing and iSTFT postprocessing done outside model
    """

    @beartype
    def __init__(
            self,
            dim,
            *,
            depth,
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            flash_attn=True,
            mask_estimator_depth=2,
            mlp_expansion_factor=4,
            skip_connection=False,
            sage_attention=False,
            spk_embd_dim=192,
    ):
        super().__init__()

        self.audio_channels = 2
        self.num_stems = 3
        self.skip_connection = skip_connection
        self.freqs_per_bands = freqs_per_bands
        self.dim = dim
        self.spk_embd_dim = spk_embd_dim

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
            sage_attention=sage_attention,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ScaleTransformer(depth=1, rotary_embed=time_rotary_embed, **transformer_kwargs),
                ScaleTransformer(depth=1, rotary_embed=freq_rotary_embed, **transformer_kwargs),
            ]))

        self.final_norm = RMSNorm(dim)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([
            MaskEstimator(dim=dim, dim_inputs=freqs_per_bands_with_complex, 
                         depth=mask_estimator_depth, mlp_expansion_factor=mlp_expansion_factor)
            for _ in range(3)
        ])

        self.spk_embd_to_dim = nn.Linear(spk_embd_dim, dim)
        self.band_cond_embedding = BandEmbedder(len(freqs_per_bands), dim, dim)

        self.register_buffer(
            "band_indices",
            torch.arange(len(self.freqs_per_bands), dtype=torch.long),
            persistent=False
        )

    def _complex_mul(self, stft_repr: Tensor, mask: Tensor) -> Tensor:
        """Manual complex multiplication: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i"""
        s_real, s_imag = stft_repr[..., 0], stft_repr[..., 1]
        m_real, m_imag = mask[..., 0], mask[..., 1]
        res_real = s_real * m_real - s_imag * m_imag
        res_imag = s_real * m_imag + s_imag * m_real
        return torch.stack([res_real, res_imag], dim=-1)

    def forward(
            self,
            stft_repr: Tensor,
            speaker_embedding: Tensor,
    ) -> Tensor:
        """
        Args:
            stft_repr: (batch, freq_bins*2, time_frames, 2) - STFT as real tensor
            speaker_embedding: (batch, spk_embd_dim)
        
        Returns:
            result: (batch, 3, freq_bins*2, time_frames, 2) - masked STFT for 3 stems
        """
        batch, freq_bins_stereo, t_stft, _ = stft_repr.shape
        device = stft_repr.device

        # (b, f*2, t, 2) -> (b, t, f*2*2)
        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, t_stft, -1)

        x = self.band_split(x)  # (b, t, num_bands, dim)

        b = x.shape[0]
        t = x.shape[1]
        f = x.shape[2]
        d = x.shape[3]

        # Band conditioning
        band_cond = self.band_cond_embedding(self.band_indices)
        band_cond = band_cond.unsqueeze(0).expand(b, -1, -1)
        band_embedding = band_cond.reshape(b * f, -1)

        # Speaker conditioning
        spk_emb = self.spk_embd_to_dim(speaker_embedding)
        # 预计算 freq transformer 的 speaker conditioning: (b, d) -> (b*t, d)
        spk_f_cond = spk_emb.unsqueeze(1).expand(b, t, -1).reshape(b * t, -1)

        store = [None] * len(self.layers)
        for i, (time_transformer, freq_transformer) in enumerate(self.layers):
            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            # Time transformer: (b, t, f, d) -> (b*f, t, d)
            # 合并 permute + reshape: (b, t, f, d) -> (b, f, t, d) -> (b*f, t, d)
            x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
            x = time_transformer(x, band_embedding)
            
            # Freq transformer: (b*f, t, d) -> (b*t, f, d)
            # 原始: view(b,f,t,d) -> permute(0,2,1,3) -> reshape(b*t,f,d)
            # 优化: (b*f, t, d) -> (b, f, t, d) -> transpose(1,2) -> (b*t, f, d)
            x = x.view(b, f, t, d).transpose(1, 2).reshape(b * t, f, d)
            x = freq_transformer(x, spk_f_cond)
            
            # 恢复: (b*t, f, d) -> (b, t, f, d)
            x = x.view(b, t, f, d)

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        # Mask estimation
        masks = [fn(x) for fn in self.mask_estimators]
        mask = torch.stack(masks, dim=1)  # (b, 3, t, f*c)

        # (b, 3, t, f_total) -> (b, 3, f*2, t, 2)
        # 合并 view + permute
        mask = mask.view(batch, 3, t_stft, freq_bins_stereo, 2).permute(0, 1, 3, 2, 4)

        # Apply mask: (b, 1, f*2, t, 2) * (b, 3, f*2, t, 2)
        stft_repr = stft_repr.unsqueeze(1)
        result = self._complex_mul(stft_repr, mask)

        return result


def convert_full_to_exportable(full_model, exportable_model, verbose=True):
    """Copy weights from full model to exportable model"""
    full_state = full_model.state_dict()
    exportable_state = exportable_model.state_dict()
    
    matched = 0
    mismatched = 0
    missing = 0
    
    for key in exportable_state.keys():
        if key in full_state:
            if full_state[key].shape == exportable_state[key].shape:
                exportable_state[key] = full_state[key]
                matched += 1
            else:
                if verbose:
                    print(f"[Shape Mismatch] {key}: {full_state[key].shape} vs {exportable_state[key].shape}")
                mismatched += 1
        else:
            if verbose:
                print(f"[Missing] {key}")
            missing += 1
    
    if verbose:
        print(f"\n=== Weight Copy Summary ===")
        print(f"  Matched: {matched}")
        print(f"  Mismatched: {mismatched}")
        print(f"  Missing: {missing}")
        print(f"  Total in exportable: {len(exportable_state)}")
    
    exportable_model.load_state_dict(exportable_state)
    return exportable_model




class SpeakerBSRoformerExportable(Module):
    """
    Exportable BS-Roformer Core without STFT/iSTFT
    Uses speaker embedding as condition for both time and freq transformers
    """

    @beartype
    def __init__(
            self,
            dim,
            *,
            depth,
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            flash_attn=True,
            mask_estimator_depth=2,
            mlp_expansion_factor=4,
            skip_connection=False,
            sage_attention=False,
            spk_embd_dim=192,
    ):
        super().__init__()

        self.audio_channels = 2
        self.num_stems = 3
        self.skip_connection = skip_connection
        self.freqs_per_bands = freqs_per_bands
        self.dim = dim
        self.spk_embd_dim = spk_embd_dim

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
            sage_attention=sage_attention,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ScaleTransformer(depth=1, rotary_embed=time_rotary_embed, **transformer_kwargs),
                ScaleTransformer(depth=1, rotary_embed=freq_rotary_embed, **transformer_kwargs),
            ]))

        self.final_norm = RMSNorm(dim)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([
            MaskEstimator(dim=dim, dim_inputs=freqs_per_bands_with_complex, 
                         depth=mask_estimator_depth, mlp_expansion_factor=mlp_expansion_factor)
            for _ in range(3)
        ])

        self.spk_embd_to_dim = nn.Linear(spk_embd_dim, dim)

    def _complex_mul(self, stft_repr: Tensor, mask: Tensor) -> Tensor:
        """Manual complex multiplication: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i"""
        s_real, s_imag = stft_repr[..., 0], stft_repr[..., 1]
        m_real, m_imag = mask[..., 0], mask[..., 1]
        res_real = s_real * m_real - s_imag * m_imag
        res_imag = s_real * m_imag + s_imag * m_real
        return torch.stack([res_real, res_imag], dim=-1)

    def forward(
            self,
            stft_repr: Tensor,
            speaker_embedding: Tensor,
    ) -> Tensor:
        """
        Args:
            stft_repr: (batch, freq_bins*2, time_frames, 2) - STFT as real tensor
            speaker_embedding: (batch, spk_embd_dim)
        
        Returns:
            result: (batch, 3, freq_bins*2, time_frames, 2) - masked STFT for 3 stems
        """
        batch, freq_bins_stereo, t_stft, _ = stft_repr.shape
        device = stft_repr.device

        # (b, f*2, t, 2) -> (b, t, f*2*2)
        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, t_stft, -1)

        x = self.band_split(x)  # (b, t, num_bands, dim)

        b = x.shape[0]
        t = x.shape[1]
        f = x.shape[2]
        d = x.shape[3]

        # Speaker conditioning: project to model dimension
        spk_emb = self.spk_embd_to_dim(speaker_embedding)  # (b, d)
        
        # Prepare speaker conditioning for time transformer: (b, d) -> (b*f, d)
        spk_t_cond = spk_emb.unsqueeze(1).expand(b, f, -1).reshape(b * f, -1)
        
        # Prepare speaker conditioning for freq transformer: (b, d) -> (b*t, d)
        spk_f_cond = spk_emb.unsqueeze(1).expand(b, t, -1).reshape(b * t, -1)
        
        store = [None] * len(self.layers)
        for i, (time_transformer, freq_transformer) in enumerate(self.layers):
            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            # Time transformer: (b, t, f, d) -> (b*f, t, d)
            x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
            x = time_transformer(x, spk_t_cond)
            
            # Freq transformer: (b*f, t, d) -> (b*t, f, d)
            x = x.view(b, f, t, d).transpose(1, 2).reshape(b * t, f, d)
            x = freq_transformer(x, spk_f_cond)
            
            # Restore: (b*t, f, d) -> (b, t, f, d)
            x = x.view(b, t, f, d)

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        # Mask estimation
        masks = [fn(x) for fn in self.mask_estimators]
        mask = torch.stack(masks, dim=1)  # (b, 3, t, f*c)

        # (b, 3, t, f_total) -> (b, 3, f*2, t, 2)
        mask = mask.view(batch, 3, t_stft, freq_bins_stereo, 2).permute(0, 1, 3, 2, 4)

        # Apply mask: (b, 1, f*2, t, 2) * (b, 3, f*2, t, 2)
        stft_repr = stft_repr.unsqueeze(1)
        result = self._complex_mul(stft_repr, mask)

        return result


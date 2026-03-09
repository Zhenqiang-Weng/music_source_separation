from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from models.bs_roformer.attend import Attend
from models.bs_roformer.conditioner import BandEmbedder
try:
    from models.bs_roformer.attend_sage import Attend as AttendSage
except:
    pass

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

# helper functions

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


# norm

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# Custom layer to replace Rearrange for QKV split
class RearrangeQKV(Module):
    """Replace einops Rearrange('b n (qkv h d) -> qkv b h d n', qkv=3, h=heads)"""
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
    
    def forward(self, x):
        # x: (b, n, qkv * h * d)
        b, n, _ = x.shape
        # reshape to (b, n, 3, h, d)
        x = x.view(b, n, 3, self.heads, -1)
        # permute to (3, b, h, d, n)
        x = x.permute(2, 0, 3, 4, 1)
        return x


class RearrangeOut(Module):
    """Replace einops Rearrange('b h d n -> b n (h d)')"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x: (b, h, d, n)
        b, h, d, n = x.shape
        # permute to (b, n, h, d) then reshape to (b, n, h*d)
        return x.permute(0, 3, 1, 2).reshape(b, n, h * d)


# attention

class FeedForward(Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            flash=True,
            sage_attention=False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        if sage_attention:
            self.attend = AttendSage(flash=flash, dropout=dropout)
        else:
            self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        b, n, _ = x.shape

        # rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        qkv = self.to_qkv(x)  # (b, n, 3 * h * d)
        qkv = qkv.view(b, n, 3, self.heads, self.dim_head)  # (b, n, 3, h, d)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, h, n, d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        # rearrange(gates, 'b n h -> b h n 1')
        gates = gates.permute(0, 2, 1).unsqueeze(-1)  # (b, h, n, 1)
        out = out * gates.sigmoid()

        # rearrange(out, 'b h n d -> b n (h d)')
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)


class LinearAttention(Module):
    """
    this flavor of linear attention proposed in https://arxiv.org/abs/2106.09681 by El-Nouby et al.
    """

    @beartype
    def __init__(
            self,
            *,
            dim,
            dim_head=32,
            heads=8,
            scale=8,
            flash=False,
            dropout=0.,
            sage_attention=False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)

        self.to_qkv_linear = nn.Linear(dim, dim_inner * 3, bias=False)

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        if sage_attention:
            self.attend = AttendSage(
                scale=scale,
                dropout=dropout,
                flash=flash
            )
        else:
            self.attend = Attend(
                scale=scale,
                dropout=dropout,
                flash=flash
            )

        self.to_out_linear = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        b, n, _ = x.shape

        # Rearrange('b n (qkv h d) -> qkv b h d n', qkv=3, h=heads)
        qkv = self.to_qkv_linear(x)  # (b, n, 3 * h * d)
        qkv = qkv.view(b, n, 3, self.heads, self.dim_head)  # (b, n, 3, h, d)
        qkv = qkv.permute(2, 0, 3, 4, 1)  # (3, b, h, d, n)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        # Rearrange('b h d n -> b n (h d)')
        out = out.permute(0, 3, 1, 2).reshape(b, n, -1)
        return self.to_out_linear(out)


class Transformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True,
            linear_attn=False,
            sage_attention=False,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            if linear_attn:
                attn = LinearAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    flash=flash_attn,
                    sage_attention=sage_attention
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                    sage_attention=sage_attention
                )

            self.layers.append(ModuleList([
                attn,
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


                
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ScaleTransformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True,
            linear_attn=False,
            sage_attention=False,
    ):
        super().__init__()
        self.layers = ModuleList([])
        self.adaLN_modulations = nn.ModuleList([]) 
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()
        
        for _ in range(depth):
            norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            if linear_attn:
                attn = LinearAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    flash=flash_attn,
                    sage_attention=sage_attention
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                    sage_attention=sage_attention
                )

            norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.layers.append(ModuleList([
                norm1,
                attn,
                norm2,
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))
            
            ada = nn.Sequential(
                nn.SELU(),
                nn.Linear(dim, 6 * dim, bias=True),
            )
            self.adaLN_modulations.append(ada)
        self.init_weights()
            
    def forward(self, x, band_embedding=None):
        # x: (b*f, t, d)
        # band_embedding: (b*f, d)
        if band_embedding is None:
            band_embedding = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)

        for (norm1, attn, norm2, ff), ada in zip(self.layers, self.adaLN_modulations):
            shift_scale = ada(band_embedding)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = shift_scale.chunk(6, dim=-1)
            x = x + attn(modulate(norm1(x), shift_msa, scale_msa)) * gate_msa.unsqueeze(1)
            x = x + ff(modulate(norm2(x), shift_mlp, scale_mlp)) * gate_mlp.unsqueeze(1)
        return self.norm(x)
    
    def init_weights(self):
        """Initialize only adaLN_modulations."""
        GATE_BIAS_INIT = 1.0

        for ada in self.adaLN_modulations:
            linear = ada[1]

            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)

            with torch.no_grad():
                D = linear.bias.shape[0] // 6
                linear.bias[2*D:3*D].fill_(GATE_BIAS_INIT)
                linear.bias[5*D:6*D].fill_(GATE_BIAS_INIT)


# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


# main class

DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)      
    


class BDCSGBSRoformer(Module):

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
            dim_freqs_in=1025,
            stft_n_fft=2048,
            stft_hop_length=512,
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=2,
            mlp_expansion_factor=4,
            skip_connection=False,
            sage_attention=False,
            spk_embd_dim=192,
    ):
        super().__init__()

        self.audio_channels = 2  # Always 2 for stereo
        self.num_stems = 3  # Fixed to 3 for TensorRT
        self.skip_connection = skip_connection
        self.freqs_per_bands = freqs_per_bands
        self.dim = dim
        self.layers = ModuleList([])

        if sage_attention:
            print("Use Sage Attention")

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

        for _ in range(depth):
            tran_modules = []
            tran_modules.append(
                ScaleTransformer(
                    depth=1,  # Fixed to 1 for TensorRT
                    rotary_embed=time_rotary_embed,
                    **transformer_kwargs
                )
            )
            tran_modules.append(
                ScaleTransformer(
                    depth=1,  # Fixed to 1 for TensorRT
                    rotary_embed=freq_rotary_embed,
                    **transformer_kwargs
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = RMSNorm(dim)

        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.stft_normalized = stft_normalized

        # Register STFT window as buffer for TensorRT
        window_fn = default(stft_window_fn, torch.hann_window)
        stft_window = window_fn(stft_win_length)
        self.register_buffer('stft_window', stft_window, persistent=True)

        # Calculate freq bins
        freqs = stft_n_fft // 2 + 1  # 1025 for n_fft=2048

        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs, (
            f'the number of freqs in the bands must equal {freqs} based on the STFT settings, '
            f'but got {sum(freqs_per_bands)}'
        )

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])
        for _ in range(3):  # Fixed to 3 for TensorRT
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )
            self.mask_estimators.append(mask_estimator)

        self.spk_embd_to_dim = nn.Linear(spk_embd_dim, dim)

        self.band_cond_embedding = BandEmbedder(len(freqs_per_bands), dim, dim)

        # band indices buffer
        self.register_buffer(
            "band_indices",
            torch.arange(len(self.freqs_per_bands), dtype=torch.long),
            persistent=False
        )

    def create_band_conditioning(self, batch_size: int, device: torch.device):
        """
        Create band conditioning embeddings.

        Returns:
            torch.Tensor: (batch_size * num_bands, dim)
        """
        band_cond = self.band_cond_embedding(self.band_indices.to(device))  # (num_bands, dim)
        band_cond = band_cond.unsqueeze(0).expand(batch_size, -1, -1)       # (b, num_bands, dim)
        band_cond = band_cond.reshape(batch_size * band_cond.shape[1], -1)  # (b*num_bands, dim)
        return band_cond

    def create_speaker_conditioning_for_fre_roformer(self, speaker_embedding: Optional[Tensor], num_time_steps: int):
        if speaker_embedding is None:
            return None
        speaker_embedding = self.spk_embd_to_dim(speaker_embedding)  # (b, d)
        b = speaker_embedding.shape[0]
        # speaker_embedding.unsqueeze(1).expand -> (b, t, d)
        speaker_embedding = speaker_embedding.unsqueeze(1).expand(b, num_time_steps, -1)
        # pack to (b * t, d)
        speaker_embedding = speaker_embedding.reshape(b * num_time_steps, -1)
        return speaker_embedding

    def _complex_mul(self, stft_repr: Tensor, mask: Tensor) -> Tensor:
        """
        Manual complex multiplication for TensorRT compatibility.
        Both tensors have shape (..., 2) where last dim is [real, imag]
        
        (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        """
        s_real = stft_repr[..., 0]
        s_imag = stft_repr[..., 1]
        m_real = mask[..., 0]
        m_imag = mask[..., 1]

        res_real = s_real * m_real - s_imag * m_imag
        res_imag = s_real * m_imag + s_imag * m_real
        
        return torch.stack([res_real, res_imag], dim=-1)

    def forward(
            self,
            raw_audio: Tensor,
            speaker_embedding: Optional[Tensor] = None,
    ):
        """
        raw_audio: (b, 2, t) - always stereo (2 channels)
        speaker_embedding: (b, spk_dim)
        """

        device = raw_audio.device
        dtype = raw_audio.dtype
        batch, channels, audio_length = raw_audio.shape

        # Flatten batch and channels: (b, 2, t) -> (b*2, t)
        raw_audio_flat = raw_audio.reshape(batch * 2, audio_length)

        # Get STFT window (registered as buffer)
        stft_window = self.stft_window.to(dtype=dtype)

        # STFT: (b*2, t) -> (b*2, f, t_stft) complex
        stft_repr = torch.stft(
            raw_audio_flat,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_win_length,
            window=stft_window,
            normalized=self.stft_normalized,
            return_complex=True
        )

        # Convert complex to real: (b*2, f, t_stft) -> (b*2, f, t_stft, 2)
        stft_repr = torch.view_as_real(stft_repr)
        
        freq_bins = stft_repr.shape[1]
        t_stft = stft_repr.shape[2]
        
        # Reshape: (b*2, f, t_stft, 2) -> (b, 2, f, t_stft, 2)
        stft_repr = stft_repr.view(batch, 2, freq_bins, t_stft, 2)

        # Rearrange: (b, 2, f, t, c) -> (b, f, 2, t, c) -> (b, f*2, t, c)
        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(batch, freq_bins * 2, t_stft, 2)

        # Flatten for band split: (b, f*2, t, 2) -> (b, t, f*2*2)
        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, t_stft, -1)

        x = self.band_split(x)  # (b, t, num_bands, dim)

        b = x.shape[0]
        num_bands = x.shape[2]
        band_embedding = self.create_band_conditioning(b, device)  # (b*num_bands, dim)

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):
            time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            b, t, f, d = x.shape

            # Time transformer: (b, t, f, d) -> (b, f, t, d) -> (b*f, t, d)
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b * f, t, d)
            x = time_transformer(x, band_embedding)
            x = x.view(b, f, t, d)
            
            # Freq transformer: (b, f, t, d) -> (b, t, f, d) -> (b*t, f, d)
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b * t, f, d)

            spk_f_cond = self.create_speaker_conditioning_for_fre_roformer(speaker_embedding, t)
            x = freq_transformer(x, spk_f_cond)

            x = x.view(b, t, f, d)

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        # Mask estimation: (b, t, num_bands, dim) -> (b, 3, t, f*c)
        masks = []
        for fn in self.mask_estimators:
            masks.append(fn(x))
        mask = torch.stack(masks, dim=1)

        # Reshape mask: (b, 3, t, f_total) -> (b, 3, t, f*2, 2) -> (b, 3, f*2, t, 2)
        mask = mask.view(batch, 3, t_stft, freq_bins * 2, 2)
        mask = mask.permute(0, 1, 3, 2, 4)

        # Expand stft_repr for stems: (b, f*2, t, 2) -> (b, 1, f*2, t, 2)
        stft_repr = stft_repr.unsqueeze(1)

        # Manual complex multiplication (TensorRT compatible)
        # stft_repr: (b, 1, f*2, t, 2), mask: (b, 3, f*2, t, 2)
        # Broadcast stft_repr to (b, 3, f*2, t, 2)
        stft_repr = self._complex_mul(stft_repr, mask)

        # Reshape for iSTFT: (b, 3, f*2, t, 2) -> (b, 3, f, 2, t, 2) -> (b*3*2, f, t, 2)
        stft_repr = stft_repr.view(batch, 3, freq_bins, 2, t_stft, 2)
        stft_repr = stft_repr.permute(0, 1, 3, 2, 4, 5).reshape(batch * 3 * 2, freq_bins, t_stft, 2)

        # Convert back to complex for iSTFT
        stft_repr = torch.view_as_complex(stft_repr.contiguous())

        # iSTFT: (b*3*2, f, t_stft) -> (b*3*2, audio_length)
        recon_audio = torch.istft(
            stft_repr,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_win_length,
            window=stft_window,
            normalized=self.stft_normalized,
            return_complex=False,
            length=audio_length
        )

        # Reshape: (b*3*2, t) -> (b, 3, 2, t)
        recon_audio = recon_audio.view(batch, 3, 2, -1)

        return recon_audio





class SpeakerBSRoformer(Module):

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
            dim_freqs_in=1025,
            stft_n_fft=2048,
            stft_hop_length=512,
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=2,
            mlp_expansion_factor=4,
            skip_connection=False,
            sage_attention=False,
            spk_embd_dim=192,
            **kwargs
    ):
        super().__init__()

        self.audio_channels = 2  # Always 2 for stereo
        self.num_stems = 3  # Fixed to 3 for TensorRT
        self.skip_connection = skip_connection
        self.freqs_per_bands = freqs_per_bands
        self.dim = dim
        self.layers = ModuleList([])

        if sage_attention:
            print("Use Sage Attention")

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

        for _ in range(depth):
            tran_modules = []
            tran_modules.append(
                ScaleTransformer(
                    depth=1,  # Fixed to 1 for TensorRT
                    rotary_embed=time_rotary_embed,
                    **transformer_kwargs
                )
            )
            tran_modules.append(
                ScaleTransformer(
                    depth=1,  # Fixed to 1 for TensorRT
                    rotary_embed=freq_rotary_embed,
                    **transformer_kwargs
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = RMSNorm(dim)

        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.stft_normalized = stft_normalized

        # Register STFT window as buffer for TensorRT
        window_fn = default(stft_window_fn, torch.hann_window)
        stft_window = window_fn(stft_win_length)
        self.register_buffer('stft_window', stft_window, persistent=True)

        # Calculate freq bins
        freqs = stft_n_fft // 2 + 1  # 1025 for n_fft=2048

        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs, (
            f'the number of freqs in the bands must equal {freqs} based on the STFT settings, '
            f'but got {sum(freqs_per_bands)}'
        )

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])
        for _ in range(3):  # Fixed to 3 for TensorRT
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )
            self.mask_estimators.append(mask_estimator)

        self.spk_embd_to_dim = nn.Linear(spk_embd_dim, dim)


    def create_speaker_conditioning_for_fre_roformer(self, speaker_embedding: Optional[Tensor], num_time_steps: int):
        if speaker_embedding is None:
            return None
        speaker_embedding = self.spk_embd_to_dim(speaker_embedding)  # (b, d)
        b = speaker_embedding.shape[0]
        # speaker_embedding.unsqueeze(1).expand -> (b, t, d)
        speaker_embedding = speaker_embedding.unsqueeze(1).expand(b, num_time_steps, -1)
        # pack to (b * t, d)
        speaker_embedding = speaker_embedding.reshape(b * num_time_steps, -1)
        return speaker_embedding

    def _complex_mul(self, stft_repr: Tensor, mask: Tensor) -> Tensor:
        """
        Manual complex multiplication for TensorRT compatibility.
        Both tensors have shape (..., 2) where last dim is [real, imag]
        
        (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        """
        s_real = stft_repr[..., 0]
        s_imag = stft_repr[..., 1]
        m_real = mask[..., 0]
        m_imag = mask[..., 1]

        res_real = s_real * m_real - s_imag * m_imag
        res_imag = s_real * m_imag + s_imag * m_real
        
        return torch.stack([res_real, res_imag], dim=-1)

    def forward(
            self,
            raw_audio: Tensor,
            speaker_embedding: Optional[Tensor] = None,
    ):
        """
        raw_audio: (b, 2, t) - always stereo (2 channels)
        speaker_embedding: (b, spk_dim)
        """

        device = raw_audio.device
        dtype = raw_audio.dtype
        batch, channels, audio_length = raw_audio.shape

        # Flatten batch and channels: (b, 2, t) -> (b*2, t)
        raw_audio_flat = raw_audio.reshape(batch * 2, audio_length)

        # Get STFT window (registered as buffer)
        stft_window = self.stft_window.to(dtype=dtype)

        # STFT: (b*2, t) -> (b*2, f, t_stft) complex
        stft_repr = torch.stft(
            raw_audio_flat,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_win_length,
            window=stft_window,
            normalized=self.stft_normalized,
            return_complex=True
        )

        # Convert complex to real: (b*2, f, t_stft) -> (b*2, f, t_stft, 2)
        stft_repr = torch.view_as_real(stft_repr)
        
        freq_bins = stft_repr.shape[1]
        t_stft = stft_repr.shape[2]
        
        # Reshape: (b*2, f, t_stft, 2) -> (b, 2, f, t_stft, 2)
        stft_repr = stft_repr.view(batch, 2, freq_bins, t_stft, 2)

        # Rearrange: (b, 2, f, t, c) -> (b, f, 2, t, c) -> (b, f*2, t, c)
        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(batch, freq_bins * 2, t_stft, 2)

        # Flatten for band split: (b, f*2, t, 2) -> (b, t, f*2*2)
        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, t_stft, -1)

        x = self.band_split(x)  # (b, t, num_bands, dim)

        b = x.shape[0]


        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):
            time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            b, t, f, d = x.shape
            spk_f_cond = self.create_speaker_conditioning_for_fre_roformer(speaker_embedding, t)

            # Time transformer: (b, t, f, d) -> (b, f, t, d) -> (b*f, t, d)
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b * f, t, d)
            x = time_transformer(x, spk_f_cond)
            x = x.view(b, f, t, d)
            
            # Freq transformer: (b, f, t, d) -> (b, t, f, d) -> (b*t, f, d)
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b * t, f, d)

            x = freq_transformer(x, spk_f_cond)

            x = x.view(b, t, f, d)

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        # Mask estimation: (b, t, num_bands, dim) -> (b, 3, t, f*c)
        masks = []
        for fn in self.mask_estimators:
            masks.append(fn(x))
        mask = torch.stack(masks, dim=1)

        # Reshape mask: (b, 3, t, f_total) -> (b, 3, t, f*2, 2) -> (b, 3, f*2, t, 2)
        mask = mask.view(batch, 3, t_stft, freq_bins * 2, 2)
        mask = mask.permute(0, 1, 3, 2, 4)

        # Expand stft_repr for stems: (b, f*2, t, 2) -> (b, 1, f*2, t, 2)
        stft_repr = stft_repr.unsqueeze(1)

        # Manual complex multiplication (TensorRT compatible)
        # stft_repr: (b, 1, f*2, t, 2), mask: (b, 3, f*2, t, 2)
        # Broadcast stft_repr to (b, 3, f*2, t, 2)
        stft_repr = self._complex_mul(stft_repr, mask)

        # Reshape for iSTFT: (b, 3, f*2, t, 2) -> (b, 3, f, 2, t, 2) -> (b*3*2, f, t, 2)
        stft_repr = stft_repr.view(batch, 3, freq_bins, 2, t_stft, 2)
        stft_repr = stft_repr.permute(0, 1, 3, 2, 4, 5).reshape(batch * 3 * 2, freq_bins, t_stft, 2)

        # Convert back to complex for iSTFT
        stft_repr = torch.view_as_complex(stft_repr.contiguous())

        # iSTFT: (b*3*2, f, t_stft) -> (b*3*2, audio_length)
        recon_audio = torch.istft(
            stft_repr,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_win_length,
            window=stft_window,
            normalized=self.stft_normalized,
            return_complex=False,
            length=audio_length
        )

        # Reshape: (b*3*2, t) -> (b, 3, 2, t)
        recon_audio = recon_audio.view(batch, 3, 2, -1)

        return recon_audio
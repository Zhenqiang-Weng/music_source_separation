from functools import partial

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from models.bs_roformer.attend import Attend
from models.bs_roformer.conditioner import BandEmbedder
try:
    from models.bs_roformer.attend_sage import Attend as AttendSage
except:
    pass
from torch.utils.checkpoint import checkpoint

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


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

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
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
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv=3, h=heads)
        )

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

        self.to_out = nn.Sequential(
            Rearrange('b h d n -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(
            self,
            x
    ):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)


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
            band_embedding = torch.zeros(x.shape[0], x.shape[2], device=x.device)

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
            net = []

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
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
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

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.skip_connection = skip_connection
        self.freqs_per_bands = freqs_per_bands
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
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    **transformer_kwargs
                )
            )
            tran_modules.append(
                ScaleTransformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    **transformer_kwargs
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(
            torch.randn(1, 4096),
            **self.stft_kwargs,
            window=torch.ones(stft_win_length),
            return_complex=True
        ).shape[1]

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
        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )
            self.mask_estimators.append(mask_estimator)

        self.spk_embd_to_dim = nn.Linear(spk_embd_dim, dim)

        self.band_cond_embedding = BandEmbedder(len(freqs_per_bands), dim, dim)

        # ============================
        # PERF: cache / buffers
        # ============================
        # 1) band indices 常驻 buffer，避免每次 forward arange
        self.register_buffer(
            "band_indices",
            torch.arange(len(self.freqs_per_bands), dtype=torch.long),
            persistent=False
        )

        # 2) stft window cache: key = (device_str, dtype)
        #    只缓存推理常用的 window，避免每次 forward 重新创建 hann_window
        self._stft_window_cache = {}

    # ----------------------------
    # PERF: cached stft window
    # ----------------------------
    def _get_stft_window(self, device: torch.device, dtype: torch.dtype):
        key = (str(device), dtype)
        w = self._stft_window_cache.get(key, None)
        if w is None or (w.device != device) or (w.dtype != dtype):
            w = self.stft_window_fn(device=device).to(dtype=dtype)
            self._stft_window_cache[key] = w
        return w

    def create_band_conditioning(self, batch_size, device):
        """
        Create band conditioning embeddings.

        Returns:
            torch.Tensor: (batch_size * num_bands, dim)
        """
        # PERF: 用 buffer band_indices，避免 arange
        # PERF: 用 expand+reshape，避免 repeat+pack 的额外开销
        band_cond = self.band_cond_embedding(self.band_indices.to(device))  # (num_bands, dim)
        band_cond = band_cond.unsqueeze(0).expand(batch_size, -1, -1)       # (b, num_bands, dim) no copy
        band_cond = band_cond.reshape(batch_size * band_cond.shape[1], -1)  # (b*num_bands, dim)
        return band_cond

    def create_speaker_conditioning_for_fre_roformer(self, speaker_embedding, num_time_steps):
        if speaker_embedding is None:
            return None
        speaker_embedding = self.spk_embd_to_dim(speaker_embedding)  # (b, d)
        speaker_embedding = speaker_embedding.unsqueeze(1).repeat(1, num_time_steps, 1)  # (b, t, d)
        speaker_embedding, _ = pack([speaker_embedding], '* d')  # (b * t, d)
        return speaker_embedding

    def create_speaker_conditioning_for_time_roformer(self, speaker_embedding, num_bands):
        if speaker_embedding is None:
            return None
        speaker_embedding = self.spk_embd_to_dim(speaker_embedding)  # (b, d)
        speaker_embedding = speaker_embedding.unsqueeze(1).repeat(1, num_bands, 1)  # (b, f, d)
        speaker_embedding, _ = pack([speaker_embedding], '* d')  # (b * f, d)
        return speaker_embedding

    def forward(
            self,
            raw_audio,
            speaker_embedding=None,
    ):
        """
        speaker_embedding: (b, dim)
        """

        device = raw_audio.device
        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), \
            'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). ' \
            'also need to be False if mono (channel dimension of 1)'

        # to stft
        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        # PERF: cached window (device+dtype)
        stft_window = self._get_stft_window(device=device, dtype=raw_audio.dtype)

        try:
            stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        except:
            stft_repr = torch.stft(
                raw_audio.cpu() if x_is_mps else raw_audio,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True
            ).to(device)

        stft_repr = torch.view_as_real(stft_repr)
        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        # merge stereo / mono into frequency for band splitting
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')
        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        x = self.band_split(x)

        # ============================
        # PERF: band embedding computed once per forward
        # ============================
        b = x.shape[0]
        band_embedding = self.create_band_conditioning(b, device)  # (b*num_bands, dim)

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):

            if len(transformer_block) == 3:
                time_transformer, freq_transformer = transformer_block

                x, ft_ps = pack([x], 'b * d')
                x = time_transformer(x, band_embedding)
                x, = unpack(x, ft_ps, 'b * d')
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            b, t, f, d = x.shape

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            x = time_transformer(x, band_embedding)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            spk_f_cond = self.create_speaker_conditioning_for_fre_roformer(speaker_embedding, t)  # (b*t, d)

            x = freq_transformer(x, spk_f_cond)

            x, = unpack(x, ps, '* f d')

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)

        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        try:
            recon_audio = torch.istft(
                stft_repr,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=False,
                length=raw_audio.shape[-1]
            )
        except:
            recon_audio = torch.istft(
                stft_repr.cpu() if x_is_mps else stft_repr,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False,
                length=raw_audio.shape[-1]
            ).to(device)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        return recon_audio


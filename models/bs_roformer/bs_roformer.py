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
            drop_prob=0.5,
    ):
        super().__init__()
        self.layers = ModuleList([])
        self.adaLN_modulations = nn.ModuleList([]) 
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()
        self.drop_prob = drop_prob
        
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

        if self.training:
            # 50% drop all conditioning
            if torch.rand(1).item() <  self.drop_prob:
                band_embedding = torch.zeros_like(band_embedding)
            # 30% probability to drop condition (keep 70%)
            # mask = (torch.rand(band_embedding.shape[0], 1, device=band_embedding.device) > self.drop_prob).float()
            # band_embedding = band_embedding * mask

        for (norm1, attn, norm2, ff), ada in zip(self.layers, self.adaLN_modulations):
            shift_scale = ada(band_embedding)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = shift_scale.chunk(6, dim=-1)
            x = x + attn(modulate(norm1(x), shift_msa, scale_msa)) * gate_msa.unsqueeze(1)
            x = x + ff(modulate(norm2(x), shift_mlp, scale_mlp)) * gate_mlp.unsqueeze(1)
        return self.norm(x)
    
    def init_weights(self):
        """Initialize only adaLN_modulations."""
        GATE_BIAS_INIT = 1.0  # sigmoid(1.0)≈0.73，可让残差分支一开始就有梯度

        for ada in self.adaLN_modulations:
            linear = ada[1]  # SELU→Linear(dim,6*dim)

            # 全零权重、全零偏置
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)

            # 调整 gate 的 bias，让残差分支非完全关闭
            with torch.no_grad():
                D = linear.bias.shape[0] // 6
                linear.bias[2*D:3*D].fill_(GATE_BIAS_INIT)  # gate_msa
                linear.bias[5*D:6*D].fill_(GATE_BIAS_INIT)  # gate_mlp

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


class BandConditionalBSRoformer(Module):

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
            linear_transformer_depth=0,
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
            # in the paper, they divide into ~60 bands, test with 1 for starters
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            flash_attn=True,
            dim_freqs_in=1025,
            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=2,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            mlp_expansion_factor=4,
            use_torch_checkpoint=True,
            skip_connection=False,
            sage_attention=False,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
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
            if linear_transformer_depth > 0:
                tran_modules.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs))
            tran_modules.append(
                ScaleTransformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs)
            )
            tran_modules.append(
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs)
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

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_win_length), return_complex=True).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(
            freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        self.band_cond_embedding = BandEmbedder(len(freqs_per_bands), dim, dim)

        
        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )

            self.mask_estimators.append(mask_estimator)

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

    
    def create_band_conditioning(self, batch_size, device):
        """Create band conditioning embeddings.
        
        Args:
            batch_size (int): Batch size.
            device (torch.device): Device to create the embeddings on.
        
        Returns:
            torch.Tensor: Band conditioning embeddings of shape (batch_size, dim).
        """
        band_indices = torch.arange(len(self.freqs_per_bands), device=device)
        band_cond = self.band_cond_embedding(band_indices)  # (num_bands, dim)
        band_cond = band_cond.repeat(batch_size, 1, 1)  # (batch_size, num_bands, dim)
        band_cond, _ = pack([band_cond], '* d') # (batch_size * num_bands, dim)
        return band_cond

    
    def forward(
            self,
            raw_audio,
            target=None,
            return_loss_breakdown=False
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """
        
        batch = raw_audio.shape[0]
        device = raw_audio.device

        # defining whether model is loaded on MPS (MacOS GPU accelerator)
        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = self.stft_window_fn(device=device)

        # RuntimeError: FFT operations are only supported on MacOS 14+
        # Since it's tedious to define whether we're on correct MacOS version - simple try-catch is used
        try:
            stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        except:
            stft_repr = torch.stft(raw_audio.cpu() if x_is_mps else raw_audio, **self.stft_kwargs,
                                   window=stft_window.cpu() if x_is_mps else stft_window, return_complex=True).to(
                device)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        stft_repr = rearrange(stft_repr,'b s f t c -> b (f s) t c')

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # axial / hierarchical attention

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):

            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block

                x, ft_ps = pack([x], 'b * d')
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                x, = unpack(x, ft_ps, 'b * d')
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                # Sum all previous
                for j in range(i):
                    x = x + store[j]

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            band_embedding = self.create_band_conditioning(batch, device)
            
            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, band_embedding, use_reentrant=False)
            else:
                x = time_transformer(x, band_embedding)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            x, = unpack(x, ps, '* f d')

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        if self.use_torch_checkpoint:
            mask = torch.stack([checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1)
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)

        # modulate frequency representation

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        # istft

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        # same as torch.stft() fix for MacOS MPS above
        try:
            recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=raw_audio.shape[-1])
        except:
            recon_audio = torch.istft(stft_repr.cpu() if x_is_mps else stft_repr, **self.stft_kwargs, window=stft_window.cpu() if x_is_mps else stft_window, return_complex=False, length=raw_audio.shape[-1]).to(device)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.

        # for window_size in self.multi_stft_resolutions_window_sizes:
        #     res_stft_kwargs = dict(
        #         n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
        #         win_length=window_size,
        #         return_complex=True,
        #         window=self.multi_stft_window_fn(window_size, device=device),
        #         **self.multi_stft_kwargs,
        #     )

        #     recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
        #     target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

        #     multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        # weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        # total_loss = loss + weighted_multi_resolution_loss

        total_loss = loss  # temporarily disable multi stft loss

        if not return_loss_breakdown:
            return total_loss, recon_audio

        return total_loss, (loss, multi_stft_resolution_loss), recon_audio
    
    def freeze_transformer_layers(self):
        """Freeze all transformer blocks (time/freq/linear) in self.layers."""
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze_transformer_layers(self):
        """Unfreeze previously frozen transformer blocks in self.layers."""
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = True
                    
                    
                    
                
    
    
class BSRoformer(Module):

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
            linear_transformer_depth=0,
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
            # in the paper, they divide into ~60 bands, test with 1 for starters
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            flash_attn=True,
            dim_freqs_in=1025,
            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=2,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
            sage_attention=False,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection

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
            if linear_transformer_depth > 0:
                tran_modules.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs))
            tran_modules.append(
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs)
            )
            tran_modules.append(
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs)
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

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_win_length), return_complex=True).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(
            freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

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

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

    def forward(
            self,
            raw_audio,
            target=None,
            return_loss_breakdown=False
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device

        # defining whether model is loaded on MPS (MacOS GPU accelerator)
        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = self.stft_window_fn(device=device)

        # RuntimeError: FFT operations are only supported on MacOS 14+
        # Since it's tedious to define whether we're on correct MacOS version - simple try-catch is used
        try:
            stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        except:
            stft_repr = torch.stft(raw_audio.cpu() if x_is_mps else raw_audio, **self.stft_kwargs,
                                   window=stft_window.cpu() if x_is_mps else stft_window, return_complex=True).to(
                device)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        stft_repr = rearrange(stft_repr,'b s f t c -> b (f s) t c')

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # axial / hierarchical attention

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):

            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block

                x, ft_ps = pack([x], 'b * d')
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                x, = unpack(x, ft_ps, 'b * d')
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                # Sum all previous
                for j in range(i):
                    x = x + store[j]

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            x, = unpack(x, ps, '* f d')

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        if self.use_torch_checkpoint:
            mask = torch.stack([checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1)
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)

        # modulate frequency representation

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        # istft

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        # same as torch.stft() fix for MacOS MPS above
        try:
            recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=raw_audio.shape[-1])
        except:
            recon_audio = torch.istft(stft_repr.cpu() if x_is_mps else stft_repr, **self.stft_kwargs, window=stft_window.cpu() if x_is_mps else stft_window, return_complex=False, length=raw_audio.shape[-1]).to(device)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss, recon_audio

        return total_loss, (loss, multi_stft_resolution_loss), recon_audio
    
    def freeze_transformer_layers(self):
        """Freeze all transformer blocks (time/freq/linear) in self.layers."""
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze_transformer_layers(self):
        """Unfreeze previously frozen transformer blocks in self.layers."""
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = True
                    
                    
                    
                
                
    
class SpeakerGuideBSRoformer(Module):

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
            linear_transformer_depth=0,
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
            # in the paper, they divide into ~60 bands, test with 1 for starters
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            flash_attn=True,
            dim_freqs_in=1025,
            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=2,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
            sage_attention=False,
            spk_embd_dim=192,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection

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
            if linear_transformer_depth > 0:
                tran_modules.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs))
            tran_modules.append(
                ScaleTransformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs)
            )
            tran_modules.append(
                ScaleTransformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs)
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

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_win_length), return_complex=True).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(
            freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

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

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

        self.spk_embd_to_dim = nn.Linear(spk_embd_dim, dim)  # assuming speaker embedding dim is 256
        
    def create_speaker_conditioning_for_fre_roformer(self, speaker_embedding, num_time_steps):
        if speaker_embedding is None:
            return None
        # construct (b*t) * d speaker conditioning for (b*t) * f * d input
        speaker_embedding = self.spk_embd_to_dim(speaker_embedding)  # (b, d)
        speaker_embedding = speaker_embedding.unsqueeze(1).repeat(1, num_time_steps, 1)  # (b, t, d)
        speaker_embedding, _ = pack([speaker_embedding], '* d')  # (b * t, d)
        return speaker_embedding
    
    def create_speaker_conditioning_for_time_roformer(self, speaker_embedding, num_bands):
        if speaker_embedding is None:
            return None
        # construct (b*f) * d speaker conditioning for (b*f) * t * d input
        speaker_embedding = self.spk_embd_to_dim(speaker_embedding)  # (b, d)
        speaker_embedding = speaker_embedding.unsqueeze(1).repeat(1, num_bands, 1)  # (b, f, d)
        speaker_embedding, _ = pack([speaker_embedding], '* d')  # (b * f, d)
        return speaker_embedding
    
    def forward(
            self,
            raw_audio,
            speaker_embedding=None,
            target=None,
            return_loss_breakdown=False
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        
        speaker_embedding: (b, dim)
        """

        device = raw_audio.device

        # defining whether model is loaded on MPS (MacOS GPU accelerator)
        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = self.stft_window_fn(device=device)

        # RuntimeError: FFT operations are only supported on MacOS 14+
        # Since it's tedious to define whether we're on correct MacOS version - simple try-catch is used
        try:
            stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        except:
            stft_repr = torch.stft(raw_audio.cpu() if x_is_mps else raw_audio, **self.stft_kwargs,
                                   window=stft_window.cpu() if x_is_mps else stft_window, return_complex=True).to(
                device)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        stft_repr = rearrange(stft_repr,'b s f t c -> b (f s) t c')

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # axial / hierarchical attention

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):

            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block

                x, ft_ps = pack([x], 'b * d')
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                x, = unpack(x, ft_ps, 'b * d')
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                # Sum all previous
                for j in range(i):
                    x = x + store[j]

            b, t, f, d = x.shape
            
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            
            spk_t_cond = self.create_speaker_conditioning_for_time_roformer(speaker_embedding, f)  # (b * f, d)
            
            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, spk_t_cond, use_reentrant=False)
            else:
                x = time_transformer(x, spk_t_cond)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            spk_f_cond = self.create_speaker_conditioning_for_fre_roformer(speaker_embedding, t)  # (b * t, d)
            
            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, spk_f_cond, use_reentrant=False)
            else:
                x = freq_transformer(x, spk_f_cond)

            x, = unpack(x, ps, '* f d')

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        if self.use_torch_checkpoint:
            mask = torch.stack([checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1)
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)

        # modulate frequency representation

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        # istft

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        # same as torch.stft() fix for MacOS MPS above
        try:
            recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=raw_audio.shape[-1])
        except:
            recon_audio = torch.istft(stft_repr.cpu() if x_is_mps else stft_repr, **self.stft_kwargs, window=stft_window.cpu() if x_is_mps else stft_window, return_complex=False, length=raw_audio.shape[-1]).to(device)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss, recon_audio

        return total_loss, (loss, multi_stft_resolution_loss), recon_audio
    
    def freeze_transformer_layers(self):
        """Freeze all transformer blocks (time/freq/linear) in self.layers."""
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze_transformer_layers(self):
        """Unfreeze previously frozen transformer blocks in self.layers."""
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = True






    

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
            linear_transformer_depth=0,
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
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
            sage_attention=False,
            spk_embd_dim=192,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
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
            if linear_transformer_depth > 0:
                tran_modules.append(
                    Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs)
                )
            tran_modules.append(
                ScaleTransformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    drop_prob=0.0,
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

        # multi-resolution stft loss (training only)
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

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
            target=None,
            return_loss_breakdown=False
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

        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # ============================
        # PERF: band embedding computed once per forward
        # ============================
        b = x.shape[0]
        band_embedding = self.create_band_conditioning(b, device)  # (b*num_bands, dim)

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):

            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block

                x, ft_ps = pack([x], 'b * d')
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                x, = unpack(x, ft_ps, 'b * d')
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            b, t, f, d = x.shape

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            # PERF: 复用 band_embedding（不再每层 create）
            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, band_embedding, use_reentrant=False)
            else:
                x = time_transformer(x, band_embedding)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            spk_f_cond = self.create_speaker_conditioning_for_fre_roformer(speaker_embedding, t)  # (b*t, d)

            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, spk_f_cond, use_reentrant=False)
            else:
                x = freq_transformer(x, spk_f_cond)

            x, = unpack(x, ps, '* f d')

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        if self.use_torch_checkpoint:
            mask = torch.stack([checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1)
        else:
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

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.
        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss, recon_audio

        return total_loss, (loss, multi_stft_resolution_loss), recon_audio

    def freeze_transformer_layers(self):
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze_transformer_layers(self):
        for layer_group in self.layers:
            for module in layer_group:
                for param in module.parameters():
                    param.requires_grad = True







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
            **args
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
                ScaleTransformer(depth=1, rotary_embed=time_rotary_embed, drop_prob=0.,**transformer_kwargs),
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





class SpeakerRoformerExportable(Module):
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
            **args
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

        spk_emb = self.spk_embd_to_dim(speaker_embedding)
        spk_f_cond = spk_emb.unsqueeze(1).expand(b, t, -1).reshape(b * t, -1)
        spk_t_cond = spk_emb.unsqueeze(1).expand(b, f, -1).reshape(b * f, -1)

        store = [None] * len(self.layers)
        for i, (time_transformer, freq_transformer) in enumerate(self.layers):
            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            # Time transformer: (b, t, f, d) -> (b*f, t, d)
            # 合并 permute + reshape: (b, t, f, d) -> (b, f, t, d) -> (b*f, t, d)
            x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
            x = time_transformer(x, spk_t_cond)
            
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




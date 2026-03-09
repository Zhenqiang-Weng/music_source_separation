import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
from .modules import Conv1d, ZeroTemporalPad
from .constants import ACTIVATIONS, NORMALIZATIONS

class ResidualBlock(nn.Module):
    """
    Implements conv -> PReLU -> norm n-times
    """

    def __init__(self, channels, kernel_size, dilation, n=2, causal=False, norm='freq', activation='relu'):
        super(ResidualBlock, self).__init__()

        self.blocks = [
            nn.Sequential(
                Conv1d(channels, channels, kernel_size, dilation=dilation),
                ZeroTemporalPad(kernel_size, dilation, causal=causal),
                ACTIVATIONS[activation](),
                NORMALIZATIONS[norm](channels), # Normalize after activation. if we used ReLU, half of out nurons would be dead!
            )
            for i in range(n)
        ]

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return x + self.blocks(x)

class GatedConvBlock(nn.Module):
    """
    Implements conv -> PReLU -> norm -> GLU
    """

    def __init__(self, channels, kernel_size, dilation, causal=False, norm='freq', activation='relu'):
        super(ResidualBlock, self).__init__()

        self.blocks = [
            nn.Sequential(
                Conv1d(channels, 2*channels, kernel_size, dilation=dilation),
                ZeroTemporalPad(kernel_size, dilation, causal=causal),
                ACTIVATIONS[activation](),
                NORMALIZATIONS[norm](2*channels),
                # Normalize after activation. if we used ReLU, half of out nurons would be dead!
            )
        ]
        self.blocks.extend([nn.GLU(dim=1)])

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return x + self.blocks(x)


class WaveResidualBlock(nn.Module):
    """A residual gated block based on WaveNet
                        |-------------------------------------------------------------|
                        |                                                             |
                        |                        |-- conv -- tanh --|                 |
          residual ->  -|--(pos_enc)--(dropout)--|                  * ---|--- 1x1 --- + --> residual
                                                 |-- conv -- sigm --|    |
                                                                        1x1
                                                                         |
          -------------------------------------------------------------> + ------------> skip
    """

    def __init__(self, residual_channels, block_channels, kernel_size, dilation_rate, causal=True, dropout=False,
                 skip_channels=False):
        """
        :param residual_channels: Num. of channels for resid. connections between wave blocks
        :param block_channels: Num. of channels used inside wave blocks
        :param kernel_size: Num. of branches for each convolution kernel
        :param dilation_rate: Hom much to dilate inputs before applying gate and filter
        :param causal: If causal, input is zero padded from the front, else both sides are zero padded equally
        :param dropout: If dropout>0, apply dropout on the input to the block gates (not the residual connection)
        :param skip_channels: If >0, return also skip (batch, time, skip_channels)
        """
        super(WaveResidualBlock, self).__init__()

        self.pad = ZeroTemporalPad(kernel_size, dilation_rate, causal=causal)
        self.causal = causal
        self.receptive_field = dilation_rate * (kernel_size - 1) + 1

        # tanh and sigmoid applied in forward
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.filter = Conv1d(residual_channels, block_channels, kernel_size, dilation=dilation_rate)
        self.gate = Conv1d(residual_channels, block_channels, kernel_size, dilation=dilation_rate)

        self.conv1x1_resid = Conv1d(block_channels, residual_channels, 1)
        self.conv1x1_skip = Conv1d(block_channels, skip_channels, 1) if skip_channels else None

        self.tensor_q = None
        self.generate = False

    def forward(self, residual):
        """Feed residual through the WaveBlock

        Allows layer-level caching for faster sequential inference.
        See https://github.com/tomlepaine/fast-wavenet for similar tensorflow implementation and original paper.

        Non - causal version does not support iterative generation for obvious reasons.
        WARNING: generating must be called before each generated sequence!
        Otherwise there will be an error due to stored queue from previous run.

        RuntimeError: Trying to backward through the graph a second time,
        but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.

        :param residual: Residual from previous block or from input_conv, (batch_size, channels, time_dim)
        :return: residual, skip
        """

        if self.generate and self.causal:
            if self.tensor_q is None:
                x = self.pad(residual)
                self.tensor_q = x[:, -self.receptive_field:, :].detach()
            else:
                assert residual.shape[
                           1] == 1, f'Expected residual.shape[1] == 1 during generation, but got residual.shape[1]={residual.shape[1]}'

                x = torch.cat((self.tensor_q, residual), dim=1)[:, -self.receptive_field:, :]
                self.tensor_q = x.detach()
        else:
            x = self.pad(residual)

        if self.dropout is not None:
            x = self.dropout(x)
        filter = torch.tanh(self.filter(x))
        gate = torch.sigmoid(self.gate(x))
        out = filter * gate
        residual = self.conv1x1_resid(out) + residual

        if self.conv1x1_skip is not None:
            return residual, self.conv1x1_skip(out)
        else:
            return residual

    def generating(self, mode):
        """Call before and after generating"""
        self.generate = mode
        self.reset_queue()

    def reset_queue(self):
        self.tensor_q = None


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean)**2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p_dropout=0., activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv(x * x_mask)
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        return x * x_mask

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


class RelativeFFTBlock(nn.Module):
    """ FFT Block with Relative Multi-Head Attention """

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=None, block_length=None):
        super(RelativeFFTBlock, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(RelativeSelfAttention(hidden_channels, hidden_channels, n_heads,
                                    window_size=window_size, p_dropout=p_dropout, block_length=block_length))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(
                hidden_channels, hidden_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class RelativeSelfAttention(nn.Module):
    """ Relative Multi-Head Attention """

    def __init__(self, channels, out_channels, n_heads, window_size=None, heads_share=True, p_dropout=0., block_length=None, proximal_bias=False, proximal_init=False):
        super(RelativeSelfAttention, self).__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(torch.randn(
                n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(
                n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels,
                           t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels,
                           t_s).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(self.k_channels)
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(
                rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + \
                self._attention_bias_proximal(t_s).to(
                    device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                block_mask = torch.ones_like(
                    scores).triu(-self.block_length).tril(self.block_length)
                scores = scores * block_mask + -1e4*(1 - block_mask)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(
                p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s)
            output = output + \
                self._matmul_with_relative_values(
                    relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(
            b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:,
                                                              slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, convert_pad_shape(
            [[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape(
            [[0, 0], [0, 0], [0, length-1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view(
            [batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, convert_pad_shape(
            [[0, 0], [0, 0], [0, 0], [0, length-1]]))
        x_flat = x.view([batch, heads, length**2 + length*(length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, convert_pad_shape(
            [[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2*length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """
        Bias for self-attention to encourage attention to close positions.
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class WordToPhonemeAttention(nn.Module):
    """ Word-to-Phoneme Multi-Head Attention """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.0):
        super(WordToPhonemeAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = LinearNorm(d_model, n_head * d_k)
        self.w_ks = LinearNorm(d_model, n_head * d_k)
        self.w_vs = LinearNorm(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        # self.layer_norm = nn.LayerNorm(d_model)

        self.fc = LinearNorm(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, key_mask=None, query_mask=None, mapping_mask=None, indivisual_attn=False, attn_prior=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        if key_mask is not None:
            key_mask = key_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        if query_mask is not None:
            query_mask = query_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        if mapping_mask is not None:
            mapping_mask = mapping_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        if attn_prior is not None:
            attn_prior = attn_prior.repeat(n_head, 1, 1)
        output, attns, attn_logprob = self.attention(
            q, k, v, key_mask=key_mask, query_mask=query_mask, mapping_mask=mapping_mask, attn_prior=attn_prior)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual
        # output = self.layer_norm(output)

        if indivisual_attn:
            attns = tuple([attn.view(n_head, sz_b, len_q, len_k) for attn in attns])
            attn_logprob = attn_logprob.view(n_head, sz_b, 1, len_q, len_k)

        return output, attns, attn_logprob


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, q, k, v, key_mask=None, query_mask=None, mapping_mask=None, attn_prior=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if key_mask is not None:
            attn = attn.masked_fill(key_mask == 0., -np.inf)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior.transpose(1, 2) + 1e-8)
        attn_logprob = attn.unsqueeze(1).clone()

        attn = self.softmax(attn)

        if query_mask is not None:
            attn = attn * query_mask
        attn_raw = attn.clone()
        if mapping_mask is not None:
            attn = attn * mapping_mask
        output = torch.bmm(attn, v)

        return output, (attn, attn_raw), attn_logprob

class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x

def word_level_pooling(src_seq, src_len, wb, src_w_len, reduce="sum"):
    """
    src_seq -- [batch_size, max_time, dim]
    src_len -- [batch_size,]
    wb -- [batch_size, max_time]
    src_w_len -- [batch_size,]
    """
    batch, device = [], src_seq.device
    for s, sl, w, wl in zip(src_seq, src_len, wb, src_w_len):
        m, split_size = s[:sl, :], list(w[:wl].int())
        m = nn.utils.rnn.pad_sequence(torch.split(m, split_size, dim=0))
        if reduce == "sum":
            m = torch.sum(m, dim=0)  # [src_w_len, hidden]
        elif reduce == "mean":
            m = torch.div(torch.sum(m, dim=0), torch.tensor(
                split_size, device=device).unsqueeze(-1))  # [src_w_len, hidden]
        else:
            raise ValueError()
        batch.append(m)
    return pad(batch).to(device)
def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

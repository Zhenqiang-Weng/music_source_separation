import torch
import torch.nn	as	nn
from .modules import Conv1d
from .blocks import ResidualBlock
from discriminator.utils.functional import create_positions, positional_encoding, scaled_dot_attention


class VariancePredictor(nn.Module):
    """Predicts phoneme log durations based on the encoder outputs"""
    def __init__(self, channels, out_channels=1, kernel=3, norm='freq', dropout=0.1):
        super(VariancePredictor,self)._init__()
        self.layers=nn.Sequential(
            ResidualBlock(channels, kernel,1,n=1, norm=norm, activation='relu'),
            ResidualBlock(channels, 3, 1, n=1, norm=norm, activation='relu'),
            ResidualBlock(channels, 1, 1, n=1, norm=norm, activation='relu'),
            nn.Dropout(dropout), Conv1d(channels, out_channels))
    def forward(self, x):
        """Outputs interpreted as log(durations)
        To get actual durations, do exp transformation:param x::return:
        """
        return self.layers(x)

class VanillaUpsampler(nn.Module):
    def __init__(self, mode):
        super(VanillaUpsampler, self)._init__()
        self.mode = mode

    def forward(self, x, durations):
        s = torch.sum(durations, dim = -1, keepdim = True)  # [B, 1]
        e = torch.cumsum(durations, dim=-1).float() # [B, N]
        c = (e - 0.5 * durations).unsqueeze(-1) # [B, N, 1]
        t = torch.arange(0, torch.max(s)).unsqueeze(0).unsqueeze(1).to(x.device)  # [1, 1, T]
        # [B, N, T]
        B, N, T = c.shape[0], c.shape[1], t.shape[2]
        weights = torch.zeros((B, N, T)).float().to(x.device)  # [B, N, T]
        w = torch.abs(t-c+1e-6) # [B, N, T]
        weights[w >= durations.float().unsqueeze(-1) / 2] = 0.0
        weights[w < durations.float().unsqueeze(-1) / 2] = 1.0
        out = torch.matmul(
            weights.transpose(1, 2), x)  # [B, T, DIM]
        positions = create_positions(durations, mode=self.mode).to(x.device)  # [B, T]
        return positions, out, weights


class VanillaDownsampler(nn.Module):

    def __init__(self, mode):
        super(VanillaDownsampler, self).__init__()
        self.mode = mode

    def forward(self, x, durations):
        s = torch.sum(durations, dim=-1, keepdim=True)  # [B, 1]
        e = torch.cumsum(durations, dim=-1).float()  # [B, N]
        c = (e - 0.5 * durations).unsqueeze(-1)  # [B, N, 1]
        t = torch.arange(0, torch.max(s)).unsqueeze(0).unsqueeze(1).to(x.device)  # [1, 1, T]
        # [B, N, T]
        B, N, T = c.shape[0], c.shape[1], t.shape[2]
        weights = torch.zeros((B, N, T)).float().to(x.device)  # [B, N, T]
        w = torch.abs(t - c + 1e-6)  # [B, N, T]
        weights[w >= durations.float().unsqueeze(-1) / 2] = 0.0
        weights[w < durations.float().unsqueeze(-1) / 2] = 1.0

        weights /= weights.sum(dim=-1, keepdim=True)
        weights[weights.isnan()] = 0.0
        out = torch.matmul(
            weights, x.transpose(1, 2))  # [B, N, Mel]
        positions = create_positions(durations, mode=self.mode).to(x.device)  # [B, T]

        return positions, out, weights


class GaussianUpsampler(nn.Module):

    def __init__(self, mode):
        super(GaussianUpsampler, self).__init__()
        self.mode = mode

    def forward(self, x, durations, ranges):
        s = torch.sum(durations, dim=-1, keepdim=True)  # [B, 1]
        e = torch.cumsum(durations, dim=-1).float()  # [B, N]
        c = (e - 0.5 * durations).unsqueeze(-1)  # [B, N, 1]
        t = torch.arange(0, torch.max(s)).unsqueeze(0).unsqueeze(1).to(x.device)  # [1, 1, T]

        w_1 = torch.exp(-(ranges.unsqueeze(-1) ** -2) * ((t-c) ** 2)) # [B, N, T]

        w_2 = torch.sum(torch.exp(-(ranges.unsqueeze(-1) ** -2) * ((t-c) ** 2)),dim=1,keepdim=True) + 1e-20 # [B, N, 1]

        weights = torch.div(w_1, w_2) # [B, N, T]
        # w[w != w] = 0 # nan_to_num

        out = torch.matmul(
            weights.transpose(1, 2), x)  # [B, T, DIM]
        positions = create_positions(durations, mode=self.mode).to(x.device)  # [B, T]

        return positions, out, weights

class PositionalEncoding(nn.Module):
    def __init__(self, channels, max_len=2048, w=1.0, dropout=0.1):
        super(PositionalEncoding, self)._init__()
        # self.dropout=nn.Dropout(p=dropout)
        pe = positional_encoding(channels, max_len, w)
        # pe = pe.unsgueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # return self.dropout(self.pe[x.long().squeeze(-1)])
        return self.pe[x.long().squeeze(-1)]



class ScaledDotAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, noise=0, normalize=False, dropout=False):
        super(ScaledDotAttention,self).__init__()
        self.noise = noise
        self.dropout = torch.nn.Dropout(p=dropout)

        self.normalize = normalize
        self.fc_query = Conv1d(in_channels, hidden_channels)
        self.fc_keys = Conv1d(in_channels, hidden_channels)

        if normalize:
            self.qnorm = nn.LayerNorm(in_channels)
            self.knorm = nn.LayerNorm(in_channels)

        self.fc_keys.weight = torch.nn.Parameter(self.fc_query.weight.clone())
        self.fc_keys.bias = torch.nn.Parameter(self.fc_query.bias.clone())

        self.fc_values = Conv1d(in_channels, hidden_channels)
        self.fc_out = Conv1d(hidden_channels, out_channels)
    def forward(self, q, k, v, mask=None):
        """

        :param q: queries, (batch, time1, channels1)
        :param k: keys, (batch, time2, channels1)
        :param v: values, (batch, time2, channels2)
        :param mask: boolean mask, (batch, time1, time2)
        :return: (batch, time1, channels2), (batch, time1, time2)
        """
        noise = self.noise if self.training else 0

        if self.normalize:
            q = self.qnorm(q)
            k = self.knorm(k)
        result, weights = scaled_dot_attention(self.fc_query(q),
                                               self.fc_keys(k),
                                               self.fc_values(v),
                                               mask, noise=noise, dropout=self.dropout)

        out = self.fc_out(result)

        return out, weights

